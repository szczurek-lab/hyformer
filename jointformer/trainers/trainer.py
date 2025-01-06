import os
import time
import math
import torch
import random
import json
import logging

from torch import nn
import numpy as np
from typing import Optional, Any
from contextlib import nullcontext
from torch.distributions.categorical import Categorical

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from jointformer.configs.trainer import TrainerConfig
from jointformer.models.transformer import Transformer
from jointformer.utils.loggers.wandb import WandbLogger
from jointformer.utils.datasets.base import BaseDataset

from jointformer.utils.runtime import set_seed
from jointformer.utils.collator import DataCollator
from jointformer.utils.chemistry import is_valid

from jointformer.trainers.utils import get_test_metric

console = logging.getLogger(__name__)
SNAPSHOT_FILENAME = 'snapshot.pt'
MODEL_FILENAME = 'ckpt.pt'


class Trainer:
    """Trainer for a Transformer model.

    Adapted from: https://github.com/karpathy/nanoGPT/blob/master/train.py

    """
    def __init__(
            self,
            config: TrainerConfig,
            model: Transformer,
            device: torch.device,
            out_dir: Optional[str] = None,
            seed: Optional[int] = 1337,
            train_dataset: Optional[BaseDataset] = None,
            val_dataset: Optional[BaseDataset] = None,
            test_dataset: Optional[BaseDataset] = None,
            tokenizer: Optional[Any] = None,
            logger: Optional[WandbLogger] = None,
            test_metric: Optional[str] = None
    ):

        # set args
        self.config = config
        self.model = model
        self.device = device
        self.out_dir = out_dir
        self.seed = seed
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.logger = logger
        self.test_metric = test_metric
        
        # Trainer State
        self._loss_dict = {}
        self._iter_num = 0
        self._best_val_loss = 1e9
        self._optuna_loss = 1e9 if hasattr(self.model, 'predict') else None 
        self._optuna_loss = 0.0 if hasattr(self.model, 'predict') and self.test_metric in ['roc_auc', 'prc_auc'] else self._optuna_loss
        self._snapshot_filepath = os.path.join(self.out_dir, SNAPSHOT_FILENAME) if self.out_dir else None
        self._learning_rate = None
        self._running_mfu = 0.0
        self._resumed_from_iter_num = 0

        self._is_distributed_run = dist.is_initialized()
        self._master_process = True if int(os.environ.get('LOCAL_RANK', 0)) == 0 else False
        
        torch.cuda.set_device(self.device)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        
        # Set up model and optimizer
        set_seed(self.seed)
        self.model.to(self.device)
        self.optimizer = self.model.configure_optimizers(self.config.weight_decay, self.config.learning_rate, (self.config.beta1, self.config.beta2), self.device)
        self.model = DDP(model, device_ids=[device], find_unused_parameters=False) if self.config.enable_ddp and dist.is_initialized() else self.model
        self.model = torch.compile(self.model) if self.config.compile else self.model

        self.task_distribution = Categorical(torch.Tensor(list(self.config.tasks.values())))

        # Miscellanuous
        self.ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.config.dtype]
        if self.tokenizer is not None and hasattr(self.tokenizer, '__len__') and hasattr(self.model, 'vocab_size'):
            if len(self.tokenizer) != self.model.vocab_size:
                raise ValueError(f"Tokenizer and model not compatible. Tokenizer is of length {len(self.tokenizer)}"
                                 f" while model expects vocab size {self.model.vocab_size}")
        
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
            device_type=device_type, dtype=self.ptdtype)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.config.dtype == 'float16'))
        self._init_data_loaders()

    def resume_snapshot(self):
        self.resume_from_file(self._snapshot_filepath, resume_training=True)

    def resume_from_file(self, filepath, resume_training=False):
        checkpoint = torch.load(filepath, map_location=self.device)
        state_dict = checkpoint['model']
        if not any(key.startswith("_orig_mod") for key in self.model.state_dict().keys()):
            unwanted_prefix = '_orig_mod.'  # compile artifacts
            for k, _ in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)    
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            print("Model state_dict loaded with strict=False.")
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")

        if resume_training:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            except:
                console.warning("Optimizer state not found in checkpoint. Initializing optimizer from scratch.")
            self._iter_num = checkpoint['iter_num']
            self._best_val_loss = checkpoint['best_val_loss']
            self._loss_dict = checkpoint['loss_dict']
            self._resumed_from_iter_num = self._iter_num
            if self.logger is not None:
                self.logger.set_run_id(checkpoint['run_id'] if 'run_id' in checkpoint else None)
            print(f"Resuming training from iteration {self._iter_num} with best validation loss {self._best_val_loss:.4f}")
        checkpoint = None

    def _save_ckpt(self, filename: str):
        if self.out_dir is not None and self._master_process and self.config.save_checkpoint:
            run_id = self.logger.run_id if self.logger is not None else None
            checkpoint = {
                'model': self.model.module.state_dict() if dist.is_initialized() else self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'iter_num': self._iter_num,
                'best_val_loss': self._best_val_loss,
                'loss_dict': self._loss_dict,
                'run_id': run_id
            }    
            torch.save(checkpoint, os.path.join(self.out_dir, filename))

    def _get_data_loader(self, dataset: torch.utils.data.dataset.Dataset, shuffle=True):
        collator = DataCollator(tokenizer=self.tokenizer, tasks=self.config.tasks)
        sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=int(os.environ["WORLD_SIZE"]),
                rank=int(os.environ["SLURM_PROCID"])) if self._is_distributed_run else None
        return  torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.config.batch_size,
                    shuffle=shuffle if sampler is None else False,
                    collate_fn=collator,
                    sampler=sampler,
                    num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)),
                    pin_memory=True,
                    persistent_workers=False
                )
    
    def _init_data_loaders(self):
        if self.train_dataset is not None:
            self.train_loader = self._get_data_loader(self.train_dataset, shuffle=True)
        if self.val_dataset is not None:
            self.val_loader = self._get_data_loader(self.val_dataset, shuffle=False)
        if self.test_dataset is not None:
            self.test_loader = self._get_data_loader(self.test_dataset, shuffle=False)

    def get_training_batch(self):
        if self._is_distributed_run:
            self.train_loader.sampler.set_epoch(self._iter_num)
        return next(iter(self.train_loader)).to(self.device)

    def get_validation_batch(self):
        return next(iter(self.val_loader)).to(self.device)

    def get_batch(self, split, task):
        batch = self._sample(self.train_dataset, task) if split == 'train' else self._sample(self.val_dataset, task)
        return batch.to(self.device)
        
    def _sample(self, dataset, task):
        idx = [idx for idx in range(len(dataset))]
        idx = random.sample(idx, min(self.config.batch_size, len(idx)))
        sampled = [dataset[i] for i in idx]
        return self.tokenizer(sampled, task=task)

    @torch.no_grad()
    def test(self, metric=None):
        metric = self.test_metric if metric is None else metric
        assert metric in ['rmse', 'roc_auc', 'prc_auc'], f"Metric {metric} not supported."

        self.model.eval()

        y_true = None
        y_pred = None

        for _, batch in enumerate(self.test_loader):
            
            y_true = batch['properties'] if y_true is None else torch.cat((y_true, batch['properties']))
            
            batch = batch.to(self.device)
            _y_pred = self.model.predict(**batch).cpu()
            y_pred = _y_pred if y_pred is None else torch.cat((y_pred, _y_pred))

        if metric == 'rmse':    
            if self.test_dataset.target_transform is not None:
                y_true = self.test_dataset.target_transform.inverse_transform(y_true)
                y_pred = self.test_dataset.target_transform.inverse_transform(y_pred)
            else:
                print("No target transform found. Assuming target is not transformed.")
        
        assert y_true.shape == y_pred.shape, f"Shapes of y_true and y_pred do not match: {y_true.shape} and {y_pred.shape}."
        test_metric = get_test_metric(y_true.numpy(), y_pred.numpy(), metric)
        
        if self.logger is not None:
            self.logger.init_run()
            self.logger.log({f"test/{metric}": test_metric})
            
        return test_metric

    @torch.no_grad()
    def estimate_loss(self):

        self.model.eval()
        out = {}
        splits = []
        if self.train_dataset:
            splits.append('train')
        if self.val_dataset:
            splits.append('val')
        tasks = list(self.config.tasks.keys())

        for split in splits:
            out[split] = {}
            for task in tasks:
                losses = torch.zeros(self.config.eval_iters)
                for k in range(self.config.eval_iters):
                    inputs = self.get_batch(split, task)
                    with self.ctx:
                        outputs = self.model.module.get_loss(**inputs) if dist.is_initialized() else self.model.get_loss(**inputs)
                    losses[k] = outputs["loss"].item() if outputs["loss"] is not None else torch.nan
                out[split][task] = losses.mean().item() if torch.nan not in losses else torch.nan

        for split in splits:
            for task in tasks:
                if 'combined' in out[split]:
                    out[split]['combined'] += out[split][task]
                else:
                    out[split]['combined'] = out[split][task]

        # if hasattr(self.model, 'calculate_perplexity') and self.eval_generation:
        #     for split in splits:
        #         out[split]['perplexity'] = {}
        #         losses = torch.zeros(self.config.eval_iters)
        #         for k in range(self.config.eval_iters):
        #             inputs = self.get_batch(split, task='generation')
        #             with self.ctx:
        #                 perplexity = self.model.calculate_perplexity(**inputs)
        #             losses[k] = perplexity.mean()
        #         out[split]['perplexity'] = losses.mean().item() if torch.nan not in losses else torch.nan

        if hasattr(self.model, 'generate') or hasattr(self.model.module, 'generate'):
            samples = []
            for _ in range(self.config.eval_iters):
                samples.extend(self.generate())
            if self.logger:
                self.logger.log_molecule_data(samples)
            is_valid_batch = [is_valid(sample) for sample in samples]
            out["val"]["validity"] = sum(is_valid_batch) / len(is_valid_batch)
            out["val"]["uniqueness"] = len(set(samples)) / len(samples)
            out["val"]["novelty"] = len(set(samples) - set(self.train_dataset.data)) / len(samples)
        self.model.train()
        return out

    def _get_lr(self):
        # 1) linear warmup for warmup_iters steps
        if self._iter_num < self.config.warmup_iters:
            return self.config.learning_rate * self._iter_num / self.config.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if self._iter_num > self.config.lr_decay_iters:
            return self.config.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (self._iter_num - self.config.warmup_iters) / (self.config.lr_decay_iters - self.config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)

    def _set_lr(self) -> None:
        self._learning_rate = self._get_lr() if self.config.decay_lr else self.config.learning_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._learning_rate

    def evaluate(self):
        if self._iter_num % self.config.eval_interval == 0 and (self._resumed_from_iter_num != self._iter_num or self._iter_num ==  0):
            if self._optuna_loss is not None and hasattr(self, "test_loader"):
                _optuna_running_loss = self.test(metric=self.test_metric)
                if self.test_metric in ['rmse']:            
                    self._optuna_loss = min(self._optuna_loss, _optuna_running_loss)
                elif self.test_metric in ['roc_auc', 'prc_auc']:
                    self._optuna_loss = max(self._optuna_loss, _optuna_running_loss)
                else:
                    raise ValueError(f"Test metric {self.test_metric} not supported.")
                console.info(f"Optuna loss: {self._optuna_loss:.4f}")
                
            losses = self.estimate_loss()
            self._loss_dict[self._iter_num] = losses
            info = f"Evaluation at step {self._iter_num}"
            if 'train' in losses:
                info += f": train loss {losses['train']['combined']:.4f}"
            if 'val' in losses:
                info += f", val loss {losses['val']['combined']:.4f}"
            console.info(info)
            if self.out_dir:
                with open(os.path.join(self.out_dir, 'loss_dict.json'), 'w') as fp:
                    json.dump(self._loss_dict, fp, indent=4)

            if self.logger:
                log_dict = {}
                for split in losses.keys():
                    for task in losses[split].keys():
                        log_dict[f'{split}/{task}'] = losses[split][task]
                log_dict['iter'] = self._iter_num
                log_dict['lr'] = self._learning_rate
                self.logger.log(log_dict)

            if self._iter_num > 0: # More logging here
                if 'val' in losses: # save checkpoint if validation loss is better
                    console.info(f"Validation loss: {losses['val']['combined']:.4f}")
                    console.info(f"Best validation loss: {self._best_val_loss:.4f}")
                    if losses['val']['combined'] < self._best_val_loss or self.config.always_save_checkpoint:
                        self._best_val_loss = losses['val']['combined']
                        self._save_ckpt(MODEL_FILENAME)
                        console.info(f"Checkpoint updated at iteration {self._iter_num}")
                    
                if self.config.save_checkpoint_every is not None: # save checkpoint every n iterations
                    if self._iter_num % self.config.save_checkpoint_every == 0:
                        self._save_ckpt(f"ckpt_{self._iter_num}.pt")

    def _terminate(self):
        if self._iter_num > self.config.max_iters:
            return True
        return False

    @torch.no_grad()
    def generate(self, temperature=1.0, top_k=25):
        original_model = self.model.module if dist.is_initialized() else self.model
            
        samples = original_model.generate(
            tokenizer=self.tokenizer,
            batch_size=self.config.batch_size,
            temperature = temperature,
            top_k = top_k,
            device = self.device)
        samples = self.tokenizer.decode(samples)
        return samples

    def train(self) -> None:

        if self._iter_num > self.config.max_iters:
            return

        if self.logger is not None and self._master_process:
            self.logger.init_run()
            self.logger.watch_model(self.model)

        inputs = self.get_training_batch()
        local_iter_num = 0  # number of iterations in the lifetime of this process

        start_timer = torch.cuda.Event(enable_timing=True)
        end_timer = torch.cuda.Event(enable_timing=True)
        while True:
            if self._terminate():
                if self.logger is not None:
                    self.logger.finish()
                break
                
            self._set_lr()
            if self._master_process:
                self.evaluate()
            if dist.is_initialized():
                dist.barrier()
            
            if self._iter_num == 0 and self.config.eval_only:
                if self.logger is not None:
                    self.logger.finish()
                break
            
            ### Training step
            if self._iter_num % self.config.log_interval == 0 and self._master_process and local_iter_num >= 10:
                start_timer.record()
            for micro_step in range(self.config.gradient_accumulation_steps): # gradient accumulation loop
                if self._is_distributed_run:
                    self.model.require_backward_grad_sync = (micro_step == self.config.gradient_accumulation_steps - 1)  # in ddp mode, only sync grads at the last micro-step
                with self.ctx:
                    outputs = self.model.module.get_loss(**inputs) if dist.is_initialized() else self.model.get_loss(**inputs)
                    loss = outputs["loss"] / self.config.gradient_accumulation_steps  # scale the loss to account for gradient accumulation
                inputs = self.get_training_batch()  # async prefetch next batch
                self.scaler.scale(loss).backward()  # backward pass, with gradient scaling if training in fp16
            
            if self.config.grad_clip != 0.0: # clip the gradient
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer) # step the optimizer and scaler if training in fp16
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True) # flush the gradients
            ###

            # timing and logging
            if self._iter_num % self.config.log_interval == 0 and self._master_process and local_iter_num >= 10:  # a CPU-GPU sync point
                end_timer.record()
                torch.cuda.synchronize()
                curr_iter_time = start_timer.elapsed_time(end_timer) / self.config.gradient_accumulation_steps
                lossf = loss.item() * self.config.gradient_accumulation_steps
                console.info(f"iter {self._iter_num}: loss {lossf:.6f}, lr {self._learning_rate:.6f}, time {curr_iter_time:.3f} ms")
                if self.config.save_snapshot and self.out_dir is not None:
                    self._save_ckpt(SNAPSHOT_FILENAME)
                
            self._iter_num += 1
            local_iter_num += 1
