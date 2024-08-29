import os
import time
import math
import torch
import random
import json
import logging

from torch import nn
from typing import Optional, Any
from contextlib import nullcontext
from torch.distributions.categorical import Categorical


from torch.nn.parallel import DistributedDataParallel as DDP

from jointformer.configs.trainer import TrainerConfig
from jointformer.models.transformer import Transformer
from jointformer.utils.loggers.wandb import WandbLogger
from jointformer.utils.datasets.base import BaseDataset

from jointformer.utils.runtime import set_seed
from jointformer.utils.data_collators import DataCollator
from jointformer.utils.chemistry import is_valid

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
            out_dir: Optional[str] = None,
            seed: Optional[int] = 1337,
            train_dataset: Optional[BaseDataset] = None,
            val_dataset: Optional[BaseDataset] = None,
            test_dataset: Optional[BaseDataset] = None,
            tokenizer: Optional[Any] = None,
            tasks: Optional[dict] = None,
            logger: Optional[WandbLogger] = None,
            device_type: Optional[str] = 'cuda'
    ):

        # set args
        self.out_dir = out_dir
        self.seed = seed
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.model = model
        self.tasks = tasks
        self.logger = logger
        self.device_type = 'cuda' if torch.cuda.is_available() and device_type == 'cuda' else 'cpu'
        self._loss_dict = {}

        # set config args
        self.compile = config.compile
        self.enable_ddp = config.enable_ddp
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.batch_size = config.batch_size
        self.block_size = config.block_size
        self.dtype = config.dtype
        if self.dtype == 'bfloat16':
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                self.dtype = 'float16'
        self.ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.weight_decay = config.weight_decay
        self.learning_rate = config.learning_rate
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.grad_clip = config.grad_clip
        self.eval_iters = config.eval_iters
        self.learning_rate = config.learning_rate
        self.warmup_iters = config.warmup_iters
        self.lr_decay_iters = config.lr_decay_iters
        self.min_lr = config.min_lr
        self.decay_lr = config.decay_lr
        self.always_save_checkpoint = config.always_save_checkpoint
        self.save_checkpoint = config.save_checkpoint
        self.save_checkpoint_every = config.save_checkpoint_every
        self.save_snapshot = config.save_snapshot
        self.eval_only = config.eval_only
        self.eval_interval = config.eval_interval
        self.max_iters = config.max_iters
        self.log_interval = config.log_interval
        self.tasks = config.tasks
        self.eval_generation = config.eval_generation

        self._iter_num = 0
        self._best_val_loss = 1e9
        self._snapshot_filepath = os.path.join(self.out_dir, SNAPSHOT_FILENAME) if self.out_dir else None
        self._learning_rate = None
        self._running_mfu = 0.0
        self._resumed_from_iter_num = 0

        self._post_init()

    def _set_ddp_config(self):
        """ Get the DDP configuration."""
        ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
        if self.enable_ddp and ddp:
            self.is_ddp = True
            self.ddp_rank = int(os.environ["SLURM_PROCID"])
            self.gpus_per_node = int(os.environ["SLURM_GPUS_PER_NODE"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            assert self.gpus_per_node == torch.cuda.device_count()
            self.ddp_local_rank = self.ddp_rank - self.gpus_per_node * (self.ddp_rank // self.gpus_per_node)
            self.device = f'cuda:{self.ddp_local_rank}'
            self.master_process = self.ddp_rank == 0  # this process will do logging, checkpointing etc.
            self.seed_offset = self.ddp_rank * 1234  # each process gets a different torch seed
            assert self.gradient_accumulation_steps % self.ddp_world_size == 0  # world_size number of processes will be training simultaneously
            self.gradient_accumulation_steps //= self.ddp_world_size  # hence, scale down the gradient accumulation iterations per process proportionally
        else:
            self.is_ddp = False
            self.master_process = True
            self.seed_offset = 0
            self.ddp_world_size = 1
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    def _set_device(self):
        torch.cuda.set_device(self.device)
        
    def _set_backends(self):
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    def _post_init(self):
        self._set_ddp_config()
        self._set_device()
        self._set_backends()
        
        set_seed(self.seed + self.seed_offset)
        self.model.to(self.device)
        self.optimizer = self.model.configure_optimizers(
            self.weight_decay, self.learning_rate, (self.beta1, self.beta2), self.device_type)

        self.task_distribution = Categorical(torch.Tensor(list(self.tasks.values())))
        self.tokens_per_iter = self.gradient_accumulation_steps * self.ddp_world_size * self.batch_size * self.block_size

        if self.tokenizer is not None and hasattr(self.tokenizer, '__len__') and hasattr(self.model, 'vocab_size'):
            if len(self.tokenizer) != self.model.vocab_size:
                raise ValueError(f"Tokenizer and model not compatible. Tokenizer is of length {len(self.tokenizer)}"
                                 f" while model expects vocab size {self.model.vocab_size}")
        if self.master_process:
            console.info(f"tokens per iteration set to: {self.tokens_per_iter:,}")

        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(
            device_type=self.device_type, dtype=self.ptdtype)

        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == 'float16'))

        if self.out_dir is not None:
            if not os.path.isdir(self.out_dir) and self.master_process:
                os.makedirs(self.out_dir, exist_ok=False)

    def _compile(self):
        if self.compile:
            self.model = torch.compile(self.model, mode='reduce-overhead', fullgraph=True, dynamic=True)

    def _parallelize(self):
        if self.is_ddp:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])

    def resume_snapshot(self):
        self.resume_from_file(self._snapshot_filepath)

    def resume_from_file(self, filepath, resume_training=False):
        checkpoint = torch.load(filepath, map_location=self.device)
        try:
            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'  # compile
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            self.model.load_state_dict(state_dict)
        except RuntimeError:
            self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if resume_training:
            self._iter_num = checkpoint['iter_num']
            self._best_val_loss = checkpoint['best_val_loss']
            self._loss_dict = checkpoint['loss_dict']
            self._resumed_from_iter_num = self._iter_num
            if self.logger is not None:
                self.logger.set_run_id(checkpoint['run_id'] if 'run_id' in checkpoint else None)
        checkpoint = None

    def _save_ckpt(self, filename: str):
        if self.out_dir is not None and self.master_process and self.save_checkpoint:
            run_id = self.logger.run_id if self.logger is not None else None
            checkpoint = {
                'model': self.raw_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'iter_num': self._iter_num,
                'best_val_loss': self._best_val_loss,
                'loss_dict': self._loss_dict,
                'run_id': run_id
            }    
            torch.save(checkpoint, os.path.join(self.out_dir, filename))

    @staticmethod
    def _get_num_workers():
        try:
            return int(os.environ["SLURM_CPUS_PER_TASK"])
        except KeyError:
            return 4
    
    def _get_data_loader(self, dataset, shuffle=True):
        collator = DataCollator(tokenizer=self.tokenizer, tasks=self.tasks)
        sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=self.ddp_world_size,
                rank=self.ddp_rank) if self.is_ddp else None
        return  torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=shuffle,
                    collate_fn=collator,
                    sampler=sampler,
                    num_workers=self._get_num_workers(),
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
        if self.is_ddp:
            self.train_loader.set_epoch(self._iter_num)
        return next(iter(self.train_loader)).to(self.device)

    def get_validation_batch(self):
        return next(iter(self.val_loader)).to(self.device)

    def get_batch(self, split, task):
        batch = self._sample(self.train_dataset, task) if split == 'train' else self._sample(self.val_dataset, task)
        return batch.to(self.device)
        
    def _sample(self, dataset, task):
        idx = [idx for idx in range(len(dataset))]
        idx = random.sample(idx, min(self.batch_size, len(idx)))
        sampled = [dataset[i] for i in idx]
        return self.tokenizer(sampled, task=task)

    @torch.no_grad()
    def test(self, metric: str = 'rmse'):
        
        self.model.eval()
        assert metric in ['rmse', 'mae'], f"Metric {metric} not supported."
        
        criterion = nn.MSELoss(reduction='sum') if metric == 'rmse' else nn.L1Loss(reduction='sum')

        if self.logger is not None:
            self.logger.init_run()

        n = 0
        loss = 0.
        for _, batch in enumerate(self.test_loader):
            properties = batch['properties']
            batch.to(self.device)
            
            with self.ctx:
                outputs = self.model.predict(**batch)["logits_prediction"].cpu()
            
            if outputs.dtype != torch.float32:
                outputs = torch.tensor(outputs, dtype=torch.float32)

            if hasattr(self.test_dataset, '_target_transform'):
                properties = self.test_dataset.undo_target_transform(properties)
                outputs = self.test_dataset.undo_target_transform(outputs)
            
            loss += criterion(properties, outputs)
            n += len(outputs)
        
        test_metric = torch.sqrt(loss / n).item() if metric == 'rmse' else (loss / n).item()

        if self.logger is not None:
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
        tasks = list(self.tasks.keys())

        for split in splits:
            out[split] = {}
            for task in tasks:
                losses = torch.zeros(self.eval_iters)
                for k in range(self.eval_iters):
                    inputs = self.get_batch(split, task)
                    with self.ctx:
                        outputs = self.model.get_loss(**inputs)
                    losses[k] = outputs["loss"].item() if outputs["loss"] is not None else torch.nan
                out[split][task] = losses.mean().item() if torch.nan not in losses else torch.nan

        for split in splits:
            for task in tasks:
                if 'combined' in out[split]:
                    out[split]['combined'] += out[split][task]
                else:
                    out[split]['combined'] = out[split][task]

        if hasattr(self.model, 'calculate_perplexity') and self.eval_generation:
            for split in splits:
                out[split]['perplexity'] = {}
                losses = torch.zeros(self.eval_iters)
                for k in range(self.eval_iters):
                    inputs = self.get_batch(split, task='generation')
                    with self.ctx:
                        perplexity = self.model.calculate_perplexity(**inputs)
                    losses[k] = perplexity.mean()
                out[split]['perplexity'] = losses.mean().item() if torch.nan not in losses else torch.nan

        if hasattr(self.model, 'generate') and self.eval_generation:
            for _ in range(self.eval_iters):
                samples = []
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
        if self._iter_num < self.warmup_iters:
            return self.learning_rate * self._iter_num / self.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if self._iter_num > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (self._iter_num - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)

    def _set_lr(self) -> None:
        self._learning_rate = self._get_lr() if self.decay_lr else self.learning_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._learning_rate

    def evaluate(self):
        if self._iter_num % self.eval_interval == 0 and self.master_process and (self._resumed_from_iter_num != self._iter_num or self._iter_num ==  0):
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
                log_dict['mfu'] = self._running_mfu * 100
                self.logger.log(log_dict)

            if self._iter_num > 0: # More logging here
                if 'val' in losses: # save checkpoint if validation loss is better
                    console.info(f"Validation loss: {losses['val']['combined']:.4f}")
                    console.info(f"Best validation loss: {self._best_val_loss:.4f}")
                    if losses['val']['combined'] < self._best_val_loss or self.always_save_checkpoint:
                        self._best_val_loss = losses['val']['combined']
                        self._save_ckpt(MODEL_FILENAME)
                        console.info(f"Checkpoint updated at iteration {self._iter_num}")
                if self.save_checkpoint_every is not None: # save checkpoint every n iterations
                    if self._iter_num % self.save_checkpoint_every == 0:
                        self._save_ckpt(f"ckpt_{self._iter_num}.pt")

    def _terminate(self):
        if self._iter_num > self.max_iters:
            return True
        return False

    @torch.no_grad()
    def generate(self, temperature=1.0, top_k=25):
        samples = self.model.generate(
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
            temperature = temperature,
            top_k = top_k,
            device = self.device)
        samples = self.tokenizer.decode(samples)
        return samples

    def train(self) -> None:
        
        if self._iter_num > self.max_iters:
            return

        self._compile()
        self._parallelize()
        self._init_data_loaders()

        if self.logger is not None and self.master_process:
            self.logger.init_run()
            self.logger.watch_model(self.model)

        inputs = self.get_training_batch()
        t0 = time.time()
        local_iter_num = 0  # number of iterations in the lifetime of this process
        self.raw_model = self.model.module if self.is_ddp else self.model  # unwrap DDP container if needed
        self._running_mfu = -1.0

        while True:
            if self._terminate():
                if self.logger is not None:
                    self.logger.finish()
                break
                
            self._set_lr()
            self.evaluate()
            if self._iter_num == 0 and self.eval_only:
                if self.logger is not None:
                    self.logger.finish()
                break
            
            ### Training step
            for micro_step in range(self.gradient_accumulation_steps): # gradient accumulation loop
                if self.is_ddp:
                    self.model.require_backward_grad_sync = (micro_step == self.gradient_accumulation_steps - 1)  # in ddp mode, only sync grads at the last micro-step
                with self.ctx:
                    outputs = self.model.get_loss(**inputs)
                    loss = outputs["loss"] / self.gradient_accumulation_steps  # scale the loss to account for gradient accumulation
                inputs = self.get_training_batch()  # async prefetch next batch
                self.scaler.scale(loss).backward()  # backward pass, with gradient scaling if training in fp16
            
            if self.grad_clip != 0.0: # clip the gradient
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer) # step the optimizer and scaler if training in fp16
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True) # flush the gradients
            ###

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if self._iter_num % self.log_interval == 0 and self.master_process:  # a CPU-GPU sync point
                lossf = loss.item() * self.gradient_accumulation_steps
                if local_iter_num >= 5:  # let the training loop settle a bit
                    self._running_mfu = 0.0
                    console.info(
                        f"iter {self._iter_num}: loss {lossf:.6f} on {inputs['task']} task, lr {self._learning_rate:.6f},"
                        + 
                        f" time {dt * 1000:.2f}ms, mfu {self._running_mfu * 100:.2f}%"
                        )
                    if self.save_snapshot:
                        self._save_ckpt(SNAPSHOT_FILENAME)
            self._iter_num += 1
            local_iter_num += 1
