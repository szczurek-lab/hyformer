import os
import time
import math
import torch
import random

from typing import Optional, Any

from contextlib import nullcontext

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributions.categorical import Categorical

from jointformer.configs.trainer import TrainerConfig
from jointformer.utils.loggers.wandb import WandbLogger


from jointformer.utils.datasets.base import BaseDataset

from jointformer.models.transformer import Transformer
from torch.utils.data._utils.collate import default_collate


SNAPSHOT_FILE = 'snapshot.ckpt'


class Trainer:

    def __init__(
            self,
            config: TrainerConfig,
            model: Transformer,
            out_dir: Optional[str] = None,
            init_from: Optional[str] = None,
            seed: Optional[int] = 0,
            train_dataset: Optional[BaseDataset] = None,
            val_dataset: Optional[BaseDataset] = None,
            tokenizer: Optional[Any] = None,
            tasks: Optional[dict] = None,
            wandb_logger: Optional[WandbLogger] = None
    ):
        """ Initialize the Trainer class.

        The trainer class is responsible for training the model.

        Upon initialization, the trainer will automatically resume training from a snapshot file if it exists.

        Otherwise, the trainer will resume training from the init_from file if it exists.
        """

        # set args
        self.out_dir = out_dir
        self.init_from = init_from
        self.seed = seed
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.model = model
        self.tasks = tasks

        # set config args
        self.compile = config.compile
        self.enable_ddp = config.enable_ddp
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.batch_size = config.batch_size
        self.block_size = config.block_size
        self.dtype = config.dtype
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
        self.eval_only = config.eval_only
        self.eval_interval = config.eval_interval
        self.max_iters = config.max_iters
        self.log_interval = config.log_interval
        self.tasks = config.tasks

        self._iter_num = 0
        self._best_val_loss = 1e9
        self._snapshot_filepath = os.path.join(self.out_dir, SNAPSHOT_FILE) if self.out_dir else None
        self._learning_rate = None

        self._get_ddp_config()
        self._init_run()
        self._resume()
        self._init_optimizer()
        self._post_init()

    def _get_ddp_config(self):
        """ Get the DDP configuration."""
        ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
        if self.enable_ddp and ddp:
            self.is_ddp = True
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = f'cuda:{self.ddp_local_rank}'
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0  # this process will do logging, checkpointing etc.
            self.seed_offset = self.ddp_rank  # each process gets a different seed
            # world_size number of processes will be training simultaneously, so we can scale
            # down the desired gradient accumulation iterations per process proportionally
            assert self.gradient_accumulation_steps % self.ddp_world_size == 0
            self.gradient_accumulation_steps //= self.ddp_world_size
        else:
            # if not ddp, we are running on a single gpu, and one process
            self.is_ddp = False
            self.master_process = True
            self.seed_offset = 0
            self.ddp_world_size = 1
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def _init_run(self):

        tokens_per_iter = self.gradient_accumulation_steps * self.ddp_world_size * self.batch_size * self.block_size
        print(f"tokens per iteration set to: {tokens_per_iter:,}")

        if self.master_process and self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)

        torch.manual_seed(self.seed + self.seed_offset)
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

        # note: float16 data type will automatically use a GradScaler
        if self.dtype == 'bfloat16':
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                self.dtype = 'float16'
        self.ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(
            device_type=self.device_type, dtype=self.ptdtype)

        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == 'float16'))
        self.task_distribution = Categorical(torch.Tensor(list(self.tasks.values())))

    def _sample_task(self):
        return list(self.tasks.keys())[self.task_distribution.sample().item()]

    def _resume(self):
        """ Resume training from a checkpoint/snapshot.

        First, check if a snapshot file exists. If it does, resume training from the snapshot file. Otherwise, check if
        the init_from argument is set. If it is, resume training from the init_from file. Otherwise, train from scratch.
        """
        if self._snapshot_filepath:
            if os.path.exists(self._snapshot_filepath):
                self._resume_from_file(self._snapshot_filepath)
        elif self.init_from:
            if os.path.exists(self.init_from):
                print(f"Resuming from {self.init_from}...")
                self._resume_from_file(self.init_from)
        else:
            print("Training from scratch...")
        self.model.to(self.device)
        print(f"Model device: {self.device}")

    def _resume_from_file(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'  # compile
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        self.model.load_state_dict(state_dict)
        self._iter_num = checkpoint['iter_num']
        self._best_val_loss = checkpoint['best_val_loss']
        checkpoint = None

    def _init_optimizer(self):
        """ Initialize the optimizer."""
        self.optimizer = self.model.configure_optimizers(
            self.weight_decay, self.learning_rate, (self.beta1, self.beta2), self.device_type)

        if self._snapshot_filepath:
            if os.path.exists(self._snapshot_filepath):
                self.optimizer.load_state_dict(torch.load(self._snapshot_filepath, map_location=self.device)["optimizer"])
        elif self.init_from:
            self.optimizer.load_state_dict(torch.load(self.init_from, map_location=self.device)["optimizer"])

    def _post_init(self):
        """ Compile the model and wrap the model in a DDP container. """

        if self.compile:
            if self.master_process:
                print("Compiling model..")
            # self.unoptimized_model = self.model  # is this necessary? No, not really
            self.model = torch.compile(self.model)

        if self.is_ddp:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])

    def get_batch(self, split, task=None):

        if task is None:
            task = self._sample_task()

        batch = self._sample(self.train_dataset, task) if split == 'train' else self._sample(self.val_dataset, task)

        if self.device_type != 'cpu':
            for key, value in batch.items():
                if value is not None and not isinstance(value, str):
                    batch[key] = value.pin_memory().to(self.device, non_blocking=True)
        return batch

    def _sample(self, dataset, task):
        """Acts as a data loader / collate_fn for the dataset."""
        idx = [idx for idx in range(len(dataset))]
        idx = random.sample(idx, self.batch_size)
        sampled = [dataset[i] for i in idx]

        if isinstance(sampled[0], tuple):
            x = [item[0] for item in sampled]
            y = [item[1] for item in sampled]
            inputs = self.tokenizer(x, task)
            inputs['targets'] = default_collate(y)
        else:
            inputs = self.tokenizer(sampled, task)
            inputs['targets'] = None
        return inputs

    import torch

    @torch.no_grad()
    def estimate_loss(self):
        task = 'lm'
        out = {}
        self.model.eval()
        splits = []
        if self.train_dataset:
            splits.append('train')
        if self.val_dataset:
            splits.append('val')
        for split in splits:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                inputs = self.get_batch(split, task)
                with self.ctx:
                    outputs = self.model(**inputs)
                losses[k] = outputs["loss"].item() if outputs["loss"] is not None else torch.nan
            out[split] = losses.mean().item() if torch.nan not in losses else torch.nan
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
        if self._iter_num % self.eval_interval == 0 and self.master_process:
            losses = self.estimate_loss()
            info = f"step {self._iter_num}"
            if 'train' in losses:
                info += f": train loss {losses['train']:.4f}"
            if 'val' in losses:
                info += f", val loss {losses['val']:.4f}"
            print(info)
            # if self.wandb_log:
            #     wandb.log({
            #         "iter": iter_num,
            #         "train/loss": losses['train'],
            #         "val/loss": losses['val'],
            #         "lr": lr,
            #         "mfu": running_mfu * 100,  # convert to percentage
            #     })
            if 'val' in losses:
                if losses['val'] < self._best_val_loss or self.always_save_checkpoint:
                    self._best_val_loss = losses['val']
                    if self._iter_num > 0 and self.out_dir:
                        checkpoint = {
                            'model': self.raw_model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            # 'model_args': model_args,
                            'iter_num': self._iter_num,
                            'best_val_loss': self._best_val_loss,
                            # 'config': config,
                        }
                        torch.save(checkpoint, os.path.join(self.out_dir, 'ckpt.pt'))

    def train(self):

        # training loop
        split = 'train'
        task = 'lm'
        inputs = self._get_batch(split=split, task=task)
        t0 = time.time()
        local_iter_num = 0  # number of iterations in the lifetime of this process
        self.raw_model = self.model.module if self.is_ddp else self.model  # unwrap DDP container if needed
        running_mfu = -1.0
        while True:

            # termination conditions
            if self._iter_num > self.max_iters:
                break

            # determine and set the learning rate for this iteration
            self._set_lr()

            # evaluate the loss on train/val sets and write checkpoints
            self.evaluate()
            if self._iter_num == 0 and self.eval_only:
                break

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(self.gradient_accumulation_steps):
                if self.is_ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    self.model.require_backward_grad_sync = (micro_step == self.gradient_accumulation_steps - 1)
                with self.ctx:
                    outputs = self.model(**inputs)
                    loss = outputs["loss"] / self.gradient_accumulation_steps  # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                inputs = self._get_batch(split, task)
                # backward pass, with gradient scaling if training in fp16
                self.scaler.scale(loss).backward()
            # clip the gradient
            if self.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            # step the optimizer and scaler if training in fp16
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            self.optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if self._iter_num % self.log_interval == 0 and self.master_process:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * self.gradient_accumulation_steps
                if local_iter_num >= 5:  # let the training loop settle a bit
                    mfu = self.raw_model.estimate_mfu(self.batch_size * self.gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                print(f"iter {self._iter_num}: loss {lossf:.6f}, lr {self._learning_rate:.6f},"
                      f" time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")
            self._iter_num += 1
            local_iter_num += 1

    #     # eval
    #     self.eval_interval = config.eval_interval
    #     self.log_interval = config.log_interval
    #     self.eval_iters = config.eval_iters
    #     self.eval_only = config.eval_only  # if True, script exits right after the first eval
    #     self.logger = logger
    #
    #     # load / save
    #     self.enable_save_checkpoint = config.enable_save_checkpoint
    #     self.always_save_checkpoint = config.always_save_checkpoint  # if True, always save a checkpoint after each eval
    #     self.init_from = config.init_from  # 'scratch' or 'resume' or 'gpt2*'
    #
    #     # adamw optimizer
    #     self.learning_rate = config.learning_rate  # max learning rate
    #     self.max_iters = config.max_iters  # total number of training iterations
    #     self.weight_decay = config.weight_decay
    #     self.beta1 = config.beta1
    #     self.beta2 = config.beta2
    #     self.grad_clip = config.grad_clip  # clip gradients at this value, or disable if == 0.0
    #
    #     # learning rate decay settings
    #     self.decay_lr = config.decay_lr  # whether to decay the learning rate
    #     self.warmup_iters = config.warmup_iters  # how many steps to warm up for
    #     self.lr_decay_iters = config.lr_decay_iters  # should be ~= max_iters per Chinchilla
    #     self.min_lr = config.min_lr  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    #
    #     # DDP settings
    #     self.enable_ddp = config.enable_ddp
    #
    #
    #     # runtime
    #     self.gradient_accumulation_steps = config.gradient_accumulation_steps  # used to simulate larger batch sizes
    #     self.batch_size = config.batch_size  # if gradient_accumulation_steps > 1, this is the micro-batch size
    #     self.device = config.device  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    #     self.device_type = config.device
    #     self.dtype = config.dtype  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    #     self.compile = config.compile  # use PyTorch 2.0 to compile the model to be faster
    #
    #     # data
    #     self.train_dataset = train_dataset
    #     self.eval_dataset = eval_dataset
    #     self.tokenizer = tokenizer
    #
    #     # eval
    #     self.iter_num = 0
    #     self.best_val_loss = 1e9
    #     self.config = config
    #
    #     # model
    #     self.model = model
    #
    #     # scaler
    #     self.scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == 'float16'))
    #
    #     # optimizer
    #     self.optimizer = None
    #     self.correct_bias = config.correct_bias
    #     self.optimizer_ckpt = None
    #     self.current_learning_rate = None
    #
    #
    #
    #
    #     self.master_process = True
    #     self.seed_offset = 0
    #     self.ddp_world_size = 1
    #     self.device = 'cuda:0'
    #
    #     is_ddp = int(os.environ.get('RANK', -1)) != -1
    #     if self.enable_ddp and is_ddp:
    #         self.init_ddp()
    #     self._post_init()
    #
    #
    # def init_ddp(self):
    #     ddp_rank = int(os.environ['RANK'])
    #     ddp_local_rank = int(os.environ['LOCAL_RANK'])
    #     self.ddp_world_size = int(os.environ['WORLD_SIZE'])
    #     self.device = f'cuda:{ddp_local_rank}'
    #     torch.cuda.set_device(self.device)
    #     self.master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    #     self.seed_offset = ddp_rank  # each process gets a different seed
    #     # world_size number of processes will be training simultaneously, so we can scale
    #     # down the desired gradient accumulation iterations per process proportionally
    #     assert self.gradient_accumulation_steps % self.ddp_world_size == 0
    #     self.gradient_accumulation_steps //= self.ddp_world_size
    #
    # def _post_init(self):
    #     if self.master_process and not os.path.isdir(self.out_dir) and self.enable_save_checkpoint:
    #         os.makedirs(self.out_dir)
    #     torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    #     torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    #     self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'  # for later use in torch.autocast
    #     # note: float16 data type will automatically use a GradScaler
    #     self.ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
    #     self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)
    #     print(f"Using {self.device_type} device")
    #
    # def init_run(self):
    #     if self.init_from == 'resume':
    #         self.load_checkpoint()
    #     elif self.init_from == 'scratch':
    #         pass
    #     else:
    #         raise ValueError(f"Unknown init_from value {self.init_from}")
    #
    #
    #
    # def load_checkpoint(self, path_to_checkpoint: str = None, reset_iter_num: bool = False) -> None:
    #
    #     out_dir = path_to_checkpoint if path_to_checkpoint is not None else os.path.join(self.out_dir, 'ckpt.pt')
    #     if not os.path.isfile(out_dir):
    #         raise FileNotFoundError(f"Checkpoint file {out_dir} does not exist!")
    #
    #     ckpt = torch.load(out_dir, map_location=next(self.model.parameters()).device)
    #
    #     # clean ckpt
    #     state_dict = ckpt['model']
    #     unwanted_prefix = '_orig_mod.'
    #     for k, v in list(state_dict.items()):
    #         if k.startswith(unwanted_prefix):
    #             state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    #
    #     # load
    #     self.model.load_state_dict(state_dict, strict=False)
    #     # self.optimizer.load_state_dict(ckpt['optimizer'])
    #     self.optimizer_ckpt = ckpt['optimizer']
    #     if reset_iter_num:
    #         self.iter_num = 0
    #         self.best_val_loss = 1e9
    #     else:
    #         self.iter_num = ckpt['iter_num']
    #         self.best_val_loss = ckpt['best_val_loss']
    #
    #     if 'run_id' in ckpt.keys() and self.logger is not None:
    #         self.logger.run_id = ckpt['run_id']
    #
    #     print(f"Successfully loaded checkpoint from {out_dir}...")
    #     return None
    #
    # def save_checkpoint(self, checkpoint_dir: str = None) -> None:
    #
    #     if self.enable_save_checkpoint:
    #         out_dir = checkpoint_dir if checkpoint_dir is not None else self.out_dir
    #         if not os.path.isdir(out_dir):
    #             os.makedirs(out_dir)
    #         run_id = None if self.logger is None else self.logger.run_id
    #         raw_model = self.model.module if self.is_ddp_run else self.model
    #         checkpoint = {
    #             'model': raw_model.state_dict(),
    #             'optimizer': self.optimizer.state_dict(),
    #             'iter_num': self.iter_num,
    #             'best_val_loss': self.best_val_loss,
    #             'run_id': run_id,
    #         }
    #         torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    #         print(f"Successfully saved to {self.out_dir}...")
    #
    #
    #
    #
    #
    #
    # def _train_init(self):
    #
    #     set_seed(1337 + self.seed_offset)
    #
    #     self.model.to(self.device)
    #
    #     self.optimizer = self.model.configure_optimizers(
    #         self.weight_decay, self.learning_rate, (self.beta1, self.beta2), self.device_type, self.correct_bias)
    #
    #     if self.optimizer_ckpt:
    #         self.optimizer.load_state_dict(self.optimizer_ckpt)
    #         self.optimizer_ckpt = None
    #
    #     if self.compile and not type(self.model).__name__ == 'OptimizedModule':
    #         print("Compiling model..")
    #         self.unoptimized_model = self.model
    #         self.model = torch.compile(self.model)
    #
    #     # wrap model into DDP container
    #     if self.is_ddp_run:
    #         self.model = DDP(self.model, device_ids=[self.ddp_local_rank])
    #
    # @torch.no_grad()
    # def estimate_loss(self):
    #     out = {}
    #     self.model.eval()
    #
    #     ###
    #     # log the loss of the model->used for model selection // double the num of iters to equal for two type of tasks
    #     # in a supervised setting some models may have None loss for some tasks
    #     ###
    #
    #     for split in ['train', 'val']:
    #         losses = torch.zeros(2 * self.eval_iters)
    #
    #         for k in range(self.eval_iters):
    #             inputs = self.get_batch(split, 'lm')
    #             with self.ctx:
    #                 outputs = self.model(
    #                     task='lm', input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
    #                     labels=inputs['labels'], target=inputs['target'], eos_mask=inputs['eos_mask'])
    #             losses[k] = torch.Tensor([0.]) if outputs['loss'] is None else outputs['loss'].item()
    #
    #         for k in range(self.eval_iters, 2 * self.eval_iters):
    #             inputs = self.get_batch(split, 'mlm')
    #             with self.ctx:
    #                 outputs = self.model(
    #                     task='mlm', input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
    #                     labels=inputs['labels'], target=inputs['target'], eos_mask=inputs['eos_mask'])
    #             losses[k] = torch.Tensor([0.]) if outputs['loss'] is None else outputs['loss'].item()
    #
    #         out[split] = losses[losses != 0.].mean()
    #
    #     ###
    #     # Log percent of valid molecules sampled
    #     ###
    #
    #     valid = []
    #     for k in range(self.eval_iters):
    #         idx = torch.ones(size=(self.batch_size, 1), device=self.device) * self.tokenizer.generate_token_id
    #         idx = idx.long()
    #         samples = self.model.generate(idx=idx, max_new_tokens=self.tokenizer.max_molecule_length)
    #         valid.extend(self.tokenizer.is_valid_smiles(samples))
    #     out['valid'] = sum(valid) / len(valid)
    #
    #     ###
    #     # Log supervised loss for fine-tuning purposes
    #     ###
    #
    #     model_name = type(self.model._orig_mod).__name__
    #
    #     if model_name in MLM_PREDICTION_MODELS:
    #         task = 'mlm'
    #     elif model_name in LM_PREDICTION_MODELS:
    #         task = 'lm'
    #     else:
    #         out['train_prediction'] = 0.
    #         out['val_prediction'] = 0.
    #         return out
    #
    #     for split in ['train', 'val']:
    #         losses = torch.zeros(2 * self.eval_iters)
    #         for k in range(2 * self.eval_iters):
    #             inputs = self.get_batch(split, task)
    #             with self.ctx:
    #                 outputs = self.model(
    #                     task=task, input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
    #                     labels=inputs['labels'], target=inputs['target'], eos_mask=inputs['eos_mask'])
    #             losses[k] = torch.Tensor([0.]) if outputs['supervised_loss'] is None else outputs['supervised_loss'].item()
    #
    #         out[split + '_prediction'] = losses[losses != 0.].mean()
    #
    #     self.model.train()
    #     return out
    #
    # def evaluate(self):
    #     losses = self.estimate_loss()
    #
    #     print(
    #         f"Evaluation at iter {self.iter_num}: train loss {losses['train']:.4f},"
    #         f" val loss {losses['val']:.4f},"
    #         f" percent {losses['valid']:.4f}")
    #
    #     if self.master_process and self.logger is not None:
    #         self.logger.log({
    #                 "iter": self.iter_num,
    #                 "train/loss": losses['train'],
    #                 "val/loss": losses['val'],
    #                 "train/loss_prediction": losses['train_prediction'],
    #                 "val/loss_prediction": losses['val_prediction'],
    #                 "val/valid": losses['valid'],
    #                 "lr": self.current_learning_rate,})
    #
    #     if losses['val'] < self.best_val_loss or self.always_save_checkpoint:
    #         self.best_val_loss = losses['val']
    #         if self.iter_num > 0:
    #             self.save_checkpoint()
    #
    # @torch.no_grad()
    # def test(self, dataset):
    #     batch_size = 64
    #     out = {}
    #     self._train_init()
    #     self.logger.init_run()
    #     self.model.eval()
    #     loss_l1 = torch.nn.L1Loss(reduction='mean')
    #     loss_mse = torch.nn.MSELoss(reduction='mean')
    #
    #     idx = [i for i in range(len(dataset))]
    #     idx_batched = [idx[i: i + batch_size] for i in range(0, len(idx), batch_size)]
    #
    #     model_name = type(self.model._orig_mod).__name__
    #
    #     if model_name in MLM_PREDICTION_MODELS:
    #         task = 'mlm'
    #     elif model_name in LM_PREDICTION_MODELS:
    #         task = 'lm'
    #     else:
    #         raise ValueError(f'Model {model_name} not in {MLM_PREDICTION_MODELS, LM_PREDICTION_MODELS}.')
    #
    #     predictions = []
    #     targets = []
    #
    #     for batch in range(len(idx_batched)):
    #         idx = torch.Tensor(idx_batched[batch]).to(torch.long)
    #         inputs = self.tokenizer.get_inputs(
    #             dataset=dataset, task=task, batch_size=self.batch_size, device=self.device, idx=idx)
    #         with self.ctx:
    #             outputs = self.model(
    #                 task=task, input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
    #                 labels=inputs['labels'], target=inputs['target'], eos_mask=inputs['eos_mask'])
    #
    #         predictions.extend(dataset.undo_target_transform(outputs['prediction'].cpu()))
    #         targets.extend(dataset.undo_target_transform(inputs['target'].cpu()))
    #
    #     predictions = torch.Tensor(predictions)
    #     targets = torch.Tensor(targets)
    #
    #     out['MSE'] = loss_mse(predictions, targets).item()
    #     out['RMSE'] = torch.sqrt(loss_mse(predictions, targets)).item()
    #     out['MAE'] = loss_l1(predictions, targets).item()
    #     out['y_pred'] = [item.item() for item in predictions]
    #     out['y_true'] = [item.item() for item in targets]
    #
    #     if self.master_process and self.logger is not None:
    #         self.logger.log({"test/MSE": out['MSE'], "test/RMSE": out['RMSE'], "test/MAE": out['MAE']})
    #
    #     return out
    #
    # @staticmethod
    # def get_task(task_p: float):
    #     p = torch.bernoulli(torch.Tensor([task_p]))
    #     if p == 1:
    #         return 'lm'
    #     elif p == 0:
    #         return 'mlm'
    #     else:
    #         raise ValueError("Wrong task!")
