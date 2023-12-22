import math
import torch
from hybrid_transformer.configs.trainers.trainer import TrainerConfig

import os
import time
import pickle
from contextlib import nullcontext

import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

class Trainer:

    def __init__(self, config: TrainerConfig, model: torch.nn, train_dataset, eval_dataset, tokenizer):

        # eval
        self.eval_interval = config.eval_interval
        self.log_interval = config.log_interval
        self.eval_iters = config.eval_iters
        self.eval_only = config.eval_only  # if True, script exits right after the first eval

        # load / save
        self.always_save_checkpoint = config.always_save_checkpoint  # if True, always save a checkpoint after each eval
        self.init_from = config.init_from  # 'scratch' or 'resume' or 'gpt2*'

        # adamw optimizer
        self.learning_rate = config.learning_rate  # max learning rate
        self.max_iters = config.max_iters  # total number of training iterations
        self.weight_decay = config.weight_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.grad_clip = config.grad_clip  # clip gradients at this value, or disable if == 0.0

        # learning rate decay settings
        self.decay_lr = config.decay_lr  # whether to decay the learning rate
        self.warmup_iters = config.warmup_iters  # how many steps to warm up for
        self.lr_decay_iters = config.lr_decay_iters  # should be ~= max_iters per Chinchilla
        self.min_lr = config.min_lr  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

        # DDP settings
        self.enable_ddp = config.enable_ddp
        self.ddp_backend = config.ddp_backend  # 'nccl', 'gloo', etc.

        # runtime
        self.gradient_accumulation_steps = config.gradient_accumulation_steps  # used to simulate larger batch sizes
        self.batch_size = config.batch_size  # if gradient_accumulation_steps > 1, this is the micro-batch size
        self.device = config.device  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
        self.dtype = config.dtype  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
        self.compile = compile  # use PyTorch 2.0 to compile the model to be faster

        # data
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        # model
        self.model = model

        # optimizer
        self.optimizer = model.configure_optimizers(
            self.weight_decay, self.learning_rate, (self.beta1, self.beta2), self.device)

        # scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == 'float16'))

        self._ddp()

    def _ddp(self) -> None:
        print(f"DDP is enabled: {self.enable_ddp}")
        self.ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?

        if self.ddp:
            print("DDP is enabled!")
            init_process_group(backend=self.ddp_backend)
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
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])
        else:
            # if not ddp, we are running on a single gpu, and one process
            self.master_process = True
            self.seed_offset = 0
            self.ddp_world_size = 1
        tokens_per_iter = self.gradient_accumulation_steps * self.ddp_world_size * self.batch_size * self.model.max_seq_len
        print(f"tokens per iteration will be: {tokens_per_iter:,}")


    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it: int) -> float:
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return self.learning_rate * it / self.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out



    @classmethod
    def from_config(cls, config: TrainerConfig, model: torch.nn.Module) -> 'Trainer':
        return cls(config, model)
