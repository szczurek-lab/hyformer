import os
import time
import math
import torch
import logging

from torch import nn
import numpy as np
from typing import Optional, Any, Dict
from contextlib import nullcontext
from torch.distributions.categorical import Categorical

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from hyformer.configs.trainer import TrainerConfig
from hyformer.models.transformer import Transformer
from hyformer.utils.loggers.wandb import WandbLogger
from hyformer.utils.datasets.base import BaseDataset

from hyformer.utils.collator import DataCollatorWithPadding
from hyformer.utils.chemistry import is_valid

from hyformer.trainers.utils import get_test_metric, seed_worker

console = logging.getLogger(__name__)
CHECKPOINT_FILENAME = 'checkpoint.pt'  # Single checkpoint file for both best model and training state
PAD_TO_MULTIPLE_OF = 8  # Pad sequences to multiple of 8 for better GPU utilization


class Trainer:
    """Trainer for a Transformer model."""
    def __init__(
            self,
            config: TrainerConfig,
            model: Transformer,
            device: torch.device,
            out_dir: Optional[str] = None,
            tokenizer: Optional[Any] = None,
            logger: Optional[WandbLogger] = None,
            seed: int = 42,  # Add seed parameter
    ):
        # Core setup
        self.config = config
        self.model = model
        self.device = device
        self.out_dir = out_dir
        self.tokenizer = tokenizer
        self.logger = logger
        self.seed = seed  # Store the seed

        # Training state
        self._epoch = 0
        self._best_val_loss = float('inf')
        self._best_epoch = 0
        self._not_improved_for_eval_epochs = 0
        self._learning_rate = None
        self._current_batch = 0

        # Distributed training setup
        self._is_distributed_run = dist.is_initialized()
        self._master_process = True if int(os.environ.get('LOCAL_RANK', 0)) == 0 else False
        
        # Model setup
        self.model.to(self.device)
        self.optimizer = self.model.configure_optimizers(
            self.config.weight_decay, 
            self.config.learning_rate, 
            (self.config.beta1, self.config.beta2), 
            self.device
        )
        
        # DDP and compilation
        if self.config.enable_ddp and dist.is_initialized():
            self.model = DDP(model, device_ids=[device], find_unused_parameters=False)
        if self.config.compile:
            self.model = torch.compile(self.model)
        
        # Mixed precision training
        self.ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.config.dtype]
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=self.ptdtype)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.config.dtype == 'float16'))

    def _create_collator(self, tasks: Dict[str, float]) -> DataCollatorWithPadding:
        """Create a data collator with the specified tasks.
        
        Args:
            tasks: Dictionary mapping task names to their weights
            
        Returns:
            Configured DataCollatorWithPadding instance
        """
        return DataCollatorWithPadding(
            tokenizer=self.tokenizer, 
            tasks=tasks,
            pad_to_multiple_of=PAD_TO_MULTIPLE_OF,  # Pad to multiple of 8 for better GPU utilization
            max_length=self.config.max_seq_len,  # Use max_seq_len from config
            return_tensors="pt",  # Return PyTorch tensors directly
            mask_probability=0.15  # Set mask probability for reconstruction task
        )

    def _create_sampler(self, dataset: BaseDataset) -> Optional[torch.utils.data.Sampler]:
        """Create a distributed sampler if running in distributed mode.
        
        Args:
            dataset: Dataset to create sampler for
            
        Returns:
            DistributedSampler if running in distributed mode, None otherwise
        """
        if not self._is_distributed_run:
            return None
            
        return torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=int(os.environ["WORLD_SIZE"]),
            rank=int(os.environ["SLURM_PROCID"])
        )

    def _get_num_workers(self) -> int:
        """Get the number of workers from SLURM environment or config.
        
        Returns:
            Number of workers to use for data loading
        """
        return int(os.environ.get("SLURM_CPUS_PER_TASK", self.config.num_workers))

    def create_loader(
        self,
        dataset: Optional[BaseDataset],
        shuffle: bool = True,
        tasks: Optional[Dict[str, float]] = None
    ) -> Optional[torch.utils.data.DataLoader]:
        """Create a data loader with optimized settings.
        
        Args:
            dataset: Dataset to create loader for
            shuffle: Whether to shuffle the data
            tasks: Optional dictionary of tasks to use for collation. 
                  Supported tasks are 'lm', 'prediction', and 'mlm'.
            
        Returns:
            Configured DataLoader instance or None if dataset is None
        """
        if dataset is None:
            return None
            
        # Use provided tasks or default to prediction task
        tasks = tasks if tasks is not None else (self.config.tasks if shuffle else {'prediction': 1.0})
        
        # Validate tasks
        for task in tasks:
            if task not in ['lm', 'prediction', 'mlm']:
                raise ValueError(f"Unsupported task: {task}. Supported tasks are 'lm', 'prediction', and 'mlm'.")
        
        # Create components
        collator = self._create_collator(tasks)
        sampler = self._create_sampler(dataset)
        num_workers = self._get_num_workers()
        
        # Create a generator for reproducibility
        g = torch.Generator()
        g.manual_seed(self.seed)  # Use the instance seed instead of a fixed 0
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle if sampler is None else False,
            collate_fn=collator,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,  # Prefetch 2 batches per worker
            worker_init_fn=seed_worker,  # Set seed for each worker
            generator=g  # Use fixed generator for reproducibility
        )

    def _get_lr(self):
        """Get learning rate with warmup and decay.
        
        Implements linear warmup followed by cosine decay to min_lr.
        The learning rate schedule is applied to the base learning rate,
        while weight decay remains constant throughout training.
        
        Returns:
            Current learning rate value
        """
        # Calculate current iteration based on completed epochs and current batch
        current_iter = self._epoch * len(self.train_loader) + self._current_batch
        
        # Linear warmup
        if current_iter < self.config.warmup_iters:
            return self.config.learning_rate * current_iter / self.config.warmup_iters
            
        # Cosine decay
        if current_iter > self.config.lr_decay_iters:
            return self.config.min_lr
            
        # Cosine decay from learning_rate to min_lr
        decay_ratio = (current_iter - self.config.warmup_iters) / (self.config.lr_decay_iters - self.config.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)

    def _set_lr(self):
        """Update learning rate.
        
        The learning rate schedule is applied to the base learning rate,
        while weight decay remains constant. This is the standard practice
        for transformer training, where weight decay is typically kept
        constant throughout training.
        """
        self._learning_rate = self._get_lr() if self.config.decay_lr else self.config.learning_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._learning_rate
            # Weight decay remains constant throughout training
            param_group['weight_decay'] = self.config.weight_decay

    @torch.no_grad()
    def evaluate(self):
        """Evaluate model on validation set."""
        if self._epoch % self.config.eval_interval != 0:
            return

        self.model.eval()
        val_loss = 0.
        num_batches = 0

        for batch in self.val_loader:
            batch = batch.to(self.device)
            loss = self.model.get_loss(**batch, reduction='sum')["loss"].cpu()
            val_loss += loss
            num_batches += 1

        val_loss = val_loss.mean().item()
        
        if self._master_process:
            console.info(f"Epoch {self._epoch}: val_loss {val_loss:.4f}")
            if self.logger:
                self.logger.log({'val/loss': val_loss, 'epoch': self._epoch, 'lr': self._learning_rate})

            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._best_epoch = self._epoch
                self._save_ckpt()
                self._not_improved_for_eval_epochs = 0
            else:
                self._not_improved_for_eval_epochs += 1

        self.model.train()

    def _save_ckpt(self):
        """Save checkpoint with optimized state dict handling."""
        if not self._master_process or not self.config.save_checkpoint:
            return

        checkpoint = {
            'model': self.model.module.state_dict() if dist.is_initialized() else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self._epoch,
            'best_val_loss': self._best_val_loss,
            'run_id': self.logger.run_id if self.logger else None
        }
        torch.save(checkpoint, os.path.join(self.out_dir, CHECKPOINT_FILENAME))

    def train(
            self,
            train_dataset: Optional[BaseDataset] = None,
            val_dataset: Optional[BaseDataset] = None,
            eval_metric: Optional[str] = None,
            patience: Optional[int] = None,
    ) -> None:
        """Main training loop with optimizations.
        
        Args:
            train_dataset: Dataset for training
            val_dataset: Dataset for validation
            eval_metric: Metric to use for evaluation
            patience: Number of epochs to wait before early stopping
        """
        if self._epoch >= self.config.max_epochs:
            return

        # Set up evaluation metrics
        self.eval_metric = eval_metric if eval_metric is not None else 'combined'
        self.patience = patience

        # Initialize data loaders
        self.train_loader = self.create_loader(train_dataset, shuffle=True)
        self.val_loader = self.create_loader(val_dataset, shuffle=False)

        if self.logger and self._master_process:
            self.logger.init_run()
            self.logger.watch_model(self.model)

        self.model.train()
        torch.cuda.empty_cache()  # Clear GPU cache before training

        while self._epoch < self.config.max_epochs:
            self._current_batch = 0  # Reset batch counter at start of epoch
            self._set_lr()
            
            # Training loop
            epoch_loss = 0.
            num_batches = 0
            start_time = time.time()

            for batch_idx, inputs in enumerate(self.train_loader):
                # Move batch to device - inputs is already a dict of tensors
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Gradient accumulation
                for micro_step in range(self.config.gradient_accumulation_steps):
                    if self._is_distributed_run:
                        self.model.require_backward_grad_sync = (micro_step == self.config.gradient_accumulation_steps - 1)
                    
                    with self.ctx:
                        outputs = self.model.module.get_loss(**inputs) if dist.is_initialized() else self.model.get_loss(**inputs)
                        loss = outputs["loss"] / self.config.gradient_accumulation_steps
                    
                    self.scaler.scale(loss).backward()

                # Optimizer step
                if self.config.grad_clip != 0.0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                num_batches += 1
                self._current_batch += 1  # Update current batch counter

                # Logging
                if batch_idx % self.config.log_interval == 0 and self._master_process:
                    avg_loss = epoch_loss / num_batches
                    elapsed = time.time() - start_time
                    console.info(f"Epoch {self._epoch}, Batch {batch_idx}: loss {avg_loss:.4f}, lr {self._learning_rate:.6f}, time {elapsed:.2f}s")

            # Epoch evaluation
            if val_dataset is not None:
                self.evaluate()
                if self.patience and self._not_improved_for_eval_epochs >= self.patience:
                    console.info(f"Early stopping triggered after {self._epoch} epochs")
                    break

            # Save checkpoint
            if self._master_process and self.config.save_checkpoint:
                self._save_ckpt()

            self._epoch += 1

        if self.logger and self._master_process:
            self.logger.finish()

    def resume_from_checkpoint(self, filepath: str, resume_training: bool = False):
        """Resume training from a checkpoint file.
        
        Args:
            filepath: Path to the checkpoint file
            resume_training: Whether to resume training state (optimizer, epoch, etc.)
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        state_dict = checkpoint['model']
        
        # Handle compiled model artifacts
        if not any(key.startswith("_orig_mod") for key in self.model.state_dict().keys()):
            unwanted_prefix = '_orig_mod.'
            for k, _ in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)    
        
        # Load model state
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            console.warning("Model state_dict loaded with strict=False.")
            console.warning(f"Missing keys: {missing_keys}")
            console.warning(f"Unexpected keys: {unexpected_keys}")

        if resume_training:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self._epoch = checkpoint['epoch']
                self._best_val_loss = checkpoint['best_val_loss']
                if self.logger is not None:
                    self.logger.set_run_id(checkpoint['run_id'] if 'run_id' in checkpoint else None)
                console.info(f"Resuming training from epoch {self._epoch} with best validation loss {self._best_val_loss:.4f}")
            except Exception as e:
                console.warning(f"Failed to resume training state: {e}")
                console.warning("Initializing training state from scratch.")

    @torch.no_grad()
    def test(self, test_dataset: BaseDataset, metric: Optional[str] = None):
        """Evaluate model on test set.
        
        Args:
            test_dataset: Dataset for testing
            metric: Metric to use for testing (overrides self.test_metric if provided)
        """
        metric = metric if metric is not None else self.test_metric
        if metric is None:
            raise ValueError("No test metric specified.")
        assert metric in ['rmse', 'roc_auc', 'prc_auc'], f"Metric {metric} not supported."

        # Create test data loader using the same pattern as other loaders
        test_loader = self.create_loader(test_dataset, shuffle=False, tasks={'prediction': 1.0})
        if test_loader is None:
            raise ValueError("Test dataset is required for testing.")

        self.model.eval()
        y_true = None
        y_pred = None

        for _, batch in enumerate(test_loader):
            # Move batch to device - batch is already a dict of tensors
            batch = {k: v.to(self.device) for k, v in batch.items()}
            y_true = batch['targets'] if y_true is None else torch.cat((y_true, batch['targets']))
            _y_pred = self.model.predict(**batch).cpu()
            y_pred = _y_pred if y_pred is None else torch.cat((y_pred, _y_pred))

        if metric == 'rmse':    
            if test_dataset.target_transform is not None:
                y_true = test_dataset.target_transform.inverse_transform(y_true)
                y_pred = test_dataset.target_transform.inverse_transform(y_pred)
            else:
                print("No target data_transform found. Assuming target is not transformed.")
        
        assert y_true.shape == y_pred.shape, f"Shapes of y_true and y_pred do not match: {y_true.shape} and {y_pred.shape}."
        test_metric = get_test_metric(y_true.numpy(), y_pred.numpy(), metric)
        
        if self.logger is not None:
            self.logger.init_run()
            self.logger.log({f"test/{metric}": test_metric})
        
        self.model.train()
        return test_metric
    
    @torch.no_grad()
    def get_predictions(self, split):
        self.model.eval()
        y_true = None
        y_pred = None

        if split == 'train':
            loader = self.train_loader
        elif split == 'val':
            loader = self.val_loader
        else:
            raise ValueError(f"Split {split} not supported.")

        for _, batch in enumerate(loader):
            
            y_true = batch['targets'] if y_true is None else torch.cat((y_true, batch['targets']))
            
            batch = batch.to(self.device)
            _y_pred = self.model.predict(**batch).cpu()
            y_pred = _y_pred if y_pred is None else torch.cat((y_pred, _y_pred))
        
        self.model.train()
        return {'y_true': y_true, 'y_pred': y_pred}

    @torch.no_grad()
    def estimate_loss(self, split):

        self.model.eval()
        loss = 0.
        
        for _, batch in enumerate(self.train_loader if split == 'train' else self.val_loader):
            batch = batch.to(self.device)
            loss += self.model.get_loss(**batch, reduction='sum')["loss"].cpu()

        self.model.train()
        return loss.mean().item()

    def _terminate(self):
        if self._epoch >= self.config.max_epochs:
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
