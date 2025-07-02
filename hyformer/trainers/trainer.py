import os
import math
import torch
import logging

from typing import Optional, Dict
from contextlib import nullcontext

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from hyformer.configs.trainer import TrainerConfig
from hyformer.models.trainable import TrainableModel
from hyformer.utils.loggers.wandb import WandbLogger
from hyformer.utils.datasets.base import BaseDataset

from hyformer.utils.tokenizers.base import BaseTokenizer
from hyformer.utils.collators.data_collator_task_tokens import DataCollatorWithTaskTokens

from hyformer.utils.metrics import calculate_metric
from hyformer.utils.reproducibility import seed_worker

import torch._dynamo
torch._dynamo.config.cache_size_limit = 64

console = logging.getLogger(__name__)
CHECKPOINT_FILENAME = 'ckpt.pt'  # Single checkpoint file for both best model and training state
CHECKPOINT_FILENAME_EPOCH = 'ckpt_epoch={epoch}.pt'
PAD_TO_MULTIPLE_OF = 128  # Pad sequences to multiple of x for better GPU utilization
DEFAULT_WORKER_SEED = 42  # Default seed for data loading workers
SUPPORTED_TASKS = ['lm', 'prediction', 'mlm']

class Trainer:
    """Trainer for a Transformer model."""
    def __init__(
            self,
            config: TrainerConfig,
            model: TrainableModel,
            device: torch.device,
            tokenizer: BaseTokenizer,
            out_dir: Optional[str] = None,
            logger: Optional[WandbLogger] = None,
            worker_seed: Optional[int] = None,
    ):
        # Core setup
        self.config = config
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.out_dir = out_dir
        self.logger = logger
        
        # Worker seed setup
        try:
            default_worker_seed = int(os.environ.get("PYTHONHASHSEED", DEFAULT_WORKER_SEED))
        except (ValueError, TypeError):
            default_worker_seed = DEFAULT_WORKER_SEED
            
        self.worker_seed = worker_seed if worker_seed is not None else default_worker_seed
        console.info(f"Using worker seed: {self.worker_seed}")

        # Training state
        self._epoch = 0
        self._best_val_loss = float('inf')
        self._best_epoch = 0
        self._not_improved_for_eval_epochs = 0
        self._iterations_in_epoch = 0
        self._current_lr = None

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
            self.model = torch.compile(self.model, mode="reduce-overhead")
        
        # Mixed precision training
        self.ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.config.dtype]
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=self.ptdtype)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.config.dtype == 'float16'))

    @classmethod
    def from_config(
        cls,
        config: TrainerConfig,
        model: TrainableModel,
        device: torch.device,
        tokenizer: BaseTokenizer,  # Now required, no default value
        out_dir: Optional[str] = None,
        logger: Optional[WandbLogger] = None,
        worker_seed: Optional[int] = None,
    ) -> 'Trainer':
        """
        Create a Trainer instance from a configuration.
        
        Args:
            config: TrainerConfig instance with training parameters
            model: Model to train
            device: Device to use for training
            tokenizer: Tokenizer for text processing (required)
            out_dir: Directory to save outputs
            logger: Logger for tracking metrics
            worker_seed: Random seed used for reproducibility (if None, will check for existing seed)
            
        Returns:
            Configured Trainer instance
        """
        return cls(
            config=config,
            model=model,
            device=device,
            tokenizer=tokenizer,
            out_dir=out_dir,
            logger=logger,
            worker_seed=worker_seed
        )
        
    def _create_collator(self, tasks: Dict[str, float]) -> DataCollatorWithTaskTokens:
        """Create a data collator with the specified tasks.
        
        Args:
            tasks: Dictionary mapping task names to their weights
            
        Returns:
            Configured DataCollatorWithTaskTokens instance
        """
        return DataCollatorWithTaskTokens(
            tokenizer=self.tokenizer, 
            tasks=tasks,
            pad_to_multiple_of=PAD_TO_MULTIPLE_OF,  # Use PAD_TO_MULTIPLE_OF from collator module
            max_length=self.tokenizer.max_length,  # Use max_seq_len from config
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
        _num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", self.config.num_workers))
        console.info(f"Using {_num_workers} workers for data loading")
        return _num_workers

    def create_loader(
        self,
        dataset: Optional[BaseDataset],
        tasks: Dict[str, float],
        shuffle: bool = True,
        num_workers: int = None,
        batch_size: int = None,
    ) -> Optional[torch.utils.data.DataLoader]:
        """Create a data loader with optimized settings.
        
        Args:
            dataset: Dataset to create loader for
            tasks: Dictionary of tasks to use for collation. 
                  Supported tasks are 'lm', 'prediction', and 'mlm'.
            shuffle: Whether to shuffle the data
            
        Returns:
            Configured DataLoader instance or None if dataset is None
        """
        if dataset is None:
            return None
            
        # Validate tasks
        for task in tasks:
            if task not in SUPPORTED_TASKS:
                raise ValueError(f"Unsupported task: {task}. Supported tasks are {SUPPORTED_TASKS}.")
        
        # Create components
        collator = self._create_collator(tasks)
        sampler = self._create_sampler(dataset)
        num_workers = num_workers if num_workers is not None else self._get_num_workers()
        
        # Create a generator for reproducibility
        g = torch.Generator()
        g.manual_seed(self.worker_seed)  # Use the instance seed instead of a fixed 0
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size if batch_size is not None else self.config.batch_size,
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
        current_iter = self._epoch * len(self.train_loader) + self._iterations_in_epoch
        
        # Linear warmup
        if current_iter < self._warmup_iters:
            return self._learning_rate * (current_iter + 1) / (self._warmup_iters + 1)
            
        # Cosine decay
        if current_iter > self._lr_decay_iters:
            return self._min_lr
            
        # Cosine decay from learning_rate to min_lr
        decay_ratio = (current_iter - self._warmup_iters) / (self._lr_decay_iters - self._warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self._min_lr + coeff * (self._learning_rate - self._min_lr)

    def _set_lr(self):
        """Update learning rate.
        
        The learning rate schedule is applied to the base learning rate,
        while weight decay remains constant. This is the standard practice
        for transformer training, where weight decay is typically kept
        constant throughout training.
        """
        current_lr = self._get_lr() if self.config.decay_lr else self._learning_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        self._current_lr = current_lr  # Store current lr for logging

    @torch.inference_mode()
    def evaluate(self, task_specific_validation: Optional[str] = None):
        """Evaluate model on validation set with task-specific loaders."""
        # Save original model training state
        was_training = self.model.training
        # Set model to evaluation mode
        self.model.eval()
        
        val_loss = 0.
        task_losses = {task: 0. for task in self.config.tasks}
        task_counts = {task: 0 for task in self.config.tasks}

        if task_specific_validation is not None:
            assert task_specific_validation in self.config.tasks.keys(), \
                f"task_specific_validation must be one of the task names in config.tasks, got {task_specific_validation}"
        
        # Evaluate each task separately
        for task_name, task_weight in self.config.tasks.items():
            task_loss = 0.
            processed_batch_count = 0

            for batch in self.val_loaders[task_name]:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Get model outputs including loss
                with self.ctx:
                    outputs = self.model(**batch)
                    loss = outputs['loss']
                
                if loss is not None:
                    task_loss += loss.item()
                    processed_batch_count += 1

            # Calculate average loss for this task
            if processed_batch_count > 0:
                task_loss /= processed_batch_count
                task_losses[task_name] = task_loss
                task_counts[task_name] = processed_batch_count
                val_loss += task_loss * task_weight
        
        if self._master_process:
            console.info(f"Epoch {self._epoch}: val_loss {val_loss:.4f}")
            
            # Log task-specific losses
            for task, loss in task_losses.items():
                if task_counts[task] > 0:
                    console.info(f"  - {task} loss: {loss:.4f} (from {task_counts[task]} val batches)")
            
            if self.logger:
                log_dict = {
                    'val/loss': val_loss,
                    'epoch': self._epoch,
                    'lr': self._current_lr
                }
                
                # Add task-specific losses to log
                for task, loss in task_losses.items():
                    if task_counts[task] > 0:
                        log_dict[f'val/{task}_loss'] = loss
                
                self.logger.log(log_dict)
            
            _val_loss = val_loss if task_specific_validation is None else task_losses[task_specific_validation]
            if _val_loss < self._best_val_loss:
                self._best_val_loss = _val_loss
                self._best_epoch = self._epoch
                self._save_ckpt()
                console.info("--------------------------------")
                console.info(f"New best validation loss: {self._best_val_loss:.4f} at epoch {self._best_epoch}")
                console.info("--------------------------------")
                self._not_improved_for_eval_epochs = 0
            else:
                self._not_improved_for_eval_epochs += 1
                
        if was_training:
            self.model.train()
            
        return val_loss

    def _save_ckpt(self, epoch: int = None):
        """Save checkpoint with optimized state dict handling."""
        if not self._master_process:
            return

        checkpoint = {
            'model': self.model.module.state_dict() if dist.is_initialized() else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self._epoch,
            'best_val_loss': self._best_val_loss,
            'run_id': self.logger.run_id if self.logger else None
        }
        _filename = CHECKPOINT_FILENAME if epoch is None else CHECKPOINT_FILENAME_EPOCH.format(epoch=epoch)
        torch.save(checkpoint, os.path.join(self.out_dir, _filename))

    @staticmethod
    def _get_grad_norm(model: torch.nn.Module) -> float:
        """Get the gradient norm of the model parameters.
        
        Args:
            model: The model to get the gradient norm of
            
        Returns:
            The gradient norm of the model parameters
        """
        norm_type = 2
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
        return total_norm ** (1. / norm_type)
        
    def train(
            self,
            train_dataset: BaseDataset,
            val_dataset: Optional[BaseDataset] = None,
            task_specific_validation: Optional[str] = None,
            patience: Optional[int] = None,
    ) -> None:
        """Main training loop with optimizations."""
        if self._epoch >= self.config.max_epochs:
            return

        self.patience = patience
        self.train_loader = self.create_loader(train_dataset, tasks=self.config.tasks, shuffle=True)
        
        # Set local variables for learning rate scheduling
        self._learning_rate = self.config.learning_rate
        self._min_lr = self.config.min_lr if self.config.min_lr is not None else self._learning_rate * 0.01
        self._lr_decay_iters = self.config.max_epochs * len(self.train_loader) // self.config.gradient_accumulation_steps
        self._warmup_iters = self.config.warmup_iters if self.config.warmup_iters is not None else int(0.1 * self._lr_decay_iters)
        
        # Log learning rate scheduling parameters
        if self._master_process:
            console.info(f"Max epochs: {self.config.max_epochs}")
            console.info(f"Number of iterations in a single epoch: {len(self.train_loader)}")
            console.info(f"Max number of iterations: {self._lr_decay_iters}")
            console.info(f"Warmup iterations: {self._warmup_iters}")
            console.info(f"Min learning rate: {self._min_lr}")
        
        # Create task-specific validation loaders
        self.val_loaders = {}
        if val_dataset is not None:
            for task_name in self.config.tasks:
                self.val_loaders[task_name] = self.create_loader(
                    val_dataset,
                    tasks={task_name: 1.0},  # Force this specific task
                    shuffle=False  # No shuffling for consistency
                )

        if self.logger and self._master_process:
            self.logger.init_run()
            self.logger.watch_model(self.model)

        self.model.train()
        torch.cuda.empty_cache()  # Clear GPU cache before training
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Training loop
        while self._epoch < self.config.max_epochs:
            epoch_loss = 0.
            processed_batch_count = 0

            for _iterations_in_epoch, batch in enumerate(self.train_loader):
                self._iterations_in_epoch = _iterations_in_epoch
                self._set_lr()
                self.optimizer.zero_grad(set_to_none=True)
                
                if self._master_process:
                    start_event.record()
                
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                if self._is_distributed_run and hasattr(self.model, "require_backward_grad_sync"):
                    self.model.require_backward_grad_sync = (_iterations_in_epoch + 1) % self.config.gradient_accumulation_steps == 0
                    
                with self.ctx:
                    outputs = self.model(**batch)
                    loss = outputs['loss'] / self.config.gradient_accumulation_steps
            
                self.scaler.scale(loss).backward()
                
                if (_iterations_in_epoch + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.grad_clip != 0.0:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                        
                epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                processed_batch_count += 1

                # Logging
                if _iterations_in_epoch % self.config.log_interval == 0 and self._master_process:
                    end_event.record()
                    torch.cuda.synchronize()
                    tokens_per_second = batch['input_ids'].numel() / (start_event.elapsed_time(end_event) / 1000.0)
                    grad_norm = self._get_grad_norm(self.model)
                    avg_loss = epoch_loss / processed_batch_count
                    console.info(
                        f"Epoch {self._epoch}, Step {_iterations_in_epoch}: train loss {avg_loss:.4f}, "
                        f"lr {self._current_lr:.6f}, "
                        f"grad_norm {grad_norm:.6f}, "
                        f"scaler_scale {self.scaler.get_scale():.1f}, "
                        f"tokens/s {tokens_per_second:.2f}")
                        
            # Save checkpoint every save_interval epochs
            if self._epoch % self.config.save_interval == 0 and self._epoch > 0 and self._master_process:
                self._save_ckpt(epoch=self._epoch)
            
            # Epoch evaluation and early stopping
            if val_dataset is None:
                if self._master_process:
                    self._save_ckpt()
            else:
                if self._master_process:
                    self.evaluate(task_specific_validation=task_specific_validation)
                
                # In distributed training, broadcast the not_improved counter to all processes
                if self._is_distributed_run:
                    # Create tensor for not_improved_for_eval_epochs to synchronize early stopping
                    not_improved_tensor = torch.tensor([self._not_improved_for_eval_epochs], device=self.device)
                    # Broadcast from master to all processes
                    dist.broadcast(not_improved_tensor, src=0)
                    # Update local value with broadcasted one
                    self._not_improved_for_eval_epochs = int(not_improved_tensor.item())
                
                # Now all processes have the same counter value for checking patience
                if self.patience and self._not_improved_for_eval_epochs >= self.patience:
                    if self._master_process:
                        console.info(f"Early stopping triggered after {self._epoch} epochs")
                    break

            self._epoch += 1

        if self.logger and self._master_process:
            self.logger.finish()

    def resume_from_checkpoint(self, filepath: str, resume_training: bool = False, discard_prediction_head: bool = False):
        """Resume training from a checkpoint file.
        
        Args:
            filepath: Path to the checkpoint file
            resume_training: Whether to resume training state (optimizer, epoch, etc.)
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        state_dict = checkpoint['model']    
        self.model.load_pretrained(state_dict=state_dict, discard_prediction_head=discard_prediction_head)

        if resume_training:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self._epoch = checkpoint['epoch']
                self._best_val_loss = checkpoint['best_val_loss']
                if self.logger is not None and self._master_process:
                    self.logger.set_run_id(checkpoint['run_id'] if 'run_id' in checkpoint else None)
                if self._master_process:
                    console.info(f"Resuming training from epoch {self._epoch} with best validation loss {self._best_val_loss:.4f}")
            except Exception as e:
                if self._master_process:
                    console.warning(f"Failed to resume training state: {e}")
                    console.warning("Initializing training state from scratch.")

    @torch.inference_mode()
    def test(self, test_dataset: BaseDataset, metric: str) -> float:
        """Evaluate model on test set using metrics based on model outputs (logits).
        
        Args:
            test_dataset: Dataset for testing
            metric: Metric to use for testing (perplexity, rmse, roc_auc, prc_auc)
            
        Returns:
            Float value of the requested metric
        """
        assert metric in ['rmse', 'roc_auc', 'prc_auc'], \
            f"Metric {metric} not supported for model output evaluation."
        
        test_loader = self.create_loader(test_dataset, shuffle=False, tasks={'prediction': 1.0})

        y_true = None
        y_pred = None

        for _, batch in enumerate(test_loader):
            
            # Get targets
            _targets = batch['target']
            
            # Get predictions
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            _logits = self.model(**batch)['logits'].cpu()
            
            y_true = torch.cat((y_true, _targets)) if y_true is not None else _targets
            y_pred = torch.cat((y_pred, _logits)) if y_pred is not None else _logits
            
        # Apply inverse transform for regression metrics if needed
        if metric == 'rmse' and hasattr(test_dataset, 'target_transform') and test_dataset.target_transform is not None:    
            y_true = test_dataset.target_transform.inverse_transform(y_true)
            y_pred = test_dataset.target_transform.inverse_transform(y_pred)
            
        test_metric = calculate_metric(y_true.numpy(), y_pred.numpy(), metric)
        assert isinstance(test_metric, float), f"Test metric {metric} is not a float."
        
        if self.logger is not None and self._master_process:
            self.logger.log({f"test/{metric}": test_metric})
        
        return test_metric
    