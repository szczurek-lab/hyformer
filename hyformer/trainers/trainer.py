import os
import time
import math
import torch
import logging

from typing import Optional, Any, Dict, List
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
CHECKPOINT_FILENAME = 'checkpoint.pt'  # Single checkpoint file for both best model and training state
CHECKPOINT_FILENAME_EPOCH = 'checkpoint_epoch_{epoch}.pt'
PAD_TO_MULTIPLE_OF = 64  # Pad sequences to multiple of 8 for better GPU utilization
DEFAULT_WORKER_SEED = 42  # Default seed for data loading workers
MAX_SEQ_LEN = 512
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
        self._learning_rate = None
        self._iterations_in_epoch = 0
        self._lr_decay_iters = None

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
            max_length=MAX_SEQ_LEN,  # Use max_seq_len from config
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
        return max(int(os.environ.get("SLURM_CPUS_PER_TASK", self.config.num_workers)), self.config.num_workers)

    def create_loader(
        self,
        dataset: Optional[BaseDataset],
        tasks: Dict[str, float],
        shuffle: bool = True,
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
        num_workers = self._get_num_workers()
        
        # Create a generator for reproducibility
        g = torch.Generator()
        g.manual_seed(self.worker_seed)  # Use the instance seed instead of a fixed 0
        
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
        current_iter = self._epoch * len(self.train_loader) + self._iterations_in_epoch
        assert self._lr_decay_iters is not None, "lr_decay_iters must be set before calling _get_lr"
        
        # Linear warmup
        if current_iter < self.config.warmup_iters:
            return self.config.learning_rate * current_iter / self.config.warmup_iters
            
        # Cosine decay
        if current_iter > self._lr_decay_iters:
            return self.config.min_lr
            
        # Cosine decay from learning_rate to min_lr
        decay_ratio = (current_iter - self.config.warmup_iters) / (self._lr_decay_iters - self.config.warmup_iters)
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
                    console.info(f"  - {task} loss: {loss:.4f} (from {task_counts[task]} batches)")
            
            if self.logger:
                log_dict = {
                    'val/loss': val_loss,
                    'epoch': self._epoch,
                    'lr': self._learning_rate
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
                self._not_improved_for_eval_epochs = 0
            else:
                self._not_improved_for_eval_epochs += 1

        # Restore original training mode
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

        # Set up evaluation parameters
        self.patience = patience

        # Initialize data loaders
        self.train_loader = self.create_loader(train_dataset, tasks=self.config.tasks, shuffle=True)
        self._lr_decay_iters = self.config.max_epochs * len(self.train_loader) // self.config.gradient_accumulation_steps
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
        
        while self._epoch < self.config.max_epochs:
            self._iterations_in_epoch = 0  # Reset batch counter at start of epoch
            self._set_lr()
            
            # Training loop
            epoch_loss = 0.
            processed_batch_count = 0
            start_time = time.time()

            for batch_idx, batch in enumerate(self.train_loader):
                if self._master_process:
                    start_event.record()
                
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Gradient accumulation
                for micro_step in range(self.config.gradient_accumulation_steps):
                    if self._is_distributed_run:
                        self.model.require_backward_grad_sync = (micro_step == self.config.gradient_accumulation_steps - 1)
                    
                    with self.ctx:
                        outputs = self.model(**batch)
                        loss = outputs['loss'] / self.config.gradient_accumulation_steps
                    
                    self.scaler.scale(loss).backward()

                # Optimizer step
                if self.config.grad_clip != 0.0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                    
                epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                processed_batch_count += 1
                self._iterations_in_epoch += 1  # Update iteration counter

                # Logging
                if batch_idx % self.config.log_interval == 0 and self._master_process:
                    end_event.record()
                    torch.cuda.synchronize()
                    tokens_per_second = batch['input_ids'].numel() / (start_event.elapsed_time(end_event) / 1000.0)
                    avg_loss = epoch_loss / processed_batch_count
                    console.info(f"Epoch {self._epoch}, Step {batch_idx}: loss {avg_loss:.4f}, lr {self._learning_rate:.6f}, tokens/s {tokens_per_second:.2f}")

            # Save checkpoint
            if self._epoch % self.config.save_interval == 0 and self._master_process:
                self._save_ckpt()
            
            # Epoch evaluation
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

    def resume_from_checkpoint(self, filepath: str, resume_training: bool = False):
        """Resume training from a checkpoint file.
        
        Args:
            filepath: Path to the checkpoint file
            resume_training: Whether to resume training state (optimizer, epoch, etc.)
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        state_dict = checkpoint['model']    
        
        # Load model state
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            if self._master_process:
                console.warning("Model state_dict loaded with strict=False.")
                console.warning(f"Missing keys: {missing_keys}")
                console.warning(f"Unexpected keys: {unexpected_keys}")

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

    def test_model_outputs(self, test_dataset: BaseDataset, metric: str) -> float:
        """Evaluate model on test set using metrics based on model outputs (logits).
        
        Args:
            test_dataset: Dataset for testing
            metric: Metric to use for testing (perplexity, rmse, roc_auc, prc_auc)
            
        Returns:
            Float value of the requested metric
        """
        assert metric in ['perplexity', 'rmse', 'roc_auc', 'prc_auc'], \
            f"Metric {metric} not supported for model output evaluation."
        
        # Determine task type based on metric
        task = 'lm' if metric == 'perplexity' else 'prediction'
        
        # Create test data loader for the appropriate task
        test_loader = self.create_loader(test_dataset, shuffle=False, tasks={task: 1.0})
        if test_loader is None:
            raise ValueError("Test dataset is required for testing.")

        y_true = None
        y_pred = None

        for _, batch in enumerate(test_loader):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Collect ground truth
            current_targets = batch['targets'] if 'targets' in batch else batch.get('labels')
            if y_true is None:
                y_true = current_targets
            else:
                y_true = torch.cat((y_true, current_targets))
                
            # Get model outputs
            outputs = self.model(**batch)
            current_logits = outputs['logits']
            
            if y_pred is None:
                y_pred = current_logits 
            else:
                y_pred = torch.cat((y_pred, current_logits))
            
            # Apply masking for language modeling tasks
            if task == 'lm' and 'attention_mask' in batch:
                attention_mask = batch['attention_mask']
                mask = attention_mask.unsqueeze(-1).expand_as(y_pred)
                y_pred[~mask] = -torch.inf

        # Apply inverse transform for regression metrics if needed
        if metric == 'rmse' and hasattr(test_dataset, 'target_transform') and test_dataset.target_transform is not None:    
            y_true = test_dataset.target_transform.inverse_transform(y_true)
            y_pred = test_dataset.target_transform.inverse_transform(y_pred)
            
        test_metric = calculate_metric(y_true.numpy(), y_pred.numpy(), metric)
        assert isinstance(test_metric, float), f"Test metric {metric} is not a float."
        
        if self.logger is not None and self._master_process:
            self.logger.log({f"test/{metric}": test_metric})
        
        return test_metric
    
    def test_generated_samples(self, metric: str, num_samples: int = 1000, temperature: float = 1.0, top_k: int = 25) -> float:
        """Evaluate quality of generated samples using metrics like validity, uniqueness, etc.
        
        Args:
            metric: Metric to use for testing (validity, uniqueness, novelty, etc.)
            num_samples: Number of samples to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            
        Returns:
            Float value of the requested metric
        """
        assert metric in ['validity', 'uniqueness', 'novelty', 'kl_divergence', 'fcd'], \
            f"Metric {metric} not supported for generated sample evaluation."
        
        # Generate samples
        original_model = self.model.module if dist.is_initialized() else self.model
        
        # Generate samples in batches to avoid memory issues
        all_samples = []
        batch_size = min(self.config.batch_size, 32)  # Smaller batch size for generation
        num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
        
        for _ in range(num_batches):
            curr_batch_size = min(batch_size, num_samples - len(all_samples))
            if curr_batch_size <= 0:
                break
                
            samples = original_model.generate(
                tokenizer=self.tokenizer,
                batch_size=curr_batch_size,
                temperature=temperature,
                top_k=top_k,
                device=self.device
            )
            
            # Decode samples if they're token IDs
            if isinstance(samples[0], int) or (isinstance(samples, torch.Tensor) and samples.dtype == torch.long):
                samples = self.tokenizer.batch_decode(samples)
                
            all_samples.extend(samples)
        
        # Calculate metrics on generated samples
        test_metric = calculate_metric(all_samples, None, metric)
        assert isinstance(test_metric, float), f"Test metric {metric} is not a float."
        
        if self.logger is not None and self._master_process:
            self.logger.log({f"test/{metric}": test_metric})
            
            # Log a few example generations
            if len(all_samples) > 0 and isinstance(all_samples[0], str):
                sample_examples = all_samples[:5]
                for i, sample in enumerate(sample_examples):
                    self.logger.log({f"test/sample_{i}": sample})
        
        if self._master_process:
            console.info(f"Test {metric}: {test_metric:.4f}")
        
        return test_metric
