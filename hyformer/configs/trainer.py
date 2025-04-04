import math
import logging
from typing import Dict, Optional
from dataclasses import dataclass, field
from hyformer.configs.base import BaseConfig

console = logging.getLogger(__file__)

@dataclass
class TrainerConfig(BaseConfig):
    """Configuration for model training. """
    # Basic training parameters
    batch_size: int  # Number of samples per batch
    learning_rate: float  # Base learning rate for optimization
    weight_decay: float  # Weight decay coefficient for regularization
    max_epochs: int  # Maximum number of training epochs
    tasks: Dict[str, float]  # Dictionary mapping task names to their weights
    
    # Runtime parameters
    compile: bool  # Whether to compile the model for faster execution
    enable_ddp: bool  # Whether to enable distributed data parallel training
    dtype: str  # Data type for computation (float32, float16, bfloat16)
    num_workers: int  # Number of data loading workers
    
    # Optimization parameters
    beta1: float  # Beta1 coefficient for Adam optimizer
    beta2: float  # Beta2 coefficient for Adam optimizer
    gradient_accumulation_steps: int  # Number of steps to accumulate gradients
    grad_clip: float  # Gradient clipping value
    
    # Scheduler parameters
    decay_lr: bool  # Whether to decay learning rate
    warmup_iters: int  # Number of warmup iterations
    min_lr: float  # Minimum learning rate
    
    # Logging parameters
    log_interval: int  # Interval between logging (in iterations)
    save_interval: int  # Interval between saving (in epochs)
    
    def __post_init__(self):
        """Initialize derived parameters and validate configuration."""
        self._normalize_task_probabilities()

    def _normalize_task_probabilities(self):
        """Normalize task probabilities to sum to 1."""
        total = sum(self.tasks.values())
        for task in self.tasks:
            self.tasks[task] = self.tasks[task] / total
        