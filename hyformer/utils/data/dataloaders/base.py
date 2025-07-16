"""
Base classes for data loaders.

This module provides abstract base classes and core functionality for creating
data loaders with consistent configuration and behavior across the project.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Union

import torch
from torch.utils.data import DataLoader, DistributedSampler

from hyformer.utils.datasets.base import BaseDataset
from hyformer.utils.tokenizers.base import BaseTokenizer
from hyformer.utils.data.collators.sequence import SequenceDataCollator
from hyformer.utils.reproducibility import seed_worker

console = logging.getLogger(__name__)

# Constants
PAD_TO_MULTIPLE_OF = 128  # Pad sequences to multiple of x for better GPU utilization
DEFAULT_WORKER_SEED = 42  # Default seed for data loading workers
SUPPORTED_TASKS = ['lm', 'prediction', 'mlm']


class BaseDataLoader(ABC):
    """Abstract base class for data loader factories.
    
    This class provides common functionality shared by all data loader implementations,
    including worker management, seed handling, and basic validation.
    
    Parameters
    ----------
    tokenizer : BaseTokenizer
        Tokenizer for processing text data
    worker_seed : int, optional
        Random seed for data loading workers
    """
    
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        worker_seed: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.worker_seed = self._initialize_worker_seed(worker_seed)
        console.info(f"{self.__class__.__name__} using worker seed: {self.worker_seed}")
    
    def _initialize_worker_seed(self, worker_seed: Optional[int] = None) -> int:
        """Initialize worker seed from environment or provided value."""
        if worker_seed is not None:
            return worker_seed
            
        try:
            default_worker_seed = int(os.environ.get("PYTHONHASHSEED", DEFAULT_WORKER_SEED))
        except (ValueError, TypeError):
            default_worker_seed = DEFAULT_WORKER_SEED
            
        return default_worker_seed
    
    def get_num_workers(self, num_workers: Optional[int] = None) -> int:
        """Get the number of workers from SLURM environment or provided value.
        
        Parameters
        ----------
        num_workers : int, optional
            Number of workers to use. If None, gets from SLURM_CPUS_PER_TASK
            environment variable or defaults to 0
            
        Returns
        -------
        int
            Number of workers to use for data loading
        """
        if num_workers is not None:
            return num_workers
            
        _num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
        console.info(f"Using {_num_workers} workers for data loading")
        return _num_workers
    
    def validate_tasks(self, tasks: Dict[str, float]) -> None:
        """Validate that all tasks are supported.
        
        Parameters
        ----------
        tasks : Dict[str, float]
            Dictionary of tasks to validate
            
        Raises
        ------
        ValueError
            If any task is not supported
        """
        for task in tasks:
            if task not in SUPPORTED_TASKS:
                raise ValueError(f"Unsupported task: {task}. Supported tasks are {SUPPORTED_TASKS}.")
    
    def create_generator(self) -> torch.Generator:
        """Create a PyTorch generator with the configured seed.
        
        Returns
        -------
        torch.Generator
            Generator instance with the worker seed
        """
        g = torch.Generator()
        g.manual_seed(self.worker_seed)
        return g
    
    @abstractmethod
    def create_loader(
        self,
        dataset: Optional[BaseDataset],
        tasks: Dict[str, float],
        **kwargs
    ) -> Optional[DataLoader]:
        """Create a data loader with the specified configuration.
        
        Parameters
        ----------
        dataset : BaseDataset, optional
            Dataset to create loader for
        tasks : Dict[str, float]
            Task configuration
        **kwargs
            Additional arguments for loader configuration
            
        Returns
        -------
        DataLoader or None
            Configured DataLoader instance or None if dataset is None
        """
        pass
