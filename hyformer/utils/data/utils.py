"""
Utility functions for data loading.

This module provides utility functions that are used across the data loading
infrastructure but don't belong to any specific class.
"""

import torch
from typing import Dict, Optional

from hyformer.utils.datasets.base import BaseDataset
from hyformer.utils.tokenizers.base import BaseTokenizer


SUPPORTED_TASKS = ['lm', 'prediction', 'mlm']


def create_loader(
    dataset: Optional[BaseDataset],
    tasks: Dict[str, float],
    tokenizer: BaseTokenizer,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    batch_size: int = 32,
    **kwargs
) -> Optional[torch.utils.data.DataLoader]:
    """Create a data loader with optimized settings using SequenceDataLoader.
    
    This is a convenience function that creates a SequenceDataLoader instance
    and uses it to create a data loader with the specified configuration.
    
    Parameters
    ----------
    dataset : BaseDataset, optional
        Dataset to create loader for
    tasks : Dict[str, float]
        Dictionary of tasks to use for collation. 
        Supported tasks are 'lm', 'prediction', and 'mlm'
    tokenizer : BaseTokenizer
        Tokenizer for processing data
    shuffle : bool, optional
        Whether to shuffle the data, by default True
    num_workers : int, optional
        Number of workers for data loading, by default None (auto-detected)
    batch_size : int, optional
        Batch size, by default 32
    **kwargs
        Additional arguments passed to SequenceDataLoader.create_loader
        
    Returns
    -------
    DataLoader or None
        Configured DataLoader instance or None if dataset is None
        
    Raises
    ------
    ValueError
        If any task in tasks is not supported
    """
    if dataset is None:
        return None
        
    # Validate tasks
    for task in tasks:
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task: {task}. Supported tasks are {SUPPORTED_TASKS}.")
    
    # Import here to avoid circular imports
    from hyformer.utils.data.dataloaders.sequence import SequenceDataLoader
    
    # Create SequenceDataLoader instance and use it to create the loader
    factory = SequenceDataLoader(tokenizer=tokenizer)
    
    return factory.create_loader(
        dataset=dataset,
        tasks=tasks,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )