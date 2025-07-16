"""
Data loaders for creating PyTorch DataLoaders with consistent configuration.

This package provides classes and utilities for creating data loaders with
automatic configuration, task-specific optimizations, and consistent behavior
across different use cases.

Classes
-------
AutoDataLoader
    Factory class for creating data loaders based on task configuration
BaseDataLoader
    Abstract base class for data loader factories
SequenceDataLoader
    Concrete implementation for sequence-based data loading

Functions
---------
create_prediction_loader
    Create a data loader optimized for prediction/encoding tasks
create_training_loader
    Create a data loader optimized for training
create_evaluation_loader
    Create a data loader optimized for evaluation

Examples
--------
>>> from hyformer.utils.dataloaders import AutoDataLoader, SequenceDataLoader
>>> from hyformer.utils.datasets.base import BaseDataset
>>> 
>>> # Create a prediction loader using AutoDataLoader
>>> loader = AutoDataLoader.for_prediction(
...     dataset=my_dataset,
...     tokenizer=my_tokenizer,
...     batch_size=32
... )
>>> 
>>> # Create a custom loader using SequenceDataLoader
>>> seq_loader = SequenceDataLoader(tokenizer=my_tokenizer)
>>> loader = seq_loader.create_loader(
...     dataset=my_dataset,
...     tasks={'prediction': 1.0},
...     batch_size=32
... )
"""

from .auto import (
    AutoDataLoader,
    create_prediction_loader,
    create_training_loader,
    create_evaluation_loader
)
from .base import BaseDataLoader
from .sequence import SequenceDataLoader

__all__ = [
    'AutoDataLoader',
    'BaseDataLoader',
    'SequenceDataLoader',
    'create_prediction_loader',
    'create_training_loader',
    'create_evaluation_loader'
] 