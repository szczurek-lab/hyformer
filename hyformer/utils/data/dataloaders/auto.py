"""
Automatic data loader factory selection.

This module provides factory classes for creating data loaders with automatic
configuration based on task types and use cases.
"""

from typing import Dict, Optional, Union

from hyformer.utils.datasets.base import BaseDataset
from hyformer.utils.tokenizers.base import BaseTokenizer
from hyformer.utils.data.dataloaders.sequence import SequenceDataLoader


class AutoDataLoader:
    """Factory class for automatic data loader creation based on task configuration.
    
    This class provides a simple interface for creating data loaders
    with common configurations for different use cases.
    """
    
    @classmethod
    def from_config(
        cls,
        dataset: BaseDataset,
        tokenizer: BaseTokenizer,
        task_type: str = "prediction",
        batch_size: int = 32,
        shuffle: bool = None,
        **kwargs
    ):
        """Create a data loader from configuration parameters.
        
        Parameters
        ----------
        dataset : BaseDataset
            Dataset to create loader for
        tokenizer : BaseTokenizer
            Tokenizer for processing data
        task_type : str, optional
            Type of task ('prediction', 'lm', 'mlm', 'mixed'), by default 'prediction'
        batch_size : int, optional
            Batch size, by default 32
        shuffle : bool, optional
            Whether to shuffle data. If None, auto-determines based on task_type
        **kwargs
            Additional arguments passed to the factory
            
        Returns
        -------
        DataLoader
            Configured data loader
            
        Raises
        ------
        ValueError
            If task_type is not supported
        """
        # Create factory
        factory = SequenceDataLoader(tokenizer=tokenizer)
        
        # Configure tasks based on task_type
        if task_type == "prediction":
            tasks = {'prediction': 1.0}
            default_shuffle = False  # Usually don't shuffle for prediction/encoding
        elif task_type == "lm":
            tasks = {'lm': 1.0}
            default_shuffle = True
        elif task_type == "mlm":
            tasks = {'mlm': 1.0}
            default_shuffle = True
        elif task_type == "mixed":
            # Default mixed configuration
            tasks = {'prediction': 0.3, 'lm': 0.4, 'mlm': 0.3}
            default_shuffle = True
        else:
            raise ValueError(f"Unsupported task_type: {task_type}. "
                           f"Supported types: 'prediction', 'lm', 'mlm', 'mixed'")
        
        # Use provided shuffle or default
        if shuffle is None:
            shuffle = default_shuffle
        
        return factory.create_loader(
            dataset=dataset,
            tasks=tasks,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )
    
    @classmethod
    def for_prediction(
        cls,
        dataset: BaseDataset,
        tokenizer: BaseTokenizer,
        batch_size: int = 32,
        **kwargs
    ):
        """Create a data loader optimized for prediction/encoding tasks.
        
        Parameters
        ----------
        dataset : BaseDataset
            Dataset containing sequences to encode
        tokenizer : BaseTokenizer
            Tokenizer for processing sequences
        batch_size : int, optional
            Batch size, by default 32
        **kwargs
            Additional arguments passed to the factory
            
        Returns
        -------
        DataLoader
            Data loader configured for prediction tasks
        """
        return cls.from_config(
            dataset=dataset,
            tokenizer=tokenizer,
            task_type="prediction",
            batch_size=batch_size,
            shuffle=False,  # No need to shuffle for encoding
            num_workers=kwargs.pop('num_workers', 0),  # Default to 0 for encoding
            distributed=kwargs.pop('distributed', False),  # Disable distributed by default
            **kwargs
        )
    
    @classmethod
    def for_training(
        cls,
        dataset: BaseDataset,
        tokenizer: BaseTokenizer,
        tasks: Optional[Dict[str, float]] = None,
        batch_size: int = 32,
        **kwargs
    ):
        """Create a data loader optimized for training.
        
        Parameters
        ----------
        dataset : BaseDataset
            Training dataset
        tokenizer : BaseTokenizer
            Tokenizer for processing data
        tasks : Dict[str, float], optional
            Task configuration. If None, uses mixed tasks
        batch_size : int, optional
            Batch size, by default 32
        **kwargs
            Additional arguments passed to the factory
            
        Returns
        -------
        DataLoader
            Data loader configured for training
        """
        factory = SequenceDataLoader(tokenizer=tokenizer)
        
        if tasks is None:
            tasks = {'prediction': 0.3, 'lm': 0.4, 'mlm': 0.3}
        
        return factory.create_loader(
            dataset=dataset,
            tasks=tasks,
            batch_size=batch_size,
            shuffle=True,  # Always shuffle for training
            **kwargs
        )
    
    @classmethod
    def for_evaluation(
        cls,
        dataset: BaseDataset,
        tokenizer: BaseTokenizer,
        task_type: str = "prediction",
        batch_size: int = 32,
        **kwargs
    ):
        """Create a data loader optimized for evaluation.
        
        Parameters
        ----------
        dataset : BaseDataset
            Evaluation dataset
        tokenizer : BaseTokenizer
            Tokenizer for processing data
        task_type : str, optional
            Type of evaluation task, by default "prediction"
        batch_size : int, optional
            Batch size, by default 32
        **kwargs
            Additional arguments passed to the factory
            
        Returns
        -------
        DataLoader
            Data loader configured for evaluation
        """
        return cls.from_config(
            dataset=dataset,
            tokenizer=tokenizer,
            task_type=task_type,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle for evaluation
            **kwargs
        )


# Convenience functions following the datasets pattern
def create_prediction_loader(
    dataset: BaseDataset,
    tokenizer: BaseTokenizer,
    batch_size: int = 32,
    shuffle: bool = False,
    **kwargs
):
    """Create a data loader specifically for prediction tasks.
    
    This is a convenience function for the common case of creating
    data loaders for encoding/prediction tasks.
    
    Parameters
    ----------
    dataset : BaseDataset
        Dataset containing sequences to encode
    tokenizer : BaseTokenizer
        Tokenizer for processing sequences
    batch_size : int, optional
        Batch size, by default 32
    shuffle : bool, optional
        Whether to shuffle data, by default False
    **kwargs
        Additional arguments passed to AutoDataLoader.for_prediction
        
    Returns
    -------
    DataLoader
        Data loader configured for prediction tasks
    """
    return AutoDataLoader.for_prediction(
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        **kwargs
    )


def create_training_loader(
    dataset: BaseDataset,
    tokenizer: BaseTokenizer,
    tasks: Optional[Dict[str, float]] = None,
    batch_size: int = 32,
    **kwargs
):
    """Create a data loader specifically for training.
    
    Parameters
    ----------
    dataset : BaseDataset
        Training dataset
    tokenizer : BaseTokenizer
        Tokenizer for processing data
    tasks : Dict[str, float], optional
        Task configuration
    batch_size : int, optional
        Batch size, by default 32
    **kwargs
        Additional arguments passed to AutoDataLoader.for_training
        
    Returns
    -------
    DataLoader
        Data loader configured for training
    """
    return AutoDataLoader.for_training(
        dataset=dataset,
        tokenizer=tokenizer,
        tasks=tasks,
        batch_size=batch_size,
        **kwargs
    )


def create_evaluation_loader(
    dataset: BaseDataset,
    tokenizer: BaseTokenizer,
    task_type: str = "prediction",
    batch_size: int = 32,
    **kwargs
):
    """Create a data loader specifically for evaluation.
    
    Parameters
    ----------
    dataset : BaseDataset
        Evaluation dataset
    tokenizer : BaseTokenizer
        Tokenizer for processing data
    task_type : str, optional
        Type of evaluation task, by default "prediction"
    batch_size : int, optional
        Batch size, by default 32
    **kwargs
        Additional arguments passed to AutoDataLoader.for_evaluation
        
    Returns
    -------
    DataLoader
        Data loader configured for evaluation
    """
    return AutoDataLoader.for_evaluation(
        dataset=dataset,
        tokenizer=tokenizer,
        task_type=task_type,
        batch_size=batch_size,
        **kwargs
    ) 