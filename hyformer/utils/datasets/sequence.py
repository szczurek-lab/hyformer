import os
import numpy as np
from typing import List, Callable, Optional, Union, Tuple, Any, Dict

from hyformer.configs.dataset import DatasetConfig
from hyformer.utils.datasets.base import BaseDataset
from hyformer.utils.transforms.auto import AutoTransform, AutoTargetTransform

_ALLOW_PICKLE = False

class SequenceDataset(BaseDataset):
    """Dataset for sequence data.

    Loads sequence data from numpy .npz files with configurable data and target keys.
    Supports data transformations and handles classification/regression tasks appropriately.
    """

    def __init__(
            self,
            data: List[str],
            target: Optional[np.ndarray] = None,
            data_transform: Optional[Union[Callable, List]] = None,
            target_transform: Optional[Union[Callable, List]] = None,
            prediction_task_type: Optional[str] = None,
            num_prediction_tasks: Optional[int] = None,
            test_metric: Optional[str] = None
    ) -> None:
        """Initialize a sequence dataset.
        
        Args:
            data: Input data (e.g., SMILES strings)
            target: Target values (e.g., properties)
            data_transform: Transformation to apply to input data
            target_transform: Transformation to apply to target values
            prediction_task_type: Type of task ('classification' or 'regression')
            num_prediction_tasks: Number of prediction tasks
            test_metric: Metric used for evaluation
        """
        super().__init__(
            data=data, target=target, 
            data_transform=data_transform, 
            target_transform=target_transform)
        
        # Store task-specific attributes
        self.prediction_task_type = prediction_task_type
        self.num_prediction_tasks = num_prediction_tasks
        self.test_metric = test_metric
    
    @staticmethod
    def _get_filepath(config: DatasetConfig, root: str, split: str) -> str:
        """Get the filepath for the specified split."""
        if split == 'train':
            filepath = config.train_data_path
        elif split == 'val':
            filepath = config.val_data_path
        elif split == 'test':
            filepath = config.test_data_path
        else:
            raise ValueError(f"Invalid split: '{split}'. Must be 'train', 'val', or 'test'.")
            
        return os.path.join(root, filepath) if root else filepath
    
    @staticmethod
    def _process_target(target: np.ndarray, data: np.ndarray, prediction_task_type: Optional[str] = None, 
                        num_prediction_tasks: Optional[int] = None) -> np.ndarray:
        """Process target data for consistency and type correctness.
        
        Args:
            target: Target array to process
            data: Input data array (for length validation)
            prediction_task_type: Type of task ('classification' or 'regression')
            num_prediction_tasks: Expected number of tasks
            
        Returns:
            Processed target array
            
        Raises:
            ValueError: If target shape is invalid or doesn't match expectations
        """
        # Reshape 1D targets to 2D
        if len(target.shape) == 1:
            target = target.reshape(-1, 1)
            
        # Validate target shape
        if len(target.shape) != 2:
            raise ValueError(f"Target has unexpected shape: {target.shape}")
        if len(data) != len(target):
            raise ValueError(f"Data and target lengths don't match: {len(data)} vs {len(target)}")
        if num_prediction_tasks and target.shape[1] != num_prediction_tasks:
            raise ValueError(f"Target has {target.shape[1]} tasks, expected {num_prediction_tasks}")
            
        # Convert target type based on task
        if prediction_task_type == 'classification' and target.dtype != np.int64:
            target = target.astype(np.int64)
        elif prediction_task_type == 'regression' and target.dtype != np.float32:
            target = target.astype(np.float32)
            
        return target
    
    @staticmethod
    def _load(filepath: str, data_key: str = 'sequence', target_key: str = 'properties', 
              prediction_task_type: Optional[str] = None, num_prediction_tasks: Optional[int] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load data from a file."""
        try:
            data_file = np.load(filepath, allow_pickle=_ALLOW_PICKLE)
        except Exception as e:
            raise ValueError(f"Failed to load data from {filepath}: {str(e)}")
            
        # Load data
        if data_key not in data_file:
            raise ValueError(f"Key '{data_key}' not found in {filepath}")
        data = data_file[data_key]
        
        # Load target if available
        target = data_file[target_key] if target_key in data_file else None
        
        # Process target if it exists
        if target is not None:
            target = SequenceDataset._process_target(target, data, prediction_task_type, num_prediction_tasks)

        return data, target

    def __add__(self, other):
        """Concatenate two SequenceDataset objects."""
        if not isinstance(other, SequenceDataset):
            raise ValueError("Can only add SequenceDataset to SequenceDataset")
        
        # Concatenate data and targets
        data = np.concatenate((self.data, other.data), axis=0)
        target = None
        if self.target is not None and other.target is not None:
            target = np.concatenate((self.target, other.target), axis=0)
        
        # Prefer non-None transforms
        data_transform = self.data_transform or other.data_transform
        target_transform = self.target_transform or other.target_transform
        
        return SequenceDataset(
            data=data,
            target=target,
            data_transform=data_transform,
            target_transform=target_transform,
            prediction_task_type=self.prediction_task_type,
            num_prediction_tasks=self.num_prediction_tasks,
            test_metric=self.test_metric
        )

    @classmethod
    def from_config(cls, config: DatasetConfig, split: str, root: str = None) -> 'SequenceDataset':
        """Create a SequenceDataset from a configuration."""
        # Get filepath
        filepath = cls._get_filepath(config, root, split)
        
        # Load data
        data, target = cls._load(
            filepath, 
            data_key=config.data_key,
            target_key=config.target_key,
            prediction_task_type=config.prediction_task_type, 
            num_prediction_tasks=config.num_prediction_tasks
        )

        # Set up transforms
        data_transform = None
        if hasattr(config, 'data_transform') and config.data_transform and split == 'train':
            data_transform = AutoTransform.from_config(config.data_transform)
            
        target_transform = None
        if hasattr(config, 'target_transform') and config.target_transform:
            target_transform = AutoTargetTransform.from_config(config.target_transform)

        return cls(
            data=data, 
            target=target, 
            data_transform=data_transform, 
            target_transform=target_transform,
            prediction_task_type=config.prediction_task_type,
            num_prediction_tasks=config.num_prediction_tasks,
            test_metric=config.test_metric
        )
