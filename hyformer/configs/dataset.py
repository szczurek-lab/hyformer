from typing import List, Union, Callable, Optional
from dataclasses import dataclass
from hyformer.configs.base import BaseConfig


@dataclass
class DatasetConfig(BaseConfig):
    """Configuration for dataset loading and processing. """
    # Basic dataset information
    dataset_type: str  # Name of the dataset
    train_data_path: str  # Path to training data file
    val_data_path: str  # Path to validation data file
    test_data_path: str  # Path to test data file
    
    # Data keys in files
    data_key: str  # Name of the array storing input data in the .npz file
    target_key: str  # Name of the array storing target values in the .npz file
    
    # Task-specific parameters
    task_type: str  # Type of task (e.g., 'regression', 'classification')
    evaluation_metric: str  # Metric used for evaluation
    num_tasks: int  # Number of prediction tasks
    
    # Transformations
    data_transform: Union[Callable, List, None]  # Transformation applied to input data
    target_transform: Union[Callable, List, None]  # Transformation applied to target data
