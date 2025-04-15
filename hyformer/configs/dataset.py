from typing import List, Union, Callable, Optional, Dict, Any
from dataclasses import dataclass, field
from hyformer.configs.base import BaseConfig


@dataclass
class DatasetConfig(BaseConfig):
    """Configuration for dataset loading and processing.
    
    This configuration class defines the parameters needed to load and process
    datasets of different types. It supports configuration for different file formats,
    data keys, and transformations.
    """
    # Basic dataset information
    dataset_type: str  # Name of the dataset
    train_data_path: str  # Path to training data file
    val_data_path: str  # Path to validation data file
    test_data_path: str  # Path to test data file
    
    # Data keys in files
    data_key: str  # Name of the column/array storing input data
    target_key: str  # Name of the column/array storing target values
    
    # Task-specific parameters
    prediction_task_type: str  # Type of task (e.g., 'regression', 'classification')
    test_metric: str  # Metric used for evaluation
    num_prediction_tasks: int  # Number of prediction tasks
    
    # Transformations
    data_transform: Union[Callable, List, None] = None  # Transformation applied to input data
    target_transform: Union[Callable, List, None] = None  # Transformation applied to target data
    
    # Storage options - unified field for all storage backends
    storage_options: Dict[str, Any] = field(default_factory=dict)  # Options for data storage (format-specific)
