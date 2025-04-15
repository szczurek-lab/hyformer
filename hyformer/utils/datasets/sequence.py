import os
import numpy as np
from typing import List, Callable, Optional, Union, Tuple, Any, Dict

from hyformer.configs.dataset import DatasetConfig
from hyformer.utils.datasets.base import BaseDataset
from hyformer.utils.transforms.auto import AutoTransform, AutoTargetTransform
from hyformer.utils.datasets.storage.base import DataStorage
from hyformer.utils.datasets.storage.auto import AutoStorage

# Default settings - shared with storage module
_ALLOW_PICKLE = False  # Security setting - controls whether pickled objects are allowed in NPZ files

class SequenceDataset(BaseDataset):
    """Dataset for loading and processing sequence data from various file formats.

    This class is designed to load sequence data (like SMILES strings for molecules) 
    from different file formats with configurable data and target keys. It handles:
    
    1. Loading data from various file formats (NPZ, CSV)
    2. Processing targets for classification or regression tasks
    3. Applying transformations to inputs and targets
    4. Validating data consistency and shapes
    
    Supported File Formats:
    - NPZ files: NumPy's compressed archive format
    - CSV files: Comma-separated values files
    
    Parameters
    ----------
    data : list of str
        Input sequence data (e.g., SMILES strings for molecules)
    target : numpy.ndarray, optional
        Target values (e.g., molecular properties), typically a 2D array 
        with shape (n_samples, n_tasks)
    data_transform : callable or list, optional
        Transformation(s) to apply to input data
    target_transform : callable or list, optional
        Transformation(s) to apply to target values
    prediction_task_type : {'classification', 'regression'}, optional
        Type of prediction task, affects target data type validation
    num_prediction_tasks : int, optional
        Number of prediction tasks (columns in the target array)
    test_metric : str, optional
        Metric used for evaluation (e.g., 'accuracy', 'rmse')
        
    Examples
    --------
    >>> # Load from config
    >>> dataset = SequenceDataset.from_config(config, split='train', root='data/')
    >>> 
    >>> # Create directly
    >>> dataset = SequenceDataset(
    ...     data=['CCCC', 'CCC', 'CCCN'],
    ...     target=np.array([[0], [1], [0]]),
    ...     prediction_task_type='classification'
    ... )
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
        
        Parameters
        ----------
        data : list of str
            Input sequence data (e.g., SMILES strings for molecules)
        target : numpy.ndarray, optional
            Target values (e.g., molecular properties), typically a 2D array
            with shape (n_samples, n_tasks)
        data_transform : callable or list, optional
            Transformation(s) to apply to input data
        target_transform : callable or list, optional
            Transformation(s) to apply to target values
        prediction_task_type : {'classification', 'regression'}, optional
            Type of prediction task, affects target data type validation
        num_prediction_tasks : int, optional
            Number of prediction tasks (columns in the target array)
        test_metric : str, optional
            Metric used for evaluation (e.g., 'accuracy', 'rmse')
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
        """Get the filepath for the specified data split.
        
        Parameters
        ----------
        config : DatasetConfig
            Dataset configuration containing file paths
        root : str
            Root directory to prepend to paths
        split : {'train', 'val', 'test'}
            Data split
            
        Returns
        -------
        str
            Full path to the data file for the specified split
        
        """
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
        
        This method ensures that targets have the correct shape, match the data length,
        and have the appropriate data type for the prediction task.
        
        Parameters
        ----------
        target : numpy.ndarray
            Target array to process
        data : numpy.ndarray
            Input data array (for length validation)
        prediction_task_type : {'classification', 'regression'}, optional
            Type of task
        num_prediction_tasks : int, optional
            Expected number of tasks (columns in target)
            
        Returns
        -------
        numpy.ndarray
            Processed target array with appropriate shape and dtype
            
        Raises
        ------
        ValueError
            If target shape is invalid or doesn't match expectations
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
    def _get_storage(filepath: str, **kwargs) -> DataStorage:
        """Get the appropriate storage backend for a file.
        
        Parameters
        ----------
        filepath : str
            Path to the data file
        **kwargs
            Additional arguments to pass to the storage backend
            
        Returns
        -------
        DataStorage
            Initialized storage backend
            
        Raises
        ------
        ValueError
            If the file format is not supported
        """
        # Use AutoStorage to create the appropriate storage backend
        return AutoStorage.from_path(filepath, **kwargs)
    
    @staticmethod
    def _load(filepath: str, data_key: str = 'sequence', target_key: str = 'properties', 
              prediction_task_type: Optional[str] = None, num_prediction_tasks: Optional[int] = None,
              **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load data from a file.
        
        This method detects the file format and uses the appropriate storage backend
        to load data and target values.
        
        Parameters
        ----------
        filepath : str
            Path to the data file
        data_key : str, default='sequence'
            Key/column name for input data
        target_key : str or list of str, default='properties'
            Key/column name for target values
        prediction_task_type : {'classification', 'regression'}, optional
            Type of prediction task
        num_prediction_tasks : int, optional
            Number of prediction tasks
        **kwargs
            Additional arguments to pass to the storage backend
            
        Returns
        -------
        tuple
            Tuple of (data_array, target_array), where target_array may be None
            
        Raises
        ------
        ValueError
            If the file cannot be loaded or format is not supported
        """
        # Get the appropriate storage backend
        storage = SequenceDataset._get_storage(filepath, **kwargs)
        
        # Load data and target
        data = storage.load_data(data_key)
        target = storage.load_target(target_key)
        
        # Process target if it exists
        if target is not None:
            target = SequenceDataset._process_target(target, data, prediction_task_type, num_prediction_tasks)

        return data, target

    def __add__(self, other):
        """Concatenate two SequenceDataset objects.
        
        Parameters
        ----------
        other : SequenceDataset
            Another SequenceDataset to concatenate with this one
            
        Returns
        -------
        SequenceDataset
            A new SequenceDataset containing data from both datasets
            
        Raises
        ------
        ValueError
            If other is not a SequenceDataset instance
        TypeError
            If the datasets have incompatible types
        """
        if not isinstance(other, SequenceDataset):
            raise ValueError("Can only add SequenceDataset to SequenceDataset")
            
        # Check compatibility
        if self.prediction_task_type != other.prediction_task_type:
            raise TypeError(f"Cannot combine datasets with different prediction_task_types: "
                            f"{self.prediction_task_type} vs {other.prediction_task_type}")
        
        # Concatenate data and targets
        data = np.concatenate((self.data, other.data), axis=0)
        target = None
        if self.target is not None and other.target is not None:
            if self.target.shape[1] != other.target.shape[1]:
                raise ValueError(f"Target shapes don't match: {self.target.shape[1]} vs {other.target.shape[1]}")
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
        """Create a SequenceDataset from a configuration.
        
        Parameters
        ----------
        config : DatasetConfig
            Configuration object containing dataset parameters
        split : {'train', 'val', 'test'}
            Data split to load
        root : str, optional
            Root directory to prepend to file paths
            
        Returns
        -------
        SequenceDataset
            Initialized SequenceDataset instance
            
        Examples
        --------
        >>> config = DatasetConfig(
        ...     dataset_type='sequence',
        ...     train_data_path='data/train.npz',
        ...     val_data_path='data/val.npz',
        ...     test_data_path='data/test.npz',
        ...     data_key='smiles',
        ...     target_key='activity',
        ...     prediction_task_type='classification',
        ...     num_prediction_tasks=1,
        ...     test_metric='accuracy',
        ...     data_transform=None,
        ...     target_transform=None
        ... )
        >>> dataset = SequenceDataset.from_config(config, split='train')
        """
        # Get filepath
        filepath = cls._get_filepath(config, root, split)
        
        # Get format-specific options from config
        format_kwargs = {}
        if hasattr(config, 'storage_options'):
            format_kwargs.update(config.storage_options)
        
        # Add common options
        format_kwargs['allow_pickle'] = getattr(config, 'allow_pickle', _ALLOW_PICKLE)
        
        # Load data
        data, target = cls._load(
            filepath, 
            data_key=config.data_key,
            target_key=config.target_key,
            prediction_task_type=config.prediction_task_type, 
            num_prediction_tasks=config.num_prediction_tasks,
            **format_kwargs
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
