from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List, Union

class DataStorage(ABC):
    """Abstract base class for different data storage formats.
    
    This class defines the interface that all storage backends must implement.
    Storage backends are responsible for loading data from different file formats,
    such as NPZ, CSV, LMDB, etc.
    """
    
    @abstractmethod
    def load_data(self, data_key: str) -> Any:
        """Load input data from storage.
        
        Parameters
        ----------
        data_key : str
            Key or identifier to access the input data
            
        Returns
        -------
        Any
            The loaded input data
            
        Raises
        ------
        ValueError
            If the data cannot be loaded
        """
        pass
    
    @abstractmethod
    def load_target(self, target_key: Union[str, List[str]]) -> Optional[Any]:
        """Load target data from storage.
        
        Parameters
        ----------
        target_key : str or list of str
            Key or identifier to access the target data
            
        Returns
        -------
        Any or None
            The loaded target data, or None if not available
            
        Raises
        ------
        ValueError
            If the target data cannot be loaded
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_path(cls, path: str, **kwargs) -> 'DataStorage':
        """Create a storage instance from a file path.
        
        Parameters
        ----------
        path : str
            Path to the data file
        **kwargs
            Additional backend-specific arguments
            
        Returns
        -------
        DataStorage
            Initialized storage backend
            
        Raises
        ------
        ValueError
            If the file cannot be loaded
        """
        pass 