import numpy as np

from typing import Any, List, Callable, Optional, Union, Dict
from torch.utils.data.dataset import Dataset


class BaseDataset(Dataset):
    """Base class for all datasets.
    
    This class extends PyTorch's Dataset class to provide a common interface for all datasets
    in the project. It handles data and target transformations and provides a standardized
    dictionary output format.
    
    Parameters
    ----------
    data : Any, optional
        The input data for the dataset
    target : Any, optional
        The target data for the dataset (e.g., labels)
    data_transform : callable or list, optional
        Transformation(s) to apply to input data
    target_transform : callable or list, optional
        Transformation(s) to apply to target data
    """
    def __init__(
            self,
            data: Any = None,
            target: Any = None,
            data_transform: Optional[Union[Callable, List]] = None,
            target_transform: Optional[Union[Callable, List]] = None
    ) -> None:
        """Initialize a dataset with data and transformations.
        
        Parameters
        ----------
        data : Any, optional
            The input data for the dataset
        target : Any, optional
            The target data for the dataset (e.g., labels)
        data_transform : callable or list, optional
            Transformation(s) to apply to input data
        target_transform : callable or list, optional
            Transformation(s) to apply to target data
        """
        super().__init__()
        self.data = data
        self.target = target
        self.data_transform = data_transform
        self.target_transform = target_transform

    def __len__(self):
        """Get the number of items in the dataset.
        
        Returns
        -------
        int
            Number of items in the dataset
        """
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get an item from the dataset by index.
        
        Parameters
        ----------
        idx : int
            Index of the item to get
            
        Returns
        -------
        dict
            Dictionary containing the input data and target data.
            The dictionary has the following keys:
            - 'data': The input data (transformed if transforms are set)
            - 'target': The target data (transformed if transforms are set),
                        or None if no target exists
        """
        x = self.data[idx]
        if self.data_transform is not None:
            x = self.data_transform(x)
        
        # Initialize result with both keys, setting target to None by default
        result = {'data': x, 'target': None}
        
        if self.target is not None:
            y = self.target[idx]
            if self.target_transform is not None:
                y = self.target_transform(y)
            result['target'] = y
        
        return result
