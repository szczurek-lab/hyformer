import numpy as np

from typing import Any, List, Callable, Optional, Union, Dict
from torch.utils.data.dataset import Dataset


class BaseDataset(Dataset):
    """Base class for all datasets.
    
    This class extends PyTorch's Dataset class to provide a common interface for all datasets
    in the project. It handles data and target transformations.
    """
    def __init__(
            self,
            data: Any = None,
            target: Any = None,
            data_transform: Optional[Union[Callable, List]] = None,
            target_transform: Optional[Union[Callable, List]] = None
    ) -> None:
        super().__init__()
        self.data = data
        self.target = target
        self.data_transform = data_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get an item from the dataset.
        
        Args:
            idx: Index of the item to get
            
        Returns:
            Dictionary containing the input data and target data.
            The dictionary has the following keys:
            - 'data': The input data (e.g., SMILES string)
            - 'target': The target data (e.g., properties), or None if no target exists
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
