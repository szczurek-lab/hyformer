import os
from typing import Dict, Any

from hyformer.configs.dataset import DatasetConfig
from hyformer.utils.datasets.base import BaseDataset

# Import the dataset implementations directly
from hyformer.utils.datasets.sequence import SequenceDataset

class AutoDataset:
    """Factory class for automatic dataset selection based on dataset type.
    
    This class provides a simple interface for creating dataset instances
    from configuration objects, using a simple if-else approach to select
    the appropriate dataset implementation.
    """
    @classmethod
    def from_config(
            cls,
            config: DatasetConfig,
            split: str,
            root: str = None
    ) -> BaseDataset:
        """Create a dataset from configuration.
        
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
        BaseDataset
            Initialized dataset instance
            
        Raises
        ------
        ValueError
            If the dataset type is not supported or if the config is invalid
        """
        # Validate config
        if not hasattr(config, 'dataset_type'):
            raise ValueError("Config must have a 'dataset_type' attribute")
            
        # Simple if-else factory pattern
        if config.dataset_type == 'sequence':
            return SequenceDataset.from_config(config, split=split, root=root)
        else:
            raise ValueError(f"Dataset type '{config.dataset_type}' is not supported")
    
    @classmethod
    def get_supported_dataset_types(cls) -> Dict[str, str]:
        """Get a mapping of supported dataset types to their descriptions.
        
        Returns
        -------
        dict
            Dictionary mapping dataset types to descriptions
        """
        return {
            'sequence': 'Dataset for sequence data like SMILES strings'
        }
