""" A torch dataset for sequential data.
"""

import torch

import os

import numpy as np

from tqdm import tqdm
from typing import List, Callable, Optional, Union

from jointformer.configs.dataset import DatasetConfig
from jointformer.utils.datasets.base import BaseDataset
from jointformer.utils.data import read_strings_from_file



class SequentialDataset(BaseDataset):
    """A torch dataset for sequential data designed to handle sequential data and its corresponding targets.

    Args:
        data: (Any, optional): The data.
        target: (Any, optional): The target.
        target_filepath (str, optional): The file path to the target.
        transform (callable or list, optional): A function or a list of functions to apply to the data.
        target_transform (callable or list, optional): A function or a list of functions to apply to the target.
        max_sequence_length (int, optional): The maximum sequence length.
        num_samples (int, optional): The number of samples to include in the dataset.
        task_type (str, optional): The type of task (e.g., classification, regression).
        seed (int, optional): The random seed.

    Attributes:
        max_sequence_length (int, optional): The maximum sequence length.
        num_samples (int, optional): The number of samples in the dataset.

    """

    def __init__(
            self,
            data: str = None,
            target: Optional[str] = None,
            transform: Optional[Union[Callable, List]] = None,
            target_transform: Optional[Union[Callable, List]] = None,
            max_sequence_length: Optional[int] = None,
            num_samples: int = None,
            seed: Optional[int] = None
    ) -> None:
        """
        Initializes a SequentialDataset object.

        Args:
            data_filepath (str, optional): The file path to the data.
            target_filepath (str, optional): The file path to the target.
            transform (callable or list, optional): A function or a list of functions to apply to the data.
            target_transform (callable or list, optional): A function or a list of functions to apply to the target.
            max_sequence_length (int, optional): The maximum sequence length.
            num_samples (int, optional): The number of samples to include in the dataset.
            seed (int, optional): The random seed.
        """

        super().__init__(data=data, target=target, transform=transform, target_transform=target_transform, num_samples=num_samples, seed=seed)
        self.max_sequence_length = max_sequence_length
        
        
    @staticmethod
    def _load_data(data_filepath: str):
        """
        Loads the data from the specified file.

        Args:
            data_filepath (str): The file path to the data.

        Returns:
            The loaded data.
        """
        return read_strings_from_file(data_filepath)

    @staticmethod
    def _load_target(target_filepath: str, task_type: str = None):
        """
        Loads the target from the specified file.

        Args:
            target_filepath (str): The file path to the target.
            task_type (str, optional): The type of task (e.g., classification, regression). Defaults to None.

        Returns:
            The loaded target.
        """
        target_extension = target_filepath.split('.')[-1]
        if target_extension == 'pt':
            target = torch.load(target_filepath)
        elif target_extension == 'npy':
            target = np.load(target_filepath)
            target = torch.from_numpy(target)
        else:
            raise ValueError(f"Unsupported target file extension: {target_extension}")
        
        if task_type is not None:
            if task_type == 'classification':
                target = target.long()
            elif task_type == 'regression':
                target = target.float()
        return target
    
    @classmethod
    def _from_filepath(
        cls,
        root: str = None, 
        data_filepath: str = None,
        target_filepath: Optional[str] = None,
        transform: Optional[Union[Callable, List]] = None,
        target_transform: Optional[Union[Callable, List]] = None,
        max_sequence_length: Optional[int] = None,
        num_samples: int = None,
        task_type: Optional[str] = None,
        seed: Optional[int] = None,
        ):
        """ Dowloads and loads the data and target from the specified file paths.
        """
        
        if root is not None:
            data_filepath = os.path.join(root, data_filepath)
            target_filepath = os.path.join(root, target_filepath)

        if data_filepath is not None:
            data = cls._load_data(data_filepath)
        if target_filepath is not None:
            target = cls._load_target(target_filepath, task_type)
        else:
            target = None
        
        return cls(
            data = data,
            target = target,
            transform=transform,
            target_transform=target_transform,
            seed=seed,
            num_samples=num_samples,
            max_sequence_length=max_sequence_length
        )

    @classmethod
    def from_config(cls, config: DatasetConfig, split: Optional[str] = None, seed: Optional[int] = None, root: str = None):
        
        if split is None:
            split = config.split

        if split is not None:
            config.split = split

        if split == 'train':
            data_filepath = config.path_to_train_data
            properties_filepath = config.path_to_train_properties
        elif split == 'val':
            data_filepath = config.path_to_val_data
            properties_filepath = config.path_to_val_properties
        elif split == 'test':
            data_filepath = config.path_to_test_data
            properties_filepath = config.path_to_test_properties
        else:
            raise ValueError("Provide a correct split value.")

        return cls._from_filepath(
            root=root,
            data_filepath = data_filepath,
            target_filepath = properties_filepath,
            transform=config.transform,
            target_transform=config.target_transform,
            seed=seed,
            num_samples=config.num_samples,
            task_type=config.task_type
        )
