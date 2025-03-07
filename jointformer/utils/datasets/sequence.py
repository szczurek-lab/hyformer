""" Torch dataset for SMILES data.
"""

import os
import torch 

import numpy as np

from typing import List, Callable, Optional, Union

from jointformer.configs.dataset import DatasetConfig
from jointformer.utils.datasets.base import BaseDataset
from jointformer.utils.transforms.auto import AutoTransform, AutoTargetTransform


class SequenceDataset(BaseDataset):
    """Dataset for sequential data with or without properties.

    Method `self._load` is used to load the data from a filepath. It assumes that the data is stored in a numpy `.npz` file.
    The data is expected to have a key `sequence` for the data and a key `properties` for the target.
    """

    def __init__(
            self,
            data: List[str],
            target: Optional[np.ndarray] = None,
            data_transform: Optional[Union[Callable, List]] = None,
            target_transform: Optional[Union[Callable, List]] = None,
            task_type: Optional[str] = None,
            num_tasks: Optional[int] = None,
            task_metric: Optional[str] = None
    ) -> None:
        super().__init__(data=data, target=target, data_transform=data_transform, target_transform=target_transform)
        self.task_type = task_type
        self.num_tasks = num_tasks
        self.task_metric = task_metric
    
    @staticmethod
    def _get_filepath(config: DatasetConfig, root: str, split: str) -> str:
        if split == 'train':
            filepath = config.path_to_train_data
        elif split == 'val':
            filepath = config.path_to_val_data
        elif split == 'test':
            filepath = config.path_to_test_data
        else:
            raise ValueError("Provide a correct split value.")
        if root is not None:
            filepath = os.path.join(root, filepath)
        return filepath
    
    @staticmethod
    def _load(filepath: str, task_type: str = None, num_tasks: int = None) -> np.ndarray:
        
        _df = np.load(filepath, allow_pickle=True)
        data = _df['sequence'] if 'sequence' in _df else None
        target = _df['properties'] if 'properties' in _df else None

        if data is None:
            raise ValueError("Data not loaded.")       
        if target is not None:
            assert len(target.shape) == 2, f"Target has an unexpected shape: {target.shape}."
            assert len(data) == len(target), f"Data and target have different lengths: {len(data)} and {len(target)}."
            if num_tasks is not None:
                assert target.shape[1] == num_tasks, f"Target has an unexpected shape: {target.shape}."
            if task_type == 'classification':
                if target.dtype != np.int64:
                    target = target.astype(np.int64)
                    print("Converting target to long.")
            elif task_type == 'regression':
                if target.dtype != np.float32:
                    target = target.astype(np.float32)
                    print("Converting target to float32.")
            else:
                pass

        return data, target

    def __add__(self, other):
        if not isinstance(other, SequenceDataset):
            raise ValueError("Can only add SequenceDataset to SequenceDataset")
        
        data = np.concatenate((self.data, other.data), axis=0)
        target = None if self.target is None else np.concatenate((self.target, other.target), axis=0)
        data_transform = self.data_transform if self.data_transform is not None else other.data_transform
        target_transform = self.target_transform if self.target_transform is not None else other.target_transform
        task_type = self.task_type
        num_tasks = self.num_tasks
        task_metric = self.task_metric
        return SequenceDataset(
            data=data,
            target=target,
            data_transform=data_transform,
            target_transform=target_transform,
            task_type=task_type,
            num_tasks=num_tasks,
            task_metric=task_metric
        )

    @classmethod
    def from_config(cls, config: DatasetConfig, split: str, root: str = None):

        filepath = cls._get_filepath(config, root, split)
        data, target = cls._load(filepath, task_type=config.task_type, num_tasks=config.num_tasks if config.num_tasks is not None else None)

        data_transform = AutoTransform.from_config(config.transform) if config.transform is not None and split == 'train' else None
        target_transform = AutoTargetTransform.from_config(config.target_transform) if config.target_transform is not None else None

        return cls(
            data=data, target=target, data_transform=data_transform, target_transform=target_transform,
            task_type=config.task_type, num_tasks=config.num_tasks, task_metric=config.task_metric)
