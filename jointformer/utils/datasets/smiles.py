""" Torch dataset for SMILES data.
"""

import os

import numpy as np
import pandas as pd

from typing import List, Callable, Optional, Union

from jointformer.configs.dataset import DatasetConfig
from jointformer.utils.datasets.base import BaseDataset
from jointformer.utils.transforms.auto import AutoTransform, AutoTargetTransform


class SMILESDataset(BaseDataset):

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
        """
        Initializes a SequentialDataset object.

        Args:
            data (str, optional): The data.
            target (nd.array, optional): The target.
            data_transform (callable or list, optional): A function or a list of functions to apply to the data.
            target_transform (callable or list, optional): A function or a list of functions to apply to the target.
            max_sequence_length (int, optional): The maximum sequence length.
        """

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
    def _load(filepath: str, task_type: str = None) -> np.ndarray:
        
        _df = pd.read_csv(filepath)
        assert 'smiles' in _df.columns, "Column 'smiles' not found in the dataset."
        
        data = _df['smiles'].tolist()
        _df = _df.drop(columns=['smiles'], inplace=False)
        target = _df.values if len(_df.columns) > 0 else None
        
        if target is not None:
            assert len(target.shape) == 2, f"Target has an unexpected shape: {target.shape}."
            assert len(data) == len(target), f"Data and target have different lengths: {len(data)} and {len(target)}."

        if task_type is not None:
            if task_type == 'classification':
                try:
                    target = target.long()
                except:
                    target = (target != 0).astype(int)
            elif task_type == 'regression':
                pass
                target = target.astype(np.float32)

        return data, target

    @classmethod
    def from_config(cls, config: DatasetConfig, split: str, root: str = None):

        filepath = cls._get_filepath(config, root, split)
        data, target = cls._load(filepath, task_type=config.task_type)

        data_transform = AutoTransform.from_config(config.transform) if config.transform is not None else None
        target_transform = AutoTargetTransform.from_config(config.target_transform) if config.target_transform is not None else None

        return cls(
            data=data, target=target, data_transform=data_transform, target_transform=target_transform,
            task_type=config.task_type, num_tasks=config.num_tasks, task_metric=config.task_metric)
