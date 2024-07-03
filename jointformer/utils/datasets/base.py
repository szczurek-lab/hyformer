""" Base class for PyTorch datasets.

This module contains the BaseDataset class, which is the base class for all PyTorch datasets used in JointFormer.

"""

import os

import pandas as pd
import torchvision.transforms as transforms

from typing import Any, List, Callable, Optional, Union
from torch.utils.data.dataset import Dataset

from jointformer.utils.runtime import set_seed
from jointformer.utils.transforms.auto import AutoTransform


class BaseDataset(Dataset):
    """Base class for datasets."""

    def __init__(
            self,
            data: Any = None,
            target: Any = None,
            transform: Optional[Union[Callable, List]] = None,
            target_transform: Optional[Union[Callable, List]] = None,
            seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.data = data
        self.target = target
        self.transform = AutoTransform.from_config(transform) if isinstance(transform, list) else transform
        self.target_transform = transforms.Compose(target_transform) if isinstance(target_transform,
                                                                                   list) else target_transform
        self.target_transform = target_transform
        self.seed = seed
        if self.seed is not None:
            set_seed(self.seed)
        self._current = 0

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        self._current += 1
        if self._current >= len(self.data):
            self._current = 0
            raise StopIteration
        else:
            idx = self._current - 1
            return self.__getitem__(idx)

    def __getitem__(self, idx: int):
        x = self.data[idx]
        if self.transform is not None:
            x = self.transform(x)
        if self.target is None:
            return x
        else:
            y = self.target[idx]
            if self.target_transform is not None:
                y = self.target_transform(y)
            return x, y

    def get_data_frame(self):
        return pd.DataFrame({'text': self.data, 'labels': self.target.numpy().flatten()})

    @staticmethod
    def _get_data_dir(
        data_dir: str,
        dataset_name: str,
        target_name: Optional[str] = None,
        split: Optional[str] = None,
        splitter: Optional[str] = None,
        seed: Optional[int] = None,
        num_samples: Optional[int] = None
        ):
        data_dir = os.path.join(data_dir, 'datasets', dataset_name)
        if target_name is not None:
            data_dir = os.path.join(data_dir, target_name)
        if split is not None:
            data_dir = os.path.join(data_dir, split)
        if splitter is not None:
            data_dir = os.path.join(data_dir, f'splitter_{splitter}')
        if seed is not None:
            data_dir = os.path.join(data_dir, f'seed_{seed}')
        if num_samples is not None:
            data_dir = os.path.join(data_dir, f'num_samples_{num_samples}')
        return data_dir
