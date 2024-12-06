""" Base class for datasets.

The BaseDataset class is a base class for datasets. It provides a common interface for handling datasets and their
corresponding targets, including methods for subsetting the dataset.
"""

import os
import random

import pandas as pd
import torchvision.transforms as transforms

from typing import Any, List, Callable, Optional, Union
from torch.utils.data.dataset import Dataset

from jointformer.utils.runtime import set_seed


class BaseDataset(Dataset):
    """Base class for datasets."""

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
            if self.data_transform is not None:
                x = self.data_transform(x)
            if self.target is None:
                return x
            else:
                y = self.target[idx]
                if self.target_transform is not None:
                    y = self.target_transform(y)
                return x, y
