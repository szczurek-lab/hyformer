""" Base class for datasets."""

import os

import torchvision.transforms as transforms

from typing import Any, List, Callable, Optional, Union

from torch.utils.data.dataset import Dataset

from jointformer.utils.transforms.auto import AutoTransform


class BaseDataset(Dataset):
    """Base class for datasets."""

    def __init__(
            self,
            data: Any = None,
            target: Any = None,
            transform: Optional[Union[Callable, List]] = None,
            target_transform: Optional[Union[Callable, List]] = None,
            num_samples: int = None
    ) -> None:
        super().__init__()
        self.data = data
        self.target = target
        self.transform = AutoTransform.from_config(transform) if isinstance(transform, list) else transform
        self.target_transform = transforms.Compose(target_transform) if isinstance(target_transform,
                                                                                   list) else target_transform
        self.target_transform = target_transform
        self.num_samples = num_samples
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

    def _subset_data(self):
        if self.data and self.num_samples and len(self.data) > self.num_samples:
            self.data = self.data[:self.num_samples]

    def subset_target(self):
        if self.target and self.num_samples and len(self.target) > self.num_samples:
            self.target = self.target[:self.num_samples]

