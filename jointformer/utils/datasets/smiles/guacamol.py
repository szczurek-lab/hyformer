import os

from urllib.request import urlretrieve
from typing import List, Optional, Union, Callable

import torch
import random

from jointformer.configs.task import TaskConfig
from jointformer.utils.datasets.smiles.base import SmilesDataset
from jointformer.utils.data import save_strings_to_file, read_strings_from_file
from jointformer.utils.properties.auto import AutoTarget

DATA_DIR = './data/guacamol'
DATA_FILE_NAME = 'smiles.txt'

GUACAMOL_URL = {
    'train': "https://ndownloader.figshare.com/files/13612760",
    'val': "https://ndownloader.figshare.com/files/13612766",
    'test': "https://ndownloader.figshare.com/files/13612757",
    'all': "https://ndownloader.figshare.com/files/13612745",
    'debug': None
}


class GuacamolDataset(SmilesDataset):

    def __init__(
            self,
            split: str,
            target_label: Optional[str] = None,
            transform: Optional[Union[Callable, List]] = None,
            target_transform: Optional[Union[Callable, List]] = None,
            num_samples: Optional[int] = None,
            validate: Optional[bool] = None,
            standardize: Optional[bool] = None,
            data_dir: str = DATA_DIR,
    ) -> None:

        # Download data and targets
        data_dir = self._get_data_dir(data_dir, split, num_samples)
        data_filename = os.path.join(data_dir, DATA_FILE_NAME)
        if not os.path.isfile(data_filename):
            self._download_data(data_dir=data_dir, split=split, num_samples=num_samples)
        target_filename = os.path.join(data_dir, f'{target_label}.pt') if target_label else None
        if target_label and not os.path.isfile(target_filename):
            self._download_target(data_filename=data_filename, target_filename=target_filename, target_label=target_label)

        # Initialize the dataset
        super().__init__(
            data_filename=data_filename, target_filename=target_filename,
            transform=transform, target_transform=target_transform,
            num_samples=num_samples, validate=validate, standardize=standardize
        )

    @staticmethod
    def _get_data_dir(data_dir: str, split: str = None, num_samples: int = None) -> str:
        if split is not None:
            data_dir = os.path.join(data_dir, split)
        if num_samples is not None:
            data_dir = os.path.join(data_dir, str(num_samples))
        return data_dir

    @staticmethod
    def _download_data(split: str, num_samples: int, data_dir: str) -> None:
        data_filename = os.path.join(data_dir, DATA_FILE_NAME)
        print(f"Downloading data into {data_filename}")
        os.makedirs(data_dir, exist_ok=True)
        urlretrieve(GUACAMOL_URL[split], data_filename)

        data = read_strings_from_file(data_filename)

        # Subset the data to restrict the number of samples that the target is computed on
        if num_samples and len(data) > num_samples:
            data = random.sample(data, num_samples)

        save_strings_to_file(data, data_filename)

    @staticmethod
    def _download_target(data_filename: str, target_filename: str, target_label: str) -> None:
        print(f"Downloading target into {target_filename}")
        data = read_strings_from_file(data_filename)
        oracle = AutoTarget.from_target_label(target_label)
        target = oracle(data)
        torch.save(target, target_filename)

    @classmethod
    def from_config(cls, config: TaskConfig, split: str = None, data_dir: str = None) -> SmilesDataset:

        if config is not None:
            config.split = split

        return cls(
            split=config.split,
            target_label=config.target_label,
            transform=config.transform,
            target_transform=config.target_transform,
            validate=config.validate,
            standardize=config.standardize,
            num_samples=config.num_samples,
            data_dir=data_dir
        )
