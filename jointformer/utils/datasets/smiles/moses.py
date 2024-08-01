import os
import moses

from typing import List, Optional, Union, Callable

import torch

from jointformer.configs.task import TaskConfig
from jointformer.utils.datasets.smiles.base import SmilesDataset
from jointformer.utils.data import save_strings_to_file, read_strings_from_file
from jointformer.utils.properties.auto import AutoTarget

DATA_DIR = './data/moses'
DATA_FILE_NAME = 'smiles.txt'
AVAILABLE_TARGETS = ['qed']


class MosesDataset(SmilesDataset):

    def __init__(
            self,
            split: str,
            target_label: Optional[str] = None,
            transform: Optional[Union[Callable, List]] = None,
            target_transform: Optional[Union[Callable, List]] = None,
            num_samples: Optional[int] = None,
            validate: Optional[bool] = False,
            data_dir: str = DATA_DIR,
    ) -> None:

        data_dir = self._get_data_dir(data_dir, split, num_samples)
        data_filepath = os.path.join(data_dir, DATA_FILE_NAME)
        if not os.path.isfile(data_filepath):
            self._download_data(data_dir=data_dir, split=split, num_samples=num_samples)
        target_filepath = os.path.join(data_dir, f'{target_label}.pt') if target_label else None
        if target_label and not os.path.isfile(target_filepath):
            self._download_target(data_filepath=data_filepath, target_filepath=target_filepath, target_label=target_label)

        super().__init__(
            data_filepath=data_filepath, target_filepath=target_filepath,
            transform=transform, target_transform=target_transform,
            num_samples=num_samples, validate=validate
        )

    @staticmethod
    def _download_data(split: str, num_samples: int, data_dir: str) -> None:
        """ Downloads and saves MOSES data. """
        data_filepath = os.path.join(data_dir, DATA_FILE_NAME)

        print(f"Downloading data into {data_filepath}")
        os.makedirs(data_dir, exist_ok=True)

        if split in ['train', 'test', 'test_scaffolds']:
            data = moses.get_dataset(split).tolist()
        elif split == 'all':
            data = []
            for split in ['train', 'test', 'test_scaffolds']:
                data.extend(moses.get_dataset(split).tolist())
        else:
            raise ValueError(f"split `{split}` not available")

        if num_samples and len(data) > num_samples:
            data = data[:num_samples]

        save_strings_to_file(data, data_filepath)

    @staticmethod
    def _download_target(data_filepath: str, target_filepath: str, target_label: str) -> None:
        print(f"Downloading target into {target_filepath}")
        data = read_strings_from_file(data_filepath)
        oracle = AutoTarget.from_target_label(target_label)
        target = oracle(data)
        torch.save(target, target_filepath)

    @staticmethod
    def _get_data_dir(data_dir: str, split: str = None, num_samples: int = None) -> str:
        if split is not None:
            data_dir = os.path.join(data_dir, split)
        if num_samples is not None:
            data_dir = os.path.join(data_dir, str(num_samples))
        return data_dir

    @classmethod
    def from_config(cls, config: TaskConfig, split: str = None):

        split = config.split if split is None else split

        if split is None:
            split = 'all'
        if split == 'val':
            split = 'test'  # MOSES does not have a validation set

        return cls(
            split=split,
            target_label=config.target_label,
            transform=config.transform,
            target_transform=config.target_transform,
            validate=config.validate,
            num_samples=config.num_samples
        )
