import importlib

from typing import Union
from jointformer.configs.task import TaskConfig
from jointformer.utils.datasets.base import BaseDataset


class AutoDataset:

    @classmethod
    def from_config(
            cls,
            config: TaskConfig,
            split: 'str' = None,
            num_samples: int = None,
            target_label: str = None
    ) -> BaseDataset:

        if split is not None:
            config.split = split
        if config.split in ['val', 'test', 'test_scaffolds']:
            config.num_samples = None
        if num_samples:
            config.num_samples = num_samples
        if target_label:
            config.target_label = target_label

        if config.dataset_name == 'moses':
            return getattr(importlib.import_module(
                "jointformer.utils.datasets.smiles.moses"),
                "MosesDataset").from_config(config)
        elif config.dataset_name == 'guacamol':
            return getattr(importlib.import_module(
                "jointformer.utils.datasets.smiles.guacamol"),
                "GuacamolDataset").from_config(config)
        else:
            raise ValueError(f"Dataset {config.dataset_name} not available.")
