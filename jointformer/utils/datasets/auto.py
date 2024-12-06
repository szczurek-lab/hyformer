""" AutoDataset class for automatic dataset selection based on config.

This module contains the AutoDataset class, which is used to automatically select the
 appropriate dataset class based on the dataset name specified in the config.

"""

import importlib

from jointformer.configs.dataset import DatasetConfig
from jointformer.utils.datasets.base import BaseDataset


class AutoDataset:

    @classmethod
    def from_config(
            cls,
            config: DatasetConfig,
            split: str,
            root: str = None
    ) -> BaseDataset:

        if config.dataset_name == 'smiles_dataset':
            return getattr(importlib.import_module(
                "jointformer.utils.datasets.smiles"),
                "SMILESDataset").from_config(config, root=root, split=split)
        else:
            raise ValueError(f"Dataset {config.dataset_name} not available.")
