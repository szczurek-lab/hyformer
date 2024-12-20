import importlib

from jointformer.configs.dataset import DatasetConfig
from jointformer.utils.datasets.base import BaseDataset


class AutoDataset:
    """ AutoDataset (factory) class for automatic dataset selection based on config file.
    """
    @classmethod
    def from_config(
            cls,
            config: DatasetConfig,
            split: str,
            root: str = None
    ) -> BaseDataset:

        if config.dataset_name == 'sequence_dataset':
            return getattr(importlib.import_module(
                "jointformer.utils.datasets.sequence"),
                "SequenceDataset").from_config(config, root=root, split=split)
        else:
            raise ValueError(f"Dataset {config.dataset_name} not available.")
