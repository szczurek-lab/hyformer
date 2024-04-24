import importlib

from jointformer.configs.task import TaskConfig


class AutoDataset:

    @classmethod
    def from_config(cls, config: TaskConfig, split: 'str' = None) -> "AutoDataset":

        if split is not None:
            config.split = split

        if config.dataset_name == 'guacamol':
            return getattr(importlib.import_module(
                "jointformer.utils.datasets.smiles.guacamol"),
                "Guacamol").from_config(config)
        else:
            raise ValueError(f"Dataset {config.dataset_name} not available.")
