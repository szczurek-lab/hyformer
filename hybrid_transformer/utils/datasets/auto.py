import importlib

from hybrid_transformer.configs.tasks.distribution_learning import DistributionLearningConfig


class AutoDataset:
    """Base Dataset. """

    @classmethod
    def from_config(cls, config: DistributionLearningConfig) -> "AutoDataset":

        if config.dataset_name not in ["guacamol"]:
            raise ValueError(f"`dataset` must be 'guacamol', got {config.dataset_name}.")
        if config.molecular_representation not in ["SMILES"]:
            raise ValueError(f"`molecular_representation` must be 'SMILES, got {config.molecular_representation}.")

        if config.dataset_name == 'guacamol' and config.molecular_representation == 'SMILES':
            return getattr(importlib.import_module(
                "hybrid_transformer.utils.datasets.guacamol"),
                "GuacamolSMILESDataset").from_config(config)
        else:
            raise ValueError(f"Invalid combination of `dataset` and `molecular_representation`.")
