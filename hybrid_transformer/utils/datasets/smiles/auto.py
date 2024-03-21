import importlib

from hybrid_transformer.configs.task import TaskConfig


class AutoDataset:
    """Base Dataset. """

    @classmethod
    def from_config(cls, config: TaskConfig, split: 'str' = None) -> "AutoDataset":

        if split is not None:
            config.split = split

        if config.dataset_name not in ["guacamol", "molecule_net"]:
            raise ValueError(f"`dataset` must be 'guacamol' or `molecule_net`, got {config.dataset_name}.")
        if config.molecular_representation not in ["SMILES"]:
            raise ValueError(f"`molecular_representation` must be 'SMILES, got {config.molecular_representation}.")

        if config.dataset_name == 'guacamol' and config.molecular_representation == 'SMILES':
            return getattr(importlib.import_module(
                "hybrid_transformer.utils.datasets.guacamol"),
                "GuacamolSMILESDataset").from_config(config)

        if config.dataset_name == 'molecule_net' and config.molecular_representation == 'SMILES':
            return getattr(importlib.import_module(
                "hybrid_transformer.utils.datasets.molecule_net"),
                "MoleculeNetSMILESDataset").from_config(config)

        raise ValueError(f"Invalid combination of `dataset` and `molecular_representation`.")
