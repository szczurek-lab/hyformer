import importlib

from typing import Union, Optional

from hyformer.utils.targets.base import BaseTarget


class AutoTarget:

    @classmethod
    def from_target_name(cls, target_name: str, **kwargs) -> BaseTarget:
        """ Returns the target class based on the target label. """

        if target_name == 'qed':
            return getattr(importlib.import_module(
                "hyformer.utils.targets.smiles.qed"),
                "QED")(**kwargs)
        elif target_name == 'physchem':
            return getattr(importlib.import_module(
                "hyformer.utils.targets.physchem"),
                "PhysChem")(**kwargs)
        elif target_name == 'sa_score':
            return getattr(importlib.import_module(
                "hyformer.utils.targets.smiles.physchem"),
                "PhysChem")(**kwargs) 
        elif target_name == 'logp':
            return getattr(importlib.import_module(
                "hyformer.utils.targets.smiles.plogp"),
                "PlogP")(**kwargs)
        elif target_name == 'guacamol_mpo':
            return getattr(importlib.import_module(
                "hyformer.utils.targets.smiles.guacamol_mpo"),
                "GuacamolMPO")(**kwargs)
        else:
            raise ValueError(f"Target {target_name} not available.")
