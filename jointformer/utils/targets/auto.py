import importlib

from typing import List, Union


class AutoTarget:

    @classmethod
    def from_target_label(cls, target: str) -> "AutoTarget":

        if target == 'qed':
            return getattr(importlib.import_module(
                "jointformer.utils.targets.smiles.qed"),
                "QED")()
        elif target == 'physchem':
            return getattr(importlib.import_module(
                "jointformer.utils.targets.smiles.physchem"),
                "PhysChem")()
        else:
            raise ValueError(f"Target {target} not available.")
