import importlib

from typing import Union

from jointformer.utils.targets.smiles.qed import QED
from jointformer.utils.targets.smiles.physchem import PhysChem


class AutoTarget:

    @classmethod
    def from_target_label(cls, target: str) -> Union[QED, PhysChem]:
        """ Returns the target class based on the target label. """

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
