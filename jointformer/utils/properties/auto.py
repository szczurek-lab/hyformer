import importlib

from typing import Union, Optional

from jointformer.utils.properties.smiles.qed import QED
from jointformer.utils.properties.smiles.physchem import PhysChem


class AutoTarget:

    @classmethod
    def from_target_label(cls, target: str, dtype: Optional[str] = None) -> Union[QED, PhysChem]:
        """ Returns the target class based on the target label. """

        if target == 'qed':
            return getattr(importlib.import_module(
                "jointformer.utils.properties.smiles.qed"),
                "QED")(dtype=dtype)
        elif target == 'physchem':
            return getattr(importlib.import_module(
                "jointformer.utils.properties.smiles.physchem"),
                "PhysChem")(dtype=dtype)
        elif target == 'plogp':
            return getattr(importlib.import_module(
                "jointformer.utils.properties.smiles.plogp"),
                "PlogP")(dtype=dtype)
        elif target == 'guacamol_mpo':
            return getattr(importlib.import_module(
                "jointformer.utils.properties.smiles.guacamol_mpo"),
                "GuacamolMPO")(dtype=dtype)
        else:
            raise ValueError(f"Target {target} not available.")
