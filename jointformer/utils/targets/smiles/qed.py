import numpy as np

from rdkit import Chem
from rdkit.Chem.QED import qed

from jointformer.utils.targets.smiles.base import BaseTarget


class QED(BaseTarget):
    """ QED target. """

    def _get_target(self, example: str) -> float:
        try:
            return qed(Chem.MolFromSmiles(example, sanitize=False))
        except Exception:
            return np.nan

    @property
    def target_names(self):
        return ["qed"]

    def __repr__(self):
        return "QED"

    def __str__(self):
        return "QED"
