import numpy as np

from rdkit import Chem
from rdkit.Chem.QED import qed

from jointformer.utils.properties.smiles.base import BaseTarget


class QED(BaseTarget):
    """ QED target. 
    Source: https://github.com/wengong-jin/hgraph2graph/blob/master/props/properties.py
    """

    def _get_target(self, example: str) -> float:
        try:
            return qed(Chem.MolFromSmiles(example))
        except Exception:
            return np.nan

    @property
    def target_names(self):
        return ["qed"]

    def __repr__(self):
        return "QED"

    def __str__(self):
        return "QED"
    
    def __len__(self):
        return 1
