import numpy as np

from hyformer.utils.targets.base import BaseTarget


class QED(BaseTarget):
    """ QED target. 
    Source: https://github.com/wengong-jin/hgraph2graph/blob/master/props/properties.py
    """

    def __init__(self):
        super().__init__()
        from rdkit import Chem
        from rdkit.Chem.QED import qed

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
