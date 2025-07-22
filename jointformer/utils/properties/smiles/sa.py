import networkx as nx
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors

from hyformer.utils.properties.smiles import sascorer
from hyformer.utils.properties.smiles.base import BaseTarget


class SA(BaseTarget):
    """ SA target.
    Source: https://github.com/wengong-jin/hgraph2graph/blob/master/props/properties.py
    """
    
    def _get_target(self, example: str) -> float:
        try: 
            mol = Chem.MolFromSmiles(example)
            return sascorer.calculateScore(mol)
        except Exception:
            return np.nan

    @property
    def target_names(self):
        return ["SA"]

    def __repr__(self):
        return "SA"

    def __str__(self):
        return "SA"
    
    def __len__(self):
        return 1
