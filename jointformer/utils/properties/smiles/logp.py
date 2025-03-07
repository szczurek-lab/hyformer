import networkx as nx
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors

from jointformer.utils.properties.smiles import sascorer
from jointformer.utils.properties.smiles.base import BaseTarget

logP_mean = 2.4570953396190123
logP_std = 1.434324401111988
sa_mean = -3.0525811293166134
sa_std = 0.8335207024513095
cycle_mean = -0.0485696876403053
cycle_std = 0.2860212110245455


class LogP(BaseTarget):
    """ LogP target.
    Source: https://github.com/wengong-jin/hgraph2graph/blob/master/props/properties.py
    """
    
    def _get_target(self, example: str) -> float:
        try: 
            mol = Chem.MolFromSmiles(example)
            log_p = Descriptors.MolLogP(mol)

            return log_p
        except Exception:
            return np.nan

    @property
    def target_names(self):
        return ["logp"]

    def __repr__(self):
        return "LogP"

    def __str__(self):
        return "LogP"
    
    def __len__(self):
        return 1
