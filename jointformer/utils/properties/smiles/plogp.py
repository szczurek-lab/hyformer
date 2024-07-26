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


class PlogP(BaseTarget):
    """ Penalized LogP target.
    Source: https://github.com/wengong-jin/hgraph2graph/blob/master/props/properties.py
    """
    
    def _get_target(self, example: str) -> float:
        try: 
            mol = Chem.MolFromSmiles(example)
            log_p = Descriptors.MolLogP(mol)
            sa = -sascorer.calculateScore(mol)

            # cycle score
            cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
            if len(cycle_list) == 0:
                cycle_length = 0
            else:
                cycle_length = max([len(j) for j in cycle_list])
            if cycle_length <= 6:
                cycle_length = 0
            else:
                cycle_length = cycle_length - 6
            cycle_score = -cycle_length

            # normalize
            normalized_log_p = (log_p - logP_mean) / logP_std
            normalized_sa = (sa - sa_mean) / sa_std
            normalized_cycle = (cycle_score - cycle_mean) / cycle_std
            
            return normalized_log_p + normalized_sa + normalized_cycle
        except Exception:
            return np.nan

    @property
    def target_names(self):
        return ["plogp"]

    def __repr__(self):
        return "PenalizedLogP"

    def __str__(self):
        return "PenalizedLogP"
