import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors

from jointformer.utils.targets.smiles import sascorer
from jointformer.utils.targets.smiles.base import BaseTarget


class PlogP(BaseTarget):
    """ pLogP target. """

    def _get_target(self, example: str) -> float:
        try:
            return Descriptors.MolLogP(Chem.MolFromSmiles(example)) \
                     - sascorer.calculateScore(Chem.MolFromSmiles(example))
        except Exception:
            return np.nan

    @property
    def target_names(self):
        return ["plogp"]

    def __repr__(self):
        return "pLogP"

    def __str__(self):
        return "pLogP"
