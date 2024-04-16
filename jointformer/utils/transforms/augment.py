import random

import numpy as np

from rdkit import Chem
from guacamol.utils.chemistry import is_valid


class AugmentSMILES:

    def __init__(self, augmentation_prob: float = 0.8):
        self.augmentation_prob = augmentation_prob

    @staticmethod
    def _get_random_smiles_from_molecule(mol: object) -> str:
        mol.SetProp("_canonicalRankingNumbers", "True")
        idxs = list(range(0, mol.GetNumAtoms()))
        random.shuffle(idxs)
        for i, v in enumerate(idxs):
            mol.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(v))
        return Chem.MolToSmiles(mol)

    def __call__(self, smiles: str) -> str:

        smiles_initial = smiles
        p = np.random.uniform()
        if p < self.augmentation_prob:
            mol = Chem.MolFromSmiles(smiles)
            smiles_augmented = self._get_random_smiles_from_molecule(mol)
            if (smiles_augmented is not None) and (len(smiles_augmented) > 0) and (is_valid(smiles_augmented)):
                return smiles_augmented
        else:
            return smiles
