"""  Implements enumeration of SMILES strings.

Method described in Bjerrum, E. J. SMILES enumeration as data augmentation for neural network modeling of molecules.
Code adapted from https://github.com/EBjerrum/SMILES-enumeration/tree/master.

"""

import numpy as np

from rdkit import Chem
from typing import Any

RETURN_CANONICAL_SMILES = False
RETURN_ISOMERIC_SMILES = True


class SmilesEnumerator:
    """SMILES Enumerator.

    Performs non-canonical SMILES enumeration.
    """

    def __init__(self, enumeration_probability: float = 0.8):
        self.enumeration_probability = enumeration_probability

    def __call__(self, smiles: str) -> str:

        p = np.random.uniform()
        if p <= self.enumeration_probability:
            return self.randomize(smiles)
        else:
            return smiles

    @staticmethod
    def randomize(smiles: str) -> str:
        molecule = Chem.MolFromSmiles(smiles)
        num_atoms_molecule = list(range(molecule.GetNumAtoms()))
        np.random.shuffle(num_atoms_molecule)
        enumerated_molecule = Chem.RenumberAtoms(molecule, num_atoms_molecule)
        return Chem.MolToSmiles(
            enumerated_molecule,
            canonical=RETURN_CANONICAL_SMILES,
            isomericSmiles=RETURN_ISOMERIC_SMILES)

    @classmethod
    def from_config(cls, config: Any) -> "SmilesEnumerator":
        return cls(enumeration_probability=config.get('enumeration_probability'))
