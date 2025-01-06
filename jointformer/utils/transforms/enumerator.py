""" Enumerates SMILES strings as described in 
Bjerrum, E. J. (2017). SMILES enumeration as data augmentation for neural network modeling of molecules.

"""

import numpy as np

from rdkit import Chem


class SmilesEnumerator:
    
    def __init__(self, enumeration_probability: float = 0.9):
        self.enumeration_probability = enumeration_probability

    def __call__(self, smiles: str) -> str:
        p = np.random.uniform()
        if p <= self.enumeration_probability:
            return self.randomize(smiles)
        return smiles

    def randomize(self, smiles: str) -> str:
        try:
            return self._randomize(smiles)
        except:
            print(f"SMILES {smiles} cannot be augmented.")
            return smiles
    
    @staticmethod
    def _randomize(smiles: str) -> str:
        """Source: https://github.com/EBjerrum/SMILES-enumeration"""
        mol = Chem.MolFromSmiles(smiles)
        ans = list(range(mol.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(mol,ans)
        return Chem.MolToSmiles(nm, canonical=False, isomericSmiles=True)

    @classmethod
    def from_config(cls, config) -> "SmilesEnumerator":
        return cls(
            enumeration_probability=config.get('enumeration_probability')
        )
