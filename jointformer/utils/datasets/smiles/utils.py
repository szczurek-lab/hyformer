from typing import Optional
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.CRITICAL) # Mute RDKit logger


def is_valid(smiles: str) -> bool:
    """
    Verifies whether a SMILES string corresponds to a valid molecule. Source: GuacaMol

    Args:
        smiles: SMILES string

    Returns:
        True if the SMILES strings corresponds to a valid, non-empty molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    valid = smiles != '' and mol is not None and mol.GetNumAtoms() > 0

    return valid


def canonicalize(smiles: str, include_stereocenters=True) -> Optional[str]:
    """
    Canonicalize the SMILES strings with RDKit.

    The algorithm is detailed under https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00543

    Args:
        smiles: SMILES string to canonicalize
        include_stereocenters: whether to keep the stereochemical information in the canonical SMILES string

    Returns:
        Canonicalized SMILES string, None if the molecule is invalid.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)
    else:
        return None
