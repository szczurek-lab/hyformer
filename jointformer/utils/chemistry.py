""" A collection of utility functions for chemistry-related tasks.

Source:
    [1] https://github.com/BenevolentAI/guacamol/blob/master/guacamol/utils/chemistry.py
    [2] https://github.com/BenevolentAI/MolBERT/blob/main/molbert/utils/featurizer/molfeaturizer.py#L1346
"""

from typing import Iterable, List, Optional
from rdkit import Chem

from jointformer.utils.data import remove_duplicates

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)


def standardize(smiles: str, canonicalize: bool = False) -> Optional[str]:
    """
            Standardise a SMILES string if valid (canonical + kekulized)

            Args:
                smiles: SMILES string
                canonicalise: optional flag to override `self.canonicalise`

            Returns: standard version the SMILES if valid, None otherwise

            """
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_CLEANUP
        Chem.SanitizeMol(mol, flags, catchErrors=True)
        if canonicalize:
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            if mol is None:
                return None
        Chem.Kekulize(mol, clearAromaticFlags=True)
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=canonicalize)
    except:
         return None

    return smiles

def is_valid(smiles: str):
    """
    Verifies whether a SMILES string corresponds to a valid molecule.

    Args:
        smiles: SMILES string

    Returns:
        True if the SMILES strings corresponds to a valid, non-empty molecule.
    """

    mol = Chem.MolFromSmiles(smiles)

    return smiles != '' and mol is not None and mol.GetNumAtoms() > 0


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


def canonicalize_list(smiles_list: Iterable[str], include_stereocenters=True) -> List[str]:
    """
    Canonicalize a list of smiles. Filters out repetitions and removes corrupted molecules.

    Args:
        smiles_list: molecules as SMILES strings
        include_stereocenters: whether to keep the stereochemical information in the canonical SMILES strings

    Returns:
        The canonicalized and filtered input smiles.
    """

    canonicalized_smiles = [canonicalize(smiles, include_stereocenters) for smiles in smiles_list]

    # Remove None elements
    canonicalized_smiles = [s for s in canonicalized_smiles if s is not None]

    return remove_duplicates(canonicalized_smiles)


def smiles_to_rdkit_mol(smiles: str) -> Optional[Chem.Mol]:
    """
    Converts a SMILES string to a RDKit molecule.

    Args:
        smiles: SMILES string of the molecule

    Returns:
        RDKit Mol, None if the SMILES string is invalid
    """
    mol = Chem.MolFromSmiles(smiles)

    #  Sanitization check (detects invalid valence)
    if mol is not None:
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None

    return mol
