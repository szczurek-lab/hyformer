from rdkit import Chem


def is_valid(smiles: str):
    """
    Verifies whether a SMILES string corresponds to a valid molecule. Source: GuacaMol

    Args:
        smiles: SMILES string

    Returns:
        True if the SMILES strings corresponds to a valid, non-empty molecule.
    """

    mol = Chem.MolFromSmiles(smiles)

    return smiles != '' and mol is not None and mol.GetNumAtoms() > 0
