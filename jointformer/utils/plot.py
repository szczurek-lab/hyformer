from rdkit import Chem


def mol_to_pil_image(molecule: Chem.rdchem.Mol, width: int = 300, height: int = 300) -> "PIL.Image":
    Chem.AllChem.Compute2DCoords(molecule)
    Chem.AllChem.GenerateDepictionMatching2DStructure(molecule, molecule)
    pil_image = Chem.Draw.MolToImage(molecule, size=(width, height))
    return pil_image
