from rdkit import Chem
from rdkit.Chem import AllChem, Crippen
from rdkit.Chem.QED import qed

from jointformer.utils.properties.smiles.sascorer import compute_sa_score


def smiles_to_mol(smiles):
    return Chem.MolFromSmiles(smiles)


def is_valid(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return False
        return True
    except:
        return False


def get_qed(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        try:
            AllChem.Kekulize(mol, clearAromaticFlags=True)
        except:
            pass
        mol.UpdatePropertyCache(strict=False)
        return float(qed(mol))
    except:
        return float("nan")


def get_sa(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        try:
            AllChem.Kekulize(mol, clearAromaticFlags=True)
        except:
            pass
        mol.UpdatePropertyCache(strict=False)
        return float(compute_sa_score(mol))
    except:
        return float('nan')


def get_logp(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        try:
            AllChem.Kekulize(mol, clearAromaticFlags=True)
        except:
            pass
        mol.UpdatePropertyCache(strict=False)
        return float(Crippen.MolLogP(mol))
    except:
        return float('nan')
