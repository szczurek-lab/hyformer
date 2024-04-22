""" Get Guacamol objectives.

Source: https://github.com/BenevolentAI/guacamol/blob/master/guacamol/standard_benchmarks.py
"""

import math
import numpy as np

import networkx as nx
from tqdm import tqdm

from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, rdmolops
from rdkit.Chem.QED import qed
from rdkit.Chem.Fingerprints import FingerprintMols
from guacamol import standard_benchmarks

from rdkit import Chem
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
import sascorer

from jointformer.utils.objectives.guacamol.sasscorer import calculateScore

med1 = standard_benchmarks.median_camphor_menthol()  # 'Median molecules 1'
med2 = standard_benchmarks.median_tadalafil_sildenafil()  # 'Median molecules 2',
pdop = standard_benchmarks.perindopril_rings()  # 'Perindopril MPO',
osmb = standard_benchmarks.hard_osimertinib()  # 'Osimertinib MPO',
adip = standard_benchmarks.amlodipine_rings()  # 'Amlodipine MPO'
siga = standard_benchmarks.sitagliptin_replacement()  # 'Sitagliptin MPO'
zale = standard_benchmarks.zaleplon_with_other_formula()  # 'Zaleplon MPO'
valt = standard_benchmarks.valsartan_smarts()  # 'Valsartan SMARTS',
dhop = standard_benchmarks.decoration_hop()  # 'Deco Hop'
shop = standard_benchmarks.scaffold_hop()  # Scaffold Hop'
rano= standard_benchmarks.ranolazine_mpo()  # 'Ranolazine MPO'
fexo = standard_benchmarks.hard_fexofenadine()  # 'Fexofenadine MPO'... 'make fexofenadine less greasy'


guacamol_objs = {"med1": med1, "pdop": pdop, "adip": adip, "rano": rano, "osmb": osmb, "siga": siga, "zale": zale,
                 "valt": valt, "med2": med2, "dhop": dhop, "shop": shop, 'fexo': fexo}


GUACAMOL_TASK_NAMES = [
    'med1', 'pdop', 'adip', 'rano', 'osmb', 'siga',
    'zale', 'valt', 'med2', 'dhop', 'shop', 'fexo'
]


def smile_is_valid_mol(smile):
    if smile is None or len(smile)==0:
        return False
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return False
    return True


def smile_to_guacamole_score(obj_func_key, smile):
    if smile is None or len(smile)==0:
        return None
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    func = guacamol_objs[obj_func_key]
    score = func.objective.score(smile)
    if score is None:
        return None
    if score < 0:
        return None
    return score


def smile_to_rdkit_mol(smile):
    return Chem.MolFromSmiles(smile)


def smile_to_QED(smile):
    """
    Computes RDKit's QED score
    """
    if smile is None:
        return None
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    qed_score = qed(mol)
    return qed_score


def smile_to_sa(smile):
    """Synthetic Accessibility Score (SA):
    a heuristic estimate of how hard (10)
    or how easy (1) it is to synthesize a given molecule."""
    if smile is None:
        return None
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    return sascorer.calculateScore(mol)


def smile_to_penalized_logP(smile):
    """ calculate penalized logP for a given smiles_tokenizers string """
    if smile is None:
        return None
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    logp = Crippen.MolLogP(mol)
    sa = sascorer.calculateScore(mol)
    cycle_length = _cycle_score(mol)
    """
    Calculate final adjusted score.
    These magic numbers are the empirical means and
    std devs of the dataset.

    I agree this is a weird way to calculate a score...
    but this is what previous papers did!
    """
    score = (
            (logp - 2.45777691) / 1.43341767
            + (-sa + 3.05352042) / 0.83460587
            + (-cycle_length - -0.04861121) / 0.28746695
    )
    return max(score, -float("inf"))


def _cycle_score(mol):
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    return cycle_length


def smiles_to_desired_scores(smiles_list, task_id="logp", verbose=False):
    if verbose:
        return smiles_to_desired_scores_with_verbose(smiles_list, task_id)
    else:
        return smiles_to_desired_scores_without_verbose(smiles_list, task_id)


def smiles_to_desired_scores_with_verbose(smiles_list, task_id="logp"):
    scores = []
    for smiles_str in tqdm(smiles_list):
        if task_id == "logp":
            score_ = smile_to_penalized_logP(smiles_str)
        elif task_id == "qed":
            score_ = smile_to_QED(smiles_str)
        else:  # otherwise, assume it is a guacamol task
            score_ = smile_to_guacamole_score(task_id, smiles_str)
        if (score_ is not None) and (math.isfinite(score_)):
            scores.append(score_)
        else:
            scores.append(np.nan)

    return np.array(scores)


def smiles_to_desired_scores_without_verbose(smiles_list, task_id="logp"):
    scores = []
    for smiles_str in tqdm(smiles_list):
        if task_id == "logp":
            score_ = smile_to_penalized_logP(smiles_str)
        elif task_id == "qed":
            score_ = smile_to_QED(smiles_str)
        else:  # otherwise, assume it is a guacamol task
            score_ = smile_to_guacamole_score(task_id, smiles_str)
        if (score_ is not None) and (math.isfinite(score_)):
            scores.append(score_)
        else:
            scores.append(np.nan)

    return np.array(scores)


def get_fingerprint_similarity(smile1, smile2):
    mol1 = Chem.MolFromSmiles(smile1)
    mol2 = Chem.MolFromSmiles(smile2)
    if (mol1 is None) or (mol2 is None):
        print("one of the input smiles_tokenizers is not a valid molecule!")
        return None
    fp1 = FingerprintMols.FingerprintMol(mol1)
    fp2 = FingerprintMols.FingerprintMol(mol2)
    fps = DataStructs.FingerprintSimilarity(fp1, fp2)
    return fps
