import logging

import numpy as np

from typing import List

from jointformer.utils.chemistry import is_valid, canonicalize_list


logger = logging.getLogger(__name__)

REQUIRED_GUACAMOL_DATA_LENGTH = 10000


def calculate_validity(smiles_list: List[str]) -> float:
    if len(smiles_list) > REQUIRED_GUACAMOL_DATA_LENGTH:
        smiles_list = smiles_list[:REQUIRED_GUACAMOL_DATA_LENGTH]
    valid_smiles = [smiles for smiles in smiles_list if is_valid(smiles)]
    return len(valid_smiles) / len(smiles_list)


def calculate_uniqueness(smiles_list: List[str]) -> float:
    if len(smiles_list) > REQUIRED_GUACAMOL_DATA_LENGTH:
        smiles_list = smiles_list[:REQUIRED_GUACAMOL_DATA_LENGTH]
    unique_smiles_list = canonicalize_list(smiles_list, include_stereocenters=False)
    return len(unique_smiles_list) / len(smiles_list)


def calculate_novelty(smiles_list: List[str], reference_smiles_list: List[str]) -> float:
    if len(smiles_list) > REQUIRED_GUACAMOL_DATA_LENGTH:
        smiles_list = smiles_list[:REQUIRED_GUACAMOL_DATA_LENGTH]
    novel_molecules = set(smiles_list).difference(set(reference_smiles_list))
    return len(novel_molecules) / len(smiles_list)


def calculate_kl_div(smiles_list: List[str], reference_smiles_list: List[str]) -> float:
    from guacamol.utils.chemistry import calculate_pc_descriptors, continuous_kldiv, discrete_kldiv, calculate_internal_pairwise_similarities

    pc_descriptor_subset = [
        'BertzCT',
        'MolLogP',
        'MolWt',
        'TPSA',
        'NumHAcceptors',
        'NumHDonors',
        'NumRotatableBonds',
        'NumAliphaticRings',
        'NumAromaticRings'
    ]

    generated_distribution = calculate_pc_descriptors(smiles_list, pc_descriptor_subset)
    reference_distribution = calculate_pc_descriptors(reference_smiles_list, pc_descriptor_subset)

    kldivs = {}

    for i in range(4):
        kldiv = continuous_kldiv(X_baseline=reference_distribution[:, i], X_sampled=generated_distribution[:, i])
        kldivs[pc_descriptor_subset[i]] = kldiv

    for i in range(4, 9):
        kldiv = discrete_kldiv(X_baseline=reference_distribution[:, i], X_sampled=generated_distribution[:, i])
        kldivs[pc_descriptor_subset[i]] = kldiv

    chembl_sim = calculate_internal_pairwise_similarities(reference_smiles_list)
    chembl_sim = chembl_sim.max(axis=1)

    sampled_sim = calculate_internal_pairwise_similarities(smiles_list)
    sampled_sim = sampled_sim.max(axis=1)

    kldiv_int_int = continuous_kldiv(X_baseline=chembl_sim, X_sampled=sampled_sim)
    kldivs['internal_similarity'] = kldiv_int_int

    partial_scores = [np.exp(-score) for score in kldivs.values()]
    return sum(partial_scores) / len(partial_scores)


def calculate_fcd(smiles_list: List[str], reference_smiles_list: List[str]) -> float:

    import fcd, pkgutil, tempfile, os

    model_name = 'ChemNet_v0.13_pretrained.h5'

    model_bytes = pkgutil.get_data('fcd', model_name)
    assert model_bytes is not None

    tmpdir = tempfile.gettempdir()
    model_path = os.path.join(tmpdir, model_name)

    with open(model_path, 'wb') as f:
        f.write(model_bytes)

    logger.info(f'Saved ChemNet model to \'{model_path}\'')

    chemnet = fcd.load_ref_model(model_path)

    mu_ref, cov_ref = _calculate_fcd_distribution_statistics(chemnet, reference_smiles_list)
    mu, cov = _calculate_fcd_distribution_statistics(chemnet, smiles_list)

    FCD = fcd.calculate_frechet_distance(mu1=mu_ref, mu2=mu,
                                         sigma1=cov_ref, sigma2=cov)
    return np.exp(-0.2 * FCD)


def _calculate_fcd_distribution_statistics(model, molecules: List[str]):
    import fcd
    sample_std = fcd.canonical_smiles(molecules)
    gen_mol_act = fcd.get_predictions(model, sample_std)

    mu = np.mean(gen_mol_act, axis=0)
    cov = np.cov(gen_mol_act.T)
    return mu, cov
