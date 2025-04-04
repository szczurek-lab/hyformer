import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    roc_auc_score,
    average_precision_score,
    r2_score,
    mean_absolute_error
)
from scipy.stats import spearmanr

from typing import List, Union, Dict, Optional

from hyformer.utils.chemistry import is_valid, canonicalize_list


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


def calculate_metric(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    metric: str,
    task_type: str = 'predictive',
    **kwargs
) -> Union[float, Dict[str, float]]:
    """Calculate specified metric for given predictions and targets.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        metric: Metric to calculate ('rmse', 'roc_auc', 'prc_auc', 'perplexity')
        task_type: Type of task ('generative' or 'predictive')
        **kwargs: Additional arguments for specific metrics
        
    Returns:
        Calculated metric value
    """
    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()
    
    if task_type == 'generative':
        if metric == 'perplexity':
            return np.exp(np.mean(y_true))  # y_true contains losses
        else:
            raise ValueError(f"Metric {metric} not supported for generative tasks")
    
    elif task_type == 'predictive':
        if metric == 'rmse':
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric == 'roc_auc':
            if y_true.shape[1] == 1:
                assert y_true.min() >= 0 and y_true.max() <= 1, "y_true should be binary"
                return roc_auc_score(y_true, y_pred)
            else:
                _num_tasks = y_true.shape[1]
                _aucs = []
                for i in range(_num_tasks):
                    if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                        _is_labeled = y_true[:, i] >= 0
                        _aucs.append(roc_auc_score(y_true[_is_labeled, i], y_pred[_is_labeled, i]))
                return np.mean(_aucs)
        elif metric == 'prc_auc':
            assert y_pred.min() >= 0 and y_pred.max() <= 1, "y_pred should be probabilities"
            return average_precision_score(y_true, y_pred)
        elif metric == 'lo':
            return calculate_lo_metrics(y_true, y_pred, kwargs.get('cluster_assignment'))
        else:
            raise ValueError(f"Metric {metric} not supported for predictive tasks")
    
    else:
        raise ValueError(f"Task type {task_type} not supported")


def calculate_lo_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cluster_assignment: np.ndarray
) -> Dict[str, float]:
    """Calculate Local Order metrics for clustered data."""
    if cluster_assignment is None:
        raise ValueError("cluster_assignment required for LO metrics")
        
    data = pd.DataFrame({
        'preds': y_pred,
        'cluster': cluster_assignment,
        'value': y_true
    })

    metrics = {
        'r2': [],
        'spearman': [],
        'mae': []
    }
    
    for cluster_idx in data['cluster'].unique():
        cluster = data[data['cluster'] == cluster_idx]
        
        # Calculate metrics for this cluster
        metrics['r2'].append(r2_score(cluster['value'], cluster['preds']))
        
        spearman, _ = spearmanr(cluster['value'], cluster['preds'])
        metrics['spearman'].append(0.0 if np.isnan(spearman) else spearman)
        
        metrics['mae'].append(mean_absolute_error(cluster['value'], cluster['preds']))
    
    # Return mean of each metric across clusters
    return {k: np.mean(v) for k, v in metrics.items()}
