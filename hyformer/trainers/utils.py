import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
import torch
import random


def seed_worker(worker_id):
    """Set seed for each worker to ensure reproducibility.
    
    This function is used as a worker_init_fn in PyTorch DataLoader to ensure
    that each worker has a different but deterministic seed. It sets seeds for
    NumPy, Python's random module, and PyTorch.
    
    Args:
        worker_id: ID of the worker process
    """
    # PyTorch's initial_seed() returns a 64-bit integer, but NumPy's random number
    # generator expects a 32-bit integer. The modulo operation % 2**32 truncates
    # the 64-bit seed to 32 bits, ensuring compatibility with NumPy and preventing
    # potential overflow issues.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_test_metric(y_true, y_pred, metric):
    if metric == 'rmse':
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif metric == 'roc_auc':
        if y_true.shape[1] == 1:
            assert y_true.min() >= 0 and y_true.max() <= 1, "y_true should be binary"
            return roc_auc_score(y_true, y_pred)
        elif y_true.shape[1] > 1:
            _num_tasks = y_true.shape[1]
            _aucs = []
            for i in range(_num_tasks):
                if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                    _is_labeled = y_true[:, i] >= 0
                    _aucs.append(roc_auc_score(y_true[_is_labeled, i], y_pred[_is_labeled, i]))
            return np.mean(_aucs)
    elif metric == 'prc_auc':
        # Source: https://github.com/SteshinSS/lohi_neurips2023/blob/main/code/metrics.py
        assert y_pred.min() >= 0 and y_pred.max() <= 1, "y_pred should be probabilities"
        return average_precision_score(y_true, y_pred)
    else: #TODO: Implement LO metrics
        raise ValueError(f"Metric {metric} not supported")
    
    
def get_lo_metrics(y: np.ndarray, cluster_assignment: np.ndarray, y_pred: np.ndarray):
    data = pd.DataFrame({'preds': y_pred, 'cluster': cluster_assignment, 'value': y})

    r2_scores = []
    spearman_scores = []
    maes = []
    for cluster_idx in data['cluster'].unique():
        cluster = data[data['cluster'] == cluster_idx]
        r2 = r2_score(cluster['value'], cluster['preds'])
        r2_scores.append(r2)

        spearman, _ = spearmanr(cluster['value'], cluster['preds'])
        if np.isnan(spearman):
            spearman = 0.0
        spearman_scores.append(spearman)

        mae = mean_absolute_error(cluster['value'], cluster['preds'])
        maes.append(mae)

    r2_scores = np.array(r2_scores)
    spearman_scores = np.array(spearman_scores)
    maes = np.array(maes)

    metrics = {
        'r2': r2_scores.mean(),
        'spearman': spearman_scores.mean(),
        'mae': maes.mean()
    }
    return metrics
