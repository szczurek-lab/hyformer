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

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Square Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate ROC AUC score.
    
    Args:
        y_true: Ground truth values (binary or multi-label)
        y_pred: Predicted probabilities
        
    Returns:
        ROC AUC value
    """
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

def calculate_prc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Precision-Recall AUC score.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted probabilities
        
    Returns:
        PRC AUC value
    """
    assert y_pred.min() >= 0 and y_pred.max() <= 1, "y_pred should be probabilities"
    return average_precision_score(y_true, y_pred)

def calculate_lo_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cluster_assignment: np.ndarray
) -> dict:
    """Calculate Local Order metrics for clustered data.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        cluster_assignment: Cluster assignments for each sample
        
    Returns:
        Dictionary containing mean R², Spearman correlation, and MAE across clusters
    """
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