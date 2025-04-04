import numpy as np
import pandas as pd
from typing import Dict, Union, List, Optional, Type
import torch
from sklearn.metrics import (
    mean_squared_error, 
    roc_auc_score, 
    average_precision_score,
    r2_score,
    mean_absolute_error
)
from scipy.stats import spearmanr
from hyformer.metrics.base import Metric
from hyformer.metrics.predictive.regression import RMSE
from hyformer.metrics.predictive.classification import ROCAUC, PRCAUC
from hyformer.metrics.generative.language import Perplexity

# Registry of available metrics
METRIC_REGISTRY: Dict[str, Type[Metric]] = {
    'rmse': RMSE,
    'roc_auc': ROCAUC,
    'prc_auc': PRCAUC,
    'perplexity': Perplexity
}

def get_metric(metric_name: str) -> Metric:
    """Get a metric instance by name.
    
    Args:
        metric_name: Name of the metric
        
    Returns:
        Metric instance
        
    Raises:
        ValueError: If metric name is not found in registry
    """
    if metric_name not in METRIC_REGISTRY:
        raise ValueError(f"Metric {metric_name} not found in registry. Available metrics: {list(METRIC_REGISTRY.keys())}")
    return METRIC_REGISTRY[metric_name]()

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