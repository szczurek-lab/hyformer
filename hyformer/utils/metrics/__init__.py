import numpy as np

from hyformer.utils.metrics.generative import calculate_perplexity
from hyformer.utils.metrics.predictive import (
    calculate_rmse,
    calculate_roc_auc,
    calculate_prc_auc,
    calculate_lo_metrics
)


def calculate_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    **kwargs
) -> float:
    """Calculate specified metric for given predictions and targets.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        metric: Metric to calculate ('rmse', 'roc_auc', 'prc_auc', 'perplexity')
        **kwargs: Additional arguments for specific metrics
        
    Returns:
        Calculated metric value
    """
    if metric == 'perplexity':
        return calculate_perplexity(y_true)
    elif metric == 'rmse':
        return calculate_rmse(y_true, y_pred)
    elif metric == 'roc_auc':
        return calculate_roc_auc(y_true, y_pred)
    elif metric == 'prc_auc':
        return calculate_prc_auc(y_true, y_pred)
    elif metric == 'lo':
        return calculate_lo_metrics(y_true, y_pred, kwargs.get('cluster_assignment'))
    else:
        raise ValueError(f"Metric {metric} not implemented.")
