import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr


def get_test_metric(y_true, y_pred, metric):
    if metric == 'rmse':
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif metric == 'roc_auc':
        assert y_pred.min() >= 0 and y_pred.max() <= 1, "y_pred should be probabilities"
        return roc_auc_score(y_true, y_pred)
    elif metric == 'prc_auc':
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
