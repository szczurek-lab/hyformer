"""
K-Nearest Neighbors probe (classification and regression) using ProbeBase.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from typing import Dict, Any, List, Union
from .base import ProbeBase


def temperature_scaled_softmax_weights(distances: np.ndarray, temperature: float = 0.1) -> np.ndarray:
    """
    Apply temperature-scaled softmax weighting to distances.
    
    Parameters
    ----------
    distances : np.ndarray
        Array of distances to neighbors. Shape: (n_neighbors,)
    temperature : float, default=0.1
        Temperature parameter for scaling. Lower values create sharper distributions.
        Typical values range from 0.01 to 0.1.
    
    Returns
    -------
    weights : np.ndarray
        Softmax-normalized weights summing to 1. Shape: (n_neighbors,)
    """
    similarities = -distances
    scaled_logits = similarities / temperature
    scaled_logits_stable = scaled_logits - np.max(scaled_logits)
    exp_logits = np.exp(scaled_logits_stable)
    weights = exp_logits / np.sum(exp_logits)
    return weights


def temp_scaled_01(distances: np.ndarray) -> np.ndarray:
    """Very sharp temperature-scaled softmax weights (T=0.1)."""
    return temperature_scaled_softmax_weights(distances, temperature=0.1)


class KNNProbe(ProbeBase):
    """
    K-Nearest Neighbors probe with GridSearchCV-based hyperparameter optimization.
    
    Inherits from ProbeBase and provides KNN-specific implementations.
    All hyperparameters are specified through param_grid for consistency with other probes.
    
    Parameters
    ----------
    param_grid : dict, optional
        Dictionary of hyperparameters to search over. If None, uses default grid.
        Expected keys:
        - 'n_neighbors': list of int, number of neighbors
        - 'weights': list of str/callable, weight functions
        - 'metric': list of str, distance metrics
    cv_folds : int, default=5
        Number of cross-validation folds.
    random_state : int, default=42
        Random state for reproducibility.
    n_jobs : int, default=4
        Number of parallel jobs for GridSearchCV.
    """
    
    def _get_base_estimator(self) -> Union[KNeighborsClassifier, KNeighborsRegressor]:
        """Return the base KNN estimator based on task type."""
        if getattr(self, 'task_type', 'classification') == 'regression':
            return KNeighborsRegressor(n_jobs=self.n_jobs)
        return KNeighborsClassifier(n_jobs=self.n_jobs)
    
    def _get_default_param_grid(self) -> Dict[str, Any]:
        """Return the default parameter grid for KNN."""
        return {
            'n_neighbors': [1, 5, 10, 20, 50, 100],
            'metric': ['euclidean', 'cosine'],
            'weights': ['uniform', 'distance', temp_scaled_01]  # temp_scaled_01 = 'soft-distance'
        }
    
    def __repr__(self) -> str:
        return f"KNNProbe(param_grid={self.param_grid})"
    
    def __str__(self) -> str:
        return f"KNNProbe(param_grid={self.param_grid})"
    