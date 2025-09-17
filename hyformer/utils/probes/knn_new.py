"""
K-Nearest Neighbors probe (classification and regression) using ProbeBase.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from typing import Dict, Any, Union, Optional
from .base import ProbeBase


def temperature_scaled_softmax_weights(
    distances: np.ndarray, temperature: float = 0.1
) -> np.ndarray:
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
    d = np.asarray(distances, dtype=float)
    squeeze_back = False
    if d.ndim == 1:
        d = d[None, :]
        squeeze_back = True
    similarities = -d
    logits = similarities / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    weights = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return weights[0] if squeeze_back else weights


def temp_scaled_05(distances: np.ndarray) -> np.ndarray:
    """Temperature-scaled softmax weights (T=0.5)."""
    return temperature_scaled_softmax_weights(distances, temperature=0.5)


class KNNProbe(ProbeBase):
    """
    K-Nearest Neighbors probe with GridSearchCV-based hyperparameter optimization.

    Inherits from ProbeBase and provides KNN-specific implementations.
    All hyperparameters are specified through param_grid for consistency with other probes.

    Parameters
    ----------
    task_type : str
        Task type. Must be 'regression' or 'classification'.
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

    def __init__(
        self,
        task_type: str,
        param_grid: Optional[Dict[str, Any]] = None,
        cv_folds: int = 5,
        random_state: int = 42,
        n_jobs: int = 4,
        verbose: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the KNNProbe with task type and hyperparameter grid.

        Parameters
        ----------
        task_type : str
            Task type, either 'classification' or 'regression'.
        param_grid : Optional[Dict[str, Any]]
            Dictionary of hyperparameters to search over. Defaults to None.
        cv_folds : int
            Number of cross-validation folds. Defaults to 5.
        random_state : int
            Random state for reproducibility. Defaults to 42.
        n_jobs : int
            Number of parallel jobs for GridSearchCV. Defaults to 4.
        verbose : int
            Verbosity level. Defaults to 0.
        *args : Any
            Additional positional arguments.
        **kwargs : Any
            Additional keyword arguments.
        """
        self.task_type = task_type
        super().__init__(
            param_grid=param_grid,
            cv_folds=cv_folds,
            random_state=random_state,
            n_jobs=n_jobs,
            task_type=task_type,
            verbose=verbose,
            *args,
            **kwargs
        )

    def _get_base_estimator(
        self,
    ) -> Union[KNeighborsClassifier, KNeighborsRegressor]:
        """Return the base KNN estimator based on task type."""
        if self.task_type == "regression":
            return KNeighborsRegressor(n_jobs=self.n_jobs)
        elif self.task_type == "classification":
            return KNeighborsClassifier(n_jobs=self.n_jobs)
        else:
            raise ValueError(
                f"Invalid task type: {self.task_type}. Must be 'regression' or 'classification'."
            )

    def _get_default_param_grid(self) -> Dict[str, Any]:
        """Return the default parameter grid for KNN (kept small by default)."""
        return {
            "n_neighbors": [1, 5, 10, 20],
            "metric": ["euclidean", "cosine", "jaccard"],
            "weights": [
                "uniform",
                "distance",
                temp_scaled_05,
            ], 
        }

    def fit(
        self, X: np.ndarray, y: np.ndarray, selection_metric: Optional[str] = None
    ) -> "KNNProbe":
        """Cap n_neighbors to the max train-fold size to avoid CV errors, then fit."""
        if selection_metric is None:
            selection_metric = "rmse" if self.task_type == "regression" else "auprc"
        # Maximum neighbors allowed per training fold size
        max_k = max(1, int(np.floor(len(X) * (self.cv_folds - 1) / self.cv_folds)))
        # Build a filtered grid
        filtered_grid: Dict[str, Any] = {}
        for key, values in self.param_grid.items():
            if key == "n_neighbors":
                filtered_values = [k for k in values if k <= max_k]
                # Ensure at least one value
                filtered_grid[key] = filtered_values or [min(max_k, 1)]
            else:
                filtered_grid[key] = values
        # Temporarily replace grid, call base fit
        original_grid = self.param_grid
        self.param_grid = filtered_grid
        try:
            return super().fit(X, y, selection_metric=selection_metric)
        finally:
            self.param_grid = original_grid

    def __repr__(self) -> str:
        return f"KNNProbe(param_grid={self.param_grid})"

    def __str__(self) -> str:
        return f"KNNProbe(param_grid={self.param_grid})"
