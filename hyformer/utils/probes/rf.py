"""
Random Forest Classifier Implementation using ProbeBase.
"""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import Dict, Any, Union, Optional
from .base import ProbeBase


class RFProbe(ProbeBase):
    """
    Random Forest probe with GridSearchCV-based hyperparameter optimization.

    Inherits from ProbeBase and provides RandomForest-specific implementations.
    Uses the same interface and behavior as the original RFProbe but with
    cleaner code organization through inheritance.

    Parameters
    ----------
    task_type : str
        Task type. Must be 'regression' or 'classification'.
    param_grid : dict, optional
        Dictionary of hyperparameters to search over. If None, uses default grid.
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
        verbose: int = 0
    ) -> None:
        """
        Initialize the RFProbe with task type and hyperparameter grid.

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
        """
        self.task_type = task_type
        super().__init__(
            param_grid=param_grid,
            cv_folds=cv_folds,
            random_state=random_state,
            n_jobs=n_jobs,
            task_type=task_type,
            verbose=verbose,
        )

    def _get_base_estimator(
        self,
    ) -> Union[RandomForestClassifier, RandomForestRegressor]:
        """Return the base RandomForest estimator based on task type."""
        if self.task_type == "regression":
            return RandomForestRegressor(
                random_state=self.random_state, n_jobs=self.n_jobs
            )
        elif self.task_type == "classification":
            return RandomForestClassifier(
                random_state=self.random_state, n_jobs=self.n_jobs
            )
        else:
            raise ValueError(
                f"Invalid task type: {self.task_type}. Must be 'regression' or 'classification'."
            )

    def _get_default_param_grid(self) -> Dict[str, Any]:
        """
        Return the default parameter grid for RandomForest based on task type.
        """
        if self.task_type == "regression":
            return {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, None],
                "min_samples_split": [2, 10],
            }
        elif self.task_type == "classification":
            return {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, None],
                "min_samples_split": [2, 10],
                "class_weight": [None, "balanced"],
            }
        else:
            raise ValueError(
                f"Invalid task type: {self.task_type}. Must be 'regression' or 'classification'."
            )
