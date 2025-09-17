"""
Random Forest Classifier Implementation using ProbeBase.
"""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import Dict, Any, Union
from .base import ProbeBase


class RFProbe(ProbeBase):
    """
    Random Forest probe with GridSearchCV-based hyperparameter optimization.
    
    Inherits from ProbeBase and provides RandomForest-specific implementations.
    Uses the same interface and behavior as the original RFProbe but with
    cleaner code organization through inheritance.
    
    Parameters
    ----------
    param_grid : dict, optional
        Dictionary of hyperparameters to search over. If None, uses default grid.
    cv_folds : int, default=5
        Number of cross-validation folds.
    random_state : int, default=42
        Random state for reproducibility.
    n_jobs : int, default=4
        Number of parallel jobs for GridSearchCV.
    """
    
    def __init__(self, param_grid: Dict[str, Any] = None, cv_folds: int = 5, 
                 random_state: int = 42, n_jobs: int = 4, task_type: str = 'classification', *args, **kwargs) -> None:
        self.task_type = task_type  # Set task_type first
        super(RFProbe, self).__init__(param_grid, cv_folds, random_state, n_jobs, task_type, *args, **kwargs)
    
    def _get_base_estimator(self) -> Union[RandomForestClassifier, RandomForestRegressor]:
        """Return the base RandomForest estimator based on task type."""
        if self.task_type == 'regression':
            return RandomForestRegressor(random_state=self.random_state, n_jobs=self.n_jobs)
        else:
            return RandomForestClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
    
    def _get_default_param_grid(self) -> Dict[str, Any]:
        """
        Return the default parameter grid for RandomForest based on task type.
        """
        if self.task_type == 'regression':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, None],
                'min_samples_split': [2, 10]
            }
        else:
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, None],
                'min_samples_split': [2, 10],
                'class_weight': [None, 'balanced']
            }
        