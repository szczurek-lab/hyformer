"""
Multi-Layer Perceptron probe (classification and regression) using ProbeBase.
"""

from sklearn.neural_network import MLPClassifier, MLPRegressor
from typing import Dict, Any
from .base import ProbeBase


class MLPProbe(ProbeBase):
    """
    Multi-Layer Perceptron probe with GridSearchCV-based hyperparameter optimization.
    
    Inherits from ProbeBase and provides MLP-specific implementations.
    All hyperparameters are specified through param_grid for consistency with other probes.
    
    Parameters
    ----------
    task_type : str
        Task type. Must be 'regression' or 'classification'.
    param_grid : dict, optional
        Dictionary of hyperparameters to search over. If None, uses default grid.
        Expected keys:
        - 'hidden_layer_sizes': list of tuples, network architectures
        - 'activation': list of str, activation functions
        - 'alpha': list of float, L2 regularization strengths
        - 'learning_rate': list of str, learning rate schedules
        - 'learning_rate_init': list of float, initial learning rates
        - 'max_iter': list of int, maximum iterations
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
        hidden_dim: int = 512,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Initialize the MLPProbe with task type and hidden dimension.

        Parameters
        ----------
        task_type : str
            Task type, either 'classification' or 'regression'.
        hidden_dim : int
            Dimension of the hidden layer. Defaults to 512.
        *args : Any
            Additional positional arguments.
        **kwargs : Any
            Additional keyword arguments.
        """
        self._hidden_dim = hidden_dim
        self.task_type = task_type
        super().__init__(task_type=task_type, *args, **kwargs)

    def _get_base_estimator(self):
        """Return the base MLP estimator with early stopping always enabled."""
        if self.task_type == 'regression':
            return MLPRegressor(
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                max_iter=1000,
                solver='adam'
            )
        elif self.task_type == 'classification':
            return MLPClassifier(
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                max_iter=1000,
                solver='sgd'
            )
        else:
            raise ValueError(f"Invalid task type: {self.task_type}. Must be 'regression' or 'classification'.")
        
    def _get_default_param_grid(self) -> Dict[str, Any]:
        """Return the default parameter grid for MLP (task-aware)."""
        base_grid = {
            'hidden_layer_sizes': [
                (self._hidden_dim,),
                (self._hidden_dim, self._hidden_dim)
            ],
            'alpha': [0.001, 0.01, 0.1],
            'learning_rate_init': [0.0001, 0.0006, 0.001, 0.006]
        }
        if self.task_type == 'regression':
            # MLPRegressor does not use 'activation'='identity' in the same way; keep defaults
            base_grid.update({
                'activation': ['relu', 'tanh'],
                'learning_rate': ['adaptive']
            })
        elif self.task_type == 'classification':
            base_grid.update({
                'activation': ['relu', 'tanh', 'identity'],
                'learning_rate': ['adaptive']
            })
        else:
            raise ValueError(f"Invalid task type: {self.task_type}. Must be 'regression' or 'classification'.")
        return base_grid
