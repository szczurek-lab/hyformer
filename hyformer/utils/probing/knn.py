import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import warnings
import itertools


class KNNProbe:
    """
    K-Nearest Neighbors classifier for probing.
    
    Parameters
    ----------
    k : int or list of int
        Number of neighbors to use for classification. If a list is provided,
        the best k will be selected using 5-fold cross-validation.
    metric : str or callable
        Distance metric to use for the tree.
        See scikit-learn's KNeighborsClassifier for valid metrics.
    weights : str, callable, or list of str/callable
        Weight function used in prediction. If a list is provided,
        the best weights will be selected using 5-fold cross-validation.
        Possible values:
        - 'uniform': uniform weights (all points in each neighborhood are weighted equally)
        - 'distance': weight points by the inverse of their distance
        - 'soft-distance': temperature-scaled softmax weighting (T=0.07, DINOv2-style)
        - callable: a user-defined function which accepts an array of distances
          and returns an array of the same shape containing the weights
    random_state : int, default=42
        Random state for reproducibility.
           
    """
    
    def __init__(self, k: int | list[int], metric: str, weights: str | list[str], random_state: int = 42) -> None:
        self.k = k if isinstance(k, list) else [k]
        self.metric = metric
        self.weights = weights if isinstance(weights, list) else [weights]
        self.best_params = {}
        self.model = None
        self.cv_scores = {}
        self.selection_metric = None
        self.random_state = random_state
    
    def _init_model(self, k_value: int, weights_value: str) -> KNeighborsClassifier:
        if weights_value == 'soft-distance':
            weights_value = temp_scaled_01
        return KNeighborsClassifier(n_neighbors=k_value, weights=weights_value, metric=self.metric, n_jobs=-1)
    
    def _evaluate_model(self, model: KNeighborsClassifier, X: np.array, y: np.array, selection_metric: str = 'auprc') -> float:
        if selection_metric == 'auprc':
            return average_precision_score(y, model.predict_proba(X)[:, 1])
        else:
            raise ValueError(f"Unknown selection metric: {selection_metric}")
        
    def fit(self, X: np.array, y: np.array, selection_metric: str = 'accuracy'):
        """
        Fit the KNN model with hyperparameter optimization using 5-fold cross-validation.
        
        Parameters
        ----------
        X : np.array
            Feature matrix of shape (n_samples, n_features)
        y : np.array
            Target vector of shape (n_samples,)
        selection_metric : str, default='accuracy'
            Metric to use for hyperparameter selection. Options:
            - 'accuracy': Classification accuracy
            - 'auprc': Area Under Precision-Recall Curve
            - 'auroc': Area Under ROC Curve
            - 'f1': F1 score
        """
        self.selection_metric = selection_metric
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        _best_score = -1
        _best_k = self.k[0]
        _best_weights = self.weights[0]
        
        param_combinations = list(itertools.product(self.k, self.weights))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for _k_val, _weights_val in tqdm(param_combinations, desc="KNN probe hyperparameter search", unit="model"):
                fold_scores = []
                
                for train_idx, val_idx in cv.split(X, y):
                    X_fold_train, X_fold_val = X[train_idx], X[val_idx]
                    y_fold_train, y_fold_val = y[train_idx], y[val_idx]
                    
                    model = self._init_model(_k_val, _weights_val)
                    model.fit(X_fold_train, y_fold_train)
                    
                    fold_score = self._evaluate_model(model, X_fold_val, y_fold_val, selection_metric)
                    fold_scores.append(fold_score)
                
                mean_cv_score = np.mean(fold_scores)
                std_cv_score = np.std(fold_scores)
                
                self.cv_scores[(_k_val, _weights_val)] = {  # Store CV scores for the current parameter combination
                    'mean': mean_cv_score,
                    'std': std_cv_score,
                    'folds': fold_scores
                }
                
                if mean_cv_score > _best_score:  # Update best parameters if this combination is better
                    _best_score = mean_cv_score
                    _best_k = _k_val
                    _best_weights = _weights_val
        
        self.best_params = {  # Store the best parameters in a dictionary
            'k': _best_k,
            'weights': _best_weights,
            'metric': self.metric,
            'selection_metric': selection_metric,
            f'cv_{selection_metric}_mean': _best_score,
            f'cv_{selection_metric}_std': self.cv_scores[(_best_k, _best_weights)]['std']
        }
        
        # Report best hyperparameters
        weight_name = _best_weights if isinstance(_best_weights, str) else getattr(_best_weights, '__name__', str(_best_weights))
        print(f"Best hyperparameters: k={_best_k}, weights={weight_name}, cv_{selection_metric}={_best_score:.4f}±{self.cv_scores[(_best_k, _best_weights)]['std']:.4f}")
        
        # Refit the model with the best parameters on all data
        self.model = self._init_model(_best_k, _best_weights)
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.array) -> np.array:
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.model.predict_proba(X)[:, 1]
    
    def __repr__(self) -> str:
        return f"KNNProbe(k={self.k}, weights={self.weights}, metric={self.metric})"
    
    def __str__(self) -> str:
        return f"KNNProbe(k={self.k}, weights={self.weights}, metric={self.metric})"


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
    """Very sharp temperature-scaled softmax weights (T=0.01)."""
    return temperature_scaled_softmax_weights(distances, temperature=0.1)
