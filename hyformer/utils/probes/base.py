"""
Base class for probing models using GridSearchCV pattern.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union, Tuple, Optional
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    mean_squared_error, r2_score
)
import matplotlib.pyplot as plt


class ProbeBase(ABC):
    """
    Abstract base class for probing models using GridSearchCV.
    
    This class follows the same pattern as RFProbe, providing a clean interface
    for hyperparameter optimization using GridSearchCV with StratifiedKFold.
    
    Parameters
    ----------
    param_grid : dict, optional
        Dictionary of hyperparameters to search over. If None, uses default grid
        from _get_default_param_grid().
    cv_folds : int, default=5
        Number of cross-validation folds.
    random_state : int, default=42
        Random state for reproducibility.
    n_jobs : int, default=4
        Number of parallel jobs for GridSearchCV.
    """
    
    def __init__(self, param_grid: Dict[str, Any] = None, cv_folds: int = 5, 
                 random_state: int = 42, n_jobs: int = 4, task_type: str = 'classification', *args, **kwargs) -> None:
        
        if param_grid is None:
            self.param_grid = self._get_default_param_grid()
        else:
            self.param_grid = param_grid
            
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.best_score_ = None
        self.best_estimator_ = None
        self.best_params_ = None
        self.cv_results_ = None
        self.n_jobs = n_jobs
        self.task_type = task_type
    
    @abstractmethod
    def _get_base_estimator(self) -> Any:
        """
        Return the base estimator (model) for GridSearchCV.
        
        Returns
        -------
        estimator : object
            Base model instance with random_state and n_jobs set appropriately.
        """
        pass
    
    @abstractmethod
    def _get_default_param_grid(self) -> Dict[str, Any]:
        """
        Return the default parameter grid for this probe type.
        
        Returns
        -------
        param_grid : Dict[str, Any]
            Default parameter grid for GridSearchCV.
        """
        pass
    
    def _get_scoring_metric(self, selection_metric: str) -> str:
        """
        Convert selection metric to sklearn scoring string.
        
        Parameters
        ----------
        selection_metric : str
            Selection metric name.
            
        Returns
        -------
        scoring : str
            Sklearn scoring metric name.
            
        Raises
        ------
        ValueError
            If selection_metric is not supported.
        """
        if self.task_type == 'regression':
            metric_mapping = {
                'rmse': 'neg_root_mean_squared_error',  # Changed from 'mse' to 'rmse'
                'r2': 'r2'
            }
        else:
            metric_mapping = {
                'auprc': 'average_precision',
                'auroc': 'roc_auc',
                'accuracy': 'accuracy',
                'f1': 'f1'
            }
        
        if selection_metric not in metric_mapping:
            raise ValueError(f"Unknown selection metric: {selection_metric}. "
                           f"Supported metrics: {list(metric_mapping.keys())}")
        
        return metric_mapping[selection_metric]
    
    def _calculate_enrichment_factor(self, y_true: np.ndarray, y_scores: np.ndarray, fraction: float) -> float:
        """
        Calculate enrichment factor at a given fraction.
        
        Enrichment factor measures how much better the model performs compared to random
        selection when looking at the top fraction of predictions.
        
        Parameters
        ----------
        y_true : np.ndarray
            True binary labels.
        y_scores : np.ndarray
            Predicted scores/probabilities.
        fraction : float
            Fraction of top predictions to consider (e.g., 0.01 for top 1%).
            
        Returns
        -------
        ef : float
            Enrichment factor. Values > 1 indicate better than random performance.
            EF = (TPR at fraction) / fraction
        """
        if len(y_true) == 0 or np.sum(y_true) == 0:
            return 0.0
        
        # Sort predictions in descending order
        sorted_indices = np.argsort(-y_scores)
        
        # Get top fraction of predictions
        n_top = max(1, int(fraction * len(y_true)))
        top_indices = sorted_indices[:n_top]
        
        # Calculate true positive rate in top fraction
        tp_in_top = np.sum(y_true[top_indices])
        total_positives = np.sum(y_true)
        
        if total_positives == 0:
            return 0.0
        
        # TPR = true positives found / total positives
        tpr_at_fraction = tp_in_top / total_positives
        
        # Enrichment factor = TPR / fraction
        # (How much better than random selection)
        ef = tpr_at_fraction / fraction if fraction > 0 else 0.0
        
        return ef
    
    def fit(self, X: np.ndarray, y: np.ndarray, selection_metric: str = 'auprc') -> 'ProbeBase':
        """
        Fit probe with hyperparameter search using cross-validation.
        
        Parameters
        ----------
        X : np.ndarray
            Training features.
        y : np.ndarray
            Training labels.
        selection_metric : str, default='auprc'
            Metric to use for hyperparameter selection.
            Options: 'auprc', 'auroc', 'accuracy', 'f1'
            
        Returns
        -------
        self : ProbeBase
            Fitted probe instance.
        """
        if self.task_type == 'regression':
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        else:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Get scoring metric
        scoring = self._get_scoring_metric(selection_metric)
        
        # Get base estimator
        base_estimator = self._get_base_estimator()
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_estimator,
            param_grid=self.param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=self.n_jobs,
            verbose=4
        )
        
        # Fit the grid search
        grid_search.fit(X, y)
        
        # Store results
        self.best_estimator_ = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        self.cv_results_ = grid_search.cv_results_
        
        # Print best parameters and score
        self.best_score_ = grid_search.best_score_
        print(f"\nBest hyperparameters: {self.best_params_}")
        print(f"Best CV {selection_metric}: {self.best_score_:.4f}")
        
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for positive class.
        
        Parameters
        ----------
        X : np.ndarray
            Test features.
            
        Returns
        -------
        y_pred : np.ndarray
            Predicted probabilities for positive class.
            
        Raises
        ------
        ValueError
            If model has not been fitted yet.
        """
        if self.best_estimator_ is None:
            raise ValueError("Model must be fitted before making predictions")
            
        return self.best_estimator_.predict_proba(X)[:, 1]
    
    def get_best_params(self) -> Dict[str, Any]:
        """
        Get the best parameters found during cross-validation.
        
        Returns
        -------
        best_params : Dict[str, Any]
            Best parameter combination.
        """
        if self.best_params_ is None:
            raise ValueError("Model must be fitted before accessing best parameters")
        return self.best_params_.copy()
    
    def get_cv_results(self) -> Dict[str, Any]:
        """
        Get detailed cross-validation results.
        
        Returns
        -------
        cv_results : Dict[str, Any]
            Cross-validation results from GridSearchCV.
        """
        if self.cv_results_ is None:
            raise ValueError("Model must be fitted before accessing CV results")
        return self.cv_results_
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, 
                 plot_curves: bool = True, figsize: Tuple[int, int] = (18, 5), 
                 verbose: int = 1, threshold: float = None) -> Dict[str, float]:
        """
        Comprehensive evaluation of the fitted probe on test data.
        
        Now compatible with evaluate_probe() function - uses 95th percentile threshold
        and returns metrics with the same naming convention.
        
        Parameters
        ----------
        X_test : np.ndarray
            Test features.
        y_test : np.ndarray
            Test labels.
        plot_curves : bool, default=True
            Whether to plot ROC curve, Precision-Recall curve, and class histogram.
        figsize : Tuple[int, int], default=(18, 5)
            Figure size for plots (width, height).
        verbose : int, default=1
            Verbosity level. 1 = print evaluation results, 0 = suppress all printed output.
        threshold : float, optional
            Custom threshold for binary classification. If None, uses 95th percentile.
            
        Returns
        -------
        metrics : Dict[str, float]
            Dictionary of computed metrics compatible with evaluate_probe format.
            
        Raises
        ------
        ValueError
            If model has not been fitted yet.
        """
        if self.best_estimator_ is None:
            raise ValueError("Model must be fitted before evaluation")
        
        if self.task_type == 'regression':
            y_pred = self.best_estimator_.predict(X_test)
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            if verbose > 0:
                print("\n" + "=" * 60)
                print(f"{self.__class__.__name__} Evaluation Results")
                print("=" * 60)
                for metric_name, value in metrics.items():
                    print(f"{metric_name.upper():>10}: {value:.3f}")
                print("=" * 60)
            return metrics
        else:
            # Get predictions
            y_pred_proba = self.predict(X_test)
            
            # Use 95th percentile threshold (compatible with evaluate_probe)
            if threshold is None:
                threshold = np.percentile(y_pred_proba, 95)
                if verbose > 0:
                    print(f"Automatically selected threshold: {threshold:.3f}")
            
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # TPR and FPR at threshold
            tn = ((y_test == 0) & (y_pred == 0)).sum()
            fp = ((y_test == 0) & (y_pred == 1)).sum()
            fn = ((y_test == 1) & (y_pred == 0)).sum()
            tp = ((y_test == 1) & (y_pred == 1)).sum()
            
            tpr_at_threshold = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr_at_threshold = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # Precision@100
            sorted_indices = np.argsort(-y_pred_proba)
            top_100_indices = sorted_indices[:100]
            precision_at_100 = precision_score(y_test[top_100_indices], 
                                             (y_pred_proba[top_100_indices] >= threshold).astype(int),
                                             zero_division=0)
            
            # Enrichment Factor @1% and @5%
            n = len(y_test)
            top_1_percent = max(1, int(np.ceil(0.01 * n)))
            top_5_percent = max(1, int(np.ceil(0.05 * n)))
            
            ef1 = (y_test[sorted_indices[:top_1_percent]].sum() / top_1_percent) / (y_test.sum() / n) if y_test.sum() > 0 else 0
            ef5 = (y_test[sorted_indices[:top_5_percent]].sum() / top_5_percent) / (y_test.sum() / n) if y_test.sum() > 0 else 0
            
            # Compute metrics (compatible with evaluate_probe format)
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'auroc': roc_auc_score(y_test, y_pred_proba),
                'auprc': average_precision_score(y_test, y_pred_proba),
                'ef1': ef1,
                'ef5': ef5
            }
            
            # Display results if verbose > 0
            if verbose > 0:
                self._display_evaluation_results(metrics, y_test, y_pred_proba, threshold)
            
                # Plot curves if requested
                if plot_curves:
                    self._plot_evaluation_curves(y_test, y_pred_proba, figsize, self.task_type)
            
            return metrics
    
    def _display_evaluation_results(self, metrics: Dict[str, float], 
                                   y_test: np.ndarray, y_pred_proba: np.ndarray, 
                                   threshold: float = None) -> None:
        """
        Display evaluation results in a formatted table.
        
        Parameters
        ----------
        metrics : Dict[str, float]
            Computed metrics.
        y_test : np.ndarray
            True test labels.
        y_pred_proba : np.ndarray
            Predicted probabilities.
        threshold : float, optional
            Classification threshold used.
        """
        print("\n" + "=" * 60)
        print(f"{self.__class__.__name__} Evaluation Results")
        print("=" * 60)
        
        print(f"Test set size: {len(y_test)} samples")
        print(f"Class distribution: {np.bincount(y_test)}")
        print(f"Prediction range: [{y_pred_proba.min():.3f}, {y_pred_proba.max():.3f}]")
        if threshold is not None:
            print(f"Classification threshold: {threshold:.3f}")
        print()
        
        print("Performance Metrics:")
        print("-" * 30)
        for metric_name, value in metrics.items():
            print(f"{metric_name.upper():>10}: {value:.3f}")
        
        print("\nBest hyperparameters:")
        print("-" * 30)
        for param, value in self.get_best_params().items():
            if not param.startswith('cv_'):
                print(f"{param:>15}: {value}")
        print("=" * 60)
    
    def _plot_evaluation_curves(self, y_test: np.ndarray, y_pred: np.ndarray, 
                               figsize: Tuple[int, int], task_type: str) -> None:
        """
        Plot evaluation curves based on task type.
        """
        if task_type == 'regression':
            # Scatter plot for regression
            fig, ax = plt.subplots(figsize=figsize)
            ax.scatter(y_test, y_pred, alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            r2 = r2_score(y_test, y_pred)
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predictions')
            ax.set_title(f'{self.__class__.__name__} - Scatter Plot (R^2 = {r2:.3f})')
            ax.grid(True, alpha=0.3)
        else:
            # Existing classification plots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auroc = roc_auc_score(y_test, y_pred_proba)
            
            ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auroc:.3f})')
            ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.set_title(f'{self.__class__.__name__} - ROC Curve')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            auprc = average_precision_score(y_test, y_pred_proba)
            baseline = np.sum(y_test) / len(y_test)  # Random classifier baseline
            
            ax2.plot(recall, precision, 'r-', linewidth=2, label=f'PR Curve (AUPRC = {auprc:.3f})')
            ax2.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, 
                       label=f'Random Baseline (Precision = {baseline:.3f})')
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title(f'{self.__class__.__name__} - Precision-Recall Curve')
            ax2.legend(loc='lower left')  # Better legend positioning
            ax2.grid(True, alpha=0.3)
            
            # Class Histogram
            pos_probs = y_pred_proba[y_test == 1]
            neg_probs = y_pred_proba[y_test == 0]
            
            ax3.hist(neg_probs, bins=30, alpha=0.7, label=f'Negative Class (n={len(neg_probs)})', color='red')
            ax3.hist(pos_probs, bins=30, alpha=0.7, label=f'Positive Class (n={len(pos_probs)})', color='green')
            ax3.set_xlabel('Predicted Probability')
            ax3.set_ylabel('Frequency')
            ax3.set_title(f'{self.__class__.__name__} - Prediction Histogram')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
