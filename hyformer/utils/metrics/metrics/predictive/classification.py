from typing import Union
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from hyformer.metrics.base import Metric

class ROCAUC(Metric):
    """Area Under ROC Curve metric."""
    
    def __init__(self):
        super().__init__(name='roc_auc', prediction_task_type='predictive')
        
    def compute(self, y_true: Union[torch.Tensor, np.ndarray], 
                y_pred: Union[torch.Tensor, np.ndarray], 
                **kwargs) -> float:
        """Compute ROC AUC.
        
        Args:
            y_true: Ground truth values (binary or multi-label)
            y_pred: Predicted values
            
        Returns:
            ROC AUC value
        """
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.numpy()
            
        if y_true.shape[1] == 1:
            assert y_true.min() >= 0 and y_true.max() <= 1, "y_true should be binary"
            return roc_auc_score(y_true, y_pred)
        else:
            _num_prediction_tasks = y_true.shape[1]
            _aucs = []
            for i in range(_num_prediction_tasks):
                if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                    _is_labeled = y_true[:, i] >= 0
                    _aucs.append(roc_auc_score(y_true[_is_labeled, i], y_pred[_is_labeled, i]))
            return np.mean(_aucs)

class PRCAUC(Metric):
    """Area Under Precision-Recall Curve metric."""
    
    def __init__(self):
        super().__init__(name='prc_auc', prediction_task_type='predictive')
        
    def compute(self, y_true: Union[torch.Tensor, np.ndarray], 
                y_pred: Union[torch.Tensor, np.ndarray], 
                **kwargs) -> float:
        """Compute PRC AUC.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted probabilities
            
        Returns:
            PRC AUC value
        """
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.numpy()
            
        assert y_pred.min() >= 0 and y_pred.max() <= 1, "y_pred should be probabilities"
        return average_precision_score(y_true, y_pred) 