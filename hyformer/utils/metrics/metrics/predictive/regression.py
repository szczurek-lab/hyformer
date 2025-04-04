from typing import Union
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from hyformer.metrics.base import Metric

class RMSE(Metric):
    """Root Mean Square Error metric."""
    
    def __init__(self):
        super().__init__(name='rmse', prediction_task_type='predictive')
        
    def compute(self, y_true: Union[torch.Tensor, np.ndarray], 
                y_pred: Union[torch.Tensor, np.ndarray], 
                **kwargs) -> float:
        """Compute RMSE.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            RMSE value
        """
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.numpy()
            
        return np.sqrt(mean_squared_error(y_true, y_pred)) 