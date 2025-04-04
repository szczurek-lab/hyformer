from abc import ABC, abstractmethod
from typing import Union, Dict, Any
import torch
import numpy as np

class Metric(ABC):
    """Base class for all metrics."""
    
    def __init__(self, name: str, task_type: str):
        self.name = name
        self.task_type = task_type
        
    @abstractmethod
    def compute(self, y_true: Union[torch.Tensor, np.ndarray], 
                y_pred: Union[torch.Tensor, np.ndarray], 
                **kwargs) -> float:
        """Compute the metric value.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            **kwargs: Additional arguments specific to the metric
            
        Returns:
            Computed metric value
        """
        pass
    
    def __call__(self, y_true: Union[torch.Tensor, np.ndarray], 
                 y_pred: Union[torch.Tensor, np.ndarray], 
                 **kwargs) -> float:
        """Alias for compute method."""
        return self.compute(y_true, y_pred, **kwargs) 