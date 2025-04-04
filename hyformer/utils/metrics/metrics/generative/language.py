from typing import Union
import torch
import numpy as np
from hyformer.metrics.base import Metric

class Perplexity(Metric):
    """Perplexity metric for language modeling."""
    
    def __init__(self):
        super().__init__(name='perplexity', task_type='generative')
        
    def compute(self, y_true: Union[torch.Tensor, np.ndarray], 
                y_pred: Union[torch.Tensor, np.ndarray], 
                **kwargs) -> float:
        """Compute perplexity.
        
        Args:
            y_true: Loss values
            y_pred: Not used for perplexity
            
        Returns:
            Perplexity value
        """
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.numpy()
        return np.exp(np.mean(y_true)) 