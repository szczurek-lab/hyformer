import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch

class TrainableModel(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        task: str,
        attention_mask: Optional[torch.Tensor] = None,
        next_token_only: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        input_labels: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        loss_fn_reduction: str = 'mean',
        nan_target_idx: int = -1,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            task: Task type ('lm', 'mlm', or 'prediction')
            attention_mask: Attention mask for the input
            next_token_only: Whether to return only the next token
            use_cache: Whether to use caching
            input_labels: Labels for language modeling tasks
            target: Target values for prediction tasks
            loss_fn_reduction: Reduction method for loss calculation
            nan_target_idx: Index for NaN targets in prediction tasks
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing model outputs including loss
        """
        pass

    @abstractmethod
    def configure_optimizers(self, *args, **kwargs):
        pass

    @abstractmethod
    def load_pretrained(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_num_params(self):
        pass 