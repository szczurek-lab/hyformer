from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union, Literal
import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam, AdamW, SGD, RMSprop

from hyformer.models.base import BaseModel


class TrainableModel(BaseModel, ABC):
    """Abstract base class for models that can be trained using the Trainer.
    
    This class defines the interface that all trainable models must implement.
    It ensures that models have the necessary methods for training, evaluation,
    and prediction.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def configure_optimizers(
        self,
        optimizer_type: Literal['adam', 'adamw', 'sgd', 'rmsprop'] = 'adamw',
        learning_rate: float = 1e-4,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None
    ) -> Optimizer:
        """Configure and return the optimizer for training.
        
        Args:
            optimizer_type: Type of optimizer to use ('adam', 'adamw', 'sgd', 'rmsprop')
            learning_rate: Initial learning rate
            optimizer_kwargs: Dictionary of optimizer-specific parameters
            device: Device to place the optimizer on
            
        Returns:
            Configured optimizer instance
            
        Example:
            >>> # For AdamW with weight decay
            >>> optimizer_kwargs = {
            ...     'weight_decay': 0.01,
            ...     'betas': (0.9, 0.999),
            ...     'eps': 1e-8
            ... }
            >>> optimizer = model.configure_optimizers(
            ...     optimizer_type='adamw',
            ...     learning_rate=1e-4,
            ...     optimizer_kwargs=optimizer_kwargs
            ... )
            
            >>> # For SGD with momentum
            >>> optimizer_kwargs = {
            ...     'momentum': 0.9,
            ...     'nesterov': True
            ... }
            >>> optimizer = model.configure_optimizers(
            ...     optimizer_type='sgd',
            ...     learning_rate=0.1,
            ...     optimizer_kwargs=optimizer_kwargs
            ... )
        """
        pass
    
    @abstractmethod
    def get_loss(self, **inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute the loss for the given inputs.
        
        Args:
            **inputs: Dictionary of input tensors
            
        Returns:
            Dictionary containing at least a 'loss' key with the computed loss value
        """
        pass
    
    @abstractmethod
    def predict(self, **inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate predictions for the given inputs.
        
        Args:
            **inputs: Dictionary of input tensors
            
        Returns:
            Tensor containing the model's predictions
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        tokenizer: Any,
        batch_size: int,
        temperature: float = 1.0,
        top_k: int = 25,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Generate samples from the model.
        
        Args:
            tokenizer: Tokenizer for encoding/decoding
            batch_size: Number of samples to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            device: Device to generate on
            
        Returns:
            Tensor containing generated samples
        """
        pass
    
    @abstractmethod
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get the model's state dictionary.
        
        Returns:
            Dictionary containing the model's parameters
        """
        pass
    
    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True) -> Any:
        """Load the model's state dictionary.
        
        Args:
            state_dict: State dictionary to load
            strict: Whether to enforce exact matching of keys
            
        Returns:
            Missing keys and unexpected keys if strict=False
        """
        pass
    
    @abstractmethod
    def to(self, device: torch.device) -> 'TrainableModel':
        """Move the model to the specified device.
        
        Args:
            device: Device to move the model to
            
        Returns:
            The model instance
        """
        pass
    
    @abstractmethod
    def train(self, mode: bool = True) -> 'TrainableModel':
        """Set the model's training mode.
        
        Args:
            mode: Whether to set training mode (True) or evaluation mode (False)
            
        Returns:
            The model instance
        """
        pass
    
    @abstractmethod
    def eval(self) -> 'TrainableModel':
        """Set the model to evaluation mode.
        
        Returns:
            The model instance
        """
        pass

    @abstractmethod
    def get_num_params(self):
        pass
    
    @abstractmethod
    def get_loss(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            properties: Optional[torch.Tensor] = None,
            task: Optional[str] = None):
        pass
    