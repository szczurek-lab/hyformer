from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from hyformer.configs.base import BaseConfig
from hyformer.utils.experiments import find_multiple

EMBEDDING_DIM_HIDDEN_FACTOR = 4
EMBEDDING_DIM_MULTIPLE_OF = 256


@dataclass
class ModelConfig(BaseConfig):
    """Configuration for the Hyformer model.

    Parameters
    ----------
    vocab_size : int
        The size of the vocabulary.
    embedding_dim : int
        The dimension of the embedding.
    hidden_embedding_dim : int
        The dimension of the hidden embedding.
    attention_dropout_p : float
        The dropout rate for the attention.
    num_transformer_layers : int
        The number of transformer layers.
    num_attention_heads : int
        The number of attention heads.
    layer_norm_eps : float
        The epsilon for the layer normalization.
    num_prediction_tasks : int, optional
        The number of prediction tasks, by default None.
    prediction_task_type : str, optional
        The type of prediction task, by default None.
    prediction_head_dropout_p : float, optional
        The dropout rate for the prediction head, by default None.
    """

    model_type: str
    vocab_size: Optional[int] = None
    embedding_dim: Optional[int] = None
    num_transformer_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None
    layer_norm_eps: Optional[float] = 1e-6
    attention_dropout_p: Optional[float] = 0.0
    hidden_embedding_dim: Optional[int] = None
    num_prediction_tasks: Optional[int] = None
    prediction_task_type: Optional[str] = None
    prediction_head_dropout_p: Optional[float] = 0.0

    def __post_init__(self):
        """Initialize derived parameters based on provided values."""
        if self.model_type not in ["Moler", "UniMol", "RegressionTransformer", "MolGPT"]:
            self._post_init()
        
    def _post_init(self):
        """Initialize derived parameters based on provided values."""
        
        
        # Validate embedding dimension is compatible with number of heads
        assert self.embedding_dim % self.num_attention_heads == 0, f"Embedding dimension {self.embedding_dim} must be 0 modulo number of attention heads {self.num_attention_heads}."
        
        # Calculate derived parameters if not provided
        if self.hidden_embedding_dim is None:
            self.hidden_embedding_dim = find_multiple(self.embedding_dim * EMBEDDING_DIM_HIDDEN_FACTOR, EMBEDDING_DIM_MULTIPLE_OF)
    