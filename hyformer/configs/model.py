from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from hyformer.configs.base import BaseConfig
from hyformer.utils.runtime import find_multiple

EMBEDDING_DIM_HIDDEN_FACTOR = 4
EMBEDDING_DIM_MULTIPLE_OF = 256


@dataclass
class ModelConfig(BaseConfig):
    """Configuration for model architecture and parameters. """
    # Required parameters
    model_type: str  # Type/name of the model architecture (e.g., "Hyformer")
    embedding_dim: int  # Dimension of token embeddings
    num_attention_heads: int  # Number of attention heads
    num_layers: int  # Number of transformer layers
    vocab_size: int  # Size of the vocabulary
    max_sequence_length: int  # Maximum sequence length
    
    # Optional parameters with defaults
    local_attention_heads: Optional[int] = None  # Number of local attention heads
    attention_head_size: Optional[int] = None  # Dimension of each attention head
    hidden_embedding_dim: Optional[int] = None  # Hidden dimension in feed-forward layers
    use_bias: bool = True  # Whether to use bias in linear layers
    attention_dropout: float = 0.0  # Dropout rate for attention
    hidden_dropout: float = 0.0  # Dropout rate for feed-forward layers
    classifier_dropout: float = 0.0  # Dropout rate for prediction head
    layer_norm_eps: float = 1e-5  # Epsilon for layer normalization
    
    # Predictor/Pooler parameters
    predictor_hidden_dim: Optional[int] = None  # Hidden dimension for predictor/pooler
    predictor_dropout: float = 0.0  # Dropout rate for predictor/pooler
    predictor_num_heads: Optional[int] = None  # Number of heads in predictor
    
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
        
        if self.predictor_hidden_dim is None:
            self.predictor_hidden_dim = self.embedding_dim
        
        if self.local_attention_heads is None:
            self.local_attention_heads = self.num_attention_heads
        
        if self.attention_head_size is None:
            self.attention_head_size = self.embedding_dim // self.num_attention_heads
            