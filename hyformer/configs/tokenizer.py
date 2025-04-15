from typing import Optional, Dict, Any
from dataclasses import dataclass
from hyformer.configs.base import BaseConfig


@dataclass
class TokenizerConfig(BaseConfig):
    """Configuration for tokenizers. """
    vocabulary_path: str  # Path to the vocabulary file or HF model name
    tokenizer_type: str  # Type of tokenizer to use
    
    # Common settings for all tokenizers
    max_length: int = 512  # Maximum sequence length
    task_tokens: Optional[Dict[str, str]] = None  # Optional task tokens dictionary
    
    # Additional configuration options stored as a dictionary
    kwargs: Dict[str, Any] = None
