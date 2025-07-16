from typing import Optional, Dict, Any
from dataclasses import dataclass
from hyformer.configs.base import BaseConfig


@dataclass
class TokenizerConfig(BaseConfig):
    """Configuration for tokenizers. 
    
    Note: max_length is no longer a tokenizer property, it should be specified
    at runtime during tokenization calls for HuggingFace compatibility.
    """
    vocabulary_path: str  # Path to the vocabulary file or HF model name
    tokenizer_type: str  # Type of tokenizer to use
    
    # Common settings for all tokenizers
    task_tokens: Optional[Dict[str, str]] = None  # Optional task tokens dictionary
    
    # Additional configuration options stored as a dictionary
    kwargs: Dict[str, Any] = None
