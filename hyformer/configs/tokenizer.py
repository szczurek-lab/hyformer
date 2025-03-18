from typing import Optional
from dataclasses import dataclass
from hyformer.configs.base import BaseConfig


@dataclass
class TokenizerConfig(BaseConfig):
    """Configuration for tokenizers. """
    path_to_vocabulary: str  # Path to the vocabulary file
    tokenizer_type: str  # Type of tokenizer to use
