from typing import Dict, List, Union, Any
from abc import ABC, abstractmethod

IGNORE_TOKEN_IDX = -100


class BaseTokenizer(ABC):
    """ Abstract class for tokenizers compatible with the trainer's DataCollatorWithTaskTokens. """
    
    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        """ID for the padding token."""
        pass
        
    @property
    @abstractmethod
    def bos_token_id(self) -> int:
        """ID for the beginning of sequence token."""
        pass
        
    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        """ID for the end of sequence token."""
        pass
        
    @property
    @abstractmethod
    def unk_token_id(self) -> int:
        """ID for the unknown token."""
        pass
        
    @property
    @abstractmethod
    def mask_token_id(self) -> int:
        """ID for the mask token (used in MLM)."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the size of the vocabulary."""
        pass
    
    @abstractmethod
    def __call__(
        self, 
        inputs: Union[str, List[str]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process inputs and prepare them for the model.
        
        Args:
            inputs: String or list of strings to tokenize
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing at minimum:
            - input_ids: Token IDs
            - attention_mask: Attention mask (1 for tokens, 0 for padding)
        """
        pass
    
    @abstractmethod
    def decode(
        self, 
        token_ids: Union[List[int], Any],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Convert token IDs back to a string.
        
        Args:
            token_ids: List or tensor of token IDs
            skip_special_tokens: Whether to remove special tokens
            
        Returns:
            Decoded string
        """
        pass
