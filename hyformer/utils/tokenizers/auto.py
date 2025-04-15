import importlib
from typing import Dict, Any

from hyformer.configs.tokenizer import TokenizerConfig
from hyformer.utils.tokenizers.base import BaseTokenizer


class AutoTokenizer:
    """Factory class for creating tokenizers.
    
    This class provides a unified interface for creating tokenizers based on configuration,
    using a simple if-else approach to select the appropriate implementation.
    
    Currently supported tokenizers:
    - SMILESTokenizer: For SMILES molecular representations
    - HFTokenizer: For using Hugging Face tokenizers
    """

    @classmethod
    def from_config(cls, config: TokenizerConfig) -> BaseTokenizer:
        """Create a tokenizer from configuration.
        
        Parameters
        ----------
        config : TokenizerConfig
            Tokenizer configuration
            
        Returns
        -------
        BaseTokenizer
            Configured tokenizer instance
            
        Raises
        ------
        ValueError
            If the specified tokenizer type is not supported
        """
        # Simple if-else factory pattern
        if config.tokenizer_type == 'SMILESTokenizer':
            from hyformer.utils.tokenizers.smiles import SMILESTokenizer
            return SMILESTokenizer.from_config(config)
        elif config.tokenizer_type == "HFTokenizer":
            from hyformer.utils.tokenizers.hf import HFTokenizer
            return HFTokenizer.from_config(config)
        else:
            raise ValueError(f"Tokenizer type '{config.tokenizer_type}' is not supported. "
                             f"Supported types: 'SMILESTokenizer', 'HFTokenizer'")
    
    @classmethod
    def get_supported_tokenizer_types(cls) -> Dict[str, str]:
        """Get a mapping of supported tokenizer types to their descriptions.
        
        Returns
        -------
        dict
            Dictionary mapping tokenizer types to descriptions
        """
        return {
            'SMILESTokenizer': 'Tokenizer for SMILES molecular representations',
            'HFTokenizer': 'Wrapper for Hugging Face tokenizers'
        }
