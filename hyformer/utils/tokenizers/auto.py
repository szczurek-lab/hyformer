import importlib

from hyformer.configs.tokenizer import TokenizerConfig


class AutoTokenizer:
    """Factory class for creating tokenizers.
    
    This class provides a unified interface for creating tokenizers based on configuration.
    By default, it returns a SMILESTokenizer instance.
    """

    @classmethod
    def from_config(cls, config: TokenizerConfig):
        """Create a tokenizer from configuration.
        
        Args:
            config: Tokenizer configuration
            
        Returns:
            Configured tokenizer instance
            
        Raises:
            ValueError: If the specified tokenizer is not available
        """     
        if config.tokenizer_type == 'SMILESTokenizer':
            return getattr(importlib.import_module(
                "hyformer.utils.tokenizers.smiles"),
                "SMILESTokenizer").from_config(config)
        elif config.tokenizer_type == "HFTokenizer":
            return getattr(importlib.import_module(
                "hyformer.utils.tokenizers.hf"),
                "HFTokenizer").from_config(config)
        else:
            raise ValueError(f"Tokenizer {config.tokenizer_type} not available. Available options: 'SMILESTokenizer', 'HFTokenizer'")
