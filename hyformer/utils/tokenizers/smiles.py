import re
import os
from typing import Dict, List, Optional

from hyformer.utils.tokenizers.base import BaseTokenizer, TASK_TOKEN_DICT


# SMILES regex pattern for tokenization
SMILES_REGEX_PATTERN = r"""(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|%[0-9]{2}|[0-9])"""


class SMILESTokenizer(BaseTokenizer):
    """A tokenizer specialized for SMILES strings using regex-based tokenization.
    
    This tokenizer implements the BaseTokenizer interface and uses a regex pattern
    to split SMILES strings into tokens. Special token handling is managed by the 
    base tokenizer.
    
    Parameters
    ----------
    vocabulary_path : str
        Path to the vocabulary file
    regex_pattern : str, default=SMILES_REGEX_PATTERN
        Regex pattern for SMILES tokenization
    **kwargs
        Additional parameters passed to the base tokenizer, including:
        max_length : int, default=512
            Maximum sequence length
        task_tokens : dict, optional
            Optional dictionary of task tokens to override defaults in TASK_TOKEN_DICT
    """
    
    def __init__(
        self,
        vocabulary_path: str,
        regex_pattern: str = SMILES_REGEX_PATTERN,
        **kwargs
    ) -> None:
        """Initialize the SMILES tokenizer.
        
        Parameters
        ----------
        vocabulary_path : str
            Path to the vocabulary file
        regex_pattern : str, default=SMILES_REGEX_PATTERN
            Regex pattern for SMILES tokenization
        **kwargs
            Additional keyword arguments passed to the parent class
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)
        super().__init__(vocabulary_path=vocabulary_path, **kwargs)
    
    def _load_vocab(self, vocab_file: str) -> Dict[str, int]:
        """Load vocabulary from file.
        
        Parameters
        ----------
        vocab_file : str
            Path to the vocabulary file
            
        Returns
        -------
        dict
            Dictionary mapping tokens to their IDs
            
        Raises
        ------
        FileNotFoundError
            If the vocabulary file does not exist
        """
        vocab = {}
        with open(vocab_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                token = line.strip()
                if token:  # Skip empty lines
                    vocab[token] = i
        
        return vocab
    
    def tokenize(self, text: str) -> List[str]:
        """Convert a SMILES string into a list of tokens using regex pattern.
        
        This method only handles the core tokenization logic without adding
        special tokens, which is handled by the base tokenizer.
        
        Parameters
        ----------
        text : str
            The SMILES string to tokenize
            
        Returns
        -------
        list of str
            The list of tokens
        """
        # Simple regex-based tokenization
        tokens = self.regex.findall(text)
        return tokens
    
    @classmethod
    def from_config(cls, config, **kwargs) -> 'SMILESTokenizer':
        """Create a SMILESTokenizer from configuration.
        
        Parameters
        ----------
        config : TokenizerConfig
            Tokenizer configuration
        **kwargs
            Additional keyword arguments to override configuration
            
        Returns
        -------
        SMILESTokenizer
            Initialized tokenizer
        """
        # Extract kwargs from config if present
        config_kwargs = {}
        if hasattr(config, 'kwargs') and config.kwargs:
            config_kwargs = config.kwargs
            
        # Get other common parameters from config
        init_kwargs = {
            'vocabulary_path': config.vocabulary_path,
            'max_length': getattr(config, 'max_length', 512),
            'task_tokens': getattr(config, 'task_tokens', None)
        }
        
        # Override with config_kwargs and then with explicit kwargs
        init_kwargs.update(config_kwargs)
        init_kwargs.update(kwargs)
        
        return cls(**init_kwargs) 
    