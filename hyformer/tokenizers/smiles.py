import re
from typing import Dict, List, Optional

from hyformer.configs.tokenizer import TokenizerConfig
from hyformer.tokenizers.base import BaseTokenizer, TOKEN_DICT, TASK_TOKEN_DICT


# SMILES regex pattern for tokenization
SMILES_REGEX_PATTERN = r"""(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|%[0-9]{2}|[0-9])"""


class SMILESTokenizer(BaseTokenizer):
    """A tokenizer specialized for SMILES strings using regex-based tokenization.
    
    Parameters
    ----------
    vocabulary_path : str
        Path to the vocabulary file containing SMILES tokens
    regex_pattern : str, optional
        Regex pattern for SMILES tokenization. Uses SMILES_REGEX_PATTERN by default.
    bos_token : str, default=TOKEN_DICT["bos"]
        Beginning of sequence token
    eos_token : str, default=TOKEN_DICT["eos"]
        End of sequence token
    pad_token : str, default=TOKEN_DICT["pad"]
        Padding token
    unk_token : str or None, default=None
        Unknown token. If None, tokenizer will fail on unknown tokens.
    mask_token : str or None, default=TOKEN_DICT["mask"]
        Masking token for masked language modeling
    task_tokens : dict, optional
        Dictionary of task-specific tokens
    
    Examples
    --------
    Basic usage:
    ```python
    tokenizer = SMILESTokenizer("vocab.txt")
    result = tokenizer("CCO", task="lm")
    ```

    """
    
    def __init__(
        self,
        vocabulary_path: str,
        regex_pattern: Optional[str] = SMILES_REGEX_PATTERN,
        bos_token: str = TOKEN_DICT["bos"],
        eos_token: str = TOKEN_DICT["eos"],
        pad_token: str = TOKEN_DICT["pad"],
        unk_token: Optional[str] = None,
        mask_token: Optional[str] = TOKEN_DICT["mask"],
        task_tokens: Optional[Dict[str, str]] = TASK_TOKEN_DICT,
        **kwargs
    ) -> None:
        """Initialize the SMILES tokenizer.
        
        Parameters
        ----------
        vocabulary_path : str
            Path to the vocabulary file
        regex_pattern : str, default=SMILES_REGEX_PATTERN
            Regex pattern for SMILES tokenization
        bos_token : str, default=TOKEN_DICT["bos"]
            Beginning of sequence token. Required.
        eos_token : str, default=TOKEN_DICT["eos"]
            End of sequence token. Required.
        pad_token : str, default=TOKEN_DICT["pad"]
            Padding token. Required.
        unk_token : str or None, default=None
            Unknown token. If None, tokenizer will fail on unknown tokens.
        mask_token : str or None, default=TOKEN_DICT["mask"]
            Masking token for masked language modeling. If None, masking will not be used.
        task_tokens : dict, optional
            Optional dictionary of task tokens.
        **kwargs
            Additional keyword arguments passed to the parent class
        """
        self.regex_pattern = regex_pattern
        try:
            self.regex = re.compile(self.regex_pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{self.regex_pattern}': {e}")
        
        super().__init__(
            vocabulary_path=vocabulary_path,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            mask_token=mask_token,
            task_tokens=task_tokens,
            **kwargs
        )
    
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
        """Simple regex-based tokenization of SMILES strings
        
        Parameters
        ----------
        text : str
            The SMILES string to tokenize
            
        Returns
        -------
        list of str
            The list of tokens
        """
        return self.regex.findall(text)
    
    @classmethod
    def from_config(cls, config: TokenizerConfig, **kwargs) -> 'SMILESTokenizer':
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
            'task_tokens': getattr(config, 'task_tokens', None)
        }
        
        # Override with config_kwargs and then with explicit kwargs
        init_kwargs.update(config_kwargs)
        init_kwargs.update(kwargs)
        
        return cls(**init_kwargs) 
    