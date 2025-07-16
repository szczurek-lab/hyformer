from hyformer.tokenizers.smiles import SMILESTokenizer

# Amino acid regex (simple one-character tokenization)
AA_REGEX_PATTERN = r"([ACDEFGHIKLMNPQRSTVWYX]|[BZO]|U|\-|\.)"


class AATokenizer(SMILESTokenizer):
    """A tokenizer specialized for amino acid sequences using regex-based tokenization.
    
    This tokenizer implements the BaseTokenizer interface and uses a regex pattern
    to split amino acid sequences into tokens. Special token handling is managed by the 
    base tokenizer.
    
    Parameters
    ----------
    vocabulary_path : str
        Path to the vocabulary file
    regex_pattern : str, default=AA_REGEX_PATTERN
        Regex pattern for amino acid tokenization
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
        regex_pattern: str = AA_REGEX_PATTERN,
        **kwargs
    ) -> None:
        """Initialize the amino acid tokenizer.
        
        Parameters
        ----------
        vocabulary_path : str
            Path to the vocabulary file
        regex_pattern : str, default=AA_REGEX_PATTERN
            Regex pattern for amino acid tokenization
        **kwargs
            Additional keyword arguments passed to the parent class
        """
        super().__init__(vocabulary_path=vocabulary_path, regex_pattern=regex_pattern, **kwargs)
    