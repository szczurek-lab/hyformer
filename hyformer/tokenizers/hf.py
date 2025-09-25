import os
import warnings
from typing import Dict, List, Optional, Union, Any
import importlib.metadata

import torch
try:
    # Check tokenizers version
    tokenizers_version = importlib.metadata.version("tokenizers")
    
    # Import transformers without downgrading tokenizers
    from transformers import AutoTokenizer as HFAutoTokenizer
    import transformers
    
    # Log versions for debugging
    _version_info = {
        "tokenizers": tokenizers_version,
        "transformers": transformers.__version__
    }
except ImportError as e:
    raise ImportError(f"Required packages are not installed: {e}")

from hyformer.tokenizers.base import BaseTokenizer, TASK_TOKEN_DICT


class HFTokenizer(BaseTokenizer):
    """An adapter for Hugging Face tokenizers that implements the BaseTokenizer interface.
    
    This tokenizer wraps a Hugging Face tokenizer, allowing it to be used with the
    same interface as other tokenizers in this codebase. Special token handling is
    managed by the base tokenizer.
    
    Parameters
    ----------
    vocabulary_path : str
        Path or identifier of the Hugging Face tokenizer (e.g., 'facebook/esm2_t33_650M_UR50D')
    use_fast : bool, default=True
        Whether to use the fast tokenizer implementation
    trust_remote_code : bool, default=False
        Whether to trust remote code when loading the tokenizer
    **kwargs
        Additional parameters passed to the base tokenizer, including:
        task_tokens : dict, optional
            Optional dictionary of task tokens to override defaults in TASK_TOKEN_DICT
    """
    
    def __init__(
        self,
        vocabulary_path: str,
        use_fast: bool = True,
        trust_remote_code: bool = False,
        subfolder: Optional[str] = None,
        repo_type: Optional[str] = None,
        **kwargs
    ) -> None:
        """Initialize the HF tokenizer adapter.
        
        Parameters
        ----------
        vocabulary_path : str
            Path or identifier of the Hugging Face tokenizer
        use_fast : bool, default=True
            Whether to use the fast tokenizer implementation
        trust_remote_code : bool, default=False
            Whether to trust remote code when loading the tokenizer
        **kwargs
            Additional keyword arguments passed to the parent class
        """
        # Store configuration options
        self.use_fast = use_fast
        self.trust_remote_code = trust_remote_code
        
        # Initialize the HF tokenizer first, with compatibility options
        try:
            self.hf_tokenizer = HFAutoTokenizer.from_pretrained(
                vocabulary_path,
                use_fast=use_fast,
                trust_remote_code=trust_remote_code,
                legacy=False,  # Use the new tokenizers API
                subfolder=subfolder,
                repo_type=repo_type
            )
        except TypeError as e:
            if "legacy" in str(e):
                # Try without the legacy parameter (older transformers)
                self.hf_tokenizer = HFAutoTokenizer.from_pretrained(
                    vocabulary_path,
                    use_fast=use_fast,
                    trust_remote_code=trust_remote_code,
                    subfolder=subfolder,
                    repo_type=repo_type
                )
            else:
                raise
        
        # Initialize the base tokenizer
        super().__init__(vocabulary_path=vocabulary_path, **kwargs)
    
    def _load_vocab(self, vocab_file: str) -> Dict[str, int]:
        """Load vocabulary from the Hugging Face tokenizer.
        
        Parameters
        ----------
        vocab_file : str
            Path or identifier of the Hugging Face tokenizer
            
        Returns
        -------
        dict
            Dictionary mapping tokens to their IDs
        """
        # Use the HF tokenizer's vocabulary
        # Special tokens will be added by the base class
        return self.hf_tokenizer.get_vocab()
    
    def tokenize(self, text: str) -> List[str]:
        """Convert a text string into a list of tokens using the HF tokenizer.
        
        This method only handles the core tokenization logic without adding
        special tokens, which is handled by the base tokenizer.
        
        Parameters
        ----------
        text : str
            The text to tokenize
            
        Returns
        -------
        list of str
            The list of tokens
        """
        # Use the HF tokenizer to tokenize the text without special tokens
        try:
            tokens = self.hf_tokenizer.tokenize(text, add_special_tokens=False)
        except TypeError:
            # Some newer tokenizers might not have add_special_tokens param
            tokens = self.hf_tokenizer.tokenize(text)
            
            # Remove any special tokens that might have been added
            special_tokens = set(self.hf_tokenizer.all_special_tokens)
            tokens = [t for t in tokens if t not in special_tokens]
            
        return tokens
    
    def __call__(
        self, 
        inputs: Union[str, List[str]],
        task: str,
        padding: bool = False,
        truncation: bool = True,
        max_length: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process inputs and prepare them for the model.
        
        This override uses the base tokenizer implementation to maintain interface consistency.
        
        Parameters
        ----------
        inputs : str or list of str
            String or list of strings to tokenize
        task : str
            Task identifier to prepend a task-specific token
        padding : bool, default=False
            Whether to pad sequences to max_length (if provided)
        truncation : bool, default=True
            Whether to truncate sequences longer than max_length (if provided)
        max_length : int, optional
            Maximum sequence length for truncation and padding. If None, no length constraints.
        **kwargs
            Additional arguments for the HF tokenizer
            
        Returns
        -------
        dict
            Dictionary containing at minimum:
            - input_ids: List of token ID lists
            - attention_mask: List of attention mask lists (1 for tokens, 0 for padding)
        """
        # Use the base implementation which returns lists
        result = super().__call__(
            inputs,
            task=task,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            **kwargs
        )
        
        return result
    
    def decode(
        self, 
        token_ids: Union[torch.Tensor, List[int]],
        skip_special_tokens: bool = True
    ) -> str:
        """Convert token IDs back to a string using the HF tokenizer.
        
        Parameters
        ----------
        token_ids : torch.Tensor or list of int
            Token IDs to decode
        skip_special_tokens : bool, default=True
            Whether to remove special tokens from the output
            
        Returns
        -------
        str
            Decoded string
        """
        # Handle tensor inputs
        if isinstance(token_ids, torch.Tensor):
            if token_ids.dim() > 1:
                # Handle batched inputs - just take the first item for now
                token_ids = token_ids[0].cpu().tolist()
            else:
                token_ids = token_ids.cpu().tolist()
        
        # Use the HF tokenizer to decode
        decoded = self.hf_tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )
        
        return decoded
    
    def _join_tokens(self, tokens: List[str]) -> str:
        """Join tokens into a string with appropriate spacing.
        
        Many HF tokenizers need spaces between tokens when joining.
        This method can be overridden by specific HF tokenizer adapters.
        
        Parameters
        ----------
        tokens : list of str
            Tokens to join
            
        Returns
        -------
        str
            Joined string
        """
        # Default implementation uses HF tokenizer's conversion
        # or falls back to space-joining for common tokenizers
        token_ids = self.convert_tokens_to_ids(tokens)
        return self.hf_tokenizer.decode(token_ids, skip_special_tokens=True)
    
    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_path: str,
        revision: str = "main",
        tokenizer_config: Optional[Dict[str, Any]] = None,
        local_dir: Optional[str] = None,
        local_dir_use_symlinks: str = "auto",
        subfolder: Optional[str] = None,
        repo_type: Optional[str] = None,
        **kwargs
    ) -> 'HFTokenizer':
        """Load a pretrained HFTokenizer.
        
        For HFTokenizer, this method directly loads the HF tokenizer from the path/repo
        and wraps it, rather than loading our custom config files.
        
        Parameters
        ----------
        repo_id_or_path : str
            Path or HuggingFace model identifier for the tokenizer
        revision : str, optional
            Git revision, by default "main"
        tokenizer_config : dict, optional
            Additional configuration, by default None
        local_dir : str, optional
            Local directory, by default None  
        local_dir_use_symlinks : str, optional
            Symlink usage, by default "auto"
        **kwargs
            Additional arguments passed to the tokenizer constructor
            
        Returns
        -------
        HFTokenizer
            Loaded HF tokenizer instance
        """
        # For HFTokenizer, we use the repo_id_or_path directly as vocabulary_path
        # since it points to a HuggingFace model identifier
        init_kwargs = kwargs.copy()
        if tokenizer_config:
            init_kwargs.update(tokenizer_config)
        
        # For HF tokenizers, store the repo id and pass subfolder/repo_type through
        init_kwargs['vocabulary_path'] = repo_id_or_path
        init_kwargs['subfolder'] = subfolder
        init_kwargs['repo_type'] = repo_type
        
        return cls(**init_kwargs)

    @classmethod
    def from_config(cls, config, **kwargs) -> 'HFTokenizer':
        """Create a HFTokenizer from configuration.
        
        Parameters
        ----------
        config : TokenizerConfig
            Tokenizer configuration
        **kwargs
            Additional keyword arguments to override configuration
            
        Returns
        -------
        HFTokenizer
            Initialized tokenizer
        """
        # Extract kwargs from config if present
        config_kwargs = {}
        if hasattr(config, 'kwargs') and config.kwargs:
            config_kwargs = config.kwargs
            
        # Get other common parameters from config
        init_kwargs = {
            'vocabulary_path': config.vocabulary_path,
            'task_tokens': getattr(config, 'task_tokens', None),
            'trust_remote_code': config_kwargs.get('trust_remote_code', False)
        }
        
        # Override with config_kwargs and then with explicit kwargs
        init_kwargs.update(config_kwargs)
        init_kwargs.update(kwargs)
        
        return cls(**init_kwargs) 