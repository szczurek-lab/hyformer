from typing import Dict, List, Union, Any, Optional
from abc import ABC, abstractmethod
import os
import warnings
import torch

try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError
except ImportError:
    hf_hub_download = None
    RepositoryNotFoundError = Exception
    warnings.warn("HuggingFace Hub is not installed. Loading tokenizers from HuggingFace not available.")

IGNORE_TOKEN_IDX = -100

TOKEN_DICT = {
    'bos': '<s>',      
    'eos': '</s>',      
    'pad': '<pad>',     
    'unk': '<unk>',     
    'mask': '<mask>',  
}

TASK_TOKEN_DICT = {
    'lm': '<lm>',                # Language modeling
    'prediction': '<cls>',       # Prediction task
    'mlm': '<mlm>'               # Masked language modeling task
}

# Default filenames for tokenizer files
TOKENIZER_CONFIG_FILENAME = "tokenizer_config.json"
VOCABULARY_FILENAME = "vocab.txt"


class BaseTokenizer(ABC):
    """Base class for tokenizers.
    
    Parameters
    ----------
    vocabulary_path : str
        Path to the vocabulary file
    bos_token : str
        Beginning of sequence token. Required.
    eos_token : str
        End of sequence token. Required.
    pad_token : str
        Padding token. Required.
    unk_token : str or None, default=None
        Unknown token. If None, tokenizer will fail on unknown tokens.
    mask_token : str or None, default=TOKEN_DICT["mask"]
        Masking token for masked language modeling. If None, masking will not be used.
    task_tokens : dict, optional
        Optional dictionary of task tokens.
    """
    
    def __init__(
        self,
        vocabulary_path: str,
        bos_token: str = TOKEN_DICT["bos"],
        eos_token: str = TOKEN_DICT["eos"],
        pad_token: str = TOKEN_DICT["pad"],
        unk_token: Optional[str] = None,
        mask_token: Optional[str] = TOKEN_DICT["mask"],
        task_tokens: Optional[Dict[str, str]] = TASK_TOKEN_DICT,
        **kwargs
    ) -> None:
        """Initialize the base tokenizer.
        
        Parameters
        ----------
        vocabulary_path : str
            Path to the vocabulary file
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
        """
        self.vocab_file = vocabulary_path
        self.vocab = self._load_vocab(vocabulary_path)
        self._init_special_tokens(bos_token, eos_token, unk_token, pad_token, mask_token, task_tokens)
    
    def _init_special_tokens(
        self,
        bos_token: str,
        eos_token: str,
        unk_token: Optional[str],
        pad_token: str,
        mask_token: Optional[str],
        task_tokens: Optional[Dict[str, str]]
    ) -> None:
        """Initialize special tokens and add them to vocabulary.
        
        This method sets up the special tokens dictionary and ensures that all 
        special tokens (including task tokens) are present in the vocabulary, 
        adding them if necessary. It also builds mappings for token IDs and 
        pre-computes commonly used token IDs.
        
        Parameters
        ----------
        bos_token : str
            Beginning of sequence token
        eos_token : str
            End of sequence token
        unk_token : str or None
            Unknown token
        pad_token : str
            Padding token
        mask_token : str or None
            Masking token
        task_tokens : dict or None
            Dictionary of task tokens
        """
        # Set up special tokens dictionary
        self.special_tokens = {
            "bos": bos_token,
            "eos": eos_token,
            "pad": pad_token
        }
        
        if unk_token is not None:
            self.special_tokens["unk"] = unk_token
        if mask_token is not None:
            self.special_tokens["mask"] = mask_token
        
        task_dict = TASK_TOKEN_DICT.copy() if task_tokens is None else task_tokens.copy()
        self.special_tokens.update(task_dict)
        
        # Add special tokens to vocabulary if not present
        next_id = len(self.vocab)
        for _, token in self.special_tokens.items():
            if token is not None and token not in self.vocab:
                self.vocab[token] = next_id
                next_id += 1
        
        # Build reverse mapping and caches
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        
        self._token_id_cache = {}
        self._special_token_ids = {
            "bos": self.bos_token_id,
            "eos": self.eos_token_id,
            "pad": self.pad_token_id
        }
        
        if self.mask_token_id is not None:
            self._special_token_ids["mask"] = self.mask_token_id
        if self.unk_token_id is not None:
            self._special_token_ids["unk"] = self.unk_token_id
    
    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_path: str,
        revision: str = "main",
        tokenizer_config: Optional[Dict[str, Any]] = None,
        local_dir: Optional[str] = None,
        local_dir_use_symlinks: str = "auto",
        **kwargs
    ) -> "BaseTokenizer":
        """Load a pretrained tokenizer from HuggingFace Hub or a local path.

        Parameters
        ----------
        repo_id_or_path : str
            Path to local directory containing tokenizer files or HuggingFace Hub repository ID.
        revision : str, optional
            Git revision for HuggingFace Hub repositories, by default "main".
        tokenizer_config : dict, optional
            Tokenizer configuration dictionary. If None, will attempt to load from
            tokenizer_config.json, by default None.
        local_dir : str, optional
            Local directory to download the tokenizer files from HuggingFace Hub,
            by default None.
        local_dir_use_symlinks : str, optional
            Whether to use symlinks for local directory, by default "auto".
        **kwargs
            Additional keyword arguments passed to the tokenizer constructor.

        Returns
        -------
        BaseTokenizer
            Loaded tokenizer instance.

        Raises
        ------
        ValueError
            If tokenizer config or vocabulary files are not found.
        NotImplementedError
            If the specific tokenizer class doesn't implement from_config.

        Examples
        --------
        Load from HuggingFace Hub:
        ```
        tokenizer = SMILESTokenizer.from_pretrained("SzczurekLab/hyformer-tokenizer")
        ```
        
        Load from local directory:
        ```
        tokenizer = SMILESTokenizer.from_pretrained("./my_tokenizer")
        ```
        """
        # Load tokenizer config
        if tokenizer_config is None:
            config_path_local = os.path.join(repo_id_or_path, TOKENIZER_CONFIG_FILENAME)
            if os.path.exists(config_path_local):
                import json
                with open(config_path_local, 'r') as f:
                    tokenizer_config = json.load(f)
            else:
                try:
                    if hf_hub_download is None:
                        raise ValueError("HuggingFace Hub is not available and no local config found")
                    config_path_hf = hf_hub_download(
                        repo_id=repo_id_or_path, 
                        filename=TOKENIZER_CONFIG_FILENAME, 
                        revision=revision,
                        local_dir=local_dir, 
                        local_dir_use_symlinks=local_dir_use_symlinks
                    )
                    import json
                    with open(config_path_hf, 'r') as f:
                        tokenizer_config = json.load(f)
                except (Exception, RepositoryNotFoundError) as e:
                    # If no config found, use defaults
                    tokenizer_config = {}
        
        # Determine vocabulary path
        vocab_path_local = os.path.join(repo_id_or_path, VOCABULARY_FILENAME)
        if os.path.exists(vocab_path_local):
            vocabulary_path = vocab_path_local
        else:
            try:
                if hf_hub_download is None:
                    raise ValueError(f"HuggingFace Hub is not available and vocabulary not found at {vocab_path_local}")
                vocabulary_path = hf_hub_download(
                    repo_id=repo_id_or_path, 
                    filename=VOCABULARY_FILENAME, 
                    revision=revision,
                    local_dir=local_dir, 
                    local_dir_use_symlinks=local_dir_use_symlinks
                )
            except (Exception, RepositoryNotFoundError) as e:
                raise ValueError(f"Vocabulary file not found in {repo_id_or_path}")

        # Merge config with kwargs
        init_kwargs = tokenizer_config.copy()
        init_kwargs.update(kwargs)
        init_kwargs['vocabulary_path'] = vocabulary_path

        # Create tokenizer instance
        return cls(**init_kwargs)

    def save_pretrained(
        self,
        save_directory: str,
        save_vocabulary: bool = True,
        **kwargs
    ) -> None:
        """Save the tokenizer configuration and vocabulary to a directory.

        Parameters
        ----------
        save_directory : str
            Directory where the tokenizer will be saved.
        save_vocabulary : bool, optional
            Whether to save the vocabulary file, by default True.
        **kwargs
            Additional keyword arguments (reserved for future use).

        Raises
        ------
        OSError
            If the save directory cannot be created.

        Examples
        --------
        ```
        tokenizer.save_pretrained("./my_tokenizer")
        ```
        """
        os.makedirs(save_directory, exist_ok=True)

        # Save tokenizer configuration
        config = {
            "tokenizer_type": self.__class__.__name__,
            "special_tokens": self.special_tokens.copy(),
        }
        
        # Add any additional config attributes that might be useful
        if hasattr(self, 'regex_pattern'):
            config["regex_pattern"] = self.regex_pattern

        config_path = os.path.join(save_directory, TOKENIZER_CONFIG_FILENAME)
        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Save vocabulary if requested
        if save_vocabulary:
            vocab_path = os.path.join(save_directory, VOCABULARY_FILENAME)
            with open(vocab_path, 'w', encoding='utf-8') as f:
                # Sort by token ID to maintain order
                sorted_vocab = sorted(self.vocab.items(), key=lambda x: x[1])
                for token, _ in sorted_vocab:
                    f.write(f"{token}\n")

        print(f"Tokenizer saved to {save_directory}")

    @abstractmethod
    def _load_vocab(self, vocab_file: str) -> Dict[str, int]:
        """Load vocabulary from file.
        
        Vocabulary shouldn't contain special tokens.
        
        Parameters
        ----------
        vocab_file : str
            Path to the vocabulary file or model identifier
            
        Returns
        -------
        dict
            Dictionary mapping tokens to their IDs
        """
        pass
    
    @property
    def pad_token_id(self) -> int:
        """ID for the padding token.
        
        Returns
        -------
        int
            The token ID for padding
        
        Raises
        ------
        KeyError
            If the pad token is not in the vocabulary
        """
        pad_token = self.special_tokens["pad"]
        if pad_token not in self.vocab:
            raise KeyError(f"Pad token '{pad_token}' not found in vocabulary")
        return self.vocab[pad_token]
    
    @property
    def bos_token_id(self) -> int:
        """ID for the beginning of sequence token.
        
        Returns
        -------
        int
            The token ID for beginning of sequence
            
        Raises
        ------
        KeyError
            If the BOS token is not in the vocabulary
        """
        bos_token = self.special_tokens["bos"]
        if bos_token not in self.vocab:
            raise KeyError(f"BOS token '{bos_token}' not found in vocabulary")
        return self.vocab[bos_token]
    
    @property
    def eos_token_id(self) -> int:
        """ID for the end of sequence token.
        
        Returns
        -------
        int
            The token ID for end of sequence
            
        Raises
        ------
        KeyError
            If the EOS token is not in the vocabulary
        """
        eos_token = self.special_tokens["eos"]
        if eos_token not in self.vocab:
            raise KeyError(f"EOS token '{eos_token}' not found in vocabulary")
        return self.vocab[eos_token]
    
    @property
    def unk_token_id(self) -> Optional[int]:
        """ID for the unknown token.
        
        Returns
        -------
        int or None
            The token ID for unknown tokens, or None if unk_token is not defined
            
        Raises
        ------
        KeyError
            If the UNK token is not in the vocabulary but is defined
        """
        unk_token = self.special_tokens.get("unk")
        if unk_token is None:
            return None
        if unk_token not in self.vocab:
            raise KeyError(f"UNK token '{unk_token}' not found in vocabulary")
        return self.vocab[unk_token]
    
    @property
    def mask_token_id(self) -> Optional[int]:
        """ID for the mask token (used in MLM).
        
        Returns
        -------
        int or None
            The token ID for the mask token, or None if mask_token is not defined
            
        Raises
        ------
        KeyError
            If the mask token is not in the vocabulary but is defined
        """
        mask_token = self.special_tokens.get("mask")
        if mask_token is None:
            return None
        if mask_token not in self.vocab:
            raise KeyError(f"Mask token '{mask_token}' not found in vocabulary")
        return self.vocab[mask_token]
    
    def get_task_token(self, task: str) -> str:
        """Get a task token by name.
        
        Parameters
        ----------
        task : str
            The task identifier to retrieve a token for
            
        Returns
        -------
        str
            The task token string
            
        Raises
        ------
        ValueError
            If the task is not recognized
        """
        if task not in self.special_tokens:
            standard_tokens = {"bos", "eos", "pad", "unk", "mask"}
            available_tasks = [k for k in self.special_tokens if k not in standard_tokens]
            raise ValueError(f"Unknown task '{task}'. Available tasks: {available_tasks}")
        return self.special_tokens[task]
    
    def task_token_id(self, task: str) -> int:
        """Get the token ID for a task token.
        
        Parameters
        ----------
        task : str
            The task identifier
            
        Returns
        -------
        int
            The token ID for the task
            
        Raises
        ------
        ValueError
            If the task is not recognized
        KeyError
            If the task token is not in the vocabulary
        """
        task_token = self.get_task_token(task)
        if task_token not in self.vocab:
            raise KeyError(f"Task token '{task_token}' for task '{task}' not found in vocabulary")
        return self.vocab[task_token]
    
    def __len__(self) -> int:
        """Return the size of the vocabulary.
        
        Returns
        -------
        int
            The number of tokens in the vocabulary
        """
        return len(self.vocab)
    
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Convert a text string into a list of tokens.
        
        Parameters
        ----------
        text : str
            The text to tokenize
            
        Returns
        -------
        list of str
            The list of tokens
        """
        pass
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs.
        
        Parameters
        ----------
        tokens : list of str
            List of tokens to convert
            
        Returns
        -------
        list of int
            List of token IDs
            
        Raises
        ------
        KeyError
            If a token is not in the vocabulary and no UNK token is defined
        """
        result = []
        for token in tokens:
            # Use cache for faster lookups
            if token in self._token_id_cache:
                result.append(self._token_id_cache[token])
            elif token in self.vocab:
                token_id = self.vocab[token]
                self._token_id_cache[token] = token_id
                result.append(token_id)
            elif "unk" in self.special_tokens:
                # Use UNK token if available
                if "unk" not in self._token_id_cache:
                    self._token_id_cache["unk"] = self.unk_token_id
                result.append(self.unk_token_id)
            else:
                # Otherwise fail fast
                raise KeyError(f"Token '{token}' not found in vocabulary and no UNK token defined")
        
        return result
    
    @property
    def all_special_ids(self) -> List[int]:
        """Get all special token IDs.
        
        Returns
        -------
        list of int
            List of all special token IDs
        """
        return self.convert_tokens_to_ids(list(self.special_tokens.values()))
    
    def __call__(
        self, 
        inputs: Union[str, List[str]],
        task: str
    ) -> Dict[str, Any]:
        """Process inputs and prepare them for the model.
        
        Parameters
        ----------
        inputs : str or list of str
            String or list of strings to tokenize
        task : str
            Task identifier to prepend a task-specific token
            
        Returns
        -------
        dict
            Dictionary containing at minimum:
            - input_ids: List of token ID lists
            - attention_mask: List of attention mask lists (1 for tokens, 0 for padding)
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        
        batch_input_ids = []
        for text in inputs:
            tokens = self.tokenize(text)
            tokens.insert(0, self.get_task_token(task)) # add task token
            tokens.insert(1, self.special_tokens["bos"]) # add BOS token
            tokens.append(self.special_tokens["eos"]) # add EOS token
            batch_input_ids.append(self.convert_tokens_to_ids(tokens))
        
        return {
            "input_ids": batch_input_ids,
            "attention_mask": [[1] * len(ids) for ids in batch_input_ids]
        }
    
    def _decode_single_sequence(
        self, 
        token_ids: Union[torch.Tensor, List[int]],
        skip_special_tokens: bool = True
    ) -> str:
        """Convert token IDs back to a string.
        
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
        if isinstance(token_ids, torch.Tensor):
            if token_ids.dim() > 1:
                token_ids = token_ids[0].cpu().tolist()
            else:
                token_ids = token_ids.cpu().tolist()
        
        # Convert IDs to tokens
        tokens = [self.ids_to_tokens[token_id] for token_id in token_ids if token_id in self.ids_to_tokens]
        
        if skip_special_tokens:
            special_tokens = set(self.special_tokens.values())
            tokens = [t for t in tokens if t not in special_tokens]
        
        return self._join_tokens(tokens)
    
    def decode(
        self,
        token_ids: Union[torch.Tensor, List[int]],
        skip_special_tokens: bool = True
    ) -> str:
        """Convert token IDs back to a string.
        
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
        if isinstance(token_ids, torch.Tensor):
            if token_ids.dim() > 1:
                return [self._decode_single_sequence(ids, skip_special_tokens=skip_special_tokens) for ids in token_ids]
            else:
                return self._decode_single_sequence(token_ids, skip_special_tokens=skip_special_tokens)
        else:
            return self._decode_single_sequence(token_ids, skip_special_tokens=skip_special_tokens)
            
    def _join_tokens(self, tokens: List[str]) -> str:
        """Join tokens into a string.
        
        The default implementation is simple concatenation, which works for
        character-level or subword tokenization like SMILES. Override this
        method for tokenizers that need different joining behavior.
        
        Parameters
        ----------
        tokens : list of str
            Tokens to join
            
        Returns
        -------
        str
            Joined string
        """
        return ''.join(tokens)
