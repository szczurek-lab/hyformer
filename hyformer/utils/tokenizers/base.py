from typing import Dict, List, Union, Any, Optional
from abc import ABC, abstractmethod
import torch

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

# Maximum default length
MAX_LENGTH = 512


class BaseTokenizer(ABC):
    """Abstract base class for tokenizers compatible with the trainer's DataCollatorWithTaskTokens.
    
    This class defines the interface that all tokenizers must implement and provides
    common functionality that can be used by all tokenizer implementations. It handles
    all special token functionality, allowing derived classes to focus solely on
    tokenizing their specific string formats.
    
    Parameters
    ----------
    vocabulary_path : str
        Path to the vocabulary file
    max_length : int, default=MAX_LENGTH
        Maximum sequence length
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
        Optional dictionary of task tokens to override defaults in TASK_TOKEN_DICT
    """
    
    def __init__(
        self,
        vocabulary_path: str,
        max_length: int = MAX_LENGTH,
        bos_token: str = TOKEN_DICT["bos"],
        eos_token: str = TOKEN_DICT["eos"],
        pad_token: str = TOKEN_DICT["pad"],
        unk_token: Optional[str] = None,
        mask_token: Optional[str] = TOKEN_DICT["mask"],
        task_tokens: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> None:
        """Initialize the base tokenizer.
        
        Parameters
        ----------
        vocabulary_path : str
            Path to the vocabulary file
        max_length : int, default=MAX_LENGTH
            Maximum sequence length
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
            Optional dictionary of task tokens to override defaults in TASK_TOKEN_DICT
        """
        self.vocab_file = vocabulary_path
        self.max_length = max_length
        self._setup_special_tokens(bos_token, eos_token, unk_token, pad_token, mask_token, task_tokens)
        self.vocab = self._load_vocab(vocabulary_path)
        self._add_special_tokens_to_vocab()
    
    def _setup_special_tokens(
        self,
        bos_token: str,
        eos_token: str,
        unk_token: Optional[str],
        pad_token: str,
        mask_token: Optional[str],
        task_tokens: Optional[Dict[str, str]]
    ) -> None:
        """Set up special tokens and task tokens.
        
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
    
    @abstractmethod
    def _load_vocab(self, vocab_file: str) -> Dict[str, int]:
        """Load vocabulary from file.
        
        This method should only load the base vocabulary without worrying about 
        special tokens, which are handled by the base class.
        
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
    
    def _add_special_tokens_to_vocab(self) -> None:
        """Add special tokens and task tokens to the vocabulary.
        
        This method ensures that all special tokens (including task tokens) are present
        in the vocabulary, adding them if necessary. It also builds mappings for
        token IDs and pre-computes commonly used token IDs.
        """
        next_id = len(self.vocab)
        for _, token in self.special_tokens.items():
            if token is not None and token not in self.vocab:
                self.vocab[token] = next_id
                next_id += 1
        
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
        task: str,
        padding: bool = False,
        truncation: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Process inputs and prepare them for the model.
        
        Parameters
        ----------
        inputs : str or list of str
            String or list of strings to tokenize
        task : str
            Task identifier to prepend a task-specific token
        padding : bool, default=False
            Whether to pad sequences to max_length
        truncation : bool, default=True
            Whether to truncate sequences longer than max_length
        **kwargs
            Additional arguments for specific tokenizer implementations
            
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
            tokens.insert(0, self.get_task_token(task))
            tokens.insert(1, self.special_tokens["bos"])
            tokens.append(self.special_tokens["eos"])
            
            # Truncate if needed
            if truncation and len(tokens) > self.max_length:
                tokens = tokens[:self.max_length-1] + [tokens[-1]]
            
            # Convert to ids
            input_ids = self.convert_tokens_to_ids(tokens)
            batch_input_ids.append(input_ids)
        
        max_len = max(len(ids) for ids in batch_input_ids)
        
        # Create attention masks and pad if needed
        attention_mask = []
        if padding:
            pad_token_id = self.pad_token_id
            padded_input_ids = []
            for input_ids in batch_input_ids:
                mask = [1] * len(input_ids) + [0] * (max_len - len(input_ids))
                attention_mask.append(mask)
                
                padded_input_ids.append(
                    input_ids + [pad_token_id] * (max_len - len(input_ids))
                )
            batch_input_ids = padded_input_ids
        else:
            attention_mask = [[1] * len(ids) for ids in batch_input_ids]
        
        return {
            "input_ids": batch_input_ids,
            "attention_mask": attention_mask
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
