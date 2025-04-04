import re
import os
import torch
from typing import Dict, List, Union, Optional, Any
from collections import OrderedDict
from hyformer.utils.tokenizers.base import BaseTokenizer, IGNORE_TOKEN_IDX


# Maximum length of the tokenized sequence
MAX_LENGTH = 512

# Standard token dictionary
TOKEN_DICT = {
    'bos': '<s>',       # Beginning of sequence
    'eos': '</s>',      # End of sequence
    'pad': '<pad>',     # Padding token
    'unk': '<unk>',     # Unknown token
    'mask': '<mask>',   # Mask token for MLM
    'ignore': IGNORE_TOKEN_IDX      # Ignore index for loss calculation
}

# Task token dictionary
TASK_TOKEN_DICT = {
    'lm': '<lm>',                # Language modeling
    'prediction': '<cls>',       # Prediction task
    'mlm': '<mlm>'               # Masked language modeling task
}

# SMILES regex pattern for tokenization
SMILES_REGEX_PATTERN = r"""(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|%[0-9]{2}|[0-9])"""


class SMILESTokenizer(BaseTokenizer):
    """
    A tokenizer specialized for SMILES strings using regex-based tokenization.
    
    This tokenizer implements the BaseTokenizer interface and uses a regex pattern
    to split SMILES strings into tokens, then maps those tokens to IDs based on
    a vocabulary loaded from a file.
    """
    
    def __init__(
        self,
        vocabulary_path: str,
        regex_pattern: str = SMILES_REGEX_PATTERN,
        max_length: int = MAX_LENGTH,
        bos_token: str = TOKEN_DICT["bos"],
        eos_token: str = TOKEN_DICT["eos"],
        unk_token: str = TOKEN_DICT["unk"],
        pad_token: str = TOKEN_DICT["pad"],
        mask_token: str = TOKEN_DICT["mask"],
        task_tokens: Optional[Dict[str, str]] = TASK_TOKEN_DICT,
        **kwargs
    ):
        """
        Initialize the SMILES tokenizer.
        
        Args:
            vocabulary_path: Path to the vocabulary file
            regex_pattern: Regex pattern for SMILES tokenization
            max_length: Maximum sequence length
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
            unk_token: Unknown token
            pad_token: Padding token
            mask_token: Masking token for masked language modeling
            task_tokens: Optional dictionary of task tokens to override defaults
        """
        self.vocab_file = vocabulary_path
        self.regex_pattern = regex_pattern
        self.max_length = max_length
        self.regex = re.compile(self.regex_pattern)
        
        # Special tokens setup
        self.special_tokens = {
            "bos_token": bos_token,
            "eos_token": eos_token,
            "pad_token": pad_token,
            "unk_token": unk_token,
            "mask_token": mask_token
        }
        
        # Task tokens
        self.task_tokens = task_tokens.copy() if task_tokens else TASK_TOKEN_DICT.copy()
        
        # Load vocabulary from file
        self.vocab = self._load_vocab(vocabulary_path)
        
        # Create the reverse mapping (id -> token)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        
        # Cache for token IDs to avoid repeated lookups
        self._token_id_cache = {}
        
        # Pre-compute special token IDs
        self._special_token_ids = {
            "bos": self.bos_token_id,
            "eos": self.eos_token_id,
            "pad": self.pad_token_id,
            "unk": self.unk_token_id,
            "mask": self.mask_token_id
        }
        
        # Pre-compute task token IDs
        self._task_token_ids = {
            task: self.get_task_token_id(task)
            for task in self.task_tokens
        }
    
    def _load_vocab(self, vocab_file: str) -> Dict[str, int]:
        """
        Load vocabulary from file.
        
        Args:
            vocab_file: Path to the vocabulary file
            
        Returns:
            Dictionary mapping tokens to their IDs
        """
        vocab = OrderedDict()
        with open(vocab_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                token = line.strip()
                if token:  # Skip empty lines
                    vocab[token] = i
        
        # Ensure special tokens are in vocabulary
        next_id = len(vocab)
        for token in self.special_tokens.values():
            if token not in vocab:
                vocab[token] = next_id
                next_id += 1
        
        # Ensure task tokens are in vocabulary
        for token in self.task_tokens.values():
            if token not in vocab:
                vocab[token] = next_id
                next_id += 1
        
        return vocab
    
    @property
    def pad_token_id(self) -> int:
        """ID for the padding token."""
        return self.vocab.get(self.special_tokens["pad_token"], 0)
    
    @property
    def bos_token_id(self) -> int:
        """ID for the beginning of sequence token."""
        return self.vocab.get(self.special_tokens["bos_token"], 1)
    
    @property
    def eos_token_id(self) -> int:
        """ID for the end of sequence token."""
        return self.vocab.get(self.special_tokens["eos_token"], 2)
    
    @property
    def unk_token_id(self) -> int:
        """ID for the unknown token."""
        return self.vocab.get(self.special_tokens["unk_token"], 3)
    
    @property
    def mask_token_id(self) -> int:
        """ID for the mask token (used in MLM)."""
        return self.vocab.get(self.special_tokens["mask_token"], 4)
    
    def get_task_token(self, task: str) -> str:
        """
        Get a task token by name.
        
        Args:
            task: The task identifier to retrieve a token for
            
        Returns:
            The task token string
        """
        assert task in self.task_tokens, f"Task '{task}' not found in task_tokens. Available tasks: {list(self.task_tokens.keys())}"
        return self.task_tokens[task]
    
    def get_task_token_id(self, task: str) -> int:
        """
        Get a task token ID by name.
        
        Args:
            task: The task identifier to retrieve a token ID for
            
        Returns:
            The task token ID
        """
        token = self.get_task_token(task)
        return self.vocab.get(token, self.unk_token_id)
    
    def __len__(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.vocab)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a SMILES string into tokens using regex pattern.
        
        Args:
            text: SMILES string to tokenize
            
        Returns:
            List of tokens
        """
        # Use regex.findall directly - it's already optimized
        return self.regex.findall(text)
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Convert tokens to their IDs using caching for efficiency.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of token IDs
        """
        # Use list comprehension with cached lookups
        return [self._token_id_cache.get(token, self.vocab.get(token, self.unk_token_id)) for token in tokens]
    
    def __call__(
        self, 
        inputs: Union[str, List[str]],
        task: str,
        padding: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process inputs and prepare them for the model.
        
        Args:
            inputs: String or list of strings to tokenize
            task: Task type (e.g., "lm", "mlm", "prediction")
            padding: Whether to pad sequences to the same length
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing:
            - input_ids: Token IDs tensor
            - attention_mask: Attention mask tensor
        """
        if isinstance(inputs, str):
            inputs = [inputs]
            
        # Get task token ID from cache
        task_token_id = self._task_token_ids[task]
        
        # Process all inputs at once
        batch_tokens = [self.tokenize(text) for text in inputs]
        
        # Truncate all sequences at once
        max_tokens = self.max_length - 3  # -3 for task, BOS, and EOS tokens
        batch_tokens = [tokens[:max_tokens] for tokens in batch_tokens]
        
        # Convert all sequences to IDs at once
        batch_ids = [self.convert_tokens_to_ids(tokens) for tokens in batch_tokens]
        
        # Add special tokens to all sequences at once
        batch_inputs = []
        batch_attention_masks = []
        for token_ids in batch_ids:
            input_ids = [task_token_id, self._special_token_ids["bos"]] + token_ids + [self._special_token_ids["eos"]]
            attention_mask = [1] * len(input_ids)
            batch_inputs.append(input_ids)
            batch_attention_masks.append(attention_mask)
        
        # Convert to tensors if requested
        # Tensor conversion is now handled by the collator
        return {
            "input_ids": batch_inputs,
            "attention_mask": batch_attention_masks
        }
    
    def decode(
        self, 
        token_ids: Union[List[int], Any],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Convert token IDs back to a SMILES string.
        
        Args:
            token_ids: List or tensor of token IDs
            skip_special_tokens: Whether to remove special tokens
            
        Returns:
            Decoded SMILES string
        """
        # Convert tensor to list if needed
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        
        # Handle nested lists (batched input)
        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]  # Take first sequence from batch
        
        # Convert IDs to tokens
        tokens = [self.ids_to_tokens.get(id, self.special_tokens["unk_token"]) for id in token_ids]
        
        # Remove special tokens if requested
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.all_special_tokens]
        
        # Join tokens without spaces (SMILES format)
        return "".join(tokens)
    
    @classmethod
    def from_config(cls, config, **kwargs) -> 'SMILESTokenizer':
        return cls(
            vocabulary_path=config.vocabulary_path,
            **kwargs
        )

    # Add HuggingFace-style getter/setter properties for special tokens
    @property
    def bos_token(self) -> str:
        """Get the beginning of sequence token."""
        return self.special_tokens["bos_token"]
    
    @bos_token.setter
    def bos_token(self, value: str):
        """Set the beginning of sequence token."""
        self.special_tokens["bos_token"] = value
        # Update vocab if the token doesn't exist
        if value not in self.vocab:
            self.vocab[value] = len(self.vocab)
            self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
    
    @property
    def eos_token(self) -> str:
        """Get the end of sequence token."""
        return self.special_tokens["eos_token"]
    
    @eos_token.setter
    def eos_token(self, value: str):
        """Set the end of sequence token."""
        self.special_tokens["eos_token"] = value
        # Update vocab if the token doesn't exist
        if value not in self.vocab:
            self.vocab[value] = len(self.vocab)
            self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
    
    @property
    def pad_token(self) -> str:
        """Get the padding token."""
        return self.special_tokens["pad_token"]
    
    @pad_token.setter
    def pad_token(self, value: str):
        """Set the padding token."""
        self.special_tokens["pad_token"] = value
        # Update vocab if the token doesn't exist
        if value not in self.vocab:
            self.vocab[value] = len(self.vocab)
            self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
    
    @property
    def unk_token(self) -> str:
        """Get the unknown token."""
        return self.special_tokens["unk_token"]
    
    @unk_token.setter
    def unk_token(self, value: str):
        """Set the unknown token."""
        self.special_tokens["unk_token"] = value
        # Update vocab if the token doesn't exist
        if value not in self.vocab:
            self.vocab[value] = len(self.vocab)
            self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
    
    @property
    def mask_token(self) -> str:
        """Get the mask token."""
        return self.special_tokens["mask_token"]
    
    @mask_token.setter
    def mask_token(self, value: str):
        """Set the mask token."""
        self.special_tokens["mask_token"] = value
        # Update vocab if the token doesn't exist
        if value not in self.vocab:
            self.vocab[value] = len(self.vocab)
            self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
    
    @property
    def all_special_tokens(self) -> List[str]:
        """Get all special tokens as a list."""
        special_tokens = list(self.special_tokens.values())
        # Add task tokens
        special_tokens.extend(self.task_tokens.values())
        # Filter out empty strings
        return [token for token in special_tokens if token]
    
    @property
    def all_special_ids(self) -> List[int]:
        """Get the IDs of all special tokens."""
        return [self.vocab.get(token, self.unk_token_id) for token in self.all_special_tokens] 
    