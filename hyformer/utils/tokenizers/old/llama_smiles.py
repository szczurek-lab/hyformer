#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import os
from typing import Dict, List, Optional, Union, Tuple, Any
import torch
from collections import OrderedDict
from transformers import LlamaTokenizer, PreTrainedTokenizerBase

from .base import BaseTokenizer

# SMILES regex pattern
SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\%|\d+)"""

# Default maximum sequence length
MAX_LENGTH = 512

# Default token dictionary
TOKEN_DICT = {
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
    "bos_token": "[BOS]",
    "eos_token": "[EOS]",
    "mask_token": "[MASK]",
}

# Default task token dictionary
TASK_TOKEN_DICT = {
    "lm": "[LM]",       # Language modeling
    "mlm": "[MLM]",     # Masked language modeling
    "prediction": "[PRED]",  # Property prediction
    "property": "[PROP]"  # Property modeling
}

class LlamaSMILESTokenizer(BaseTokenizer):
    """
    A SMILES tokenizer that uses a regex pattern for tokenization and a custom vocabulary.
    Optionally, it can also use the LlamaTokenizer for non-SMILES text.
    """
    
    def __init__(
        self,
        tokenizer_path: str,
        pretrained_model_name_or_path: Optional[str] = None,
        regex_pattern: str = SMI_REGEX_PATTERN,
        task_tokens: Optional[Dict[str, str]] = None,
        max_length: int = MAX_LENGTH,
        **kwargs
    ):
        """Initialize the tokenizer.
        
        Args:
            tokenizer_path: Path to the custom vocabulary file
            pretrained_model_name_or_path: Optional path to pretrained LlamaTokenizer or model name
            regex_pattern: Regex pattern for SMILES tokenization
            task_tokens: Dictionary of task tokens
            max_length: Maximum sequence length
            **kwargs: Additional arguments
        """
        super().__init__(
            tokenizer_path=tokenizer_path,
            task_tokens=task_tokens,
            max_length=max_length
        )
        
        # Store SMILES regex pattern
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        
        # Load custom vocabulary
        self.vocab = self._load_vocab(tokenizer_path)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        
        # Add special tokens to vocabulary
        self._add_special_tokens()
        
        # Initialize task token IDs dictionary
        self._task_token_ids = {}
        for task, token in self.task_tokens.items() if self.task_tokens else {}:
            self._task_token_ids[task] = self.vocab.get(token, -1)
            if self._task_token_ids[task] == -1:
                # Add to vocabulary if not present
                self.vocab[token] = len(self.vocab)
                self.ids_to_tokens[self.vocab[token]] = token
                self._task_token_ids[task] = self.vocab[token]
        
        # Optionally load LlamaTokenizer for non-SMILES text
        self.llama_tokenizer = None
        if pretrained_model_name_or_path:
            self.llama_tokenizer = LlamaTokenizer.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
            
            # Add task tokens to LlamaTokenizer if provided
            if task_tokens:
                special_tokens_dict = {}
                for task, token in task_tokens.items():
                    if token not in self.llama_tokenizer.all_special_tokens:
                        special_tokens_dict["additional_special_tokens"] = \
                            special_tokens_dict.get("additional_special_tokens", []) + [token]
                
                if special_tokens_dict:
                    self.llama_tokenizer.add_special_tokens(special_tokens_dict)
    
    @property
    def bos_token_id(self) -> int:
        """Get the ID of the beginning of sequence token."""
        return self.vocab.get(TOKEN_DICT["bos_token"], 1)  # Default to 1 if not found
    
    @property
    def eos_token_id(self) -> int:
        """Get the ID of the end of sequence token."""
        return self.vocab.get(TOKEN_DICT["eos_token"], 2)  # Default to 2 if not found
    
    @property
    def pad_token_id(self) -> int:
        """Get the ID of the padding token."""
        return self.vocab.get(TOKEN_DICT["pad_token"], 0)  # Default to 0 if not found
    
    @property
    def unk_token_id(self) -> int:
        """Get the ID of the unknown token."""
        return self.vocab.get(TOKEN_DICT["unk_token"], 3)  # Default to 3 if not found
    
    @property
    def mask_token_id(self) -> int:
        """Get the ID of the mask token."""
        return self.vocab.get(TOKEN_DICT["mask_token"], 4)  # Default to 4 if not found
    
    def __len__(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.vocab)
    
    def task_token_id(self, task: str) -> int:
        """
        Get the token ID for a specific task.
        
        Args:
            task: Task name
            
        Returns:
            Token ID for the task
        """
        if task not in self.task_tokens:
            raise ValueError(f"Unsupported task: {task}. Supported tasks are: {', '.join(self.task_tokens.keys())}")
            
        task_token = self.task_tokens[task]
        
        # Add the token to the vocabulary if it doesn't exist
        if task not in self._task_token_ids:
            token_id = self.convert_tokens_to_ids(task_token)
            if token_id == self.unk_token_id:
                # Token not in vocabulary, add it
                self.vocab[task_token] = len(self.vocab)
                self.ids_to_tokens[self.vocab[task_token]] = task_token
                token_id = self.vocab[task_token]
            self._task_token_ids[task] = token_id
            
        return self._task_token_ids[task]
        
    def _load_vocab(self, vocab_file: str) -> Dict[str, int]:
        """Load vocabulary from file.
        
        Args:
            vocab_file: Path to vocabulary file
            
        Returns:
            Dictionary mapping tokens to indices
        """
        vocab = OrderedDict()
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for index, token in enumerate(f):
                token = token.strip()
                vocab[token] = index
        return vocab
    
    def _add_special_tokens(self):
        """Add special tokens to the vocabulary."""
        special_tokens = {
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
            "mask_token": "[MASK]"
        }
        
        # Add task tokens
        if self.task_tokens:
            for task, token in self.task_tokens.items():
                special_tokens[f"{task}_token"] = token
            
        # Add special tokens to vocabulary if they don't exist
        for name, token in special_tokens.items():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.ids_to_tokens[self.vocab[token]] = token
                
        # Update special tokens mapping
        self.special_tokens = {
            name: token for name, token in special_tokens.items()
        }
    
    def _tokenize_smiles(self, text: str) -> List[str]:
        """Tokenize SMILES string using regex pattern.
        
        Args:
            text: SMILES string
            
        Returns:
            List of tokens
        """
        return self.regex.findall(text)
    
    def _tokenize_text(
        self, 
        texts: Union[str, List[str]],
        max_length: int,
        is_smiles: bool = True,
        return_tensors: str = "pt",
        padding: Union[bool, str] = True,
        truncation: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Convert text to token IDs.
        
        Args:
            texts: Input text or list of texts
            max_length: Maximum sequence length
            is_smiles: Whether to use SMILES tokenization (default: True)
            return_tensors: Return type of tensors
            padding: Padding strategy
            truncation: Whether to truncate sequences
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with tokenized outputs
        """
        # If not SMILES and LlamaTokenizer is available, use it
        if not is_smiles and self.llama_tokenizer:
            return self.llama_tokenizer(
                texts,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors=return_tensors,
                **kwargs
            )
        
        # Otherwise use custom vocabulary with regex tokenization
        if isinstance(texts, str):
            texts = [texts]
            
        # Tokenize each string using regex
        tokenized_texts = [self._tokenize_smiles(text) if is_smiles else text.split() for text in texts]
        
        # Convert to token IDs
        input_ids = []
        for tokens in tokenized_texts:
            ids = [self.vocab.get(t, self.vocab[self.special_tokens["unk_token"]]) for t in tokens]
            input_ids.append(ids)
        
        # Handle padding and truncation
        if truncation:
            input_ids = [ids[:max_length] for ids in input_ids]
            
        if padding == "max_length":
            pad_id = self.vocab[self.special_tokens["pad_token"]]
            input_ids = [ids + [pad_id] * (max_length - len(ids)) 
                        for ids in input_ids]
        elif padding:
            pad_id = self.vocab[self.special_tokens["pad_token"]]
            max_len = max(len(ids) for ids in input_ids)
            input_ids = [ids + [pad_id] * (max_len - len(ids)) 
                        for ids in input_ids]
        
        attention_mask = [[1] * len(ids) for ids in input_ids]
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    
    def _add_task_tokens(self, tokenized, task):
        """
        Add task tokens to tokenized output.
        
        Args:
            tokenized: Dictionary with tokenized outputs
            task: Task type to use for tokenization
            
        Returns:
            Dictionary with tokenized outputs including task tokens
        """
        # Get the task token ID and BOS token ID
        task_token_id = self.task_token_id(task)
        bos_token_id = self.bos_token_id
        
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # Handle single sequence vs batch
        if isinstance(input_ids, torch.Tensor):
            # Tensor version
            if input_ids.dim() == 1 or (input_ids.dim() == 2 and input_ids.size(0) == 1):
                # For a single sequence
                prefix_tokens = torch.tensor([task_token_id, bos_token_id], dtype=input_ids.dtype, device=input_ids.device)
                
                # Flatten if needed
                if input_ids.dim() == 2 and input_ids.size(0) == 1:
                    input_ids = input_ids.flatten()
                    attention_mask = attention_mask.flatten()
                
                tokenized["input_ids"] = torch.cat([prefix_tokens, input_ids])
                
                # Update attention mask
                attention_prefix = torch.tensor([1, 1], dtype=attention_mask.dtype, device=attention_mask.device)
                tokenized["attention_mask"] = torch.cat([attention_prefix, attention_mask])
            else:
                # For batched sequences
                batch_size = input_ids.size(0)
                
                # Create prefix tensors for task and BOS tokens
                task_tokens = torch.full((batch_size, 1), task_token_id, dtype=input_ids.dtype, device=input_ids.device)
                bos_tokens = torch.full((batch_size, 1), bos_token_id, dtype=input_ids.dtype, device=input_ids.device)
                
                # Concatenate task token, BOS token, and input_ids
                tokenized["input_ids"] = torch.cat([task_tokens, bos_tokens, input_ids], dim=1)
                
                # Update attention mask
                attention_task = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_bos = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)
                tokenized["attention_mask"] = torch.cat([attention_task, attention_bos, attention_mask], dim=1)
        else:
            # List version
            if isinstance(input_ids[0], int):
                # Single sequence
                tokenized["input_ids"] = [task_token_id, bos_token_id] + input_ids
                tokenized["attention_mask"] = [1, 1] + attention_mask
            else:
                # Batch of sequences
                batch_size = len(input_ids)
                tokenized["input_ids"] = [[task_token_id, bos_token_id] + seq for seq in input_ids]
                tokenized["attention_mask"] = [[1, 1] + mask for mask in attention_mask]
        
        return tokenized
            
    def __call__(
        self, 
        inputs: Union[str, List[str]],
        task: str = "lm",
        is_smiles: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Process inputs and prepare them for the model.
        
        Args:
            inputs: String or list of strings to tokenize
            task: Task type (e.g., "lm", "mlm", "prediction", "property")
            is_smiles: Whether inputs are SMILES strings
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing at minimum:
            - input_ids: Token IDs
            - attention_mask: Attention mask (1 for tokens, 0 for padding)
        """
        # First tokenize the text without task tokens
        tokenized = self._tokenize_text(
            texts=inputs,
            max_length=self.max_length - 2,  # Reserve space for task and BOS tokens
            is_smiles=is_smiles,
            **kwargs
        )
        
        # Then add task tokens to the tokenized output
        tokenized = self._add_task_tokens(tokenized, task)
        
        return tokenized
    
    def tokenize(
        self, 
        text: Union[str, List[str]], 
        is_smiles: bool = True,
        **kwargs
    ) -> Union[List[str], List[List[str]]]:
        """
        Tokenize text into tokens.
        
        Args:
            text: Input text or list of texts
            is_smiles: Whether to use SMILES tokenization (default: True)
            **kwargs: Additional arguments
            
        Returns:
            List of tokens or list of token lists
        """
        if not is_smiles and self.llama_tokenizer:
            return self.llama_tokenizer.tokenize(text, **kwargs)
        
        if is_smiles:
            if isinstance(text, str):
                return self._tokenize_smiles(text)
            return [self._tokenize_smiles(t) for t in text]
        else:
            # Fallback to basic whitespace tokenization
            if isinstance(text, str):
                return text.split()
            return [t.split() for t in text]
    
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Convert tokens to token IDs.
        
        Args:
            tokens: Token or list of tokens
            
        Returns:
            Token ID or list of token IDs
        """
        if isinstance(tokens, str):
            return self.vocab.get(tokens, self.vocab[self.special_tokens["unk_token"]])
        return [self.vocab.get(t, self.vocab[self.special_tokens["unk_token"]]) for t in tokens]
    
    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """Convert token IDs to tokens.
        
        Args:
            ids: Token ID or list of token IDs
            
        Returns:
            Token or list of tokens
        """
        if isinstance(ids, int):
            return self.ids_to_tokens.get(ids, self.special_tokens["unk_token"])
        return [self.ids_to_tokens.get(i, self.special_tokens["unk_token"]) for i in ids]
    
    def decode(
        self,
        token_ids: Union[torch.Tensor, List[int]],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs back to a string.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to remove special tokens
            
        Returns:
            Decoded string
        """
        # Handle various input types
        if isinstance(token_ids, torch.Tensor):
            # Convert tensor to list
            if token_ids.dim() > 1:
                # Handle batched input - decode first sequence only
                token_ids = token_ids[0].tolist()
            else:
                token_ids = token_ids.tolist()
        
        # Convert IDs to tokens
        tokens = [self.ids_to_tokens.get(id, TOKEN_DICT["unk_token"]) for id in token_ids]
        
        # Remove special tokens if requested
        if skip_special_tokens:
            special_tokens = list(TOKEN_DICT.values()) + list(self.task_tokens.values())
            tokens = [t for t in tokens if t not in special_tokens]
            
        # For SMILES, join tokens without spaces
        return "".join(tokens)
    
    def to_huggingface(self) -> PreTrainedTokenizerBase:
        """
        Create a HuggingFace-compatible tokenizer interface for use with collators.
        
        This creates a lightweight wrapper around the LlamaSMILESTokenizer that follows
        the HuggingFace PreTrainedTokenizerBase interface expected by DataCollatorWithTaskTokens.
        
        Returns:
            A HuggingFace-compatible tokenizer interface
        """
        # If we have a LlamaTokenizer, return it with additional methods
        if self.llama_tokenizer:
            # Create a copy of the LlamaTokenizer
            hf_tokenizer = self.llama_tokenizer
            
            # Inject our task handling and SMILES functionality
            original_call = hf_tokenizer.__call__
            
            def enhanced_call(inputs, task="lm", is_smiles=True, **kwargs):
                if is_smiles:
                    # Use our SMILES tokenization
                    return self(inputs, task=task, is_smiles=True, **kwargs)
                else:
                    # Use original LlamaTokenizer for text
                    return original_call(inputs, **kwargs)
            
            hf_tokenizer.__call__ = enhanced_call
            
            return hf_tokenizer
            
        # Otherwise, create a minimal adapter class that implements the expected interface
        class HuggingFaceAdapter(PreTrainedTokenizerBase):
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
                # Required attributes
                self.pad_token_id = tokenizer.pad_token_id
                self.bos_token_id = tokenizer.bos_token_id 
                self.eos_token_id = tokenizer.eos_token_id
                self.unk_token_id = tokenizer.unk_token_id
                self.mask_token_id = tokenizer.mask_token_id
                self.vocab_size = len(tokenizer)
                
            def __len__(self):
                return len(self.tokenizer)
                
            def __call__(self, inputs, task="lm", is_smiles=True, **kwargs):
                # Handle inputs - always expecting strings, not dictionaries
                return self.tokenizer(inputs, task=task, is_smiles=is_smiles, **kwargs)
                
            def decode(self, *args, **kwargs):
                return self.tokenizer.decode(*args, **kwargs)
                
            def convert_tokens_to_ids(self, *args, **kwargs):
                return self.tokenizer.convert_tokens_to_ids(*args, **kwargs)
                
            def convert_ids_to_tokens(self, *args, **kwargs):
                return self.tokenizer.convert_ids_to_tokens(*args, **kwargs)
                
            def get_vocab(self):
                return self.tokenizer.vocab
                
            # Implement required methods for PreTrainedTokenizerBase
            def save_vocabulary(self, save_directory):
                # Not implemented but required by the interface
                return [os.path.join(save_directory, "vocab.json")]
                
            def _add_tokens(self, new_tokens, special_tokens=False):
                # Not fully implemented but required by the interface
                return 0
                
        return HuggingFaceAdapter(self)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LlamaSMILESTokenizer":
        """
        Create a tokenizer from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            LlamaSMILESTokenizer instance
        """
        # Handle both dictionary-like and object-like config
        if hasattr(config, 'tokenizer_path'):
            # Object-like config
            return cls(
                tokenizer_path=config.tokenizer_path,
                pretrained_model_name_or_path=getattr(config, 'pretrained_model_name_or_path', None),
                regex_pattern=getattr(config, 'regex_pattern', SMI_REGEX_PATTERN),
                task_tokens=getattr(config, 'task_tokens', None),
                max_length=getattr(config, 'max_length', MAX_LENGTH),
                **(getattr(config, 'tokenizer_kwargs', {}) or {})
            )
        else:
            # Dictionary-like config
            return cls(
                tokenizer_path=config["tokenizer_path"],
                pretrained_model_name_or_path=config.get("pretrained_model_name_or_path"),
                regex_pattern=config.get("regex_pattern", SMI_REGEX_PATTERN),
                task_tokens=config.get("task_tokens"),
                max_length=config.get("max_length", MAX_LENGTH),
                **(config.get("tokenizer_kwargs", {}) or {})
            ) 