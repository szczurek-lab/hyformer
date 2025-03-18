#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from typing import Dict, List, Optional, Union, Tuple
import torch
from collections import OrderedDict

from .base import BaseTokenizer, TOKEN_DICT, TASK_TOKEN_DICT, MAX_LENGTH

# SMILES regex pattern
SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\%|\d+)"""

class LlamaSMILESTokenizer(BaseTokenizer):
    """A standalone SMILES tokenizer with custom vocabulary."""
    
    def __init__(
        self,
        tokenizer_path: str,
        regex_pattern: str = SMI_REGEX_PATTERN,
        task_tokens: Optional[Dict[str, str]] = None,
        max_length: int = MAX_LENGTH,
        **kwargs
    ):
        """Initialize the tokenizer.
        
        Args:
            tokenizer_path: Path to the vocabulary file
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
        
        # Store regex pattern
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        
        # Load vocabulary
        self.vocab = self._load_vocab(tokenizer_path)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        
        # Add special tokens to vocabulary if they don't exist
        self._add_special_tokens()
        
    def _load_vocab(self, vocab_file: str) -> OrderedDict:
        """Load vocabulary from file.
        
        Args:
            vocab_file: Path to vocabulary file
            
        Returns:
            OrderedDict mapping tokens to indices
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
            special_tokens.update(self.task_tokens)
            
        # Add special tokens to vocabulary if they don't exist
        for name, token in special_tokens.items():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.ids_to_tokens[self.vocab[token]] = token
                
        # Update special tokens mapping
        self.special_tokens = {
            name: token for name, token in special_tokens.items()
        }
        
    def _tokenize_text(self, text: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
        """Tokenize text using regex pattern.
        
        Args:
            text: Input text or list of texts
            
        Returns:
            List of tokens or list of token lists
        """
        if isinstance(text, str):
            return self.regex.findall(text)
        return [self.regex.findall(t) for t in text]
        
    def tokenize(self, text: Union[str, List[str]], **kwargs) -> Union[List[str], List[List[str]]]:
        """Tokenize text using regex pattern.
        
        Args:
            text: Input text or list of texts
            **kwargs: Additional arguments
            
        Returns:
            List of tokens or list of token lists
        """
        return self._tokenize_text(text)
        
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
        
    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        **kwargs
    ) -> Union[List[int], List[List[int]]]:
        """Encode text to token IDs.
        
        Args:
            text: Input text or list of texts
            add_special_tokens: Whether to add special tokens
            **kwargs: Additional arguments
            
        Returns:
            List of token IDs or list of token ID lists
        """
        # First tokenize using regex
        tokens = self.tokenize(text)
        
        # Convert to IDs
        ids = self.convert_tokens_to_ids(tokens)
        
        # Add special tokens if requested
        if add_special_tokens:
            if isinstance(ids, list) and isinstance(ids[0], list):
                ids = [self._add_task_tokens(seq) for seq in ids]
            else:
                ids = self._add_task_tokens(ids)
                
        return ids
        
    def decode(
        self,
        ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> Union[str, List[str]]:
        """Decode token IDs to text.
        
        Args:
            ids: Token IDs or list of token ID lists
            skip_special_tokens: Whether to skip special tokens
            **kwargs: Additional arguments
            
        Returns:
            Decoded text or list of decoded texts
        """
        # Convert IDs to tokens
        tokens = self.convert_ids_to_tokens(ids)
        
        # Join tokens
        if isinstance(tokens, list) and isinstance(tokens[0], list):
            return ["".join(t) for t in tokens]
        return "".join(tokens)
        
    @classmethod
    def from_config(cls, config: Dict) -> "LlamaSMILESTokenizer":
        """Create a tokenizer from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            LlamaSMILESTokenizer instance
        """
        return cls(
            tokenizer_path=config["tokenizer_path"],
            regex_pattern=config.get("regex_pattern", SMI_REGEX_PATTERN),
            task_tokens=config.get("task_tokens"),
            max_length=config.get("max_length", MAX_LENGTH),
            **config.get("tokenizer_kwargs", {})
        ) 