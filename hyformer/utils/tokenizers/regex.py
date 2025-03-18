import re
import os
import torch
from typing import List, Dict, Any, Optional, Union
import collections

from .base_tokenizer import BaseTokenizer, TASK_TOKEN_DICT

MAX_LENGTH = 512
SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"""

class RegexSMILESTokenizer(BaseTokenizer):
    """
    A regex-based tokenizer for SMILES strings using DeepChem's pattern.
    
    This tokenizer splits SMILES strings into tokens using the regex pattern
    developed by Schwaller et al. and used in DeepChem.
    
    References:
    ----------
    [1] Philippe Schwaller, Teodoro Laino, Théophile Gaudin, Peter Bolgar, Christopher A. Hunter, 
        Costas Bekas, and Alpha A. Lee. ACS Central Science 2019 5 (9): Molecular Transformer: 
        A Model for Uncertainty-Calibrated Chemical Reaction Prediction 1572-1583 
        DOI: 10.1021/acscentsci.9b00576
    """
    
    def __init__(
        self, 
        tokenizer_path: Optional[str] = None,
        regex_pattern: str = SMI_REGEX_PATTERN,
        task_tokens: Optional[Dict[str, str]] = TASK_TOKEN_DICT,
        max_length: int = MAX_LENGTH,
        **kwargs
    ):
        """
        Initialize the tokenizer.
        
        Parameters
        ----------
        tokenizer_path : str, optional
            Path to vocabulary file (deepchem.txt)
        regex_pattern : str
            SMILES token regex pattern
        task_tokens : Dict[str, str], optional
            Dictionary mapping task names to task token strings
        max_length : int
            Maximum sequence length
        **kwargs : dict
            Additional arguments
        """
        # Initialize the base tokenizer
        super().__init__(
            tokenizer_path=tokenizer_path or "",
            task_tokens=task_tokens or TASK_TOKEN_DICT,
            max_length=max_length,
            **kwargs
        )
        
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)
        
        # Create a simple tokenizer object to store in self.tokenizer
        # This is needed to match the BaseTokenizer interface
        self.tokenizer = SimpleTokenizer()
        
        # Initialize vocabulary with special tokens
        self.vocab = collections.OrderedDict({
            self.tokenizer.pad_token: 0,
            self.tokenizer.unk_token: 1,
            self.tokenizer.bos_token: 2,
            self.tokenizer.eos_token: 3,
            self.tokenizer.mask_token: 4
        })
        
        # Add task tokens to vocabulary
        for task, token in self.task_tokens.items():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        # Load vocabulary if provided
        if tokenizer_path and os.path.exists(tokenizer_path):
            self._load_vocab(tokenizer_path)
        
        # Create reverse mapping (id to token)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()]
        )
        
        # Update the tokenizer's vocabulary
        self.tokenizer.vocab = self.vocab
        self.tokenizer.ids_to_tokens = self.ids_to_tokens
    
    def _load_vocab(self, vocab_file: str) -> None:
        """
        Load vocabulary from file.
        
        Parameters
        ----------
        vocab_file : str
            Path to vocabulary file
        """
        try:
            # Use the load_vocab function
            loaded_vocab = load_vocab(vocab_file)
            
            # Update our vocabulary with the loaded one
            # We keep our special tokens at their original indices
            special_tokens = {
                self.tokenizer.pad_token, self.tokenizer.unk_token, 
                self.tokenizer.bos_token, self.tokenizer.eos_token, 
                self.tokenizer.mask_token
            }
            
            # Add task tokens to special tokens
            for token in self.task_tokens.values():
                special_tokens.add(token)
            
            # Add tokens from loaded vocabulary (skip special tokens)
            for token, _ in loaded_vocab.items():
                if token not in special_tokens and token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
            
            print(f"Loaded vocabulary from {vocab_file} ({len(loaded_vocab)} tokens)")
            print(f"Final vocabulary size: {len(self.vocab)} tokens")
        except Exception as e:
            print(f"Error loading vocabulary: {e}")
    
    def _tokenize_text(
        self, 
        texts: Union[str, List[str]],
        max_length: int,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Convert text to token IDs without adding task tokens.
        
        This method implements the abstract method from BaseTokenizer.
        
        Parameters
        ----------
        texts : Union[str, List[str]]
            Input text or list of texts to tokenize
        max_length : int
            Maximum length for tokenization
        **kwargs : dict
            Additional arguments
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with tokenized outputs
        """
        # Handle single string vs list of strings
        if isinstance(texts, str):
            # Tokenize a single string
            tokens = self.tokenize(texts)
            
            # Truncate if necessary
            if max_length and len(tokens) > max_length:
                tokens = tokens[:max_length]
            
            # Convert to IDs
            input_ids = self.convert_tokens_to_ids(tokens)
            
            # Create attention mask (1 for real tokens)
            attention_mask = [1] * len(input_ids)
            
            # Convert to tensors
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
            }
        else:
            # Tokenize a batch of strings
            batch_input_ids = []
            batch_attention_mask = []
            
            for text in texts:
                # Tokenize
                tokens = self.tokenize(text)
                
                # Truncate if necessary
                if max_length and len(tokens) > max_length:
                    tokens = tokens[:max_length]
                
                # Convert to IDs
                input_ids = self.convert_tokens_to_ids(tokens)
                
                # Create attention mask
                attention_mask = [1] * len(input_ids)
                
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
            
            # Pad sequences to the same length
            max_len = max(len(ids) for ids in batch_input_ids)
            
            # Pad input_ids and attention_mask
            padded_input_ids = []
            padded_attention_mask = []
            
            for input_ids, attention_mask in zip(batch_input_ids, batch_attention_mask):
                # Calculate padding length
                pad_len = max_len - len(input_ids)
                
                # Pad input_ids with pad_token_id
                padded_input_ids.append(
                    input_ids + [self.vocab[self.tokenizer.pad_token]] * pad_len
                )
                
                # Pad attention_mask with 0s
                padded_attention_mask.append(
                    attention_mask + [0] * pad_len
                )
            
            # Convert to tensors
            return {
                "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long)
            }
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a SMILES string into tokens.
        
        Parameters
        ----------
        text : str
            SMILES string to tokenize
            
        Returns
        -------
        List[str]
            List of tokens
        """
        tokens = [token for token in self.regex.findall(text)]
        return tokens
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Convert tokens to token IDs.
        
        Parameters
        ----------
        tokens : List[str]
            List of tokens
            
        Returns
        -------
        List[int]
            List of token IDs
        """
        return [self.vocab.get(token, self.tokenizer.unk_token) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """
        Convert token IDs to tokens.
        
        Parameters
        ----------
        ids : List[int]
            List of token IDs
            
        Returns
        -------
        List[str]
            List of tokens
        """
        return [self.ids_to_tokens.get(id, self.tokenizer.unk_token) for id in ids]
    
    def encode(self, text: str, add_special_tokens: bool = True, task: str = "lm") -> List[int]:
        """
        Tokenize and convert to token IDs.
        
        Parameters
        ----------
        text : str
            SMILES string to encode
        add_special_tokens : bool
            Whether to add special tokens (CLS and SEP)
        task : str
            Task type to use for tokenization
            
        Returns
        -------
        List[int]
            List of token IDs
        """
        # Use the BaseTokenizer.__call__ method if add_special_tokens is True
        if add_special_tokens:
            tokenized = self(text, task=task)
            return tokenized["input_ids"].tolist()
        else:
            # Otherwise, just tokenize without special tokens
            tokens = self.tokenize(text)
            return self.convert_tokens_to_ids(tokens)
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> Union[str, List[str]]:
        """
        Convert token IDs back to strings.
        
        Parameters
        ----------
        token_ids : Union[List[int], torch.Tensor]
            Token IDs to decode
        skip_special_tokens : bool
            Whether to skip special tokens in the output
            
        Returns
        -------
        Union[str, List[str]]
            Decoded string(s)
        """
        # Convert tensor to list if needed
        if isinstance(token_ids, torch.Tensor):
            if token_ids.dim() > 1:
                # Batch of sequences
                return [self._decode_single(ids.tolist(), skip_special_tokens) for ids in token_ids]
            else:
                # Single sequence
                token_ids = token_ids.tolist()
        
        # Handle single sequence
        return self._decode_single(token_ids, skip_special_tokens)
    
    def _decode_single(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a single sequence of token IDs.
        
        Parameters
        ----------
        ids : List[int]
            Token IDs to decode
        skip_special_tokens : bool
            Whether to skip special tokens
            
        Returns
        -------
        str
            Decoded string
        """
        tokens = self.convert_ids_to_tokens(ids)
        
        # Skip special tokens if requested
        if skip_special_tokens:
            special_tokens = {
                self.tokenizer.bos_token, self.tokenizer.eos_token,
                self.tokenizer.pad_token, self.tokenizer.unk_token,
                self.tokenizer.mask_token
            }
            
            # Add task tokens to special tokens
            for token in self.task_tokens.values():
                special_tokens.add(token)
                
            tokens = [token for token in tokens if token not in special_tokens]
        
        # Join tokens without spaces
        return ''.join(tokens)
    
    @classmethod
    def from_config(cls, config: Any) -> 'RegexSMILESTokenizer':
        """
        Create a tokenizer from a configuration object.
        
        Parameters
        ----------
        config : Any
            Configuration object with tokenizer settings
            
        Returns
        -------
        RegexSMILESTokenizer
            Initialized tokenizer
        """
        # Extract parameters from config
        tokenizer_path = getattr(config, "tokenizer_path", None)
        regex_pattern = getattr(config, "regex_pattern", SMI_REGEX_PATTERN)
        task_tokens = getattr(config, "task_tokens", None)
        max_length = getattr(config, "max_length", MAX_LENGTH)
        
        # Create tokenizer
        return cls(
            path_to_vocabulary=config.path_to_vocabulary,
            task_tokens=task_tokens,
            max_length=max_length
        )
