import torch
from typing import Dict, List, Union, Optional, Tuple, Any
from abc import ABC, abstractmethod

# Maximum length of the tokenized sequence
MAX_LENGTH = 512

# Standard token dictionary
TOKEN_DICT = {
    'bos': '<s>',       # Beginning of sequence
    'eos': '</s>',      # End of sequence
    'pad': '<pad>',     # Padding token
    'unk': '<unk>',     # Unknown token
    'mask': '<mask>',   # Mask token for MLM
    'ignore': -100      # Ignore index for loss calculation
}

# Task token dictionary
TASK_TOKEN_DICT = {
    'lm': '<lm>',                # Language modeling
    'prediction': '<cls>',       # Prediction task
    'mlm': '<mlm>'               # Masked language modeling task
}

class BaseTokenizer(ABC):
    """
    Abstract base class for tokenizers.
    
    This class defines the interface for tokenizers and provides common functionality
    for handling task tokens and tokenization.
    """
    
    def __init__(
        self,
        tokenizer_path: str,
        task_tokens: Optional[Dict[str, str]] = TASK_TOKEN_DICT,
        max_length: int = MAX_LENGTH,
        **kwargs
    ):
        """
        Initialize the base tokenizer.
        
        Args:
            tokenizer_path: Path to the tokenizer file
            task_tokens: Dictionary mapping task names to task token strings
            max_length: Maximum sequence length
            **kwargs: Additional arguments to pass to the tokenizer
        """
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length        
        self.task_tokens = task_tokens
        
        # Initialize the tokenizer (implemented by subclasses)
        self.tokenizer = None
        
        # Task token ID mapping (will be populated after tokenizer is initialized)
        self._task_token_ids = {}
    
    def __call__(
        self, 
        inputs: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]],
        task: str,  # Required - specify which task to use (e.g., 'lm', 'prediction', 'mlm')
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize inputs based on their type.
        
        This method handles different input types:
        - String or list of strings: Direct tokenization
        - Dictionary with 'data' and 'target' keys: Tokenize data and include targets
        - List of dictionaries: Tokenize each dictionary and collect targets
        
        Args:
            inputs: Input to tokenize (string, list of strings, dictionary, or list of dictionaries)
            task: Task type to use for tokenization. Must be one of: {list(self.task_tokens.keys())}
                 This determines which task token is prepended to the sequence.
            **kwargs: Additional arguments to pass to the tokenizer
            
        Returns:
            Dictionary with tokenized outputs including input_ids, attention_mask, and optionally targets
        """
        # Validate task parameter
        if task not in self.task_tokens:
            raise ValueError(f"Unsupported task: {task}. Supported tasks are: {', '.join(self.task_tokens.keys())}")
            
        # Handle dictionary input from SequenceDataset.__getitem__
        if isinstance(inputs, dict) and 'data' in inputs:
            # Extract data and target from dictionary
            data = inputs['data']
            target = inputs.get('target')
            
            # Tokenize the data
            tokenized = self._tokenize(data, task=task, **kwargs)
            
            # Include target if it exists
            if target is not None:
                # Ensure target is a tensor
                if not isinstance(target, torch.Tensor):
                    target = torch.tensor(target)
                tokenized['targets'] = target
            
            return tokenized
        
        # Handle list of dictionaries from DataLoader with SequenceDataset
        elif isinstance(inputs, list) and all(isinstance(x, dict) and 'data' in x for x in inputs):
            # Extract data from each dictionary
            data = [x['data'] for x in inputs]
            
            # Tokenize the batch of data
            tokenized = self._tokenize(data, task=task, **kwargs)
            
            # Collect targets if they exist
            targets = [x.get('target') for x in inputs]
            if any(t is not None for t in targets):
                # Filter out None values and ensure all targets are tensors
                valid_targets = [t for t in targets if t is not None]
                if valid_targets:
                    # Convert to tensors if needed
                    tensor_targets = [
                        t if isinstance(t, torch.Tensor) else torch.tensor(t)
                        for t in valid_targets
                    ]
                    
                    # If all targets have the same shape, stack them
                    if all(t.shape == tensor_targets[0].shape for t in tensor_targets):
                        tokenized['targets'] = torch.stack(tensor_targets)
                    else:
                        # Otherwise, just store the list of tensors
                        tokenized['targets'] = tensor_targets
                        
            return tokenized
        
        # Handle string or list of strings (backward compatibility)
        else:
            return self._tokenize(inputs, task=task, **kwargs)
    
    def _tokenize(
        self, 
        texts: Union[str, List[str]], 
        task: str = "lm",
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize input texts and add task tokens.
        
        This is a template method that:
        1. Calls _tokenize_text to convert text to token IDs
        2. Calls _add_task_tokens to add task-specific tokens
        
        Args:
            texts: Input text or list of texts to tokenize
            task: Task type to use for tokenization
            **kwargs: Additional arguments to pass to the tokenizer
            
        Returns:
            Dictionary with tokenized outputs including input_ids and attention_mask
        """
        # First, tokenize the text without task tokens
        tokenized = self._tokenize_text(
            texts, 
            max_length=self.max_length - 2,  # Reserve space for task and BOS tokens
            **kwargs
        )
        
        # Then add task tokens to the tokenized output
        return self._add_task_tokens(tokenized, task)
    
    @abstractmethod
    def _tokenize_text(
        self, 
        texts: Union[str, List[str]],
        max_length: int,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Convert text to token IDs without adding task tokens.
        
        This method should be implemented by subclasses to handle the
        specific tokenization logic for their token type.
        
        Args:
            texts: Input text or list of texts to tokenize
            max_length: Maximum length for tokenization
            **kwargs: Additional arguments to pass to the tokenizer
            
        Returns:
            Dictionary with tokenized outputs
        """
        pass
    
    def _add_task_tokens(
        self, 
        tokenized: Dict[str, torch.Tensor], 
        task: str
    ) -> Dict[str, torch.Tensor]:
        """
        Add task tokens to tokenized output.
        
        This method prepends the task token and BOS token to each sequence.
        
        Args:
            tokenized: Dictionary with tokenized outputs
            task: Task type to use for tokenization
            
        Returns:
            Dictionary with tokenized outputs including task tokens
        """
        # Get the task token ID and BOS token ID
        task_token_id = self.task_token_id(task)
        bos_token_id = self.tokenizer.bos_token_id
        
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # Handle single sequence vs batch
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
            
            # Update special tokens mask if present
            if "special_tokens_mask" in tokenized:
                special_prefix = torch.tensor([1, 1], dtype=tokenized["special_tokens_mask"].dtype, device=tokenized["special_tokens_mask"].device)
                tokenized["special_tokens_mask"] = torch.cat([special_prefix, tokenized["special_tokens_mask"].flatten()])
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
            
            # Update special tokens mask if present
            if "special_tokens_mask" in tokenized:
                special_task = torch.ones((batch_size, 1), dtype=tokenized["special_tokens_mask"].dtype, device=tokenized["special_tokens_mask"].device)
                special_bos = torch.ones((batch_size, 1), dtype=tokenized["special_tokens_mask"].dtype, device=tokenized["special_tokens_mask"].device)
                tokenized["special_tokens_mask"] = torch.cat(
                    [special_task, special_bos, tokenized["special_tokens_mask"]], 
                    dim=1
                )
        
        return tokenized
    
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
            token_id = self.tokenizer.convert_tokens_to_ids(task_token)
            if token_id == self.tokenizer.unk_token_id:
                # Token not in vocabulary, add it
                token_id = self.tokenizer.add_tokens([task_token])
                # If add_tokens returns the number of added tokens, get the actual ID
                if isinstance(token_id, int) and token_id > 0:
                    token_id = self.tokenizer.convert_tokens_to_ids(task_token)
            self._task_token_ids[task] = token_id
            
        return self._task_token_ids[task]
    
    @abstractmethod
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        """
        Decode token IDs back to strings.
        
        Args:
            token_ids: Tensor of token IDs
            skip_special_tokens: Whether to skip special tokens in the decoded output
            
        Returns:
            List of decoded strings
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: Any) -> 'BaseTokenizer':
        """
        Create a tokenizer from a configuration object.
        
        Args:
            config: Configuration object
            
        Returns:
            Initialized tokenizer
        """
        pass
