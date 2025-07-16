import torch
from typing import Dict, List, Union, Optional, Any, Tuple
from torch.distributions.categorical import Categorical
from hyformer.models.utils import ModelInput
from dataclasses import dataclass
import numpy as np

# Constants for token types
from hyformer.utils.tokenizers.base import BaseTokenizer
from hyformer.utils.tokenizers.base import IGNORE_TOKEN_IDX as TOKEN_IGNORE_INDEX

_LM_PREFIX_LENGTH = 2


@dataclass
class SequenceDataCollator:
    """
    Data collator for sequence data that handles padding, masking, and task-specific processing.
    
    This collator:
    1. Pads sequences to the maximum length in the batch or to `pad_to_multiple_of`
    2. Creates attention masks
    3. Applies dynamic masking for MLM tasks
    4. Handles different task types (lm, prediction, mlm)
    
    Following Hugging Face best practices, masking is handled by the collator to enable dynamic masking during training.
    """
    tokenizer: BaseTokenizer
    tasks: Dict[str, float]
    max_length: int
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    mask_probability: float = 0.15  # Probability for masking tokens in MLM task, used only if the MLM task is present in the tasks dictionary
    _lm_prefix_length: int = _LM_PREFIX_LENGTH
    
    def __post_init__(self):
        """Initialize task distribution for sampling tasks. Normalize the task probabilities to sum to 1."""
        total = sum(self.tasks.values())
        for task in self.tasks:
            self.tasks[task] = self.tasks[task] / total
        self._task_dist = Categorical(torch.Tensor(list(self.tasks.values())))
    
    def _sample_task(self) -> str:
        """Sample a task based on the task distribution."""
        return list(self.tasks.keys())[self._task_dist.sample().item()]
    
    def __call__(self, batch: List[Dict[str, Any]]) -> ModelInput:
        """
        Collate and pad a batch of examples.
        
        Prepare input_labels for specific tasks.
        
        Args:
            batch: List of examples to collate. Each example is a dictionary with 'data' and 'target' keys
                  from SequenceDataset.__getitem__. The 'target' value may be None if no target exists.
            
        Returns:
            Collated and padded batch as a ModelInput instance
        """
        # Sample a task for this batch
        task = self._sample_task()
        
        # Extract SMILES strings from the batch dictionaries
        smiles_data = [example['data'] for example in batch]
        
        # Tokenize only the SMILES strings
        tokenized_batch = self.tokenizer(smiles_data, task=task)
        
        # Get the input_ids and attention_mask
        input_ids = tokenized_batch["input_ids"]
        attention_mask = tokenized_batch["attention_mask"]
        
        # Calculate max length for padding - use the minimum of max_length and the longest sequence
        batch_max_length = min(self.max_length, max(len(ids) for ids in input_ids))
        
        # For language modeling, add one extra padding token to ensure EOS is properly handled when shifting
        if task == 'lm':
            batch_max_length += 1
        
        # Adjust to pad_to_multiple_of if specified
        if self.pad_to_multiple_of is not None:
            batch_max_length = ((batch_max_length + self.pad_to_multiple_of - 1) // 
                          self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        # Efficient padding using vectorized operations
        padded_input_ids = self._pad_sequences(
            input_ids, 
            batch_max_length, 
            pad_value=self.tokenizer.pad_token_id
        )
        
        padded_attention_mask = self._pad_sequences(
            attention_mask, 
            batch_max_length, 
            pad_value=False
        )
        
        input_labels = None
        target = None
        if task == 'lm':
            input_labels = padded_input_ids.clone()
            input_labels[:, :self._lm_prefix_length] = TOKEN_IGNORE_INDEX  # Assuming [TASK] and [BOS] are the first two tokens
            input_labels[input_labels == self.tokenizer.pad_token_id] = TOKEN_IGNORE_INDEX  # Pad tokens are ignored

        elif task == 'mlm':
            masked_input_ids, input_labels = self._mask_tokens(
                padded_input_ids.clone(),
                padded_attention_mask
            )
        
        elif task == 'prediction':
            if all(example['target'] is None for example in batch):  # if all targets are None, set target to None for getting cls token encodings
                target = None
            else:
                target = torch.from_numpy(np.stack([example['target'] for example in batch])).float()
     
        else:
            raise ValueError(f"Task {task} not supported")
              
        return ModelInput(
            input_ids=masked_input_ids if task == 'mlm' else padded_input_ids,
            attention_mask=padded_attention_mask,
            task=task,
            input_labels=input_labels,
            target=target
        )
    
    def _pad_sequences(self, sequences: List[Union[torch.Tensor, List[int]]], max_length: int, pad_value: Any) -> torch.Tensor:
        """
        Efficiently pad a list of sequences to the same length using vectorized operations.
        
        Args:
            sequences: List of tensors or lists to pad
            max_length: Maximum length to pad to
            pad_value: Value to use for padding
            
        Returns:
            Padded tensor of shape [batch_size, max_length]
        """
        # Convert sequences to tensors if they aren't already
        if not isinstance(sequences[0], torch.Tensor):
            sequences = [torch.tensor(seq) for seq in sequences]
            
        # Create a tensor filled with pad_value of shape [batch_size, max_length]
        batch_size = len(sequences)
        padded_sequences = torch.full(
            (batch_size, max_length),
            pad_value,
            dtype=sequences[0].dtype,
            device=sequences[0].device
        )
        
        # Copy each sequence into the padded tensor
        for i, seq in enumerate(sequences):
            # Truncate if longer than max_length
            length = min(len(seq), max_length)
            padded_sequences[i, :length] = seq[:length].clone()
            
        return padded_sequences
    
    def _mask_tokens(self, inputs: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling.
        
        Args:
            inputs: Input token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            
        Returns:
            Tuple of:
                - Masked input tensor
                - Labels tensor (with -100 for non-masked tokens)
        """
        labels = inputs.clone()
        
        # Create a mask for special tokens
        special_tokens_mask = self._get_special_tokens_mask(inputs)
        
        # Create probability matrix for masking
        probability_matrix = torch.full(labels.shape, self.mask_probability)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # Also mask padding tokens
        padding_mask = attention_mask.eq(0)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
        
        # Sample tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Set labels to ignore index (-100) for non-masked tokens
        labels[~masked_indices] = TOKEN_IGNORE_INDEX
        
        # 80% of the time, replace masked tokens with [MASK]
        indices_to_mask = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_to_mask] = self.tokenizer.mask_token_id
        
        # 10% of the time, keep the original token
        # This is already handled by not changing these tokens
        
        # 10% of the time, replace with random token
        indices_to_replace = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_to_mask
        random_tokens = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_to_replace] = random_tokens[indices_to_replace]
        
        return inputs, labels
    
    def _get_special_tokens_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Create a mask for special tokens.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            
        Returns:
            Boolean mask where True indicates a special token
        """
        # Get special token IDs from the tokenizer
        special_tokens = self.tokenizer.all_special_ids
        
        # Create mask where True indicates a special token
        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in special_tokens:
            if token_id is not None:
                special_tokens_mask |= (input_ids == token_id)
                
        # Also mask task tokens and BOS tokens (first two tokens in each sequence)
        if input_ids.dim() > 1:  # For batched inputs
            special_tokens_mask[:, 0] = True  # Task token
            special_tokens_mask[:, 1] = True  # BOS token
        else:  # For single sequence
            special_tokens_mask[0] = True  # Task token
            special_tokens_mask[1] = True  # BOS token
                
        return special_tokens_mask


# Alias for backward compatibility
DataCollatorWithTaskTokens = SequenceDataCollator