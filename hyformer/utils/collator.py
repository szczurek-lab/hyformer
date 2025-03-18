import torch
from typing import Dict, List, Union, Optional, Any, Tuple
from torch.distributions.categorical import Categorical
from hyformer.models.utils import ModelInput
from dataclasses import dataclass

# Constants for token types
TOKEN_IGNORE_INDEX = -100  # Standard ignore index for loss calculation

@dataclass
class DataCollatorWithPadding:
    """
    Data collator that handles padding and masking according to Hugging Face best practices.
    
    This collator:
    1. Pads sequences to the maximum length in the batch or to `pad_to_multiple_of`
    2. Creates attention masks
    3. Applies dynamic masking for MLM tasks
    4. Handles different task types (lm, prediction, mlm)
    
    Following Hugging Face best practices, masking is handled by the collator rather than
    the tokenizer to enable dynamic masking during training.
    """
    tokenizer: Any
    tasks: Dict[str, float]
    pad_to_multiple_of: Optional[int] = None
    max_length: int = 512  # Maximum sequence length
    return_tensors: str = "pt"
    mask_probability: float = 0.15  # Probability for masking tokens in MLM task
    
    def __post_init__(self):
        """Initialize task distribution for sampling tasks."""
        self._task_dist = Categorical(torch.Tensor(list(self.tasks.values())))
    
    def _sample_task(self) -> str:
        """Sample a task based on the task distribution."""
        return list(self.tasks.keys())[self._task_dist.sample().item()]
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate and pad a batch of examples.
        
        Args:
            batch: List of examples to collate. Each example is a dictionary with 'data' and 'target' keys
                  from SequenceDataset.__getitem__. The 'target' value may be None if no target exists.
            
        Returns:
            Collated and padded batch as a dictionary with appropriate tensors
        """
        # Sample a task for this batch
        task = self._sample_task()
        
        # The tokenizer now expects a list of dictionaries from SequenceDataset
        tokenized_batch = self.tokenizer(batch, task=task)
        
        # Get the input_ids and attention_mask
        input_ids = tokenized_batch["input_ids"]
        attention_mask = tokenized_batch["attention_mask"]
        
        # Calculate max length for padding - use the minimum of max_length and the longest sequence
        batch_max_length = min(self.max_length, max(len(ids) for ids in input_ids))
        
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
        
        # Create the output batch
        output_batch = {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
        }
        
        # Apply masking for MLM task
        if task == 'mlm':
            masked_inputs, labels = self._mask_tokens(
                padded_input_ids.clone(),
                padded_attention_mask
            )
            output_batch["input_ids"] = masked_inputs
            output_batch["input_labels"] = labels
        
        # Add targets if they were included by the tokenizer
        if "targets" in tokenized_batch:
            output_batch["targets"] = tokenized_batch["targets"]
        
        return output_batch
    
    def _pad_sequences(self, sequences: List[torch.Tensor], max_length: int, pad_value: Any) -> torch.Tensor:
        """
        Efficiently pad a list of sequences to the same length using vectorized operations.
        
        Args:
            sequences: List of tensors to pad
            max_length: Maximum length to pad to
            pad_value: Value to use for padding
            
        Returns:
            Padded tensor of shape [batch_size, max_length]
        """
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
            padded_sequences[i, :length] = seq[:length]
            
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
        special_tokens = {
            self.tokenizer.bos_token_id,  # Beginning of sequence
            self.tokenizer.eos_token_id,  # End of sequence
            self.tokenizer.pad_token_id,  # Padding
            self.tokenizer.unk_token_id,  # Unknown
            self.tokenizer.mask_token_id,  # Mask token
        }
        
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
        