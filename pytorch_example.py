#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch integration example for RegexSMILESTokenizer.

This script demonstrates how to use the RegexSMILESTokenizer with PyTorch
for creating datasets and dataloaders for SMILES data.

To run:
    conda activate jointformer
    python pytorch_example.py
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Union, Tuple

# Import the tokenizer
from hyformer.utils.tokenizers.regex import RegexSMILESTokenizer


class SMILESDataset(Dataset):
    """
    PyTorch Dataset for SMILES strings.
    
    This dataset handles SMILES strings and converts them to token IDs
    using the RegexSMILESTokenizer.
    """
    
    def __init__(
        self,
        data: List[str],
        tokenizer: RegexSMILESTokenizer,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True
    ):
        """
        Initialize the dataset.
        
        Parameters
        ----------
        data : List[str]
            List of SMILES strings
        tokenizer : RegexSMILESTokenizer
            Tokenizer to use for encoding SMILES strings
        max_length : int, optional
            Maximum sequence length (will truncate longer sequences)
        add_special_tokens : bool
            Whether to add special tokens (CLS and SEP)
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Parameters
        ----------
        idx : int
            Index of the sample
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with input_ids and attention_mask
        """
        smiles = self.data[idx]
        
        # Tokenize and convert to IDs
        tokens = self.tokenizer.tokenize(smiles)
        
        # Add special tokens if requested
        if self.add_special_tokens:
            tokens = self.tokenizer.add_special_tokens_single_sequence(tokens)
        
        # Convert to IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Truncate if necessary
        if self.max_length is not None and len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)
        
        # Convert to tensors
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }


class SMILESCollator:
    """
    Collator for batching SMILES data.
    
    This collator handles padding of sequences to the same length within a batch.
    """
    
    def __init__(
        self,
        tokenizer: RegexSMILESTokenizer,
        pad_to_multiple_of: Optional[int] = None
    ):
        """
        Initialize the collator.
        
        Parameters
        ----------
        tokenizer : RegexSMILESTokenizer
            Tokenizer to use for padding
        pad_to_multiple_of : int, optional
            If set, pad the sequence to a multiple of this value
        """
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        
        Parameters
        ----------
        batch : List[Dict[str, torch.Tensor]]
            List of samples from the dataset
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Batched samples with padding
        """
        # Get the maximum length in the batch
        max_length = max(len(x["input_ids"]) for x in batch)
        
        # Adjust to pad_to_multiple_of if needed
        if self.pad_to_multiple_of is not None:
            max_length = (
                (max_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )
        
        # Initialize tensors for the batch
        input_ids = []
        attention_mask = []
        
        # Pad each sample to max_length
        for sample in batch:
            # Get the current length
            curr_len = len(sample["input_ids"])
            
            # Calculate padding length
            pad_len = max_length - curr_len
            
            # Pad input_ids with pad_token_id
            padded_input_ids = torch.cat([
                sample["input_ids"],
                torch.full((pad_len,), self.tokenizer.vocab[self.tokenizer.pad_token], dtype=torch.long)
            ])
            
            # Pad attention_mask with 0s
            padded_attention_mask = torch.cat([
                sample["attention_mask"],
                torch.zeros(pad_len, dtype=torch.long)
            ])
            
            # Add to batch
            input_ids.append(padded_input_ids)
            attention_mask.append(padded_attention_mask)
        
        # Stack tensors
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask)
        }


def create_smiles_dataloader(
    smiles_list: List[str],
    tokenizer: RegexSMILESTokenizer,
    batch_size: int = 32,
    max_length: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    pad_to_multiple_of: Optional[int] = 8
) -> DataLoader:
    """
    Create a DataLoader for SMILES data.
    
    Parameters
    ----------
    smiles_list : List[str]
        List of SMILES strings
    tokenizer : RegexSMILESTokenizer
        Tokenizer to use for encoding
    batch_size : int
        Batch size
    max_length : int, optional
        Maximum sequence length
    shuffle : bool
        Whether to shuffle the data
    num_workers : int
        Number of workers for DataLoader
    pad_to_multiple_of : int, optional
        If set, pad the sequence to a multiple of this value
        
    Returns
    -------
    DataLoader
        DataLoader for SMILES data
    """
    # Create dataset
    dataset = SMILESDataset(
        data=smiles_list,
        tokenizer=tokenizer,
        max_length=max_length,
        add_special_tokens=True
    )
    
    # Create collator
    collator = SMILESCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=pad_to_multiple_of
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator
    )
    
    return dataloader


class SMILESModel(torch.nn.Module):
    """
    Simple PyTorch model for SMILES data.
    
    This is a basic transformer encoder model for processing SMILES data.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1
    ):
        """
        Initialize the model.
        
        Parameters
        ----------
        vocab_size : int
            Size of the vocabulary
        hidden_size : int
            Size of the hidden layers
        num_hidden_layers : int
            Number of hidden layers
        num_attention_heads : int
            Number of attention heads
        intermediate_size : int
            Size of the intermediate (feed-forward) layer
        hidden_dropout_prob : float
            Dropout probability for hidden layers
        attention_probs_dropout_prob : float
            Dropout probability for attention probabilities
        """
        super().__init__()
        
        # Embedding layer
        self.embeddings = torch.nn.Embedding(vocab_size, hidden_size)
        
        # Position embeddings
        self.position_embeddings = torch.nn.Embedding(512, hidden_size)
        
        # Layer normalization and dropout
        self.layer_norm = torch.nn.LayerNorm(hidden_size)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        
        # Transformer encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            dropout=hidden_dropout_prob,
            activation="gelu",
            batch_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_hidden_layers
        )
        
        # Output layer
        self.output = torch.nn.Linear(hidden_size, vocab_size)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        input_ids : torch.Tensor
            Tensor of token IDs
        attention_mask : torch.Tensor
            Tensor of attention mask
            
        Returns
        -------
        torch.Tensor
            Output logits
        """
        # Get sequence length
        seq_length = input_ids.size(1)
        
        # Create position IDs
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Get embeddings
        inputs_embeds = self.embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        
        # Add embeddings
        hidden_states = inputs_embeds + position_embeds
        
        # Apply layer norm and dropout
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Create attention mask for transformer
        # Convert from [0, 1] to [True, False] where True means "don't attend"
        transformer_attention_mask = (1.0 - attention_mask.float()) * -10000.0
        
        # Apply transformer encoder
        hidden_states = self.encoder(
            src=hidden_states,
            src_key_padding_mask=(attention_mask == 0)
        )
        
        # Apply output layer
        logits = self.output(hidden_states)
        
        return logits


def initialize_model_and_optimizer(
    tokenizer: RegexSMILESTokenizer,
    hidden_size: int = 768,
    learning_rate: float = 1e-4
) -> Tuple[SMILESModel, torch.optim.Optimizer]:
    """
    Initialize model and optimizer.
    
    Parameters
    ----------
    tokenizer : RegexSMILESTokenizer
        Tokenizer to get vocabulary size
    hidden_size : int
        Size of the hidden layers
    learning_rate : float
        Learning rate for the optimizer
        
    Returns
    -------
    Tuple[SMILESModel, torch.optim.Optimizer]
        Initialized model and optimizer
    """
    # Initialize model
    model = SMILESModel(
        vocab_size=len(tokenizer.vocab),
        hidden_size=hidden_size,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=hidden_size * 4,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    return model, optimizer


if __name__ == "__main__":
    # This script is meant to be imported, not run directly
    pass 