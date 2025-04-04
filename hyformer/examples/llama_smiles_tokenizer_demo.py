#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo script showing how to use the LlamaSMILESTokenizer with a custom vocabulary.
This script demonstrates how to tokenize SMILES strings using the regex pattern
and a custom vocabulary file.
"""

import os
import torch
from hyformer.utils.tokenizers import LlamaSMILESTokenizer, TASK_TOKEN_DICT

def main():
    # Example SMILES strings
    smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "C1=CC=C2C(=C1)C=CN2C(=O)C", # Melatonin
        "CCN(CC)C(=O)C1CN(C2CC3=CNC4=CC=CC(=C34)C2=C1)C" # LSD
    ]
    
    # Path to vocabulary file
    vocab_path = os.path.join("data", "vocabularies", "vocab.txt")
    
    # Initialize the LlamaSMILESTokenizer with a custom vocabulary
    tokenizer = LlamaSMILESTokenizer(
        tokenizer_path=vocab_path,
        task_tokens=TASK_TOKEN_DICT
    )
    
    print(f"LlamaSMILESTokenizer initialized successfully with vocabulary from {vocab_path}!")
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    
    # Tokenize SMILES strings using regex pattern
    print("\n=== Tokenizing SMILES strings using regex pattern ===")
    for i, smi in enumerate(smiles):
        tokens = tokenizer.tokenize(smi)  # is_smiles=True by default now
        print(f"SMILES {i+1}: {smi}")
        print(f"Tokens: {tokens}")
        print(f"Number of tokens: {len(tokens)}")
        print()
    
    # Encode SMILES strings for model input with task token
    print("\n=== Encoding SMILES strings with task token ===")
    encoding = tokenizer(
        smiles,
        task="lm",  # Language modeling task
        padding=True,
        return_tensors="pt"
    )
    
    # Show shape of encoded tensors
    print(f"Input IDs shape: {encoding['input_ids'].shape}")
    print(f"Attention mask shape: {encoding['attention_mask'].shape}")
    
    # Show the actual token IDs for the first example
    print(f"\nInput IDs for first SMILES: {encoding['input_ids'][0]}")
    print(f"Tokens: {[tokenizer.convert_ids_to_tokens(id.item()) for id in encoding['input_ids'][0]]}")
    
    # Decode the encoded tokens back to SMILES
    print("\n=== Decoding SMILES tokens ===")
    decoded = tokenizer.decode(encoding["input_ids"][0])
    print(f"Original: {smiles[0]}")
    print(f"Decoded: {decoded}")
    
    # Example using the tokenizer for text (if LlamaTokenizer was provided)
    print("\n=== Example with text tokenization ===")
    text = "The chemical formula for aspirin is CC(=O)OC1=CC=CC=C1C(=O)O"
    
    # Basic whitespace tokenization for text (unless LlamaTokenizer is available)
    text_tokens = tokenizer.tokenize(text, is_smiles=False)
    print(f"Text tokens: {text_tokens[:10]}...")
    print(f"Number of text tokens: {len(text_tokens)}")
    
    # Extract SMILES part for specialized tokenization
    smiles_part = "CC(=O)OC1=CC=CC=C1C(=O)O"
    smiles_tokens = tokenizer.tokenize(smiles_part)
    print(f"SMILES tokens: {smiles_tokens}")
    print(f"Number of SMILES tokens: {len(smiles_tokens)}")
    
    print("\nNotice how the specialized SMILES tokenization is more chemically informed!")
    
    # Show some special tokens from the vocabulary
    print("\n=== Special tokens in vocabulary ===")
    for name, token in tokenizer.special_tokens.items():
        token_id = tokenizer.vocab.get(token)
        print(f"{name}: '{token}' (ID: {token_id})")

if __name__ == "__main__":
    main() 