#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example showing how to use SMILESTokenizer and HFTokenizer with the same interface.

This script demonstrates how the tokenizer implementations adhere to SOLID principles,
particularly the Liskov Substitution Principle, by showing that both tokenizers
can be used interchangeably with the same interface.
"""

import os
import torch
import argparse
from pathlib import Path

from hyformer.configs.tokenizer import TokenizerConfig
from hyformer.utils.tokenizers.auto import AutoTokenizer
from hyformer.utils.tokenizers.smiles import SMILESTokenizer
from hyformer.utils.tokenizers.hf import HFTokenizer


def test_smiles_tokenizer(vocab_path):
    """Test the SMILES tokenizer."""
    print("\n----- Testing SMILESTokenizer -----")
    
    # Create config
    config = TokenizerConfig(
        tokenizer_type="SMILESTokenizer",
        vocabulary_path=vocab_path
    )
    
    # Create tokenizer from config
    tokenizer = AutoTokenizer.from_config(config)
    
    # Example SMILES string
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    print(f"Input SMILES: {smiles}")
    
    # Tokenize (without special tokens)
    tokens = tokenizer.tokenize(smiles)
    print(f"Tokens: {tokens}")
    
    # Convert to token IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"Token IDs: {token_ids}")
    
    # Full tokenization pipeline (with special tokens)
    output = tokenizer(smiles)
    print(f"Input IDs shape: {len(output['input_ids'])}x{len(output['input_ids'][0])}")
    print(f"Attention mask shape: {len(output['attention_mask'])}x{len(output['attention_mask'][0])}")
    
    # Display special tokens handling
    tokens_with_special = tokenizer.tokenize(smiles)
    input_with_special = tokenizer(smiles, add_bos_token=True, add_eos_token=True)
    print(f"First token (BOS): {tokenizer.ids_to_tokens[input_with_special['input_ids'][0][0]]}")
    print(f"Last token (EOS): {tokenizer.ids_to_tokens[input_with_special['input_ids'][0][-1]]}")
    
    # Decode back to string
    decoded = tokenizer.decode(output["input_ids"][0])
    print(f"Decoded: {decoded}")
    

def test_hf_tokenizer(model_name):
    """Test the HF tokenizer."""
    print("\n----- Testing HFTokenizer -----")
    
    # Create config
    config = TokenizerConfig(
        tokenizer_type="HFTokenizer",
        vocabulary_path=model_name,
        kwargs={"use_fast": True}
    )
    
    # Create tokenizer from config
    tokenizer = AutoTokenizer.from_config(config)
    
    # Example protein sequence
    protein = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    print(f"Input protein: {protein}")
    
    # Tokenize (without special tokens)
    tokens = tokenizer.tokenize(protein)
    print(f"Tokens: {tokens[:10]}...")  # Only print first few for brevity
    
    # Convert to token IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"Token IDs: {token_ids[:10]}...")  # Only print first few for brevity
    
    # Full tokenization pipeline (with special tokens)
    output = tokenizer(protein)
    print(f"Input IDs shape: {len(output['input_ids'])}x{len(output['input_ids'][0])}")
    print(f"Attention mask shape: {len(output['attention_mask'])}x{len(output['attention_mask'][0])}")
    
    # Display special tokens handling
    input_with_special = tokenizer(protein, add_bos_token=True, add_eos_token=True)
    print(f"First token (BOS): {tokenizer.ids_to_tokens[input_with_special['input_ids'][0][0]]}")
    print(f"Last token (EOS): {tokenizer.ids_to_tokens[input_with_special['input_ids'][0][-1]]}")
    
    # Decode back to string
    decoded = tokenizer.decode(output["input_ids"][0])
    print(f"Decoded: {decoded}")


def main():
    """Run tokenizer examples."""
    parser = argparse.ArgumentParser(description="Test tokenizers")
    parser.add_argument("--smiles-vocab", default="data/vocabulary/smiles.txt", 
                        help="Path to SMILES vocabulary file")
    parser.add_argument("--hf-model", default="facebook/esm2_t33_650M_UR50D", 
                        help="HF model name for protein tokenization")
    args = parser.parse_args()
    
    # Create vocab directory if it doesn't exist
    vocab_dir = os.path.dirname(args.smiles_vocab)
    os.makedirs(vocab_dir, exist_ok=True)
    
    # Create a simple SMILES vocabulary file if it doesn't exist
    if not os.path.exists(args.smiles_vocab):
        print(f"Creating simple SMILES vocabulary at {args.smiles_vocab}")
        smiles_tokens = [
            "C", "c", "1", "2", "3", "(", ")", "=", "O", "N", "n", "o", "H", 
            "[", "]", "S", "+", "-", "#", "F", "Cl", "Br", "I", "P", "B", "/", "\\", 
            "Si", "Se", "s", "p", ":", "@", "."
        ]
        with open(args.smiles_vocab, 'w') as f:
            for token in smiles_tokens:
                f.write(f"{token}\n")
    
    # Test both tokenizers
    test_smiles_tokenizer(args.smiles_vocab)
    test_hf_tokenizer(args.hf_model)


if __name__ == "__main__":
    main() 