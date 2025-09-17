#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script demonstrating the usage of the SMILESTokenizer.
This shows how to tokenize, encode, and decode SMILES strings.
"""

import os
import torch
from hyformer.tokenizers.smiles import SMILESTokenizer

# Example SMILES strings
examples = [
    "CC(=O)NC1=CC=C(O)C=C1",  # Acetaminophen
    "CC1=C(C)C(C)=CC=C1O",     # Thymol
    "C1=CC=C2C(=C1)C=CC=C2",   # Naphthalene
    "CC1=CC=CC=C1C(=O)O",      # 2-Methylbenzoic acid
    "C1=CC=C(C=C1)C=O"         # Benzaldehyde
]

def main():
    # Path to vocabulary file
    vocab_path = "data/vocabularies/vocab.txt"
    
    # Check if vocabulary file exists
    if not os.path.exists(vocab_path):
        print(f"Vocabulary file not found at {vocab_path}")
        print("Please ensure you have the correct path to the vocabulary file.")
        return
    
    # Initialize the tokenizer
    tokenizer = SMILESTokenizer.from_config(vocab_path)
    print(f"Loaded tokenizer with vocabulary size: {len(tokenizer)}")
    
    # Print special token IDs
    print("\nSpecial token IDs:")
    print(f"PAD token ID: {tokenizer.pad_token_id}")
    print(f"BOS token ID: {tokenizer.bos_token_id}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    print(f"UNK token ID: {tokenizer.unk_token_id}")
    print(f"MASK token ID: {tokenizer.mask_token_id}")
    
    # Initialize tokenizer with custom special tokens
    custom_tokenizer = SMILESTokenizer(
        vocabulary_path=vocab_path,
        bos_token="<start>",
        eos_token="<end>",
        task_tokens={"property": "<PROPERTY>"}  # Override property task token
    )
    
    print("\nCustom tokenizer special token IDs:")
    print(f"BOS token: {custom_tokenizer.bos_token}")
    print(f"EOS token: {custom_tokenizer.eos_token}")
    print(f"Property task token: {custom_tokenizer.get_task_token('property')}")
    print(f"Property task token ID: {custom_tokenizer.task_token_id('property')}")
    print(f"LM task token: {custom_tokenizer.get_task_token('lm')}")
    print(f"MLM task token: {custom_tokenizer.get_task_token('mlm')}")
    
    # Demonstrate HuggingFace-style properties
    print("\nDemonstrating HuggingFace-style initialization and properties:")
    hf_tokenizer = SMILESTokenizer(
        vocabulary_path=vocab_path,
        bos_token="<START>",
        eos_token="<END>",
        sep_token="<SEP>",
        cls_token="<CLS>"
    )
    
    # Verify tokens and IDs
    print(f"BOS token: {hf_tokenizer.bos_token}, ID: {hf_tokenizer.bos_token_id}")
    print(f"EOS token: {hf_tokenizer.eos_token}, ID: {hf_tokenizer.eos_token_id}")
    print(f"SEP token: {hf_tokenizer.sep_token}, ID: {hf_tokenizer.sep_token_id}")
    print(f"CLS token: {hf_tokenizer.cls_token}, ID: {hf_tokenizer.cls_token_id}")
    print(f"All special tokens: {hf_tokenizer.all_special_tokens}")
    print(f"All special token IDs: {hf_tokenizer.all_special_ids}")
    
    # Encode with the updated tokens
    encoded_with_new_tokens = hf_tokenizer(examples[0])
    decoded_with_new_tokens = hf_tokenizer.decode(encoded_with_new_tokens["input_ids"][0])
    print(f"Encoded and decoded with new tokens: {decoded_with_new_tokens}")
    
    # Tokenize a single SMILES string
    smiles = examples[0]
    print(f"\nTokenizing single SMILES: {smiles}")
    tokens = tokenizer.tokenize(smiles)
    print(f"Tokens: {tokens}")
    
    # Convert tokens to IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"Token IDs: {token_ids}")
    
    # Encode a single SMILES string with the __call__ method
    print("\nEncoding single SMILES:")
    encoded = tokenizer(smiles, task="lm")
    print(f"Input IDs shape: {encoded['input_ids'].shape}")
    print(f"Attention mask shape: {encoded['attention_mask'].shape}")
    print(f"First few input IDs: {encoded['input_ids'][0][:10].tolist()}")
    
    # Decode back to SMILES
    decoded = tokenizer.decode(encoded["input_ids"][0])
    print(f"Decoded SMILES: {decoded}")
    
    # Encode a batch of SMILES strings
    print("\nEncoding batch of SMILES:")
    batch_encoded = tokenizer(examples, task="prediction")
    print(f"Batch input IDs shape: {batch_encoded['input_ids'].shape}")
    print(f"Batch attention mask shape: {batch_encoded['attention_mask'].shape}")
    
    # Decode one example from the batch
    batch_decoded = tokenizer.decode(batch_encoded["input_ids"][2])
    print(f"Decoded example from batch: {batch_decoded}")
    print(f"Original example: {examples[2]}")
    
    # Demonstrate task-specific tokenization
    print("\nDemonstrating task-specific tokenization:")
    for task in ["lm", "mlm", "prediction"]:
        task_encoded = tokenizer(smiles, task=task)
        print(f"Task: {task}, First token ID: {task_encoded['input_ids'][0][0].item()}")
    
if __name__ == "__main__":
    main() 