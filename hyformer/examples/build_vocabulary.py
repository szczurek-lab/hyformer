#!/usr/bin/env python
"""
Script to build a vocabulary file from a dataset.

This script extracts tokens from a dataset and adds them to a vocabulary file.
It can be used to create a new vocabulary file or update an existing one.
"""

import os
import argparse
from hyformer.configs.dataset import DatasetConfig
from hyformer.configs.tokenizer import TokenizerConfig
from hyformer.utils.tokenizers.auto import AutoTokenizer
from hyformer.utils.tokenizers.utils import build_vocabulary_from_dataset
from hyformer.utils.tokenizers.smiles import SMILES_REGEX_PATTERN


def main():
    parser = argparse.ArgumentParser(description="Build vocabulary from dataset")
    
    parser.add_argument(
        "--path_to_dataset_config", 
        type=str, 
        required=True,
        help="Path to the dataset config file"
    )
    parser.add_argument(
        "--path_to_tokenizer_config", 
        type=str, 
        default=None,
        help="Path to the tokenizer config file (optional)"
    )
    parser.add_argument(
        "--vocab_file", 
        type=str, 
        required=True,
        help="Path to the vocabulary file (will be created if it doesn't exist)"
    )
    parser.add_argument(
        "--regex_pattern", 
        type=str, 
        default=None,
        help="Regex pattern for tokenization (overrides the one from tokenizer config)"
    )
    parser.add_argument(
        "--splits", 
        type=str, 
        nargs="+", 
        default=["train", "val", "test"],
        help="Dataset splits to process"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=None,
        help="Root directory for dataset files"
    )
    parser.add_argument(
        "--min_frequency", 
        type=int, 
        default=1,
        help="Minimum frequency for a token to be included"
    )
    parser.add_argument(
        "--no_create", 
        action="store_true",
        help="Don't create the vocabulary file if it doesn't exist"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Load dataset config
    dataset_config = DatasetConfig.from_config_file(args.path_to_dataset_config)
    
    # Load tokenizer config if provided
    tokenizer_config = None
    tokenizer = None
    if args.path_to_tokenizer_config:
        tokenizer_config = TokenizerConfig.from_config_file(args.path_to_tokenizer_config)
        
        # Initialize the tokenizer to validate the config
        try:
            tokenizer = AutoTokenizer.from_config(tokenizer_config)
            if not args.quiet:
                print(f"Successfully initialized tokenizer: {tokenizer_config.tokenizer_type}")
                print(f"Vocabulary size: {len(tokenizer)}")
        except Exception as e:
            print(f"Warning: Could not initialize tokenizer: {e}")
            print("Proceeding with config-only mode (limited validation)")
        
        # Update vocabulary path if not explicitly provided
        if tokenizer_config.vocabulary_path != args.vocab_file and not args.quiet:
            print(f"Note: Using provided vocab_file: {args.vocab_file}")
            print(f"      (different from tokenizer_config.vocabulary_path: {tokenizer_config.vocabulary_path})")
    
    # Build vocabulary
    success = build_vocabulary_from_dataset(
        dataset_config=dataset_config,
        vocab_file=args.vocab_file,
        tokenizer_config=tokenizer_config,
        regex_pattern=args.regex_pattern,
        splits=args.splits,
        root=args.data_dir,
        min_frequency=args.min_frequency,
        create_if_missing=not args.no_create,
        verbose=not args.quiet
    )
    
    if success:
        print(f"Successfully built vocabulary: {args.vocab_file}")
        
        # Try to initialize tokenizer with the new vocabulary
        if tokenizer_config is not None and not args.quiet:
            try:
                # Create a copy of the config with the updated vocabulary path
                updated_config = TokenizerConfig.from_dict(tokenizer_config.to_dict())
                updated_config.vocabulary_path = args.vocab_file
                
                updated_tokenizer = AutoTokenizer.from_config(updated_config)
                print(f"Successfully initialized tokenizer with new vocabulary")
                print(f"New vocabulary size: {len(updated_tokenizer)}")
            except Exception as e:
                print(f"Warning: Could not initialize tokenizer with new vocabulary: {e}")
    else:
        print("Failed to build vocabulary")
        exit(1)


if __name__ == "__main__":
    main() 