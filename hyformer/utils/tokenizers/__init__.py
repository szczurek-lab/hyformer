"""
Tokenizer utilities for processing sequence data into model-compatible formats.

This package provides tokenizers for converting text-like sequences
(e.g., SMILES strings) into token IDs for use with various models.

Classes
-------
BaseTokenizer
    Abstract base class defining the interface for all tokenizers
AutoTokenizer
    Factory class for creating tokenizers from configuration
SMILESTokenizer
    Tokenizer for SMILES molecular representations
HFTokenizer
    Adapter for using Hugging Face tokenizers with the BaseTokenizer interface
"""
