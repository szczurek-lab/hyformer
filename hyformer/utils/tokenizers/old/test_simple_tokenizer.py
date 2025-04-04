import os
import sys
from typing import List

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from hyformer.utils.tokenizers.simple_regex_tokenizer import RegexSMILESTokenizer

def test_regex_smiles_tokenizer():
    """Test the RegexSMILESTokenizer with the DeepChem vocabulary."""
    # Path to DeepChem vocabulary
    vocab_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))), "data", "vocabularies", "deepchem.txt")
    
    # Check if vocabulary file exists
    if not os.path.exists(vocab_path):
        print(f"Warning: DeepChem vocabulary file not found at {vocab_path}")
        print("Using default vocabulary instead.")
        vocab_path = None
    
    # Create tokenizer
    tokenizer = RegexSMILESTokenizer(vocab_file=vocab_path)
    
    # Test with aspirin
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    
    # Tokenize
    tokens = tokenizer.tokenize(smiles)
    print("Basic tokenization:")
    print(f"SMILES: {smiles}")
    print(f"Tokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}")
    
    # Encode (with special tokens)
    ids = tokenizer.encode(smiles, add_special_tokens=True)
    print("\nEncoding with special tokens:")
    print(f"Token IDs: {ids}")
    print(f"Number of IDs: {len(ids)}")
    
    # Decode (skipping special tokens)
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    print("\nDecoding (skipping special tokens):")
    print(f"Decoded: {decoded}")
    
    # Verify
    assert decoded == smiles, "Tokenization is not reversible!"
    print("Tokenization is reversible!")
    
    # Test with a more complex SMILES
    complex_smiles = "C[C@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)CCCC[NH3+])C(=O)[O-]"  # A peptide
    
    # Tokenize
    complex_tokens = tokenizer.tokenize(complex_smiles)
    print("\nComplex SMILES tokenization:")
    print(f"SMILES: {complex_smiles}")
    print(f"Tokens: {complex_tokens}")
    print(f"Number of tokens: {len(complex_tokens)}")
    
    # Encode and decode
    complex_ids = tokenizer.encode(complex_smiles)
    complex_decoded = tokenizer.decode(complex_ids)
    print(f"Decoded: {complex_decoded}")
    
    # Verify
    assert complex_decoded == complex_smiles, "Complex tokenization is not reversible!"
    print("Complex tokenization is reversible!")
    
    # Test with a reaction SMILES
    reaction = "CC(=O)O.CN>>CC(=O)NC"  # Acetylation reaction
    
    # Tokenize
    reaction_tokens = tokenizer.tokenize(reaction)
    print("\nReaction SMILES tokenization:")
    print(f"SMILES: {reaction}")
    print(f"Tokens: {reaction_tokens}")
    print(f"Number of tokens: {len(reaction_tokens)}")
    
    # Encode and decode
    reaction_ids = tokenizer.encode(reaction)
    reaction_decoded = tokenizer.decode(reaction_ids)
    print(f"Decoded: {reaction_decoded}")
    
    # Verify
    assert reaction_decoded == reaction, "Reaction tokenization is not reversible!"
    print("Reaction tokenization is reversible!")
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_regex_smiles_tokenizer() 