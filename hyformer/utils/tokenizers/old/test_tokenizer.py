import torch
import os
import sys
from typing import Dict, List, Union

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from hyformer.utils.tokenizers.old.smiles import SMILESTokenizer

def test_smiles_tokenizer():
    """Test the SMILESTokenizer with the new BaseTokenizer implementation."""
    # Create a tokenizer
    tokenizer = SMILESTokenizer(
        tokenizer_path="path/to/tokenizer.json",  # Replace with actual path in your environment
        max_length=512
    )
    
    # Test tokenizing a single SMILES string
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    
    # Mock the tokenizer methods since we don't have an actual tokenizer file
    tokenizer.tokenizer.convert_tokens_to_ids = lambda x: 1  # Mock token ID
    tokenizer.tokenizer.bos_token_id = 2
    tokenizer.tokenizer.unk_token_id = 0
    
    # Mock the _tokenize_text method
    def mock_tokenize_text(texts, max_length, **kwargs):
        if isinstance(texts, str):
            return {
                "input_ids": torch.tensor([3, 4, 5, 6]),
                "attention_mask": torch.tensor([1, 1, 1, 1])
            }
        else:
            batch_size = len(texts)
            return {
                "input_ids": torch.tensor([[3, 4, 5, 6]] * batch_size),
                "attention_mask": torch.tensor([[1, 1, 1, 1]] * batch_size)
            }
    
    tokenizer._tokenize_text = mock_tokenize_text
    
    # Test single string tokenization
    result = tokenizer(smiles, task="lm")
    print("Single string tokenization:")
    print(f"Input IDs: {result['input_ids']}")
    print(f"Attention Mask: {result['attention_mask']}")
    
    # Expected: [task_token_id, bos_token_id, 3, 4, 5, 6]
    assert result["input_ids"][0] == 1, "First token should be task token"
    assert result["input_ids"][1] == 2, "Second token should be BOS token"
    assert len(result["input_ids"]) == 6, "Should have 6 tokens (2 special + 4 content)"
    
    # Test batch tokenization
    batch = ["CC(=O)OC1=CC=CC=C1C(=O)O", "CC(=O)OC1=CC=CC=C1C(=O)O"]
    result = tokenizer(batch, task="prediction")
    print("\nBatch tokenization:")
    print(f"Input IDs shape: {result['input_ids'].shape}")
    print(f"Attention Mask shape: {result['attention_mask'].shape}")
    
    # Expected: batch_size x [task_token_id, bos_token_id, 3, 4, 5, 6]
    assert result["input_ids"].shape == (2, 6), "Should have shape (2, 6)"
    assert result["input_ids"][0, 0] == 1, "First token should be task token"
    assert result["input_ids"][0, 1] == 2, "Second token should be BOS token"
    
    # Test with dictionary input (from SequenceDataset)
    data_dict = {"data": "CC(=O)OC1=CC=CC=C1C(=O)O", "target": torch.tensor([0.5])}
    result = tokenizer(data_dict, task="mlm")
    print("\nDictionary input tokenization:")
    print(f"Input IDs: {result['input_ids']}")
    print(f"Target: {result['targets']}")
    
    # Expected: [task_token_id, bos_token_id, 3, 4, 5, 6] and target tensor
    assert result["input_ids"][0] == 1, "First token should be task token"
    assert result["input_ids"][1] == 2, "Second token should be BOS token"
    assert torch.allclose(result["targets"], torch.tensor([0.5])), "Target should be preserved"
    
    # Test with list of dictionaries (from DataLoader with SequenceDataset)
    batch_dicts = [
        {"data": "CC(=O)OC1=CC=CC=C1C(=O)O", "target": torch.tensor([0.5])},
        {"data": "CC(=O)OC1=CC=CC=C1C(=O)O", "target": torch.tensor([0.7])}
    ]
    result = tokenizer(batch_dicts, task="lm")
    print("\nList of dictionaries tokenization:")
    print(f"Input IDs shape: {result['input_ids'].shape}")
    print(f"Targets shape: {result['targets'].shape}")
    
    # Expected: batch_size x [task_token_id, bos_token_id, 3, 4, 5, 6] and stacked target tensors
    assert result["input_ids"].shape == (2, 6), "Should have shape (2, 6)"
    assert result["targets"].shape == (2, 1), "Targets should be stacked with shape (2, 1)"
    assert torch.allclose(result["targets"][0], torch.tensor([0.5])), "First target should be preserved"
    assert torch.allclose(result["targets"][1], torch.tensor([0.7])), "Second target should be preserved"
    
    # Test with mixed target types
    batch_dicts_mixed = [
        {"data": "CC(=O)OC1=CC=CC=C1C(=O)O", "target": 0.5},  # Not a tensor
        {"data": "CC(=O)OC1=CC=CC=C1C(=O)O", "target": torch.tensor([0.7])}
    ]
    result = tokenizer(batch_dicts_mixed, task="lm")
    print("\nMixed target types tokenization:")
    print(f"Input IDs shape: {result['input_ids'].shape}")
    print(f"Targets shape: {result['targets'].shape}")
    
    # Expected: targets converted to tensors and stacked
    assert result["targets"].shape == (2, 1), "Targets should be stacked with shape (2, 1)"
    assert torch.allclose(result["targets"][0], torch.tensor([0.5])), "First target should be converted to tensor"
    
    # Test with missing targets
    batch_dicts_missing = [
        {"data": "CC(=O)OC1=CC=CC=C1C(=O)O"},  # No target
        {"data": "CC(=O)OC1=CC=CC=C1C(=O)O", "target": torch.tensor([0.7])}
    ]
    result = tokenizer(batch_dicts_missing, task="lm")
    print("\nMissing targets tokenization:")
    print(f"Input IDs shape: {result['input_ids'].shape}")
    if "targets" in result:
        print(f"Targets: {result['targets']}")
        # Only one valid target
        assert len(result["targets"]) == 1, "Should have only one target"
        assert torch.allclose(result["targets"][0], torch.tensor([0.7])), "Valid target should be preserved"
    
    # Test invalid task
    try:
        result = tokenizer(smiles, task="invalid_task")
        assert False, "Should have raised ValueError for invalid task"
    except ValueError as e:
        print(f"\nCorrectly caught invalid task: {e}")
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_smiles_tokenizer() 