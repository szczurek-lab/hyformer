import torch
from typing import List, Optional, Union, Dict, Any, Tuple

from hyformer.utils.tokenizers.base import BaseTokenizer, TASK_TOKEN_DICT
from transformers import PreTrainedTokenizerFast
import transformers
transformers.logging.set_verbosity_error()

class SMILESTokenizer(BaseTokenizer):
    """
    Tokenizer for SMILES strings based on the Hugging Face PreTrainedTokenizerFast.
    
    This tokenizer handles SMILES strings and prepends task tokens to the tokenized sequences.
    """
    
    def __init__(
        self,
        tokenizer_path: str,
        task_tokens: Optional[Dict[str, str]] = None,
        max_length: int = 512,
        **kwargs
    ):
        """
        Initialize the SMILES tokenizer.
        
        Args:
            tokenizer_path: Path to the pretrained tokenizer
            task_tokens: Dictionary mapping task names to task token strings
            max_length: Maximum sequence length
            **kwargs: Additional arguments to pass to the tokenizer
        """
        # Initialize the base tokenizer first
        super().__init__(tokenizer_path, task_tokens, max_length, **kwargs)
        
        # Initialize the actual tokenizer
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            unk_token="<unk>",
            mask_token="<mask>",
            **kwargs
        )
        
        # Add task tokens to the tokenizer vocabulary
        self.tokenizer.add_tokens(list(self.task_tokens.values()))
    
    def _tokenize_text(
        self, 
        texts: Union[str, List[str]],
        max_length: int,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Convert SMILES text to token IDs without adding task tokens.
        
        Args:
            texts: Input SMILES text or list of SMILES texts to tokenize
            max_length: Maximum length for tokenization
            **kwargs: Additional arguments to pass to the tokenizer
            
        Returns:
            Dictionary with tokenized outputs
        """
        # Tokenize the input without padding (padding will be handled by the collator)
        return self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            **kwargs
        )
    
    def decode(self, x: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        """Decode token IDs to SMILES strings.
        
        Args:
            x: Tensor of token IDs
            skip_special_tokens: Whether to skip special tokens in the output
            
        Returns:
            List of decoded SMILES strings
        """
        smiles_data = []
        for row in x:
            smiles = self.tokenizer.decode(row.tolist(), skip_special_tokens=skip_special_tokens).replace(' ', '')
            smiles_data.append(smiles)
        return smiles_data
    
    @classmethod
    def from_config(cls, config: Any) -> 'SMILESTokenizer':
        """Create a tokenizer from a configuration object.
        
        Args:
            config: Configuration object with tokenizer parameters
            
        Returns:
            Initialized SMILESTokenizer
        """
        # Extract parameters from config with defaults
        tokenizer_path = getattr(config, 'tokenizer_path', None)
        if tokenizer_path is None:
            # For backward compatibility
            tokenizer_path = getattr(config, 'path_to_vocabulary', None)
            
        if tokenizer_path is None:
            raise ValueError("No tokenizer_path or path_to_vocabulary found in config")
            
        # Get task tokens if provided, otherwise use default
        task_tokens = getattr(config, 'task_tokens', None)
        max_length = getattr(config, 'max_length', 512)
        
        return cls(
            tokenizer_path=tokenizer_path,
            task_tokens=task_tokens,
            max_length=max_length
        )
