from deepchem.feat.smiles_tokenizer import SmilesTokenizer as DeepChemSmilesTokenizer
from typing import List, Optional, Union

from hyformer.utils.tokenizers.base import BaseTokenizer, TOKEN_DICT

from transformers import AutoTokenizer


class HFTokenizer(BaseTokenizer):

    def __init__(
        self,
        vocabulary_path: str,
        max_molecule_length: int,
        mlm_probability: Optional[float] = 0.15,
        ignore_index: Optional[int] = TOKEN_DICT['ignore']
    ):

        super().__init__(
            vocabulary_path=vocabulary_path,
            max_molecule_length=max_molecule_length,
            mlm_probability=mlm_probability,
            ignore_index=ignore_index
        )

    @property
    def cls_token_id(self):
        return self.tokenizer.cls_token_id
    
    @property
    def mask_token_id(self):
        return self.tokenizer.mask_token_id
    
    @property
    def cls_token_id(self):
        return self.tokenizer.cls_token_id
    
    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id
    
    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id
    
    @property
    def sep_token_id(self):
        return self.tokenizer.sep_token_id

    def _set_generation_prefix(self):
        self.generation_prefix = self.tokenizer.cls_token_id
    
    def _init_tokenizer(self, vocabulary_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(vocabulary_path)

    def __len__(self):
        return len(self.tokenizer)

    def _tokenize(self, data: Union[str, List[str]], task: str, padding: bool = False):
        """Tokenize the input data.
        
        Args:
            data: Input data to tokenize
            task: Task to perform
            padding: Whether to pad the sequences (default: False)
            
        Returns:
            Tokenized data
        """
        return self.tokenizer(
            data,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            padding=padding,  # Padding is now handled by the collator
            max_length=self.max_molecule_length,
            return_special_tokens_mask=True
        )
        
    @classmethod
    def from_config(cls, config):
        return cls(
            vocabulary_path=config.vocabulary_path,
            max_molecule_length=config.max_molecule_length
        )