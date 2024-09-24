from deepchem.feat.smiles_tokenizer import SmilesTokenizer as DeepChemSmilesTokenizer
from typing import List, Optional, Union

from jointformer.utils.tokenizers.base import BaseTokenizer, TOKEN_DICT


class FancyTokenizer(BaseTokenizer):

    def __init__(
        self,
        path_to_vocabulary: str,
        max_molecule_length: int,
        mlm_probability: Optional[float] = 0.15,
        ignore_index: Optional[int] = TOKEN_DICT['ignore']
    ):

        super().__init__(
            path_to_vocabulary=path_to_vocabulary,
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

    def _init_tokenizer(self, path_to_vocabulary: str):
        self.tokenizer = DeepChemSmilesTokenizer(
            vocab_file=path_to_vocabulary,
            cls_token=TOKEN_DICT['cls'],
            sep_token=TOKEN_DICT['sep'],
            mask_token=TOKEN_DICT['mask'],
            pad_token=TOKEN_DICT['pad'],
            unk_token=TOKEN_DICT['unknown'])

    def __len__(self):
        return len(self.tokenizer)

    def _tokenize(self, data: Union[str, List[str]]):
        return self.tokenizer(
            data, truncation=True, padding='max_length', max_length=self.max_molecule_length,
            return_special_tokens_mask=True, return_token_type_ids=False, return_tensors='pt')
        
    @classmethod
    def from_config(cls, config):
        return cls(
            path_to_vocabulary=config.path_to_vocabulary,
            max_molecule_length=config.max_molecule_length
        )
