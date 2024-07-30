import torch

from deepchem.feat.smiles_tokenizer import SmilesTokenizer as DeepChemSmilesTokenizer
from typing import List, Tuple, Any, Optional, Union

from jointformer.utils.tokenizers.base import BaseTokenizer, TOKEN_DICT, TASK_TOKEN_DICT


class SmilesTokenizer(BaseTokenizer):

    def __init__(
        self,
        path_to_vocabulary: str,
        max_molecule_length: int,
        mlm_probability: Optional[float] = 0.15,
        ignore_index: Optional[int] = TOKEN_DICT['ignore'],
        set_separate_task_tokens: Optional[bool] = False
    ):

        super().__init__(
            path_to_vocabulary=path_to_vocabulary,
            max_molecule_length=max_molecule_length,
            mlm_probability=mlm_probability,
            ignore_index=ignore_index,
            set_separate_task_tokens=set_separate_task_tokens
        )

    def _init_tokenizer(self, path_to_vocabulary: str):
        self.tokenizer = DeepChemSmilesTokenizer(
            vocab_file=path_to_vocabulary,
            cls_token=TOKEN_DICT['cls'],
            sep_token=TOKEN_DICT['sep'],
            mask_token=TOKEN_DICT['mask'],
            pad_token=TOKEN_DICT['pad'],
            unk_token=TOKEN_DICT['unknown'])

    def _init_separate_task_tokens(self):
        special_tokens_dict = {'additional_special_tokens': list(TASK_TOKEN_DICT.values())}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.max_molecule_length = self.max_molecule_length - 1

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
            max_molecule_length=config.max_molecule_length,
            set_separate_task_tokens=config.set_separate_task_tokens
        )
