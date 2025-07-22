from typing import List, Optional, Union
from transformers import T5Tokenizer

from hyformer.utils.tokenizers.base import BaseTokenizer, TOKEN_DICT
from hyformer.utils.tokenizers.smiles_separate_task_token import TASK_TOKEN_DICT


class SmilesBPETokenizer(BaseTokenizer):

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
        self.generation_prefix = [self.tokenizer.convert_tokens_to_ids(TASK_TOKEN_DICT['generation']), self.tokenizer.cls_token_id]

    def _init_tokenizer(self, path_to_vocabulary: str):
        self.tokenizer = T5Tokenizer.from_pretrained(
            path_to_vocabulary,
            model_max_length=self.max_molecule_length,
            cls_token=TOKEN_DICT['cls'],
            sep_token=TOKEN_DICT['sep'],
            mask_token=TOKEN_DICT['mask'],
            pad_token=TOKEN_DICT['pad'],
            unk_token=TOKEN_DICT['unknown']
            )
        self.tokenizer.add_special_tokens({'additional_special_tokens': list(TASK_TOKEN_DICT.values())})
        self.tokenizer.eos_token = self.tokenizer.sep_token
        self.tokenizer.eos_token_id = self.tokenizer.sep_token_id
        
    def __len__(self):
        return len(self.tokenizer)

    def _tokenize(self, data: Union[str, List[str]], task: str):
        
        _prefix_token = f"{TASK_TOKEN_DICT[task]}{self.tokenizer.cls_token}"
        if isinstance(data, str):
            data = _prefix_token + " " + data
        else:
            data = [_prefix_token + " " + _data for _data in data]
            
        return self.tokenizer(
            data, truncation=True, padding='max_length', max_length=self.max_molecule_length,
            return_special_tokens_mask=True, return_token_type_ids=False, return_tensors='pt')
    
    def __call__(self, x, task):
        inputs = super().__call__(x, task)
        if task in ['generation', 'reconstruction', 'mlm']:
            labels = inputs["input_labels"][:, 1:].clone()
            inputs["input_labels"] = labels
        return inputs

    @classmethod
    def from_config(cls, config):
        return cls(
            path_to_vocabulary=config.path_to_vocabulary,
            max_molecule_length=config.max_molecule_length
        )
