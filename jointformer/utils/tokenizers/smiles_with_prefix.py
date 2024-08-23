import torch

from typing import List, Optional, Tuple, Union

from jointformer.models.utils import ModelInput
from jointformer.utils.tokenizers.base import TOKEN_DICT
from jointformer.utils.tokenizers.smiles import SmilesTokenizer

PREFIX_TOKEN_DICT = {'prefix': '[PREFIX]'}


class SmilesTokenizerWithPrefix(SmilesTokenizer):

    def __init__(
        self,
        path_to_vocabulary: str,
        max_molecule_length: int,
        mlm_probability: Optional[float] = 0.15,
        ignore_index: Optional[int] = TOKEN_DICT['ignore']
    ) -> None:
        """ Smiles tokenizer with prefix token.
        
        Returns tokenized SMILES with a prefix token at the beginning of the sequence.

        Example:
        >>> from jointformer.utils.tokenizers import SmilesTokenizerWithPrefix
        >>> tokenizer = SmilesTokenizerWithPrefix(path_to_vocabulary='path/to/vocab.txt', max_molecule_length=128)
        >>> tokenizer('CCO')

        Args:
            path_to_vocabulary (str): Path to the vocabulary file.
            max_molecule_length (int): Maximum length of the SMILES sequence.
            mlm_probability (float, optional): Probability of masking tokens for MLM. Defaults to 0.15.
            ignore_index (int, optional): Index of the ignore token. Defaults to TOKEN_DICT['ignore'].
        """

        super().__init__(
            path_to_vocabulary=path_to_vocabulary,
            max_molecule_length=max_molecule_length,
            mlm_probability=mlm_probability,
            ignore_index=ignore_index
        )
    
    def _set_generation_prefix(self):
        self.generation_prefix = [self.prefix_token_id, self.tokenizer.cls_token_id]

    def _init_tokenizer(self, path_to_vocabulary: str):
        super()._init_tokenizer(path_to_vocabulary)
        self.tokenizer.add_special_tokens({'additional_special_tokens': [PREFIX_TOKEN_DICT['prefix']]})

    def _tokenize(self, data: Union[str, List[str]]):
        prefix = [' ' for _ in range(len(data))]
        inputs = self.tokenizer(
            text=prefix, text_pair=data, truncation=True, padding='max_length', max_length=self.max_molecule_length,
            return_special_tokens_mask=True, return_token_type_ids=False, return_tensors='pt')
        # Add prefix token to the beginning of the encoded tokens
        inputs['input_ids'][:, 0] = self.prefix_token_id
        # Add cls token to the beginning of the sequence
        inputs['input_ids'][:, 1] = self.tokenizer.cls_token_id
        # The rest of the tokens stay the same
        inputs['input_ids'] = inputs['input_ids'].contiguous()
        return inputs
    
    def __call__(self, x: Union[str, List[str], Tuple[str, torch.Tensor], List[Tuple[str, torch.Tensor]]], task: str) -> ModelInput:
        inputs = super().__call__(x, task)
        if task == 'generation':
            labels = inputs["input_labels"][:, 1:].clone()
            labels[labels == self.tokenizer.convert_tokens_to_ids(PREFIX_TOKEN_DICT['prefix'])] = self.ignore_index
            inputs["input_labels"] = labels
        return inputs
    
    @property
    def prefix_token_id(self):
        return self.tokenizer.convert_tokens_to_ids(PREFIX_TOKEN_DICT['prefix'])
    
    @classmethod
    def from_config(cls, config):
        return cls(
            path_to_vocabulary=config.path_to_vocabulary,
            max_molecule_length=config.max_molecule_length
        )
