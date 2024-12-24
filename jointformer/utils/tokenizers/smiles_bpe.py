from transformers import T5Tokenizer

from jointformer.utils.tokenizers.base import BaseTokenizer, TOKEN_DICT
from jointformer.utils.tokenizers.smiles_separate_task_token import SmilesTokenizerSeparateTaskToken, TASK_TOKEN_DICT

class SmilesBPETokenizer(SmilesTokenizerSeparateTaskToken):

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
        