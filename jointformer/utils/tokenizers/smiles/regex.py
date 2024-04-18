""" This module defines a PyTorch compatible SMILES Tokenizer based on regular expressions.
"""

import re
import torch
import warnings
from typing import List, Tuple

from jointformer.utils.datasets.utils import read_strings_from_file

REGEX_PATTERN = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
DEFAULT_MAX_LENGTH = 128


class SMILESTokenizer:

    def __init__(
            self,
            path_to_vocabulary: str = None,
            max_length: int = None,
            pad_token: str = '[PAD]',
            bos_token: str = '[BOS]',
            eos_token: str = '[EOS]',
            mask_token: str = '[MASK]',
            unk_token: str = '[UNK]',
            rec_token: str = '[REC]',
            pred_token: str = '[PRED]',
            additional_special_tokens: List[str] = None
    ):
        self.path_to_vocabulary = path_to_vocabulary
        self.max_length = max_length if max_length is not None else DEFAULT_MAX_LENGTH
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.mask_token = mask_token
        self.unk_token = unk_token
        self.rec_token = rec_token
        self.pred_token = pred_token
        self.additional_special_tokens = additional_special_tokens if additional_special_tokens is not None else []

        self.vocabulary = self._load_vocabulary()
        self.regex = re.compile(REGEX_PATTERN)
        self.token_to_index = {token: index for index, token in enumerate(self.vocabulary)}
        self.index_to_token = {index: token for token, index in self.token_to_index.items()}
        self.num_tokens = len(self.vocabulary)

    def __call__(self, smiles: str) -> List[int]:
        return self.tokenize(smiles)

    @property
    def special_tokens(self) -> List[str]:
        return [
            self.pad_token, self.bos_token, self.eos_token, self.mask_token,
            self.unk_token, self.rec_token, self.pred_token] + self.additional_special_tokens

    def add_bos_token(self, tokens: List[int]) -> List[int]:
        return [self.token_to_index[self.bos_token]] + tokens

    def add_eos_token(self, tokens: List[int]) -> List[int]:
        return tokens + [self.token_to_index[self.eos_token]]

    def validate_string(self, smiles: str) -> None:
        if not self.regex.fullmatch(smiles):
            warnings.warn(f"Input string '{smiles}' contains not recognized characters.")
        return None

    def tokenize(self, smiles: str) -> torch.Tensor:
        self.validate_string(smiles)
        smiles = self.regex.findall(smiles)
        tokens = [self.token_to_index.get(token, self.token_to_index[self.unk_token]) for token in smiles]
        if len(tokens) > self.max_length - 2:
            tokens = tokens[:self.max_length - 2]
        tokens = self.add_bos_token(tokens)
        tokens = self.add_eos_token(tokens)
        tokens = self.pad(tokens)
        return torch.tensor(tokens, dtype=torch.long)

    def detokenize(self, tokens: torch.Tensor) -> str:
        tokens = tokens.tolist()
        tokens = [token for token in tokens if token]
        smiles = ''.join([self.index_to_token[token] for token in tokens])
        for special_token in self.special_tokens:
            smiles = smiles.replace(special_token, '')
        return smiles

    def _load_vocabulary(self) -> List[str]:
        return self.special_tokens + read_strings_from_file(self.path_to_vocabulary)

    def pad(self, tokens: List[int]) -> List[int]:
        return tokens + [self.token_to_index[self.pad_token]] * (self.max_length - len(tokens))

    def mask(self, tokens: List[int], mask_index: int) -> List[int]:
        tokens[mask_index] = self.token_to_index[self.mask_token]
        return tokens
