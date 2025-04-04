import os
import collections

from typing import List, Optional
from transformers import BertTokenizer

from hyformer.utils.tokenizers.regex import RegexSmilesTokenizer
from hyformer.utils.tokenizers.utils import load_vocab


class SmilesTokenizer(BertTokenizer):
    
    def __init__(
        self,
        vocabulary_path: str,
        # unk_token="[UNK]",
        # sep_token="[SEP]",
        # pad_token="[PAD]",
        # cls_token="[CLS]",
        # mask_token="[MASK]",
        **kwargs):
        """Constructs a SmilesTokenizer.

        Parameters
        ----------
        vocabulary_path: str
            Path to a SMILES character per line vocabulary file.
            Default vocab file is found in deepchem/feat/tests/data/vocab.txt
        """

        super().__init__(vocabulary_path, **kwargs)

        if not os.path.isfile(vocabulary_path):
            raise ValueError(
                "Can't find a vocab file at path '{}'.".format(vocabulary_path))
        self.vocab = load_vocab(vocabulary_path)
        self.highest_unused_index = max([
            i for i, v in enumerate(self.vocab.keys())
            if v.startswith("[unused")
        ])
        self.ids_to_tokens = collections.OrderedDict([
            (ids, tok) for tok, ids in self.vocab.items()
        ])
        self.basic_tokenizer = RegexSmilesTokenizer()

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def vocab_list(self):
        return list(self.vocab.keys())

    def _tokenize(self, text: str, max_seq_length: int = 512, **kwargs):
        """Tokenize a string into a list of tokens.

        Parameters
        ----------
        text: str
            Input string sequence to be tokenized.
        """

        max_len_single_sentence = max_seq_length - 2
        split_tokens = [
            token for token in self.basic_tokenizer.tokenize(text)
            [:max_len_single_sentence]
        ]
        return split_tokens

    def _convert_token_to_id(self, token: str):
        """Converts a token (str/unicode) in an id using the vocab.

        Parameters
        ----------
        token: str
            String token from a larger sequence to be converted to a numerical id.
        """

        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index: int):
        """Converts an index (integer) in a token (string/unicode) using the vocab.

        Parameters
        ----------
        index: int
            Integer index to be converted back to a string-based token as part of a larger sequence.
        """

        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]):
        """Converts a sequence of tokens (string) in a single string.

        Parameters
        ----------
        tokens: List[str]
            List of tokens for a given string sequence.

        Returns
        -------
        out_string: str
            Single string from combined tokens.
        """

        out_string: str = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def add_special_tokens_ids_single_sequence(self,
                                               token_ids: List[Optional[int]]):
        """Adds special tokens to the a sequence for sequence classification tasks.

        A BERT sequence has the following format: [CLS] X [SEP]

        Parameters
        ----------
        token_ids: list[int]
            list of tokenized input ids. Can be obtained using the encode or encode_plus methods.
        """

        return [1111] + [self.cls_token_id] + token_ids + [self.sep_token_id]

    def add_special_tokens_single_sequence(self, tokens: List[str]):
        """Adds special tokens to the a sequence for sequence classification tasks.
        A BERT sequence has the following format: [CLS] X [SEP]

        Parameters
        ----------
        tokens: List[str]
            List of tokens for a given string sequence.
        """
        return [self.cls_token] + tokens + [self.sep_token]

    def add_special_tokens_ids_sequence_pair(
            self, token_ids_0: List[Optional[int]],
            token_ids_1: List[Optional[int]]) -> List[Optional[int]]:
        """Adds special tokens to a sequence pair for sequence classification tasks.
        A BERT sequence pair has the following format: [CLS] A [SEP] B [SEP]

        Parameters
        ----------
        token_ids_0: List[int]
            List of ids for the first string sequence in the sequence pair (A).
        token_ids_1: List[int]
            List of tokens for the second string sequence in the sequence pair (B).
        """

        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        return cls + token_ids_0 + sep + token_ids_1 + sep

    def add_padding_tokens(self,
                           token_ids: List[Optional[int]],
                           length: int,
                           right: bool = True) -> List[Optional[int]]:
        """Adds padding tokens to return a sequence of length max_length.
        By default padding tokens are added to the right of the sequence.

        Parameters
        ----------
        token_ids: list[optional[int]]
            list of tokenized input ids. Can be obtained using the encode or encode_plus methods.
        length: int
            TODO
        right: bool, default True
            TODO

        Returns
        -------
        List[int]
            TODO
        """
        padding = [self.pad_token_id] * (length - len(token_ids))

        if right:
            return token_ids + padding
        else:
            return padding + token_ids
        
