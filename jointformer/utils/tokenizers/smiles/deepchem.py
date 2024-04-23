""" Reimplements the SmilesTokenizer class from deepchem.feat.smiles_tokenizer.
 Source: https://deepchem.readthedocs.io/en/2.4.0/api_reference/tokenizers.html

 The tokenizer heavily inherits from the BertTokenizer implementation found in Huggingface’s transformers library.
  It runs a WordPiece tokenization algorithm over SMILES strings using the tokenization SMILES regex developed by [1].

 Requires huggingface's transformers and tokenizers libraries to be installed.

    References
    ----------
    [1]  Philippe Schwaller, Teodoro Laino, Théophile Gaudin, Peter Bolgar, Christopher A. Hunter, Costas Bekas, and
     Alpha A. Lee. ACS Central Science 2019 5 (9): Molecular Transformer: A Model for Uncertainty-Calibrated
      Chemical Reaction Prediction 1572-1583 DOI: 10.1021/acscentsci.9b00576.
 """


import os
import collections

from typing import List
from transformers import BertTokenizer
from logging import getLogger

from jointformer.utils.tokenizers.smiles.regex import RegexSmilesTokenizer
from jointformer.utils.tokenizers.smiles.utils import read_vocabulary

logger = getLogger(__name__)


class DeepChemSmilesTokenizer(BertTokenizer):

    def __init__(
        self,
        path_to_vocabulary: str,
        # unk_token="[UNK]",
        # sep_token="[SEP]",
        # pad_token="[PAD]",
        # cls_token="[CLS]",
        # mask_token="[MASK]",
        **kwargs
    ) -> None:
        """Constructs a SmilesTokenizer.

        Parameters
        ----------
        vocabulary_file: str
            Path to a SMILES character per line vocabulary file.
            Default vocab file can be found in deepchem/feat/tests/data/vocab.txt
        max_length: int
            Maximum length of the sequence.
        """
        vocabulary_file = path_to_vocabulary
        super().__init__(vocabulary_file, **kwargs)
        self.vocab = read_vocabulary(vocabulary_file)
        self.highest_unused_index = max([i for i, v in enumerate(self.vocab.keys()) if v.startswith("[unused")])
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.basic_tokenizer = RegexSmilesTokenizer()
        # self.init_kwargs["max_len"] = self.max_len

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def vocab_list(self):
        return list(self.vocab.keys())

    def _tokenize(self, text: str):
        split_tokens = [token for token in self.basic_tokenizer.tokenize(text)]
        return split_tokens

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]):
        out_string: str = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def add_special_tokens_ids_single_sequence(self, token_ids: List[int]):
        return [self.cls_token_id] + token_ids + [self.sep_token_id]

    def add_special_tokens_single_sequence(self, tokens: List[str]):
        return [self.cls_token] + tokens + [self.sep_token]

    def add_special_tokens_ids_sequence_pair(self, token_ids_0: List[int],
                                       token_ids_1: List[int]) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def add_padding_tokens(self,
                     token_ids: List[int],
                     length: int,
                     right: bool = True) -> List[int]:

        padding = [self.pad_token_id] * (length - len(token_ids))

        if right:
            return token_ids + padding
        else:
            return padding + token_ids

    def save_vocabulary(self, vocab_path: str):  # -> tuple[str]: doctest issue raised with this return type annotation
        """
        Save the tokenizer vocabulary to a file.

        Parameters
        ----------
        vocab_path: obj: str
            The directory in which to save the SMILES character per line vocabulary file.
            Default vocab file is found in deepchem/feat/tests/data/vocab.txt

        Returns
        ----------
        vocab_file: :obj:`Tuple(str)`:
            Paths to the files saved.
            typle with string to a SMILES character per line vocabulary file.
            Default vocab file is found in deepchem/feat/tests/data/vocab.txt

        """
        index = 0
        vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file))
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)
