""" Load and savews data. """

import os
import pandas as pd
import numpy as np

from typing import Union, List


def load(file_path: str, **kwargs) -> Union[np.ndarray, pd.DataFrame, List[str]]:

    """ Load data from a file.

    Args:
        file_path: Path to the file to load.

    Returns:
        pd.DataFrame: Data loaded from the file.

    """

    # Check if the file exists
    assert os.path.exists(file_path) and os.path.isfile(file_path), f"File {file_path} does not exist."
    
    # Load the data, depending on the file extension
    _extension = _get_file_extension(file_path)
    if _extension == '.csv':
        data = pd.read_csv(file_path, **kwargs)
    elif _extension == '.gz':
        data = pd.read_csv(file_path, compression='gzip', **kwargs)
    elif _extension == '.npy':
        data = np.load(file_path, **kwargs)
    elif _extension == '.txt' or _extension == '.smiles':
        data = np.loadtxt(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {_extension}")
    
    return data


def save(data: Union[np.ndarray, pd.DataFrame, List[str]], file_path: str, overrite: bool = False) -> None:

    """ Save data to a file.

    Args:
        data: Data to save.
        file_path: Path to the file to save the data to.

    """

    # Check if the file exists and create the directory if it does not
    if not overrite:
        assert not os.path.exists(file_path), f"File {file_path} already exists."

    dname = os.path.dirname(file_path)
    if not os.path.exists(dname):
        os.makedirs(dname, exist_ok=False)

    # Save the data, depending on the file extension
    _extension = _get_file_extension(file_path)
    if _extension == '.csv':
        data.to_csv(file_path, index=False)
    elif _extension == '.gz':
        data.to_csv(file_path, compression='gzip', index=False)
    elif _extension == '.npy':
        np.save(file_path, data)
    elif _extension == '.txt':
        np.savetxt(file_path, data)
    else:
        raise ValueError(f"Unsupported file extension: {_extension}")
    
    return None


def _get_file_extension(file_path: str) -> str:

    """ Get the file extension.

    Args:
        file_path: Path to the file.

    Returns:
        str: File extension.

    """
    return os.path.splitext(file_path)[1]

# coding=utf-8

"""
Vocabulary helper class
"""

import re
import numpy as np


# contains the data structure
class Vocabulary:
    """Stores the tokens and their conversion to vocabulary indexes."""

    def __init__(self, tokens=None, starting_id=0):
        self._tokens = {}
        self._current_id = starting_id

        if tokens:
            for token, idx in tokens.items():
                self._add(token, idx)
                self._current_id = max(self._current_id, idx + 1)

    def __getitem__(self, token_or_id):
        return self._tokens[token_or_id]

    def add(self, token):
        """Adds a token."""
        if not isinstance(token, str):
            raise TypeError("Token is not a string")
        if token in self:
            return self[token]
        self._add(token, self._current_id)
        self._current_id += 1
        return self._current_id - 1

    def update(self, tokens):
        """Adds many tokens."""
        return [self.add(token) for token in tokens]

    def __delitem__(self, token_or_id):
        other_val = self._tokens[token_or_id]
        del self._tokens[other_val]
        del self._tokens[token_or_id]

    def __contains__(self, token_or_id):
        return token_or_id in self._tokens

    def __eq__(self, other_vocabulary):
        return self._tokens == other_vocabulary._tokens  # pylint: disable=W0212

    def __len__(self):
        return len(self._tokens) // 2

    def encode(self, tokens):
        """Encodes a list of tokens as vocabulary indexes."""
        vocab_index = np.zeros(len(tokens), dtype=np.float32)
        for i, token in enumerate(tokens):
            vocab_index[i] = self._tokens[token]
        return vocab_index

    def decode(self, vocab_index):
        """Decodes a vocabulary index matrix to a list of tokens."""
        tokens = []
        for idx in vocab_index:
            tokens.append(self[idx])
        return tokens

    def _add(self, token, idx):
        if idx not in self._tokens:
            self._tokens[token] = idx
            self._tokens[idx] = token
        else:
            raise ValueError("IDX already present in vocabulary")

    def tokens(self):
        """Returns the tokens from the vocabulary"""
        return [t for t in self._tokens if isinstance(t, str)]


class SMILESTokenizer:
    """Deals with the tokenization and untokenization of SMILES.
    
    Source: https://github.com/MolecularAI/reinvent-models/blob/main/reinvent_models/reinvent_core/models/vocabulary.py
    """

    REGEXPS = {
        "brackets": re.compile(r"(\[[^\]]*\])"),
        "2_ring_nums": re.compile(r"(%\d{2})"),
        "brcl": re.compile(r"(Br|Cl)")
    }
    REGEXP_ORDER = ["brackets", "2_ring_nums", "brcl"]

    def tokenize(self, data, with_begin_and_end=True):
        """Tokenizes a SMILES string."""
        def split_by(data, regexps):
            if not regexps:
                return list(data)
            regexp = self.REGEXPS[regexps[0]]
            splitted = regexp.split(data)
            tokens = []
            for i, split in enumerate(splitted):
                if i % 2 == 0:
                    tokens += split_by(split, regexps[1:])
                else:
                    tokens.append(split)
            return tokens

        tokens = split_by(data, self.REGEXP_ORDER)
        if with_begin_and_end:
            tokens = ["^"] + tokens + ["$"]
        return tokens

    def untokenize(self, tokens):
        """Untokenizes a SMILES string."""
        smi = ""
        for token in tokens:
            if token == "$":
                break
            if token != "^":
                smi += token
        return smi


def create_vocabulary(smiles_list, tokenizer):
    """Creates a vocabulary for the SMILES syntax."""
    tokens = set()
    for smi in smiles_list:
        tokens.update(tokenizer.tokenize(smi, with_begin_and_end=False))

    vocabulary = Vocabulary()
    # vocabulary.update(["$", "^"] + sorted(tokens))  # end token is 0 (also counts as padding)
    vocabulary.update(sorted(tokens))  # end token is 0 (also counts as padding)
    return vocabulary


def load_strings_from_txt(file_path):
    """
    Loads a list of strings from a .txt file, with each line as an entry in the list.

    Args:
        file_path (str): Path to the .txt file.

    Returns:
        list: A list of strings, one for each line in the file.
    """
    try:
        with open(file_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        return lines
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return []
    
def save_strings_to_txt(strings, file_path):
    """
    Saves a list of strings to a .txt file in UTF-8 encoding, with each string written on a new line.

    Args:
        strings (list): List of strings to save.
        file_path (str): Path to the .txt file to save the strings.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            for string in strings:
                f.write(string + "\n")
        print(f"Strings successfully saved to {file_path} in UTF-8 encoding.")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")
        