""" A base PyTorch dataset for SMILES strings.
"""

import torch
import random
import numpy as np

from tqdm import tqdm
from rdkit import Chem
from typing import List, Callable, Optional, Union

from jointformer.configs.task import TaskConfig
from jointformer.utils.datasets.base import BaseDataset
from jointformer.utils.data import read_strings_from_file
from jointformer.utils.chemistry import is_valid


class SmilesDataset(BaseDataset):
    """A PyTorch dataset for SMILES strings. """

    def __init__(
            self,
            data: Optional[List[str]] = None,
            data_filename: Optional[str] = None,
            target: Optional[torch.Tensor] = None,
            target_filename: Optional[str] = None,
            transform: Optional[Union[Callable, List]] = None,
            target_transform: Optional[Union[Callable, List]] = None,
            num_samples: int = None,
            validate: Optional[bool] = None,
            standardize: Optional[bool] = None
    ) -> None:

        assert data is not None or data_filename is not None, "Either data or data_filename must be provided."

        if data_filename is not None:
            data = self._load_data(data_filename)
        if target_filename is not None:
            target = self._load_target(target_filename)

        super().__init__(
            data=data, target=target, transform=transform, target_transform=target_transform
        )
        self.num_samples = num_samples
        self.validate = validate
        self.standardize = standardize
        self._subset()
        self._validate()
        self._standardize()

    def _subset(self):
        if self.num_samples is not None:
            if self.data is not None and len(self.data) > self.num_samples:
                self.data = random.sample(self.data, self.num_samples)
            if self.target is not None and len(self.target) > self.num_samples:
                idx = torch.randperm(self.target.size()[0])
                self.target = self.target[idx[:self.num_samples]]

    def _validate(self):
        if self.validate and self.data is not None:
            self.data = [x for x in tqdm(self.data, desc="Validating SMILES data") if is_valid(x)]
            if self.target is not None:
                inputs = [
                    (x, y) for x, y in tqdm(zip(self.data, self.target), desc="Validating target labels")
                    if torch.equal(y, y)
                ]
                self.data = [x for x, y in inputs]
                self.target = [y for x, y in inputs]

    def _standardize(self):
        if self.standardize and self.data is not None:
            self.data = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=False, isomericSmiles=True)
                         for smiles in tqdm(self.data, desc="Standardizing SMILES data")]

    @staticmethod
    def _load_data(data_filename: str):
        return read_strings_from_file(data_filename)

    @staticmethod
    def _load_target(target_filename: str):
        data = torch.load(target_filename)
        return data
