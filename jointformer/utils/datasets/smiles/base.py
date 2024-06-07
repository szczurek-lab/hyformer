""" A base PyTorch dataset for SMILES strings.
"""

import torch
import random
import logging
import numpy as np

from tqdm import tqdm
from rdkit import Chem
from typing import List, Callable, Optional, Union

from jointformer.configs.task import TaskConfig
from jointformer.utils.datasets.base import BaseDataset
from jointformer.utils.data import read_strings_from_file
from jointformer.utils.chemistry import is_valid, standardize

logger = logging.getLogger(__name__)


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
            standardize: Optional[bool] = None,
            max_molecule_length: Optional[int] = None,
    ) -> None:

        if data is None and data_filename is None:
            logger.warning("Either data or data_filename must be provided.")
            raise AssertionError

        if data_filename is not None:
            data = self._load_data(data_filename)
        if target_filename is not None:
            target = self._load_target(target_filename)

        super().__init__(
            data=data, target=target, transform=transform, target_transform=target_transform
        )
        self.max_molecule_length = max_molecule_length
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
        logger.info("Validating SMILES data.")
        if self.validate and self.data is not None:

            is_valid_molecule = [is_valid(x) for x in self.data]

            if self.max_molecule_length is not None:
                is_valid_molecule = [x and len(y) <= self.max_molecule_length for x, y in zip(is_valid_molecule, self.data)]

            if self.target is not None:
                is_valid_molecule = [x and torch.equal(y, y) for x, y in zip(is_valid_molecule, self.target)]

            self.data = [x for x, y in zip(self.data, is_valid_molecule) if y]
            if self.target is not None:
                self.target = [x for x, y in zip(self.target, is_valid_molecule) if y]

            if sum(is_valid_molecule) < len(is_valid_molecule):
                logger.warning(
                    f"Removed {len(is_valid_molecule) - sum(is_valid_molecule)} invalid molecules."
                )

    def _standardize(self):
        if self.standardize and self.data is not None:
            self.data = [standardize(smiles, canonicalize=True) for smiles in self.data]
            if self.target is not None:
                self.target = [x for x, y in zip(self.target, self.data) if y is not None]
                self.data = [x for x in self.data if x is not None]

    @staticmethod
    def _load_data(data_filename: str):
        return read_strings_from_file(data_filename)

    @staticmethod
    def _load_target(target_filename: str):
        data = torch.load(target_filename)
        return data
