""" A base PyTorch dataset for SMILES strings.
"""

import torch
import random
import logging
import numpy as np

from tqdm import tqdm
from typing import List, Callable, Optional, Union

from jointformer.configs.task import TaskConfig
from jointformer.utils.datasets.base import BaseDataset
from jointformer.utils.data import read_strings_from_file
from jointformer.utils.chemistry import is_valid, standardize
from jointformer.utils.data_collators import DataCollator

logger = logging.getLogger(__name__)


class SmilesDataset(BaseDataset):
    """A PyTorch dataset for SMILES strings. """

    def __init__(
            self,
            data: Optional[List[str]] = None,
            data_filepath: Optional[str] = None,
            target: Optional[torch.Tensor] = None,
            target_filepath: Optional[str] = None,
            transform: Optional[Union[Callable, List]] = None,
            target_transform: Optional[Union[Callable, List]] = None,
            num_samples: int = None,
            validate: Optional[bool] = None,
            standardize: Optional[bool] = None,
            max_molecule_length: Optional[int] = None,
            seed: Optional[int] = None,
            task_type: Optional[str] = None
    ) -> None:

        if data is None and data_filepath is None:
            raise AssertionError("Either data or data_filepath must be provided.")

        if data_filepath is not None:
            data = self._load_data(data_filepath)
        if target_filepath is not None:
            target = self._load_target(target_filepath)

        super().__init__(
            data=data, target=target, transform=transform, target_transform=target_transform, seed=seed
        )
        self.max_molecule_length = max_molecule_length
        self.num_samples = num_samples
        self.validate = validate
        self.standardize = standardize
        self.task_type = task_type
        self._subset()
        self._validate()
        self._standardize()

    def _subset(self):
        if self.num_samples is not None:
            idx = list(range(len(self.target)))
            random.shuffle(idx)
            idx = idx[:self.num_samples] 
            if self.data is not None and len(self.data) > self.num_samples:
                self.data = [self.data[i] for i in idx]
            if self.target is not None and len(self.target) > self.num_samples:
                self.target = self.target[idx]

    def _validate(self):
        
        if self.validate and self.data is not None:

            logger.info("Validating SMILES data.")

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
    def _load_data(data_filepath: str):
        return read_strings_from_file(data_filepath)

    @staticmethod
    def _load_target(target_filepath: str, task_type: str = None):
        target = torch.load(target_filepath)
        if task_type == 'classification':
            target = target.long()
        elif task_type == 'regression':
            target = target.float()
        return target
