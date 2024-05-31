""" A base PyTorch dataset for SMILES strings. """

import torch
import numpy as np


from tqdm import tqdm
from rdkit import Chem
from typing import List, Callable, Optional, Union

from jointformer.utils.datasets.utils import read_strings_from_file
from jointformer.utils.datasets.base import BaseDataset
from jointformer.utils.datasets.smiles.utils import is_valid


class SmilesDataset(BaseDataset):
    """A PyTorch dataset for SMILES strings. """

    def __init__(
            self,
            data: List[str] = None,
            data_filename: Optional[str] = None,
            target: Union[List[float]] = None,
            target_filename: Optional[str] = None,
            transform: Optional[Union[Callable, List]] = None,
            target_transform: Optional[Union[Callable, List]] = None,
            num_samples: int = None,
            validate: Optional[bool] = False
    ) -> None:

        assert data is not None or data_filename is not None, "Either data or data_filename must be provided."

        if data_filename is not None:
            data = self._load_data(data_filename)
        if target_filename is not None:
            target = self._load_target(target_filename)

        super().__init__(
            data=data, target=target, transform=transform, target_transform=target_transform, num_samples=num_samples
        )
        self.validate = validate
        self._validate_data()
        self._validate_target()

    def _validate_data(self):
        if self.validate and self.data:
            self.data = [x for x in tqdm(self.data, desc="Validating SMILES data") if is_valid(x)]
            # self.data = [x for x in self.data if len(x) > 0]
            self.data = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=False, isomericSmiles=True)
                         for smiles in tqdm(self.data, desc="Standardizing SMILES data")]

    def _validate_target(self):
        if self.validate and self.data and self.target:
            inputs = [(x, y) for x, y in tqdm(zip(self.data, self.target), desc="Validating target labels") if y == y]
            self.data = [x for x, y in inputs]
            self.target = [y for x, y in inputs]

    @classmethod
    def from_config(cls, config, split: str = None):
        if split is not None:
            config.split = split

        return cls(
            data_filename=config.data_filename,
            target_filename=config.target_filename,
            transform=config.transform,
            target_transform=config.target_transform,
            num_samples=config.num_samples,
            validate=config.validate
        )

    @staticmethod
    def _load_data(data_filename: str):
        return read_strings_from_file(data_filename)

    @staticmethod
    def _load_target(target_filename: str):
        data = torch.load(target_filename)
        return data
