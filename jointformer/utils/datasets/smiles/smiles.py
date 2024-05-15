"""SMILES dataset.

This module defines the SMILESDataset class, which is a PyTorch dataset for SMILES strings. The dataset requires
a path to a .txt file containing the SMILES strings. The SMILES strings can be automatically validated and augmented.
Additionally, SMILESDataset supports supervised learning tasks by providing basic labels for each SMILES string.
"""
import os
import torchvision.transforms as transforms

from tqdm import tqdm
from typing import List, Callable, Optional, Union

from jointformer.utils.transforms.auto import AutoTransform
from jointformer.utils.targets.auto import AutoTarget
from jointformer.utils.datasets.utils import read_strings_from_file
from jointformer.utils.targets.utils import save_floats_to_file, read_floats_from_file

from jointformer.utils.datasets.base import BaseDataset

from jointformer.utils.datasets.smiles.utils import is_valid

AVAILABLE_TARGETS = ["qed", "physchem"]


class SmilesDataset(BaseDataset):
    """A PyTorch dataset for SMILES strings and their basic physicochemical properties. """

    def __init__(
            self,
            data_file_path: str,
            target_label: Optional[str] = None,
            target_file_path: Optional[str] = None,
            transform: Optional[Union[Callable, List]] = None,
            target_transform: Optional[Union[Callable, List]] = None,
            num_samples: Optional[int] = None,
            validate: Optional[bool] = False
    ) -> None:
        """ Initializes the dataset.

        Args:
        data_file_path: str
            The path to the .txt file containing the SMILES strings.
        transform: callable, optional
            A function/transform that takes in a SMILES string and returns a transformed version.
        target_transform: callable, optional
            A function/transform that takes in a target of a SMILES string and returns a transformed version.
        num_samples: int, optional
            The number of samples to use. If None, all samples are used.
        validate: bool, optional
            If True, the SMILES strings are checked for validity.
        """

        super().__init__()
        self.data_file_path = data_file_path
        self.target_label = target_label
        self.target_file_path = target_file_path
        assert None in [self.target_label, self.target_file_path], "Either target_label or target_file_path must be None."
        self.transform = AutoTransform.from_config(transform) if isinstance(transform, list) else transform
        self.target_transform = transforms.Compose(target_transform) if isinstance(target_transform, list) else target_transform  # todo: add target transform support
        self.num_samples = num_samples
        self.validate = validate

        self._init_data()
        self._init_target()

    def _init_data(self):
        self._read_data(self.data_file_path)
        self._subset_data()
        self._validate_data()

    def _init_target(self):
        if self.target_file_path is not None:
            self._read_target(self.target_file_path)
        elif self.target_label is not None:
            self.oracle = AutoTarget.from_target_label(self.target_label)
            self.target = self.oracle(self.data)
        else:
            self.target = None
        self._validate_target()

    def _read_data(self, data_file_path: str):
        self.data = read_strings_from_file(data_file_path)
        return self.data

    def _read_target(self, target_file_path: str):
        self.target = read_floats_from_file(target_file_path)
        return self.target

    def _validate_data(self):
        if self.validate:
            self.data = [x for x in tqdm(self.data, desc="Validating SMILES data") if is_valid(x)]
            self.data = [x for x in self.data if len(x) > 0]

    def _validate_target(self):
        if self.validate:
            inputs = [(x, y) for x, y in tqdm(zip(self.data, self.target), desc="Validating target labels") if y == y]
            self.data = [x for x, y in inputs]
            self.target = [y for x, y in inputs]

    def _subset_data(self):
        if self.num_samples is not None and len(self.data) > self.num_samples:
            self.data = self.data[:self.num_samples]

    def _calculate_targets(self, target_label: str, data_file_path: Optional[str] = None, num_samples: Optional[int] = None):
        oracle = AutoTarget.from_target_label(target_label)
        data = self._read_data(data_file_path) if data_file_path is not None else self.data
        data = data[:num_samples] if num_samples is not None else data
        target = oracle(data, dtype='float')
        return target

    @classmethod
    def from_config(cls, config, split=None):
        if split is not None:
            config.split = split

        return cls(
            data_file_path=config.data_file_path,
            target_file_path=config.target_file_path,
            transform=config.transform,
            target_transform=config.target_transform,
            num_samples=config.num_samples,
            validate=config.validate
        )
