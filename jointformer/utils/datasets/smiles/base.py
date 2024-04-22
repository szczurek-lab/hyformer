"""SMILES dataset.

This module defines the SMILESDataset class, which is a PyTorch dataset for SMILES strings. The dataset requires
a path to a .txt file containing the SMILES strings. The SMILES strings can be automatically validated and augmented.
Additionally, SMILESDataset supports supervised learning tasks by providing basic labels for each SMILES string.

Example:
    >>> from jointformer.utils.datasets.smiles_tokenizers.base import SMILESDataset
    >>> from jointformer.utils.datasets.smiles_tokenizers.utils import read_strings_from_file

    >>> file_path = "data/smiles_tokenizers.txt"
    >>> data = read_strings_from_file(file_path)

    >>> dataset = SMILESDataset(file_path, num_samples=1000, validate=True)
    >>> print(len(dataset))
    >>> print(dataset[0])
"""

import torchvision.transforms as transforms

from tqdm import tqdm
from typing import List, Callable, Optional, Union
from torch.utils.data.dataset import Dataset
from guacamol.utils.chemistry import is_valid

from jointformer.utils.datasets.utils import read_strings_from_file


class SMILESDataset(Dataset):
    """A PyTorch dataset for SMILES strings and their basic physicochemical properties."""

    def __init__(
            self,
            file_path: str,
            transform: Optional[Union[Callable, List]] = None,
            target_transform: Optional[Union[Callable, List]] = None,
            num_samples: Optional[int] = None,
            validate: Optional[bool] = False
    ) -> None:
        """ Initializes the dataset.

        Args:
        file_path: str
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
        self.data = None
        self.target = None
        self.transform = transforms.Compose(transform) if isinstance(transform, list) else transform
        self.target_transform = transforms.Compose(target_transform) if isinstance(target_transform, list) else target_transform
        self.num_samples = num_samples
        self.validate = validate
        self._read_data(file_path)
        self._validate_data()
        self._subset_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        x = self.data[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x

    def _read_data(self, file_path: str):
        self.data = read_strings_from_file(file_path)

    def _validate_data(self):
        if self.validate:
            self.data = [x for x in tqdm(self.data, desc="Validating SMILES data") if is_valid(x)]
            self.data = [x for x in self.data if len(x) > 0]

    def _subset_data(self):
        if self.num_samples is not None and len(self.data) > self.num_samples:
            self.data = self.data[:self.num_samples]
