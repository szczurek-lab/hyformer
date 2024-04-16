"""Defines a generic dataset for SMILES strings."""

import torchvision.transforms as transforms

from tqdm import tqdm
from typing import List, Callable, Optional
from torch.utils.data.dataset import Dataset
from guacamol.utils.chemistry import is_valid

from jointformer.utils.datasets.smiles.utils import read_strings_from_file


class SMILESDataset(Dataset):

    def __init__(
            self,
            file_path: str,
            transform: Optional[Callable, List] = None,
            num_samples: Optional[int] = None,
            validate: Optional[bool] = False
    ) -> None:
        """ Initializes the dataset.

        Args:
        file_path: str
            The path to the .txt file containing the SMILES strings.
        transform: callable, optional
            A function/transform that takes in a SMILES string and returns a transformed version.
        num_samples: int, optional
            The number of samples to use. If None, all samples are used.
        validate: bool, optional
            If True, the SMILES strings are checked for validity.
        """

        super().__init__()
        self.data = None
        self.transform = transforms.Compose(transform) if isinstance(transform, list) else transform
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
