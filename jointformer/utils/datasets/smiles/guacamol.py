import os

from urllib.request import urlretrieve
from typing import List, Optional, Union, Callable

from jointformer.configs.task import TaskConfig
from jointformer.utils.datasets.smiles.smiles import SmilesDataset
from jointformer.utils.targets.utils import save_floats_to_file

DATASET_SEED = 0

GUACAMOL_URL = {
    'train': "https://ndownloader.figshare.com/files/13612760",
    'val': "https://ndownloader.figshare.com/files/13612766",
    'test': "https://ndownloader.figshare.com/files/13612757",
    'all': "https://ndownloader.figshare.com/files/13612745",
    'debug': None
}

DATA_FOLDER = './data/guacamol'
DATA_FILE_NAME = 'smiles.txt'


class Guacamol(SmilesDataset):

    def __init__(
            self,
            split: str,
            target_label: Optional[str] = None,
            transform: Optional[Union[Callable, List]] = None,
            target_transform: Optional[Union[Callable, List]] = None,
            num_samples: Optional[int] = None,
            validate: Optional[bool] = False
    ) -> None:

        self.split = split
        self.num_samples = num_samples
        self.target_label = target_label
        self._path_to_data_dir = None
        self._target_filename = None
        self._set_path_to_data_dir()
        self._get_data()
        self._get_targets()
        super().__init__(
            data_file_path=self._data_filename, target_file_path=self._target_filename, transform=transform,
            target_transform=target_transform, num_samples=num_samples, validate=validate)

    def _set_path_to_data_dir(self) -> None:
        self._path_to_data_dir = os.path.join(DATA_FOLDER, self.split)
        if self.num_samples is not None:
            self._path_to_data_dir = os.path.join(self._path_to_data_dir, str(self.num_samples))
        return None

    def _get_data(self) -> None:
        self._data_filename = os.path.join(self._path_to_data_dir, DATA_FILE_NAME)
        if not os.path.exists(self._data_filename):
            self._download_guacamol_data()
            return None

    def _download_guacamol_data(self) -> None:
        print("Downloading Guacamol data...")
        os.makedirs(self._path_to_data_dir, exist_ok=True)
        urlretrieve(GUACAMOL_URL[self.split], self._data_filename)
        print(f"Guacamol {self.split} data downloaded into {self._path_to_data_dir}.")

    def _get_targets(self):
        if self.target_label is not None:
            self._target_filename = os.path.join(self._path_to_data_dir, f'{self.target_label}.txt')
            if not os.path.exists(self._target_filename):
                os.makedirs(self._path_to_data_dir, exist_ok=True)
                target = super()._calculate_targets(
                    target_label=self.target_label, data_file_path=self._data_filename, num_samples=self.num_samples)
                save_floats_to_file(target, self._target_filename)
                print(f"Guacamol {self.target_label} data obtained into {self._path_to_data_dir}.")
        return None

    @classmethod
    def from_config(cls, config: TaskConfig, split: str = None):

        split = config.split if split is None else split

        if split is None:
            raise ValueError("split must be provided either in the config or as an argument.")

        return cls(
            split=config.split, target_label=config.target_label, transform=config.transform,
            target_transform=config.target_transform, validate=config.validate, num_samples=config.num_samples)
