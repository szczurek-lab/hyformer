import os
import torch
import logging
import random

from tqdm import tqdm
from typing import Tuple, List, Any
from torch.utils.data.dataset import Dataset
from urllib.request import urlretrieve
from guacamol.utils.chemistry import is_valid
from rdkit import Chem

from hybrid_transformer.utils.datasets.utils import load_txt_into_list, save_list_into_txt
from hybrid_transformer.utils.objectives.guacamol.objective import get_objective
from hybrid_transformer.utils.objectives.guacamol.utils import GUACAMOL_TASK_NAMES
from hybrid_transformer.utils.utils import select_random_indices_from_length
from hybrid_transformer.utils.transforms.augment import AugmentSMILES

DATASET_SEED = 0

GUACAMOL_URL = {
    'train': "https://ndownloader.figshare.com/files/13612760",
    'val': "https://ndownloader.figshare.com/files/13612766",
    'test': "https://ndownloader.figshare.com/files/13612757",
    'all': "https://ndownloader.figshare.com/files/13612745",
    'debug': None
}

DATA_FOLDER = './data/guacamol'
FILE_NAME = "smiles.txt"

TARGET_LABELS = {}


class GuacamolSMILESDataset(Dataset):

    def __init__(
            self, split: str = None, target_label: str = None, transforms: List[Any] = None,
            validate: bool = True, num_samples: int = None, data_dir: str = None) -> None:

        super().__init__()
        self.data = None
        self.target = None
        self.split = split
        self.target_label = target_label
        self.transforms = transforms
        self.target_transforms = None
        self.path_to_data_dir = None
        self.num_samples = num_samples if self.split == 'train' else None  # subset only training data

        self._check_args()
        self._get_path_to_data_dir(data_dir)
        if data_dir is not None:
            self._load_data(os.path.join(data_dir, FILE_NAME))
            if validate:
                self._validate_data_only()
        else:
            self._load_guacamol_data()
        if target_label is not None:
            self._load_guacamol_target()
        if validate:
            self._validate()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:

        x = self.data[idx]

        if self.target is not None:
            if self.transforms is not None:
                for transform in self.transforms:
                    x = transform(x)

        if self.target_label is None:
            return x
        else:
            y = self.target[idx]
            return x, y

    def _check_args(self):
        if self.split not in GUACAMOL_URL and self.split is not None:
            raise ValueError('Variable `dataset` must be one of "%s"' % (list(GUACAMOL_URL.keys())))
        if (self.target_label is not None) and (self.target_label not in GUACAMOL_TASK_NAMES):
            raise ValueError('Variable `target_label` must be one of "%s"' % GUACAMOL_TASK_NAMES)

    def _load_guacamol_data(self) -> None:
        filename = os.path.join(self.path_to_data_dir, FILE_NAME)
        if not os.path.exists(filename):
            self._download_guacamol_data()

        if self.num_samples is not None:
            path_to_subset_data_dir = self.path_to_data_dir + '_' + str(self.num_samples)
            subset_filename = os.path.join(path_to_subset_data_dir, FILE_NAME)
            if not os.path.exists(subset_filename):
                os.makedirs(path_to_subset_data_dir)
                print("Creating dir '%s'" % path_to_subset_data_dir)
                random.seed(DATASET_SEED)
                data = load_txt_into_list(filename)
                data = random.sample(data, self.num_samples)
                save_list_into_txt(subset_filename, data)
            self.path_to_data_dir = path_to_subset_data_dir
            filename = subset_filename

        self._load_data(filename)
        return None

    def _load_data(self, filename: str) -> None:
        self.data = load_txt_into_list(filename)
        return None

    def _load_guacamol_target(self):

        filename = os.path.join(self.path_to_data_dir, f'{self.target_label}.pt')
        if not os.path.exists(filename):
            target = get_objective(self.data, self.target_label, verbose=True)
            torch.save(target, filename)
            print(f"Guacamol {self.split} {self.target_label} objective saved into {filename}.")
        self.target = torch.load(filename)
        return None

    def _download_guacamol_data(self) -> None:
        print(f"Downloading Guacamol data...")
        os.makedirs(self.path_to_data_dir, exist_ok=True)
        urlretrieve(GUACAMOL_URL[self.split], os.path.join(self.path_to_data_dir, FILE_NAME))
        print(f"Guacamol {self.split} data downloaded into {self.path_to_data_dir}.")
        return None

    def _get_path_to_data_dir(self, data_dir: str = None) -> None:
        self.path_to_data_dir = os.path.join(DATA_FOLDER, self.split) if data_dir is None else data_dir
        return None

    def _validate(self):
        print("Validating Guacamol dataset...")
        if self.target_label is None:
            self._validate_data_only()
        else:
            self._validate_data_and_target()
        print(f"Guacamol dataset loaded with {len(self.data)} valid examples!")
        return None

    def _validate_data_only(self):
        data_safe = []
        for idx, x in enumerate(tqdm(self.data)):
            if is_valid(x):
                data_safe.append(x)
        self.data = data_safe
        return None

    def _validate_data_and_target(self):
        data_safe = []
        target_safe = torch.ones(size=self.target.size(), dtype=self.target.dtype)

        for idx, (x, y) in enumerate(tqdm(zip(self.data, self.target))):
            if is_valid(x) and not torch.isnan(y):
                data_safe.append(x)
                target_safe[idx] = y
        self.data = data_safe
        self.target = target_safe
        return None

    def _subset_dataset(self):
        idx = select_random_indices_from_length(len(self.data), self.num_samples)
        self.data = [self.data[i] for i in idx]
        if self.target_label is not None:
            self.target = self.target[idx]
        return None

    def undo_target_transform(self, y: torch.Tensor) -> torch.Tensor:
        if self.target_transforms is None:
            return y
        y_undone = y.numpy()
        for transform in reversed(self.target_transforms):
            y_undone = transform.untransform(y_undone)
        return torch.from_numpy(y_undone).view(-1, 1)

    @classmethod
    def from_config(cls, config):
        transforms = []
        if config.augment_molecular_representation:
            transforms.append(AugmentSMILES(augmentation_prob=config.augmentation_prob))

        return cls(
            split=config.split, target_label=config.target_label, transforms=transforms,
            validate=config.validate, num_samples=config.num_samples)
