import os
import torch

import deepchem as dc

from typing import List, Optional, Union, Callable, Any
from deepchem.feat.molecule_featurizers.raw_featurizer import RawFeaturizer

from jointformer.configs.task import TaskConfig
from jointformer.utils.datasets.smiles.base import SmilesDataset
from jointformer.utils.data import save_strings_to_file

DATA_DIR = 'data/molecule_net'
DATA_FILE_NAME = 'smiles.txt'


class MoleculeNetDataset(SmilesDataset):

    def __init__(
            self,
            split: str,
            splitter: str,
            target_label: str,
            transform: Optional[Union[Callable, List]] = None,
            validate: Optional[bool] = None,
            standardize: Optional[bool] = None,
            out_dir: Optional[str] = None,
    ) -> None:

        # Download data and targets
        out_dir = out_dir if out_dir is not None else './'
        data_dir = self._get_data_dir(os.path.join(out_dir, DATA_DIR, target_label, splitter), split)
        data_filename = os.path.join(data_dir, DATA_FILE_NAME)
        target_filename = os.path.join(data_dir, f'{target_label}.pt')
        if not os.path.isfile(data_filename) or not os.path.isfile(target_filename):
            self._download(
                data_dir=data_dir,
                data_filename=data_filename,
                target_filename=target_filename,
                split=split,
                splitter=splitter,
                target_label=target_label
            )

        # Initialize the dataset
        super().__init__(
            data_filename=data_filename, target_filename=target_filename,
            transform=transform, target_transform=None,
            num_samples=None, validate=validate, standardize=standardize
        )
        self._target_transform = self.get_target_transform(target_label=target_label, splitter=splitter) # used only for validation

    @staticmethod
    def _get_data_dir(data_dir: str, split: str = None, num_samples: int = None, splitter: str = None) -> str:
        if split is not None:
            data_dir = os.path.join(data_dir, split)
        if num_samples is not None:
            data_dir = os.path.join(data_dir, str(num_samples))
        if splitter is not None:
            data_dir = os.path.join(data_dir, splitter)
        return data_dir

    @staticmethod
    def _download(
            data_dir: str,
            data_filename: str,
            target_filename: str,
            split: str,
            splitter: str,
            target_label: str
    ) -> None:

        print(f"Downloading data into {data_filename}")
        os.makedirs(data_dir, exist_ok=True)

        featurizer = RawFeaturizer(smiles=True)

        if target_label == 'esol':
            _, datasets, _ = dc.molnet.load_delaney(featurizer=featurizer, splitter=splitter, data_dir=data_dir)
        elif target_label == 'freesolv':
            _, datasets, _ = dc.molnet.load_sampl(featurizer=featurizer, splitter=splitter, data_dir=data_dir)
        elif target_label == 'lipo':
            _, datasets, _ = dc.molnet.load_lipo(featurizer=featurizer, splitter=splitter, data_dir=data_dir)
        else:
            raise ValueError(f"Unknown target label: {target_label}")

        split_idx = {'train': 0, 'val': 1, 'test': 2}

        data = datasets[split_idx[split]].X.tolist()
        save_strings_to_file(data, data_filename)

        print(f"Downloading target into {target_filename}")
        target = torch.from_numpy(datasets[split_idx[split]].y)
        torch.save(target, os.path.join(data_dir, f"{target_label}.pt"))

    @staticmethod
    def get_target_transform(target_label: str, splitter: str = 'random') -> List[Any]:
        featurizer = RawFeaturizer(smiles=True)
        if target_label == 'esol':
            _, _, target_transform = dc.molnet.load_delaney(featurizer=featurizer, splitter=splitter)
        elif target_label == 'freesolv':
            _, _, target_transform = dc.molnet.load_sampl(featurizer=featurizer, splitter=splitter)
        elif target_label == 'lipo':
            _, _, target_transform = dc.molnet.load_lipo(featurizer=featurizer, splitter=splitter)
        else:
            target_transform = None
        return target_transform

    def undo_target_transform(self, y: torch.Tensor) -> torch.Tensor:
        if self._target_transform is None:
            return y
        y_undone = y.numpy()
        for transform in reversed(self._target_transform):
            y_undone = transform.untransform(y_undone)
        return torch.from_numpy(y_undone).view(-1, 1)

    @classmethod
    def from_config(cls, config: TaskConfig, split: str = None, out_dir: str = None) -> SmilesDataset:

        if split is not None:
            config.split = split

        return cls(
            split=config.split,
            splitter=config.splitter,
            target_label=config.target_label,
            transform=config.transform,
            validate=config.validate,
            standardize=config.standardize,
            out_dir=out_dir
        )
