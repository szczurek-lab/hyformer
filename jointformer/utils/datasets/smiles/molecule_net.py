import os
import torch

from typing import List, Optional, Union, Callable, Any

from deepchem.molnet import load_delaney, load_sampl, load_lipo, load_hiv, load_bace_classification, load_bbbp, load_tox21, load_toxcast, load_sider, load_clintox
from deepchem.feat.molecule_featurizers.raw_featurizer import RawFeaturizer

from jointformer.configs.task import TaskConfig
from jointformer.utils.datasets.smiles.base import SmilesDataset
from jointformer.utils.data import save_strings_to_file

DATA_FILENAME = 'smiles.txt'
DATASET_NAME = 'molecule_net'

SPLIT_MAP = {
    'train': 0,
    'val': 1,
    'test': 2
}

LOAD_FN = {
    "lipophilicity": load_lipo,
    "esol": load_delaney,
    "freesolv": load_sampl,
    "hiv": load_hiv,
    "bace": load_bace_classification,
    "bbbp": load_bbbp,
    "tox21": load_tox21,
    "toxcast": load_toxcast,
    "sider": load_sider,
    "clintox": load_clintox
}


class MoleculeNetDataset(SmilesDataset):

    def __init__(
            self,
            split: str,
            splitter: str,
            target_label: str,
            transform: Optional[Union[Callable, List]] = None,
            validate: Optional[bool] = None,
            standardize: Optional[bool] = None,
            data_dir: Optional[str] = None,
            seed: Optional[int] = None,
            num_samples: Optional[int] = None,
            task_type: Optional[str] = None
    ) -> None:

        self._download(
            data_dir=data_dir,
            split=split,
            splitter=splitter,
            target_label=target_label,
            seed=seed
        )

        data_dir = self._get_data_dir(
            data_dir=data_dir, dataset_name=DATASET_NAME, target_name=target_label,
              split=split, splitter=splitter, seed=seed)
        data_filepath = os.path.join(data_dir, DATA_FILENAME)
        target_filepath = os.path.join(data_dir, f'{target_label}.pt')
        super().__init__(
            data_filepath=data_filepath, target_filepath=target_filepath,
            transform=transform, target_transform=None, task_type=task_type,
            num_samples=num_samples, validate=validate, standardize=standardize, seed=seed
        )
        self._target_transform = self.get_target_transform(target_label=target_label, splitter=splitter) # used only for validation

    @staticmethod
    def get_target_transform(target_label: str, splitter: str = 'random') -> List[Any]:
        featurizer = RawFeaturizer(smiles=True)
        _, _, target_transform = LOAD_FN[target_label](featurizer=featurizer, splitter=splitter)
        return target_transform
        
    def undo_target_transform(self, y: torch.Tensor) -> torch.Tensor:
        if self._target_transform is None:
            return y
        y_undone = y.numpy()
        for transform in reversed(self._target_transform):
            y_undone = transform.untransform(y_undone)
        return torch.from_numpy(y_undone).view(-1, 1)

    @classmethod
    def _download(
            cls,
            split: Optional[str] = None,
            splitter: Optional[str] = None,
            target_label: Optional[str] = None,
            data_dir: Optional[str] = None,
            seed: Optional[int] = None,
            num_samples: Optional[int] = None
    ) -> None:
        
        data_dir = cls._get_data_dir(
            data_dir=data_dir, dataset_name=DATASET_NAME, target_name=target_label,
              split=split, splitter=splitter, seed=seed, num_samples=num_samples)
        data_filepath = os.path.join(data_dir, DATA_FILENAME)
        target_filepath = os.path.join(data_dir, f'{target_label}.pt')
        if not os.path.isfile(data_filepath) or not os.path.isfile(target_filepath):
            print(f"Downloading {target_label} {DATASET_NAME} dataset...")
            os.makedirs(data_dir, exist_ok=False)

            featurizer = RawFeaturizer(smiles=True)
            _, datasets, _ = LOAD_FN[target_label](featurizer=featurizer, splitter=splitter)
            

            data = datasets[SPLIT_MAP[split]].X.tolist()
            save_strings_to_file(data, data_filepath)

            target = torch.from_numpy(datasets[SPLIT_MAP[split]].y)
            torch.save(target, os.path.join(data_dir, f"{target_label}.pt"))

    @classmethod
    def from_config(cls, config: TaskConfig, split: str = None, data_dir: str = None, seed: int = None) -> SmilesDataset:

        if split is not None:
            config.split = split

        return cls(
            split=config.split,
            splitter=config.splitter,
            target_label=config.target_label,
            transform=config.transform,
            validate=config.validate,
            standardize=config.standardize,
            data_dir=data_dir,
            seed=seed,
            num_samples=config.num_samples,
            task_type=config.task_type
        )
