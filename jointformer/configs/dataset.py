
from typing import List, Optional, Union, Callable
from jointformer.configs.base import Config


class DatasetConfig(Config):

    def __init__(
            self,
            dataset_name: str,
            target_label: str,
            validate: bool,
            standardize: bool,
            num_samples: int,
            split: str,
            splitter: Optional[str],
            transform: Optional[Union[Callable, List]],
            target_transform: Optional[Union[Callable, List]]
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.target_label = target_label
        self.validate = validate
        self.standardize = standardize
        self.num_samples = num_samples
        self.split = split
        self.splitter = splitter
        self.transform = transform
        self.target_transform = target_transform
