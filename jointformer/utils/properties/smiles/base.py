""" Base class for all properties. """

import torch
import numpy as np
from tqdm import tqdm

from typing import List, Union


class BaseTarget:

    def __init__(self, dtype='pt'):
        self.dtype = dtype

    def __call__(self, examples: List[str]) -> Union[torch.Tensor, List[float], np.ndarray]:
        targets = self.get_targets(examples)
        if self.dtype == 'pt' and not isinstance(targets, torch.Tensor):
            return torch.from_numpy(targets)
        return targets

    def get_targets(self, examples: Union[List[str], str]) -> np.ndarray:

        if isinstance(examples, str):
            examples = [examples]
        
        targets = np.zeros(shape=(len(examples), len(self)), dtype=np.float32)  # initialize targets
        for idx, example in enumerate(tqdm(examples, desc="Calculating target data")):
            targets[idx, :] = self._get_target(example)
        return targets

    def _get_target(self, example: str) -> Union[float, np.ndarray]:
        raise NotImplementedError

    @property
    def target_names(self):
        raise NotImplementedError
