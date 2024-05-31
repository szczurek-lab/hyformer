""" Base class for all properties. """

import torch
from tqdm import tqdm

from typing import List, Union

DTYPE_OBJECTIVE = torch.float32


class BaseTarget:

    def __init__(self, dtype=DTYPE_OBJECTIVE):
        self.dtype = dtype

    def __call__(self, examples: List[str], dtype='pt') -> Union[torch.Tensor, List[float]]:
        targets = self.get_targets(examples)
        if dtype == 'pt':
            if not isinstance(targets, torch.Tensor):
                targets = torch.Tensor(targets)
            return targets.to(DTYPE_OBJECTIVE).unsqueeze(1)
        else:
            return targets

    def get_targets(self, examples: Union[List[str], str]) -> List[float]:
        if isinstance(examples, str):
            examples = [examples]
        objective_values = []
        for example in tqdm(examples, desc="Calculating target data"):
            objective_values.append(self._get_target(example))
        return objective_values

    def _get_target(self, example: str) -> float:
        raise NotImplementedError

    @property
    def target_names(self):
        raise NotImplementedError
