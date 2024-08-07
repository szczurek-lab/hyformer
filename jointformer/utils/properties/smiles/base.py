""" Base class for all properties. """

import sys
import numpy as np

from tqdm import tqdm
from typing import List, Union

from jointformer.utils.properties.smiles.utils import TorchConvertMixin

class BaseTarget(TorchConvertMixin):

    def __init__(self, dtype=None, verbose=True):
        self.dtype = dtype if dtype is not None else 'pt'  # Default to torch.Tensor
        self.verbose = verbose

    def __call__(self, examples: List[str]) -> Union['torch.Tensor', np.ndarray]:
        targets = self.get_targets(examples)
        if self.dtype == 'pt':
            return self.to_tensor(targets)
        return targets

    def get_targets(self, examples: Union[List[str], str]) -> np.ndarray:

        if isinstance(examples, str):
            examples = [examples]
        
        targets = np.zeros(shape=(len(examples), len(self)), dtype=np.float32)  # initialize targets
        for idx, example in enumerate(tqdm(examples, desc="Calculating target data", disable=(not self.verbose))):
            targets[idx, :] = self._get_target(example)
        return targets

    def _get_target(self, example: str) -> Union[float, np.ndarray]:
        raise NotImplementedError

    @property
    def target_names(self):
        raise NotImplementedError
