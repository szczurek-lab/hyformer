import abc

import numpy as np


class EncoderWrapper(abc.ABC):
    @abc.abstractmethod
    def encode(self, smiles: list[str]) -> np.ndarray:
        pass
