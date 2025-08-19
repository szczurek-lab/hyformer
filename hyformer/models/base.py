import abc

import numpy as np

from torch import nn
from typing import List


class Encoder(abc.ABC):
    @abc.abstractmethod
    def encode(self, X: list[str]) -> np.ndarray:
        pass

class Generator(abc.ABC):
    @abc.abstractmethod
    def generate(self, number_samples: int) -> List[str]:
        pass


class BaseModel(nn.Module, abc.ABC):
    @abc.abstractmethod
    def to_generator(self, tokenizer, batch_size, temperature, top_k, device) -> Generator:
        pass

    @abc.abstractmethod
    def to_encoder(self, tokenizer, batch_size, device) -> Encoder:
        pass

    @abc.abstractmethod
    def load_pretrained(self, filename, device='cpu'):
        pass

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config):
        pass
