from torch import nn
import abc
from guacamol.assess_distribution_learning import DistributionMatchingGenerator
import numpy as np


class SmilesEncoder(abc.ABC):
    @abc.abstractmethod
    def encode(self, smiles: list[str]) -> np.ndarray:
        pass

class BaseModel(nn.Module, abc.ABC):
    @abc.abstractmethod
    def to_guacamole_generator(self, tokenizer, batch_size, temperature, top_k, device) -> DistributionMatchingGenerator:
        pass

    @abc.abstractmethod
    def to_smiles_encoder(self, tokenizer, batch_size, device) -> SmilesEncoder:
        pass

    @abc.abstractmethod
    def load_pretrained(self, filename, device='cpu'):
        pass

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config):
        pass

