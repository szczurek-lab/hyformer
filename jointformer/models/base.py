from torch import nn
import abc
from guacamol.assess_distribution_learning import DistributionMatchingGenerator

class BaseModel(nn.Module, abc.ABC):

    @abc.abstractmethod
    def to_guacamole_generator(self, tokenizer, batch_size, temperature, top_k, device) -> DistributionMatchingGenerator:
        pass

    @abc.abstractmethod
    def load_pretrained(self, filename, device='cpu'):
        pass
