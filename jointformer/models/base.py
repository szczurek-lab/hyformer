import abc
import inspect
import logging
import torch

from guacamol.assess_distribution_learning import DistributionMatchingGenerator
from torch import nn
from typing import Optional

console = logging.getLogger(__name__)


class BaseModel(nn.Module, abc.ABC):

    @abc.abstractmethod
    def to_guacamole_generator(self, tokenizer, batch_size, temperature, top_k, device) -> DistributionMatchingGenerator:
        pass

    @abc.abstractmethod
    def load_pretrained(self, filename, device='cpu'):
        pass
    
    @abc.abstractmethod
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        pass
    
    @abc.abstractmethod
    def initialize_parameters(self):
        pass

    @abc.abstractmethod
    def get_num_params(self, non_embedding=True):
        pass

    @abc.abstractmethod
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        pass
    
    @abc.abstractmethod
    def get_loss(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            properties: Optional[torch.Tensor] = None,
            task: Optional[str] = None):
        pass
    
    @abc.abstractmethod
    def predict(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None
           ):
        pass

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config):
        pass
