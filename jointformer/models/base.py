import logging
import inspect
from torch import nn
import abc
import torch
from guacamol.assess_distribution_learning import DistributionMatchingGenerator

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
    def set_prediction_task(self, task_type: str, num_tasks: int):
        pass
    