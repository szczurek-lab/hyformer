import abc
import torch

from typing import Optional

from jointformer.models.base import BaseModel


class TrainableModel(BaseModel, abc.ABC):

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def get_num_params(self):
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
    