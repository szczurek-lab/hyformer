import torch
from abc import ABC, abstractmethod
from typing import Optional, Any, Union


class BaseMetric(ABC):

    def __init__(self, reduction: Optional[str] = 'mean', ignore_index: Optional[int] = None):
        self.reduction = reduction
        self.ignore_index = ignore_index

    @abstractmethod
    def __call__(
        self,
        input: Any,
        mask: Optional[Any] = None,
        target: Optional[Any] = None,
        return_detached: Optional[bool] = False,
        ) -> Union[float, torch.Tensor]:
        pass

    @staticmethod
    def _map_masked_logits_to_ignore_value(logits: torch.Tensor, mask: torch.Tensor, ignore_value: Union[int, float]) -> torch.Tensor:
        assert len(logits.shape) == 3, "Logits must be of shape (batch_size, seq_len, vocab_size)"
        assert len(mask.shape) == 2, "Attention mask must be of shape (batch_size, seq_len)"
        mask = mask.unsqueeze(-1).expand_as(logits)
        logits[~mask] = ignore_value
        return logits
    
    def _reduce(self, metric: torch.Tensor) -> torch.Tensor:
        if self.reduction == 'mean':
            return metric.mean()
        elif self.reduction == 'sum':
            return metric.sum()
        elif self.reduction == 'none':
            return metric
        else:
            raise ValueError("Invalid reduction type. Choose from 'mean', 'sum', 'none'")
    
    def _detach(self, x: torch.Tensor) -> float:
        return x.cpu().detach().values()

    def __str__(self) -> str:
        return self.__class__.__name__.lower()
    
    def __repr__(self) -> str:
        return self.__str__()
