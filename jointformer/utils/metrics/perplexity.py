import torch

from torch.nn import functional as F
from typing import Optional, Optional, Union

from jointformer.utils.metrics.base import BaseMetric


class Perplexity(BaseMetric):

    def __init__(self, reduction: Optional[str] = 'mean'):
        super().__init__(reduction=reduction, ignore_index=None)

    def __call__(self, input: torch.Tensor, mask: torch.Tensor, return_detached: Optional[bool] = False, base: Optional[Union[float, str]] = 'exp') -> Union[float, torch.Tensor]:
        """ Calculates the perplexity of a sequence.
        
        Args:
            input: Logits of the sequence of shape (batch_size, seq_len, vocab_size).
            mask: Attention mask of the sequence of shape (batch_size, seq_len). Can pass None if no mask is required.
            base: Base of the exponentiation used to calculate the perplexity. Can be 'exp' or a float or int.
        """
        if mask is not None:
            input = self._map_masked_logits_to_ignore_value(input, mask, float('nan'))
        perplexity = self._calculate_perplexity(input, base)
        perplexity = self._reduce(perplexity)
        if return_detached:
             return self._detach(perplexity)
        else:
             return perplexity        

    @staticmethod
    def _calculate_perplexity(logits: torch.Tensor, base: Union[float, str]) -> torch.Tensor:
            """ Calculates the perplexity of a sequence according to: https://github.com/ETHmodlab/CLM_perplexity/blob/main/src/python/helper.py
            """
            log_probs = F.log_softmax(logits, dim=-1).max(dim=-1).values
            if base == 'exp':
                return torch.exp(-log_probs.nanmean(dim=-1))
            elif isinstance(base, float) or isinstance(base, int):
                return base ** (-log_probs.nanmean(dim=-1))
            else:
                raise ValueError("Invalid base type. Choose from 'exp' or float or int.")
            