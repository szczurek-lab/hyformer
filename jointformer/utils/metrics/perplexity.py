import torch

from torch.nn import functional as F
from typing import Optional, Any

from jointformer.utils.metrics.base import BaseMetric


class Perplexity(BaseMetric):

    def __init__(self, reduction: Optional[str] = 'mean'):
        super().__init__(reduction=reduction, ignore_index=None)

    def __call__(self, input: torch.Tensor, mask: torch.Tensor) -> float:
        """ Calculates the perplexity of a sequence.
        
        Args:
            input: Logits of the sequence of shape (batch_size, seq_len, vocab_size).
            mask: Attention mask of the sequence of shape (batch_size, seq_len). Can pass None if no mask is required.
            reduction: 'mean' or 'sum' or 'none'.
        """
        if mask is not None:
            input = self._map_masked_logits_to_ignore_value(input, mask, float('nan'))
        perplexity = self._calculate_perplexity(input)
        return self._reduce(perplexity)
            
    @staticmethod
    def _calculate_perplexity(logits: torch.Tensor, ignore_index: Optional[int] = None) -> torch.Tensor:
            """ Calculates the perplexity of a sequence according to: https://github.com/ETHmodlab/CLM_perplexity/blob/main/src/python/helper.py
            """
            log_probs = F.log_softmax(logits, dim=-1).max(dim=-1).values
            perplexity = 2 ** (-log_probs.nanmean(dim=-1))
            return perplexity
