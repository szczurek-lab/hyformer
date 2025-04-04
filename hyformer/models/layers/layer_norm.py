import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square normalization layer.

    Parameters
    ----------
    embedding_dim : int
        The dimension of the embedding.
    eps : float
        The epsilon for numerical stability.
    """

    def __init__(self, embedding_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embedding_dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
