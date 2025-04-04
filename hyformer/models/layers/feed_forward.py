import torch
import torch.nn as nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation.

    Parameters
    ----------
    embedding_dim : int
        The dimension of the embedding.
    hidden_embedding_dim : int
        The dimension of the hidden embedding.
    """

    def __init__(self, embedding_dim: int, hidden_embedding_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(embedding_dim, hidden_embedding_dim, bias=False)
        self.w3 = nn.Linear(embedding_dim, hidden_embedding_dim, bias=False)
        self.w2 = nn.Linear(hidden_embedding_dim, embedding_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
