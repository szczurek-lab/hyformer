import torch
import torch.nn as nn
from torch.nn import functional as F


class FeedForward(nn.Module):

    def __init__(self, embedding_dim: int, embedding_hidden_dim: int, bias: bool, *args, **kwargs):
        super().__init__()
        self.w1 = nn.Linear(embedding_dim, embedding_hidden_dim, bias=False)
        self.w3 = nn.Linear(embedding_dim, embedding_hidden_dim, bias=False)
        self.w2 = nn.Linear(embedding_hidden_dim, embedding_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
