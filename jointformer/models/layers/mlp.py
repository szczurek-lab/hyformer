import torch
import torch.nn as nn
from torch.nn import functional as F


class MLP(nn.Module):

    def __init__(self, embed_dim, bias, dropout):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 4 * embed_dim, bias=bias)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(4 * embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.gelu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):

    def __init__(self, embed_dim, bias, dropout):
        super().__init__()
        intermediate_size = find_multiple(int(8 * embed_dim / 3), 256)
        self.w1 = nn.Linear(embed_dim, intermediate_size, bias=False)
        self.w3 = nn.Linear(embed_dim, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)
