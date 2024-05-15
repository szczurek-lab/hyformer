import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """ LayerNorm with an optional bias. """

    def __init__(self, ndim, bias, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    """ RMSNorm normalizing function. """

    def __init__(self, ndim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight