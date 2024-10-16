import torch
import torch.nn as nn

from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm


class DownstreamPredictionHead(nn.Module):
    """https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/dino_head.py """

    def __init__(self, embedding_dim, num_tasks, hidden_dim):
        super().__init__()
        self.mlp = nn.Linear(embedding_dim, hidden_dim)
        self.apply(self._init_weights)
        self.last_layer = weight_norm(nn.Linear(hidden_dim, num_tasks, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)
        return x        
    

class RegressionHead(nn.Module):

    def __init__(self, embedding_dim: int, prediction_hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, prediction_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(prediction_hidden_dim),
            nn.Linear(prediction_hidden_dim, output_dim))

    def forward(self, x):
        return self.net(x)


class ClassificationHead(nn.Module):

    def __init__(self, embedding_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        return self.net(x)
    