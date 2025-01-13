import torch
import torch.nn as nn

from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm


class DownstreamPredictionHead(nn.Module):
    """https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/dino_head.py """

    def __init__(self, embedding_dim, num_tasks, hidden_dim, pooler_dropout):
        super().__init__()
        self.pooling_layer = nn.Dropout(p=pooler_dropout) if pooler_dropout > 0 else nn.Identity()
        self.linear = nn.Linear(embedding_dim, num_tasks, bias=True)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.linear(self.pooling_layer(x))

class DownstreamPredictionHeadDeep(nn.Module):

    def __init__(self, embedding_dim, num_tasks, hidden_dim, pooler_dropout):
        super().__init__()
        self.dense = nn.Linear(embedding_dim, embedding_dim)
        self.activation_fn = nn.GELU()
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(embedding_dim, num_tasks)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
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
    