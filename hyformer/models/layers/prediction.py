""" Prediction head for downstream and physichochemical tasks. """

import torch.nn as nn
from torch.nn.init import trunc_normal_


class DownstreamPredictionHead(nn.Module):

    def __init__(self, embedding_dim, num_tasks, pooler_dropout):
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
