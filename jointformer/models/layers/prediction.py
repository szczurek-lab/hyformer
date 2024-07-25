import torch
import torch.nn as nn


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
