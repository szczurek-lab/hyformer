import torch
import torch.nn as nn

from jointformer.models.layers.layer_norm import RMSNorm
from jointformer.models.layers.attention import Attention
from jointformer.models.layers.mlp import FeedForward


class TransformerLayer(nn.Module):

    def __init__(self, embedding_dim, embedding_hidden_dim, bias, attention_dropout, feed_forward_dropout, num_heads, block_size, layer_norm_eps):
        super().__init__()
        self.attention_layer = Attention(embedding_dim, num_heads, bias, attention_dropout, block_size)
        self.feed_forward = FeedForward(embedding_dim, embedding_hidden_dim, bias, feed_forward_dropout)
        self.attention_layer_normalization = RMSNorm(embedding_dim, layer_norm_eps)
        self.feed_forward_normalization = RMSNorm(embedding_dim, layer_norm_eps)
        
    def forward(self, x: torch.Tensor, is_causal: bool, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attention_layer(x=self.attention_layer_normalization(x), is_causal=is_causal, attn_mask=mask)
        x = x + self.feed_forward(self.feed_forward_normalization(x))
        return x
