import torch.nn as nn

from jointformer.models.layers.layer_norm import RMSNorm
from jointformer.models.layers.self_attention import SelfAttention
from jointformer.models.layers.mlp import FeedForward


class TransformerBlock(nn.Module):

    def __init__(self, embed_dim, bias, dropout, num_heads, block_size, layer_norm_eps):
        super().__init__()
        self.attention_layer = SelfAttention(embed_dim, num_heads, bias, dropout, block_size)
        self.feed_forward = FeedForward(embed_dim, bias, dropout)
        self.attention_layer_normalization = RMSNorm(embed_dim, layer_norm_eps)
        self.feed_forward_normalization = RMSNorm(embed_dim, layer_norm_eps)
        
    def forward(self, x, is_causal, mask=None):
        attn, attn_weights = self.attn_1(x=self.attention_layer_normalization(x), is_causal=is_causal, mask=mask)
        x = x + attn
        x = x + self.feed_forward(self.feed_forward_normalization(x))
        return x, attn_weights
