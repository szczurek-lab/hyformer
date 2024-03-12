import torch.nn as nn

from hybrid_transformer.layers.layer_norm import LayerNorm
from hybrid_transformer.layers.hybrid_attention import HybridSelfAttention
from hybrid_transformer.layers.mlp import MLP


class HybridTransformerBlock(nn.Module):

    def __init__(self, embed_dim, bias, dropout, num_heads, block_size):
        super().__init__()
        self.ln_1 = LayerNorm(embed_dim, bias)
        self.attn_1 = HybridSelfAttention(embed_dim, num_heads, bias, dropout, block_size)
        self.ln_2 = LayerNorm(embed_dim, bias)
        self.mlp = MLP(embed_dim, bias, dropout)

    def forward(self, x, task, mask=None):
        x = self.ln_1(x)
        attn, attn_weights = self.attn_1(x=x, mask=mask, task=task)
        x = x + attn
        x = self.ln_2(x)
        x = x + self.mlp(x)
        return x, attn_weights
