import torch.nn as nn

from jointformer.models.layers.layer_norm import LayerNorm
from jointformer.models.layers.self_attention import SelfAttention
from jointformer.models.layers.mlp import MLP


class TransformerBlock(nn.Module):

    def __init__(self, embed_dim, bias, dropout, num_heads, block_size, layer_norm_eps):
        super().__init__()
        self.ln_1 = LayerNorm(embed_dim, bias, layer_norm_eps)
        self.attn_1 = SelfAttention(embed_dim, num_heads, bias, dropout, block_size)
        self.ln_2 = LayerNorm(embed_dim, bias, layer_norm_eps)
        self.mlp = MLP(embed_dim, bias, dropout)

    def forward(self, x, is_causal, mask=None):
        x = self.ln_1(x)
        attn, attn_weights = self.attn_1(x=x, is_causal=is_causal, mask=mask)
        x = x + attn
        x = self.ln_2(x)
        x = x + self.mlp(x)
        return x, attn_weights
