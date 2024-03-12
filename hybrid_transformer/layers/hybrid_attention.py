import math
import torch

import torch.nn as nn
import torch.nn.functional as F


class HybridSelfAttention(nn.Module):
    """Hybrid Self-Attention switching between a `fully-visible` and a `causal` masking patterns."""

    def __init__(self, embed_dim, num_heads, bias, dropout, block_size):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.register_buffer("mask_causal", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))

    def forward(self, x, task, mask=None):
        """ Two types of masks are supported. A boolean mask where a value of True indicates that
            the element should take part in attention. A float mask of the same type as query, key, value
            that is added to the attention score. """

        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (embed_dim)
        q = self.q_proj(x)

        # self-attention
        k, v = self.kv_proj(x).split(self.embed_dim, dim=2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)

        # Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if task == 'lm':
            att = att.masked_fill(self.mask_causal[:, :, :T, :T] == 0, float('-inf'))
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))

        att_probs = F.softmax(att, dim=-1)
        y = self.attn_dropout(att_probs) @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        y = self.resid_dropout(self.out_proj(y))

        return y, att_probs
