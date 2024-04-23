import math
import torch
import warnings

import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, bias, dropout, block_size):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            warnings.warn("The `scaled_dot_product_attention` function is not available in PyTorch < 2.0.0.")
            self.register_buffer("mask_causal", torch.tril(torch.ones(block_size, block_size))
                                 .view(1, 1, block_size, block_size))

    def forward(self, x, is_causal: bool, mask: torch.Tensor):
        """ Parameters
            ----------
            x : torch.Tensor
                The input tensor of shape `(batch_size, seq_len, embed_dim)`.
            is_causal : bool
                If True, the self-attention mechanism is modified to only attend to the left context.
            mask : torch.Tensor
                A boolean mask where a value of True indicates that the element should take part in attention.

            Returns
            -------
            torch.Tensor
                The output tensor of shape `(batch_size, seq_len, embed_dim)`.

            """

        if is_causal and mask is not None:
            raise ValueError("You cannot provide a mask if the attention is causal.")

        att = None
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (embed_dim)

        # self-attention
        q, k, v = self.qkv_proj(x).split(self.embed_dim, dim=2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)

        # Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            dropout = self.dropout if self.training else 0.0
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if is_causal:
                att = att.masked_fill(self.mask_causal[:, :, :T, :T] == 0, float('-inf'))
            elif mask is not None:
                att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = self.attn_dropout(att) @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        y = self.resid_dropout(self.out_proj(y))

        return y, att
