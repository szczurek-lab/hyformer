import math
import torch
import warnings

import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from jointformer.models.layers.rotary import RotaryPositionalEmbedding


class Attention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, bias: bool, dropout: float, block_size: int, flash_attention: bool):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.bias = bias
        self.dropout = dropout
        self.block_size = block_size
        self.flash_attention = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and flash_attention
        
        if not self.flash_attention:
            warnings.warn("Using custom attention implementation. This may be slower than the native PyTorch implementation.")
            self.register_buffer(
                "mask_causal", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
                )
            self.attn_dropout = nn.Dropout(dropout)

        # key, query, value projections for all heads, but in a batch
        self.qkv = nn.Linear(self.embedding_dim, 3 * self.embedding_dim, bias=bias)
        self.out = nn.Linear(self.embedding_dim, self.embedding_dim, bias=bias)
        self.relative_embedding = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, is_causal: bool) -> torch.Tensor:
        """ Forward pass of the attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim)
            attn_mask (torch.Tensor): Mask tensor of shape (batch_size, seq_len) and type torch.bool
            is_causal (bool): If True, the model is autoregressive and variable `mask` is ignored
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embedding_dim)
        
        """
        batch_size, seq_len, embedding_dim = x.shape 
        
        q, k, v = self.qkv(x).split(embedding_dim, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        q = self.relative_embedding(q)
        k = self.relative_embedding(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2) 

        attn_mask = None if is_causal else attn_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.num_heads, seq_len, seq_len)
        
        if self.flash_attention:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, 
                is_causal=is_causal, dropout_p=self.dropout if self.training else 0.)
        else:
            y = self.scaled_dot_product_attention(q, k, v, seq_len, attn_mask=attn_mask, is_causal=is_causal)
            
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, embedding_dim)        
        y = self.out(y)
        
        return y
    
    def scaled_dot_product_attention(self, q, k, v, seq_len, attn_mask, is_causal) -> torch.Tensor:
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if is_causal:
            att = att.masked_fill(self.mask_causal[:, :, :seq_len, :seq_len] == 0, float('-inf'))
        else:
            att = att.masked_fill(attn_mask.int() == 0, float('-inf'))
        att_probs = F.softmax(att, dim=-1)
        y = self.attn_dropout(att_probs) @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        return y
    