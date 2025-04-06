import math
import torch
import warnings

import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple

from hyformer.models.layers.rotary import RotaryEmbedding
from hyformer.models.layers.cache import KVCache

class Attention(nn.Module):
    """Multi-head attention layer with rotary positional embeddings.

    Parameters
    ----------
    embedding_dim : int
        The dimension of the embedding.
    num_attention_heads : int
        The number of attention heads.
    attention_dropout_p : float
        The dropout rate for the attention.
    """

    def __init__(
        self, embedding_dim: int, num_attention_heads: int, attention_dropout_p: float
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.head_dim = embedding_dim // num_attention_heads
        self.dropout = attention_dropout_p
        assert hasattr(torch.nn.functional, 'scaled_dot_product_attention'), "Flash attention is not available. Please install PyTorch 2.1 or higher."

        self.q_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.k_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.v_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)        
        self.out = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        
        self.relative_embedding = RotaryEmbedding(self.head_dim) 
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        is_causal: bool,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """ Forward pass of the attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim)
            attention_mask (torch.Tensor): attention_mask tensor of shape (batch_size, seq_len) and type torch.bool
            is_causal (bool): If True, the model is autoregressive and variable `attention_mask` is ignored
            past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]]): Past key and value tensors of shape (batch_size, num_attention_heads, seq_len, head_dim)
            use_cache (bool): If True, the model is autoregressive and variable `past_key_value` is used
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embedding_dim)
            Optional[Tuple[torch.Tensor, torch.Tensor]]: Past key and value tensors of shape (batch_size, num_attention_heads, seq_len, head_dim)
        """
        batch_size, seq_len, embedding_dim = x.shape 
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)  
        v = v.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        q = self.relative_embedding.rotate_queries_or_keys(q, offset=0)
        k = self.relative_embedding.rotate_queries_or_keys(k, offset=0)
        
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)  # concat along seq_len
            v = torch.cat([past_v, v], dim=2)

        present_key_value = (k, v) if use_cache else None
        
        # Expand to (batch_size, num_attention_heads, seq_len, seq_len)
        attention_mask = None if is_causal else attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.num_attention_heads, seq_len, seq_len)
        
        # (batch, num_attention_heads, seq_len, head_dim) → (batch, num_attention_heads, seq_len, head_dim)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, is_causal=is_causal, dropout_p=self.dropout if self.training else 0.)

        # (batch, num_attention_heads, seq_len, head_dim) → (batch, seq_len, embedding_dim)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, embedding_dim)  
        
        # (batch, seq_len, embedding_dim) → (batch, seq_len, embedding_dim)
        y = self.out(y)
        
        return y, present_key_value
    