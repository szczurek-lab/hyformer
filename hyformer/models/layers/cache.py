"""KV cache for the attention layer. """

import torch
import torch.nn as nn


class KVCache(nn.Module):
    def __init__(self, batch_size, max_seq_length, num_heads, head_dim, dtype, device):
        super().__init__()
        cache_shape = (batch_size, num_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype, device=device), persistent=False)
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype, device=device), persistent=False)
        self.current_seq_len = 0
    
    def __len__(self):
        return self.current_seq_len
    
    def get(self):
        return self.k_cache[:, :, :self.current_seq_len], self.v_cache[:, :, :self.current_seq_len]
    
    def update(self, k_val, v_val):
        _seq_len = k_val.shape[2]
        assert _seq_len == v_val.shape[2], "k_val and v_val must have the same sequence length"
        _new_seq_len = self.current_seq_len + _seq_len
        assert _new_seq_len <= self.k_cache.shape[2], f"Cache is too small. Current sequence length: {_new_seq_len}, Cache size: {self.k_cache.shape[2]}"
        self.k_cache[:, :, self.current_seq_len:self.current_seq_len + _seq_len] = k_val
        self.v_cache[:, :, self.current_seq_len:self.current_seq_len + _seq_len] = v_val
        self.current_seq_len = _new_seq_len
        return self.get()
    
    def clear(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.current_seq_len = 0
    