import torch

import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from jointformer.models.base import Model
from jointformer.models.layers.layer_norm import LayerNorm
from jointformer.models.layers.transformer import TransformerBlock


class Transformer(Model):

    def __init__(
            self, vocab_size: int, max_seq_len: int, embedding_dim: int,
            dropout: float, num_layers: int, bias: int, num_heads: int,):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.bias = bias
        self.num_heads = num_heads
        self.prediction_head_output_dim = 1

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(self.vocab_size, self.embedding_dim),
            wpe  = nn.Embedding(self.max_seq_len, self.embedding_dim),
            drop = nn.Dropout(self.dropout),
            h    = nn.ModuleList(
                [TransformerBlock(
                    self.embedding_dim, self.bias, self.dropout, self.num_heads,
                    self.max_seq_len) for _ in range(self.num_layers)]),
            ln_f = LayerNorm(self.embedding_dim, bias=self.bias),
        ))
        self.initialize_parameters()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            task: str = 'lm'
    ):

        if task == 'lm':
            is_causal = True
            mask = None
        elif task in ['mlm', 'ae', 'predict']:
            is_causal = False
            mask = attention_mask
        else:
            raise ValueError(f"task must be 'lm' or 'mlm', got {task}")

        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.max_seq_len, f"Cannot forward sequence of length {t}, max_seq_length is only {self.max_seq_len}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # Forward
        tok_emb = self.transformer.wte(input_ids)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        attention_probs = []
        for block in self.transformer.h:
            x, attn = block(x, is_causal=is_causal, mask=mask)
            attn = attn.detach().cpu() if attn is not None else None
            attention_probs.append(attn)
        x = self.transformer.ln_f(x)

        return {
            'embeddings': x,
            'attention_probabilities': torch.stack(attention_probs) if attention_probs[0] is not None else None,
        }

    def load_pretrained(self, filename, device='cpu'):
        state_dict = torch.load(filename, map_location=device)['model']
        unwanted_prefix = '_orig_mod.'  # compile
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        self.load_state_dict(state_dict)
