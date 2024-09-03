import inspect
import math
import torch

import torch.nn as nn

from typing import Optional

from jointformer.models.layers.layer_norm import RMSNorm
from jointformer.models.layers.transformer import TransformerLayer
from jointformer.models.utils import ModelOutput


class Transformer(nn.Module):

    def __init__(
            self, vocab_size: int, max_seq_len: int, embedding_dim: int, embedding_hidden_dim: int, attention_dropout: float,
            feed_forward_dropout: float, num_layers: int, bias: int, num_heads: int, layer_norm_eps: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.embedding_hidden_dim = embedding_hidden_dim
        self.attention_dropout = attention_dropout
        self.feed_forward_dropout = feed_forward_dropout
        self.num_layers = num_layers
        self.bias = bias
        self.num_heads = num_heads
        self.layer_norm_eps = layer_norm_eps

        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.layers = nn.ModuleList([
            TransformerLayer(
                self.embedding_dim, self.embedding_hidden_dim, self.bias, self.attention_dropout,
                self.feed_forward_dropout, self.num_heads, self.max_seq_len, self.layer_norm_eps
                )
              for _ in range(self.num_layers)])
        self.layer_norm = RMSNorm(self.embedding_dim, self.layer_norm_eps)

    def forward(
            self,
            input_ids: torch.Tensor,
            is_causal: bool,
            attention_mask: torch.Tensor,
            **kwargs
    ):
        #assert False, (self.token_embedding, input_ids)
        x = self.token_embedding(input_ids)
        for _, layer in enumerate(self.layers):
            x = layer(x, is_causal=is_causal, mask=attention_mask)
        x = self.layer_norm(x)
        return ModelOutput(embeddings=x, attention_mask=attention_mask)

    def load_pretrained(self, filename, device='cpu'):
        state_dict = torch.load(filename, map_location=device, weights_only=True)['model']
        unwanted_prefix = '_orig_mod.'  # compile
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        self.load_state_dict(state_dict, strict=False)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def initialize_parameters(self):
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer
