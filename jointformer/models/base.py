import torch
import math
import inspect
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from jointformer.models.layers.layer_norm import LayerNorm
from jointformer.models.layers.transformer import TransformerBlock

from jointformer.utils.tokenizers.smiles.smiles import IGNORE_INDEX

from jointformer.utils.optimization import AdamW


class Transformer(nn.Module):

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
        self._post_init()

    def _post_init(self):
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            task: str = 'lm',
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

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, correct_bias):
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
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        if correct_bias:
            optimizer = AdamW(optim_groups, lr=learning_rate, betas=betas, correct_bias=correct_bias, fused=use_fused)
        else:
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        # todo: add padding after eos token
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            outputs = self(input_ids=idx_cond, task='lm')
            logits = outputs['lm_logits']
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
