import torch
import math
import inspect
import torch.nn as nn
import torch.nn.functional as F

from hybrid_transformer.configs.model import ModelConfig
from hybrid_transformer.layers.layer_norm import LayerNorm
from hybrid_transformer.layers.transformer import HybridTransformerBlock

from hybrid_transformer.utils.tokenizers.smiles import IGNORE_INDEX

from hybrid_transformer.utils.optimization import AdamW


class Transformer(nn.Module):

    def __init__(
            self, vocab_size: int, max_seq_len: int, embedding_dim: int,
            dropout: float, num_layers: int, bias: int, num_heads: int, task_p: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.bias = bias
        self.num_heads = num_heads
        self.prediction_head_output_dim = 1
        self.task_p = task_p

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.vocab_size, self.embedding_dim),
            wpe = nn.Embedding(self.max_seq_len, self.embedding_dim),
            drop = nn.Dropout(self.dropout),
            h = nn.ModuleList([HybridTransformerBlock(
                self.embedding_dim, self.bias, self.dropout, self.num_heads,
                self.max_seq_len) for _ in range(self.num_layers)]),
            ln_f = LayerNorm(self.embedding_dim, bias=self.bias),
        ))
        self.lm_head = nn.Linear(self.embedding_dim, self.vocab_size, bias=False)
        self.prediction_head = nn.Linear(self.embedding_dim, self.prediction_head_output_dim, bias=False)
        self.output_dropout = nn.Dropout(self.dropout)
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.num_layers))

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
            self, input_ids, task='lm', labels=None, target=None,
            eos_mask=None, attention_mask=None, return_attention=False):

        b, t = input_ids.size()
        assert t <= self.max_seq_len, f"Cannot forward sequence of length {t}, max_seq_length is only {self.max_seq_len}"
        pos = torch.arange(0, t, dtype=torch.long, device=input_ids.device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(input_ids)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        attention_probs = []
        for block in self.transformer.h:
            x, attn = block(x, task=task, mask=attention_mask)
            attention_probs.append(attn.detach().cpu())
        x = self.transformer.ln_f(x)

        if labels is not None:

            lm_logits = self.lm_head(x)

            if task == 'lm':
                shift_logits = lm_logits[..., :-1, :].contiguous()
            if task == 'mlm':
                shift_logits = lm_logits[..., 1:, :].contiguous()

            shift_labels = labels[..., 1:].contiguous()

            unsupervised_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX)

        else:
            lm_logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            unsupervised_loss = None

        if target is not None:  # p(y | x)
            if task == 'lm':
                # OLD: y_pred = self.prediction_head(x[:, -1, :]).flatten()
                y_pred = self.prediction_head(self.output_dropout(x[:, -1, :])).flatten()

            if task == 'mlm':
                # OLD: y_pred = self.prediction_head(x[:, 0, :]).flatten()
                y_pred = self.prediction_head(self.output_dropout(x[:, 0, :])).flatten()

            supervised_loss = F.mse_loss(y_pred, target)

        else:
            supervised_loss = None
            y_pred = None

        return {
            'loss': None,
            'unsupervised_loss': unsupervised_loss,
            'supervised_loss': supervised_loss,
            'lm_logits': lm_logits,
            'prediction': y_pred,
            'embedding': x,
            'attention_probabilities': torch.stack(attention_probs)
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

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
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

    def get_lm_loss(self, **kwargs):
        kwargs['task'] = 'lm'
        outputs = super().forward(**kwargs)
        return outputs['unsupervised_loss']

    def get_mlm_loss(self, **kwargs):
        kwargs['task'] = 'mlm'
        outputs = super().forward(**kwargs)
        return outputs['unsupervised_loss']

    def get_prediction_loss(self, **kwargs):
        kwargs['task'] = 'mlm'
        outputs = super().forward(**kwargs)
        return outputs['supervised_loss']
