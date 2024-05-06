import torch

import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from jointformer.models.transformer import Transformer
from jointformer.utils.tokenizers.smiles.smiles import IGNORE_INDEX


class GPT(Transformer):

    def __init__(
            self, vocab_size: int, max_seq_len: int, embedding_dim: int,
            dropout: float, num_layers: int, bias: int, num_heads: int):
        super().__init__(
            vocab_size=vocab_size, max_seq_len=max_seq_len, embedding_dim=embedding_dim,
            dropout=dropout, num_layers=num_layers, bias=bias, num_heads=num_heads)

        self.lm_head = nn.Linear(self.embedding_dim, self.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

        self.prediction_head = nn.Linear(self.embedding_dim, self.prediction_head_output_dim, bias=False)
        self.output_dropout = nn.Dropout(self.dropout)
        self.initialize_parameters()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                **kwargs):

        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, task='lm')
        outputs["loss"] = None

        if labels is not None:
            outputs["logits"] = self.lm_head(outputs['embeddings'])
            input = outputs['logits'][:, :-1, :].contiguous()
            target = labels[:, 1:].contiguous()
            outputs["loss"] = F.cross_entropy(
                input.view(-1, input.size(-1)), target.view(-1),
                size_average=True, ignore_index=IGNORE_INDEX, reduction='mean')
        else:
            outputs["logits"] = self.lm_head(outputs["embeddings"][:, [-1], :])

        return outputs

    def get_perplexity(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            *args,
            **kwargs
    ):
        """ Compute perplexity of the model on the input sequence.

        Reference: https://github.com/ETHmodlab/CLM_perplexity/blob/main/src/python/helper.py
        """
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, task='lm')
        outputs["logits"] = self.lm_head(outputs['embeddings'])
        log_probs = F.log_softmax(outputs["logits"], dim=-1).max(dim=-1).values
        perplexity = torch.zeros(size=(input_ids.size(0),))

        for idx, (log_prob, mask) in enumerate(zip(log_probs, attention_mask)):
            log_prob = log_prob[mask.bool()]
            log_prob = log_prob[:-2]  # Ignore the last two tokens, as they correspond to special tokens in generation
            log_prob = 2 ** (-log_prob.sum() / log_prob.size(0))
            perplexity[idx] = log_prob
        return perplexity

    def generate(self, bos_token_id, eos_token_id, pad_token_id, input_length, batch_size, temperature=1.0, top_k=None, device='cpu'):
        """
        Generate complete sequences of indices using the model.
        """

        idx = torch.ones(size=(batch_size, 1), dtype=torch.long, device=device) * bos_token_id
        idx = self.generate_single_tokens(idx, input_length - 1, temperature, top_k, eos_token_id, pad_token_id)

        # check for completion
        for sequence_idx, sequence in enumerate(idx):
            if eos_token_id not in sequence:
                idx[sequence_idx, -1] = eos_token_id
        return idx


    @torch.no_grad()
    def generate_single_tokens(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_token_id=None, pad_token_id=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        if eos_token_id is None:
            assert pad_token_id is not None, "If eos_token_id is not None, pad_token_id must be provided."
        if pad_token_id is None:
            assert eos_token_id is not None, "If pad_token_id is not None, eos_token_id must be provided."

        eos_flag = torch.zeros(size=(idx.size(0), 1), dtype=torch.bool, device=idx.device)

        for _ in range(max_new_tokens):
            if eos_token_id:
                is_end = torch.logical_or(idx[:, [-1]] == eos_token_id, idx[:, [-1]] == pad_token_id)
                eos_flag = torch.logical_or(eos_flag, is_end)
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            outputs = self(input_ids=idx_cond, attention_mask=None, task='lm')
            logits = outputs['logits']

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
            idx_next = torch.where(eos_flag, torch.ones_like(idx_next) * pad_token_id, idx_next)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @classmethod
    def from_config(cls, config):
        return cls(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            num_layers=config.num_layers,
            bias=config.bias,
            num_heads=config.num_heads
        )