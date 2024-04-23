import torch
import math
import inspect
import torch.nn as nn
import torch.nn.functional as F

from jointformer.models.base import Transformer

from jointformer.utils.tokenizers.smiles.smiles import IGNORE_INDEX

from jointformer.utils.optimization import AdamW


class Jointformer(Transformer):

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

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.num_layers))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def forward(
            self, input_ids, task='lm', labels=None, target=None,
            eos_mask=None, attention_mask=None, return_attention=False):

        outputs = super.__call__(input_ids=input_ids, attention_mask=attention_mask, return_attention=return_attention)

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
