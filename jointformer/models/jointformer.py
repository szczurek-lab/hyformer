from guacamol.assess_distribution_learning import DistributionMatchingGenerator
from jointformer.models.base import SmilesEncoder
from jointformer.models.defaults import DefaultSmilesEncoderWrapper
import torch

import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from jointformer.models.transformer import Transformer
from jointformer.utils.tokenizers.base import TOKEN_DICT
from jointformer.models.wrappers import DefaultSmilesGeneratorWrapper
from jointformer.models.trainable import TrainableModel
from jointformer.models.layers.prediction import RegressionHead, ClassificationHead
from jointformer.models.utils import ModelOutput
from jointformer.models.wrappers import DefaultSmilesEncoderWrapper


DEFAULT_NUM_PHYCHEM_TASKS = 200


class Jointformer(Transformer, TrainableModel):

    def __init__(
            self,
            vocab_size: int,
            max_seq_len: int,
            embedding_dim: int,
            embedding_hidden_dim: int,
            attention_dropout: float,
            feed_forward_dropout: float,
            num_layers: int,
            bias: int,
            num_heads: int,
            layer_norm_eps: float,
            prediction_task_type: str,
            prediction_hidden_dim: int,
            num_prediction_tasks: int,
            num_physchem_tasks: Optional[int] = DEFAULT_NUM_PHYCHEM_TASKS,
            init_weights: bool = True,
            tie_weights: bool = True,
            set_separate_task_tokens = False,
    ):

        super().__init__(
            vocab_size=vocab_size, max_seq_len=max_seq_len, embedding_dim=embedding_dim, embedding_hidden_dim=embedding_hidden_dim, attention_dropout=attention_dropout,
            feed_forward_dropout=feed_forward_dropout, num_layers=num_layers, bias=bias, num_heads=num_heads, layer_norm_eps=layer_norm_eps
            )
        
        # Hardcoding all tasks into the model definition for easier serialization
        self.prediction_task_type = prediction_task_type
        self.lm_head = nn.Linear(self.embedding_dim, self.vocab_size, bias=False)
        self.mlm_head = nn.Linear(self.embedding_dim, self.vocab_size, bias=False)
        self.physchem_head = RegressionHead(embedding_dim=self.embedding_dim, prediction_hidden_dim=prediction_hidden_dim, output_dim=num_physchem_tasks)

        # Init prediction head depending on task type
        if prediction_task_type == 'classification':
            self.prediction_head = ClassificationHead(embedding_dim=self.embedding_dim, output_dim=2 if num_prediction_tasks == 1 else num_prediction_tasks) # binary or multiclass classification
        elif prediction_task_type == 'regression':
            self.prediction_head = RegressionHead(embedding_dim=self.embedding_dim, prediction_hidden_dim=prediction_hidden_dim, output_dim=num_prediction_tasks)
        else:
            raise ValueError('Variable `prediction_task_type` must be either `classification` or `regression`.')
        
        # Weight tying https://paperswithcode.com/method/weight-tying
        if tie_weights:
            self.token_embedding.weight = self.lm_head.weight
            self.mlm_head.weight = self.lm_head.weight

        # Weight initialization
        if init_weights:
            self.initialize_parameters()

    def forward(
            self,
            input_ids: torch.Tensor,
            task: str,
            attention_mask: torch.Tensor,
            next_token_only: Optional[bool] = False,
            **kwargs
            ):
        
        if task == 'generation':
            _is_causal = True
            _attention_mask = None
        elif task in ['physchem', 'prediction', 'mlm']:
            _is_causal = False
            _attention_mask = attention_mask
        else:
            raise ValueError('Variable `task` must be either `generation`, `mlm`, `prediction` or `physchem`. Passed value: {}'.format(task))
        
        outputs = super().forward(input_ids=input_ids, attention_mask=_attention_mask, is_causal=_is_causal)
        cls_embeddings = outputs['embeddings'][:, 0]
        lm_embeddings = outputs['embeddings'][:, [-1]] if next_token_only else outputs['embeddings']
        if _is_causal:
            outputs["logits_generation"] = self.lm_head(lm_embeddings)
        else:
            outputs["logits_physchem"] = self.physchem_head(cls_embeddings)
            outputs["logits_prediction"] = self.prediction_head(cls_embeddings)

        return ModelOutput(
            attention_mask=attention_mask,
            embeddings=outputs['embeddings'],
            cls_embeddings=cls_embeddings,
            lm_embeddings=lm_embeddings[:, 1:],
            logits_generation=outputs.get('logits_generation', None),
            logits_physchem=outputs.get('logits_physchem', None),
            logits_prediction=outputs.get('logits_prediction', None),
            loss=None
        )
    
    def predict(self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        return self.forward(input_ids=input_ids, attention_mask=attention_mask, task='prediction')

    def get_loss(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            task: str,
            input_labels: Optional[torch.Tensor] = None,
            properties: Optional[torch.Tensor] = None
            ):
        if task == 'lm' or task == 'generation':
            return self.get_loss_lm(input_ids, attention_mask, input_labels)
        elif task == 'mlm':
            return self.get_loss_mlm(input_ids, attention_mask, input_labels)
        elif task == 'prediction':
            return self.get_loss_prediction(input_ids, attention_mask, properties)
        elif task == 'physchem':
            return self.get_loss_physchem(input_ids, attention_mask, properties)
        else:
            raise ValueError('Variable `task` must be either `lm`, `mlm`, `prediction` or `finetune`.')

    def get_loss_lm(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, input_labels: torch.Tensor, **kwargs):
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, task='generation', next_token_only=False)
        if input_labels is not None: 
            logits = outputs['logits_generation'][:, :-1, :].contiguous()
            labels = input_labels[:, 1:].contiguous()
            batch_size, seq_length, vocab_size = logits.size()
            outputs["loss"] = F.cross_entropy(
                logits.view(batch_size * seq_length, vocab_size),
                labels.view(batch_size * seq_length),
                ignore_index=TOKEN_DICT['ignore'],
                reduction='mean'
                )
        return outputs

    def get_loss_mlm(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, input_labels: torch.Tensor, **kwargs):
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, task='mlm')
        outputs["logits_generation"] = self.mlm_head(outputs['embeddings'])
        if input_labels is not None:
            if self.set_separate_task_tokens:
                logits = outputs['logits_generation'][:, 1:, :].contiguous()
                labels = input_labels[:, 1:].contiguous() 
            else:
                logits = outputs['logits_generation']
                labels = input_labels
            batch_size, seq_length, vocab_size = logits.size()
            outputs["loss"] = F.cross_entropy(
                logits.view(batch_size * seq_length, vocab_size),
                labels.view(batch_size * seq_length),
                ignore_index=TOKEN_DICT['ignore'],
                reduction='mean'
                )
        return outputs

    def get_loss_physchem(self, input_ids: torch.Tensor, attention_mask:  torch.Tensor, properties: torch.Tensor, **kwargs):
        outputs = self.predict(input_ids=input_ids, attention_mask=attention_mask)
        if properties is not None:
            outputs["loss"] = F.mse_loss(outputs["logits_physchem"].flatten(), properties.flatten(), reduction='mean')
        return outputs

    def get_loss_prediction(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, properties: torch.Tensor, **kwargs):
        outputs = self.predict(input_ids=input_ids, attention_mask=attention_mask)
        if properties is not None:
            if self.prediction_task_type == 'classification':
                outputs["loss"] = F.cross_entropy(outputs["logits_prediction"], properties, reduction='mean')
            elif self.prediction_task_type == 'regression':
                outputs["loss"] = F.mse_loss(outputs["logits_prediction"].flatten(), properties.flatten(), 'mean')
            else:
                raise ValueError('Variable `prediction_task_type` must be either `classification` or `regression`.')
        return outputs

    def generate(self, bos_token_id, eos_token_id, pad_token_id, input_length, batch_size, temperature=1.0, top_k=None, device='cpu', generate_token_id=None):
        """
        Generate complete sequences of indices using the model.
        """
        idx = torch.full((batch_size, 1), bos_token_id, device=device, dtype=torch.long)
        if self.set_separate_task_tokens:
            assert generate_token_id is not None, "If set_separate_task_tokens is True, task_token_id must be provided."
            idx_task_token = torch.full((batch_size, 1), generate_token_id, device=device, dtype=torch.long)  
            idx = torch.cat([idx_task_token, idx], dim=1)

        # TODO: implement caching
        idx = self.generate_single_token(idx, input_length - 1, temperature, top_k, eos_token_id, pad_token_id)

        # TODO: vectorize
        # check for completion
        for sequence_idx, sequence in enumerate(idx):
            if eos_token_id not in sequence:
                idx[sequence_idx, -1] = eos_token_id
        return idx

    @torch.no_grad()
    def generate_single_token(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_token_id=None, pad_token_id=None):
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
            outputs = self(input_ids=idx_cond, attention_mask=None, next_token_only=True, task='generation')
            logits = outputs['logits_generation']

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

    def to_guacamole_generator(self, tokenizer, batch_size, temperature, top_k, device) -> DistributionMatchingGenerator:
        return DefaultSmilesGeneratorWrapper(self, tokenizer, batch_size, temperature, top_k, device)

    @classmethod
    def from_config(cls, config):
        return cls(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            embedding_dim=config.embedding_dim,
            embedding_hidden_dim=config.embedding_hidden_dim,
            attention_dropout=config.attention_dropout,
            feed_forward_dropout=config.feed_forward_dropout,
            num_layers=config.num_layers,
            bias=config.bias,
            num_heads=config.num_heads,
            prediction_task_type=config.prediction_task_type,
            prediction_hidden_dim=config.prediction_hidden_dim,
            num_prediction_tasks=config.num_prediction_tasks,
            num_physchem_tasks=config.num_physchem_tasks,
            layer_norm_eps=config.layer_norm_eps
        )

    def to_guacamole_generator(self, tokenizer, batch_size, temperature, top_k, device) -> DistributionMatchingGenerator:
        return DefaultSmilesGeneratorWrapper(self, tokenizer, batch_size, temperature, top_k, device)
    
    def to_smiles_encoder(self, tokenizer, batch_size, device) -> SmilesEncoder:
        return DefaultSmilesEncoderWrapper(self, tokenizer, batch_size, device)
