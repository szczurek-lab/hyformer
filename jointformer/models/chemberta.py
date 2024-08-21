"""Source: https://github.com/seyonechithrananda/bert-loves-chemistry/blob/master/chemberta/utils/roberta_regression.py#L138"""


import numpy as np

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import PreTrainedModel, RobertaModel
from transformers.file_utils import ModelOutput
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

from jointformer.configs.model import ModelConfig
from transformers import RobertaConfig
from jointformer.models.base import BaseModel

import torch 
import logging
import inspect

from jointformer.models.utils import ModelOutput

from typing import Optional
from torch import nn

from jointformer.models.trainable import TrainableModel

from jointformer.configs.model import ModelConfig

from jointformer.models.base import BaseModel, SmilesEncoder

console = logging.getLogger(__name__)


class ChemBERTa(RobertaPreTrainedModel, BaseModel):

    def __init__(self, config: RobertaConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
    
    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            **kwargs
            ):
        
        outputs = self.roberta(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True,
            token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=True
        )

        return ModelOutput(
            attention_mask=attention_mask,
            embeddings=outputs.hidden_states[-1],
            cls_embeddings=outputs.last_hidden_state[:, 0, :]  # take <s> token (equiv. to [CLS])
        )
    
    def get_loss(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        properties: Optional[torch.Tensor] = None,
        **kwargs
    ):
        return self.forward(input_ids=input_ids, attention_mask=attention_mask, properties=properties)

    def to_guacamole_generator(self, tokenizer, batch_size, temperature, top_k, device) -> NotImplementedError:
        raise NotImplementedError("ChemBERTa is not a valid generative model.")
    
    def to_smiles_encoder(self, tokenizer, batch_size, device) -> SmilesEncoder:
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._device = device
        self.eval()
        self.to(self._device)
        return self
    
    @torch.no_grad
    def encode(self, smiles: list[str]) -> np.ndarray:
        """Source: https://github.com/valence-labs/mood-experiments/blob/30650c24f8518a05acd574664eec94bfaa04c047/mood/representations.py#L236 """
        hidden_states = []
        for i in tqdm(range(0, len(smiles), self._batch_size), "Encoding samples"):
            batch = smiles[i:i+self._batch_size]
            inputs = self._tokenizer(batch, task='prediction')
            inputs.to(self._device)
            outputs = self(**inputs)
            h = [h_[mask].mean(0) for h_, mask in zip(outputs['embeddings'], inputs['attention_mask'])]
            h = torch.stack(h)
            h = h.cpu().numpy()
            hidden_states.append(h)
        hidden_states = np.concatenate(hidden_states, axis=0)
        return hidden_states
    
    def load_pretrained(self, filename: str, from_hf: bool = False, map_location: str = 'cpu'):
        if from_hf:
            self.from_pretrained(filename)
        else:
            ckpt = torch.load(filename, map_location=map_location)
            self.load_state_dict(ckpt)
            del ckpt
    
    def get_num_params(self):
        return self.num_parameters()

    def initialize_parameters(self): # Parameters are initialized upon initializing the class by HF constructor
        self.init_weights()

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
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        console.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        console.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        console.info(f"using fused AdamW: {use_fused}")
        return optimizer
    
    @classmethod
    def from_config(cls, config: ModelConfig) -> PreTrainedModel:
        roberta_config = RobertaConfig.from_pretrained(config.pretrained_filepath)
        roberta_config.num_labels = config.predictor_num_heads
        return cls.from_pretrained(config.pretrained_filepath, config=roberta_config)


class RobertaForSequenceClassification(ChemBERTa):
    _keys_to_ignore_on_load_missing = ["position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.classifier = RobertaClassificationHead(config)
        # self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        properties: Optional[torch.Tensor] = None,
        **kwargs
        ):
    
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        logits_prediction = self.regression(outputs['cls_embeddings'])

        if properties is not None:
            loss_fct = MSELoss()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits_prediction.view(-1, self.num_labels), properties.long().view(-1))
        else:
            loss = None

        return ModelOutput(
            attention_mask=outputs.get('attention_mask', None),
            embeddings=outputs.get('embeddings', None),
            cls_embeddings=outputs.get('cls_embeddings', None),
            lm_embeddings=None,
            logits_generation=outputs.get('logits_generation', None),
            logits_physchem=outputs.get('logits_physchem', None),
            logits_prediction=logits_prediction,
            loss=loss
        )


class RobertaForRegression(ChemBERTa):
    _keys_to_ignore_on_load_missing = ["position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.regression = RobertaRegressionHead(config)
        # self.init_weights()
        
    def predict(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs
    ):
        """DEPRETICATED: Use forward instead."""
        output = self.forward(input_ids=input_ids, attention_mask=attention_mask,)
        return {
            "loss": None,
            "y_pred": output,
        }
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        properties: Optional[torch.Tensor] = None,
        **kwargs
        ):
    
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        logits_prediction = self.regression(outputs['cls_embeddings'])

        if properties is not None:
            loss_fct = MSELoss()
            loss = loss_fct(logits_prediction.view(-1), properties.view(-1))
        else:
            loss = None

        return ModelOutput(
            attention_mask=outputs.get('attention_mask', None),
            embeddings=outputs.get('embeddings', None),
            cls_embeddings=outputs.get('cls_embeddings', None),
            lm_embeddings=None,
            logits_generation=outputs.get('logits_generation', None),
            logits_physchem=outputs.get('logits_physchem', None),
            logits_prediction=logits_prediction,
            loss=loss
        )
    

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs): 
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaRegressionHead(nn.Module):
    """Head for multitask regression models."""

    def __init__(self, config):
        super(RobertaRegressionHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
