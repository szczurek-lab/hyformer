import torch 
import logging
import inspect

from typing import Optional
from torch import nn
from transformers import RobertaForSequenceClassification
from guacamol.assess_distribution_learning import DistributionMatchingGenerator

from jointformer.models.base import BaseModel
from jointformer.models.utils import DefaultGuacamolModelWrapper
from jointformer.configs.model import ModelConfig

console = logging.getLogger(__name__)


class ChemBERTa(RobertaForSequenceClassification, BaseModel):

    def load_pretrained(self, filename: str):
        self.from_pretrained(filename)

    def to_guacamole_generator(self, tokenizer, batch_size, temperature, top_k, device) -> DistributionMatchingGenerator:
        return DefaultGuacamolModelWrapper(self, tokenizer, batch_size, temperature, top_k, device)

    def get_num_params(self):
        return self.num_parameters()

    def initialize_parameters(self): # Parameters are initialized upon initializing the class by HF constructor
        pass

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        return 0.0

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

    def set_prediction_task(self, task_type: str, out_size: int, hidden_size: int, dropout: float):
        self.problem_type = task_type
        self.classifier = RobertaClassificationHead(hidden_size=hidden_size, num_labels=out_size, dropout=dropout, task_type=task_type)

    def get_loss(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            properties: Optional[torch.Tensor] = None,
            task: Optional[str] = None):

        return self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=properties)

    def predict(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs):
        """ Perform a forward pass through the discriminative part of the model. """
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        outputs["loss"] = None
        outputs["y_pred"] = outputs["logits"]
        return outputs

    @classmethod
    def from_config(cls, config: ModelConfig):
        return cls.from_pretrained(config.pretrained_filepath)


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size: int, dropout: float, num_labels: int, task_type: str):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        dropout = (
            dropout if dropout is not None else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.act_fn = nn.Tanh() if task_type == 'classification' else nn.ReLU()
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.linear(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
