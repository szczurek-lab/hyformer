import torch

from collections.abc import Mapping
from dataclasses import dataclass, fields
from guacamol.assess_distribution_learning import DistributionMatchingGenerator
from tqdm import tqdm
from typing import Any, List, Optional
import torch.nn.functional as F


class DefaultGuacamolModelWrapper(DistributionMatchingGenerator):
    def __init__(self, model, tokenizer, batch_size, temperature, top_k, device):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        self.model.eval()

    @torch.no_grad()
    def generate(self, number_samples: int) -> List[str]:
        generated = []
        model = self.model.to(self.device)
        for _ in tqdm(range(0, number_samples, self.batch_size), "Generating samples"):
            samples = model.generate(self.tokenizer.cls_token_id,
                                        self.tokenizer.sep_token_id,
                                        self.tokenizer.pad_token_id,
                                        self.tokenizer.max_molecule_length,
                                        self.batch_size,
                                        self.temperature,
                                        self.top_k,
                                        self.device)
            generated.extend(self.tokenizer.decode(samples))
        return generated[:number_samples]


class ModelInput(dict):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def to(self, device: str, pin_memory: bool = True) -> 'ModelInput':
        if device != 'cpu':
            for key, value in self.items():
                if isinstance(value, torch.Tensor):
                    if pin_memory:
                        self[key] = value.pin_memory().to(device, non_blocking=True)
                    else:
                        self[key] = value.to(device, non_blocking=True)
        return self

class ModelOutput(dict):
    
    def __init__(self, embeddings, loss=None, logits_generation=None, logits_prediction=None, logits_physchem=None, attention_mask=None):
        super().__init__(
            embeddings=embeddings,
            loss=loss,
            logits_generation=logits_generation,
            logits_prediction=logits_prediction,
            logits_physchem=logits_physchem,
            attention_mask=attention_mask
            )
     
    @property
    def global_embedding(self):
        if self.attention_mask is not None and self.embedding is not None:
            return torch.einsum('bsd, bs -> bsd', self.embedding, self.mask.float())
        else:
            return None


def lm_loss(logits, labels, ignore_index, reduction):
    logits = logits[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=ignore_index, reduction=reduction)


def mlm_loss(logits, labels, ignore_index, reduction):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=ignore_index, reduction=reduction)


def regression_loss(logits, labels, reduction):
    return F.mse_loss(logits.flatten(), labels.flatten(), reduction=reduction)

def classification_loss(logits, labels, reduction):
    return F.cross_entropy(logits, labels, reduction=reduction)
