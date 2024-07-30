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
        if self.attention_mask is None:
            return self.embeddings.mean(dim=-1)
        else:
            w = self.attention_mask / self.attention_mask.sum(dim=-1, keepdim=True)
            w = w.unsqueeze(-2)
            global_embedding = w @ self.embeddings
            return global_embedding.squeeze(-2)
    