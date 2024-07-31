import torch

from collections.abc import Mapping
from dataclasses import dataclass, fields
from guacamol.assess_distribution_learning import DistributionMatchingGenerator
from tqdm import tqdm
from typing import Any, List, Optional
import torch.nn.functional as F
from typing import List
from jointformer.models.base import SmilesEncoder, DistributionMatchingGenerator
from tqdm import tqdm
from jointformer.utils.tokenizers.auto import SmilesTokenizer
import numpy as np


class DefaultGuacamolModelWrapper(DistributionMatchingGenerator):
    def __init__(self, model, tokenizer, batch_size, temperature, top_k, device):
        self._model = model
        self._tokenizer: SmilesTokenizer = tokenizer
        self._batch_size = batch_size
        self._device = device
        self._temperature = temperature
        self._top_k = top_k

    @torch.no_grad()
    def generate(self, number_samples: int) -> List[str]:
        generated = []
        self._model.eval()
        model = self._model.to(self._device)
        for _ in tqdm(range(0, number_samples, self._batch_size), "Generating samples"):
            samples = model.generate(self._tokenizer.cls_token_id,
                                        self._tokenizer.sep_token_id,
                                        self._tokenizer.pad_token_id,
                                        self._tokenizer.max_molecule_length,
                                        self._batch_size,
                                        self._temperature,
                                        self._top_k,
                                        self._device)
            generated.extend(self._tokenizer.decode(samples))
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
    
    
class DefaultSmilesEncoderWrapper(SmilesEncoder):
    def __init__(self, model, tokenizer, batch_size, device):
        self._model = model
        self._tokenizer: SmilesTokenizer = tokenizer
        self._batch_size = batch_size
        self._device = device

    def encode(self, smiles: list[str]) -> np.ndarray:
        self._model.eval()
        model = self._model.to(self._device)
        embeddings = []
        for i in tqdm(range(0, len(smiles), self._batch_size), "Encoding samples"):
            batch = smiles[i:i+self._batch_size]
            batch_input = self._tokenizer(batch, task="prediction")
            for k,v in batch_input.items():
                if isinstance(v, torch.Tensor):
                    batch_input[k] = v.to(self._device)
            output = model(**batch_input, is_causal=False)
            embeddings.append(output["global_embedding"].detach().cpu().numpy())
        ret = np.concatenate(embeddings, axis=0)
        return ret
