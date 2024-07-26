import torch

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
