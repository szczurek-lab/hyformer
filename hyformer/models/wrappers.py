""" Wrappers for generation and featurization. """

import torch
import numpy as np
from tqdm import tqdm
from typing import List
from hyformer.models.base import Generator, Encoder
from hyformer.models.utils import ModelOutput
from hyformer.utils.tokenizers.base import BaseTokenizer


### GENERATION ###

class DefaultGeneratorWrapper(Generator):
    def __init__(self, model, tokenizer, batch_size, temperature, top_k, device):
        self._model = model
        self._tokenizer: BaseTokenizer = tokenizer
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
            samples: list[str] = model.generate(self._tokenizer.cls_token_id,
                                        self._tokenizer.sep_token_id,
                                        self._tokenizer.pad_token_id,
                                        self._tokenizer.max_molecule_length,
                                        self._batch_size,
                                        self._temperature,
                                        self._top_k,
                                        self._device)
            generated.extend(self._tokenizer.decode(samples))
        return generated[:number_samples]


class HyformerGeneratorWrapper(DefaultGeneratorWrapper):
    @torch.no_grad()
    def generate(self, number_samples: int) -> List[str]:
        generated = []
        self._model.eval()
        model = self._model.to(self._device)
        for _ in tqdm(range(0, number_samples, self._batch_size), "Generating samples"):
            samples: list[str] = model.generate(self._tokenizer, self._batch_size, self._temperature, self._top_k, self._device)
            generated.extend(self._tokenizer.decode(samples))
        return generated[:number_samples]
    

### FEATURIZATION ###

class DefaultEncoderWrapper(Encoder):
    def __init__(self, model, tokenizer, batch_size, device):
        self._model = model
        self._tokenizer: BaseTokenizer = tokenizer
        self._batch_size = batch_size
        self._device = device

    @torch.no_grad()
    def encode(self, X: list[str]) -> np.ndarray:
        self._model.eval()
        model = self._model.to(self._device)
        embeddings = []
        for i in tqdm(range(0, len(X), self._batch_size), "Encoding samples"):
            batch = X[i:i+self._batch_size]
            batch_input = self._tokenizer(batch, task="prediction")
            for k,v in batch_input.items():
                if isinstance(v, torch.Tensor):
                    batch_input[k] = v.to(self._device)
            output: ModelOutput = model(**batch_input, is_causal=False)
            embeddings.append(output["global_embeddings"].cpu().numpy())
        ret = np.concatenate(embeddings, axis=0)
        return ret


class HyformerEncoderWrapper(DefaultEncoderWrapper):
    _TASK = "prediction"
    _EMBEDDING_KEY = "cls_embeddings"

    @torch.no_grad()
    def encode(self, X: list[str]) -> np.ndarray:
        self._model.eval()
        model = self._model.to(self._device)
        embeddings = np.zeros((len(X), model.embedding_dim))
        for i in tqdm(range(0, len(X), self._batch_size), "Encoding samples"):
            batch = X[i:i+self._batch_size]
            model_input = self._tokenizer(batch, task=self._TASK)
            model_input = model_input.to(self._device)
            output = model(**model_input)
            batch_size = min(self._batch_size, len(X) - i)
            global_embs = output[self._EMBEDDING_KEY].cpu().numpy()
            embeddings[i:i+batch_size] = global_embs[:batch_size]
        return embeddings
    