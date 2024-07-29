from guacamol.assess_distribution_learning import DistributionMatchingGenerator
from jointformer.models.base import BaseModel, SmilesEncoder
from molecule_generation import load_model_from_directory, VaeWrapper
from molecule_generation.models.moler_vae import MoLeRVae
import os
from tqdm import tqdm
import torch
import numpy as np
from molecule_generation.utils.model_utils import load_vae_model_and_dataset
from molecule_generation.utils.moler_inference_server import _encode_from_smiles
import tensorflow as tf

class Moler(BaseModel, DistributionMatchingGenerator, SmilesEncoder):
    def __init__(self) -> None:
        super().__init__()
        self._model: MoLeRVae = None
        self._dataset = None
        self._tokenizer = None
        self._batch_size = None
        self._temperature = None
        self._top_k = None
        self._device = None

    def to_guacamole_generator(self, tokenizer, batch_size, temperature, top_k, device) -> DistributionMatchingGenerator:
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._temperature = temperature
        self._top_k = top_k
        self._device = device
        return self
    
    def to_smiles_encoder(self, tokenizer, batch_size, device) -> SmilesEncoder:
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._device = device
        return self

    def generate(self, number_samples: int):
        generated = []
        with self._model as model:
            for _ in tqdm(range(0, number_samples, self._batch_size), "Generating samples"):
                samples = model.sample(self._batch_size)
                generated.extend(samples)
        return generated

    def encode(self, smiles: list[str]) -> np.ndarray:
        rets = []
        for i in tqdm(range(0, len(smiles), self._batch_size), "Encoding samples"):
            batch = smiles[i:i+self._batch_size]
            enc = _encode_from_smiles(self._dataset, self._model, batch)
            rets.extend(enc)
        return np.stack(rets, axis=0)
        
    def load_pretrained(self, filename, *args, **kwargs):
        self._dataset, self._model = load_vae_model_and_dataset(filename)
        assert isinstance(self._model, MoLeRVae), self._model
    
    @classmethod
    def from_config(cls, config):
        return cls()