""" Implements Regression Transformer, as in: https://github.com/GT4SD/gt4sd-core/blob/main/notebooks/regression-transformer-demo.ipynb
"""

import torch
import random

import numpy as np
import pandas as pd 

from tqdm import tqdm
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from guacamol.assess_distribution_learning import DistributionMatchingGenerator
from gt4sd.algorithms.conditional_generation.regression_transformer import  RegressionTransformer as RegressionTransformerGeneratorWrapper
from gt4sd.algorithms.conditional_generation.regression_transformer import RegressionTransformerMolecules
 
from jointformer.models.base import BaseModel, SmilesEncoder


class RegressionTransformer(BaseModel, DistributionMatchingGenerator, SmilesEncoder):
    def __init__(self) -> None:
        super().__init__()
        self._model: RegressionTransformerGeneratorWrapper = None
        self._dataset = None
        self._tokenizer = None
        self._batch_size = None
        self._temperature = None
        self._top_k = None
        self._device = None
        self._tolerence  = None
        self._algorithm_version = None
        self._search_strategy = None 
        self._target = None
        self._fraction_to_mask = None

    def to_guacamole_generator(self, temperature, fraction_to_mask, device, seed_dataset_file, *args, **kwargs) -> DistributionMatchingGenerator:
        self._dataset = self._load_regression_transformer_dataset(seed_dataset_file)
        self._temperature = temperature
        self._fraction_to_mask = fraction_to_mask
        self._seed_dataset_file = seed_dataset_file
        self._device = device
        return self
    
    def to_smiles_encoder(self, tokenizer, batch_size, device) -> SmilesEncoder:
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._device = device # TODO: move to device
        return self

    def _generate_single_example(self) -> str:
        assert self._dataset is not None, "Initialize the dataset prior to generation"
        seed_property, seed_example = random.sample(self._dataset, 1)[0]  # get the sampled example
        
        generator = RegressionTransformerGeneratorWrapper(
            configuration=RegressionTransformerMolecules(
                algorithm_version=self._model,
                search="sample",
                temperature=self._temperature, 
                tolerance=100,
                sampling_wrapper={
                    'property_goal': {'<qed>': seed_property},  # TODO: property_goal qed to be change
                    'fraction_to_mask': self._fraction_to_mask
                }),
            target=seed_example)
        generation = generator.sample(1)
        generated_value = next(generation)
        return generated_value[0]

    @torch.no_grad()
    def generate(self, number_samples: int):
        generated = []
        for _ in tqdm(range(0, number_samples), "Generating samples"):
            sample = self._generate_single_example()
            generated.append(sample)
        return generated[:number_samples]

    def encode(self, smiles: list[str]) -> np.ndarray:
        #pass #TODO: implement
        rets = []
        for i in tqdm(range(0, len(smiles), self._batch_size), "Encoding samples"):
             batch = smiles[i:i+self._batch_size]
             enc = self._encode_from_smiles(self._dataset, self._model, batch)
             rets.extend(enc)
        return np.stack(rets, axis=0)

    def load_pretrained(self, filename, *args, **kwargs):
        self._model = self._load_regression_transformer_model(filename)

    @staticmethod
    def _load_regression_transformer_model(filename):
        if filename == 'qed':
            return 'qed'
        else:
            raise NotImplementedError(f"Model {filename} not implemented")

    @staticmethod
    def _load_regression_transformer_dataset(filename):
        df = pd.read_csv(filename, sep='|', header=None)
        df.columns = ['property', 'smiles'] 
        df['property'] = df['property'].str.replace('<qed>', '', regex=False)
        df['property'] = df['property'].astype(float)
        dataset = []
        for _, row in df.iterrows():
            property = row['property']
            smiles = row['smiles']
            dataset.append((property, smiles))
        return dataset
        
    @classmethod
    def from_config(cls, config):
        return cls()



