""" Implements Regression Transformer, as in: https://github.com/GT4SD/gt4sd-core/blob/main/notebooks/regression-transformer-demo.ipynb
"""

import torch
import random

import numpy as np
import pandas as pd 

from typing import Any, Dict, Union
from tqdm import tqdm
from terminator.inference import InferenceRT
from terminator.tokenization import InferenceBertTokenizer
from terminator.collators import MaskedTextCollator, PropertyCollator
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from guacamol.assess_distribution_learning import DistributionMatchingGenerator
from gt4sd.algorithms.conditional_generation.regression_transformer import  RegressionTransformer as RegressionTransformerGeneratorWrapper
from gt4sd.algorithms.conditional_generation.regression_transformer import RegressionTransformerMolecules
from gt4sd.algorithms.conditional_generation.regression_transformer.implementation import ConditionalGenerator
from gt4sd.frameworks.torch import map_tensor_dict
 
from jointformer.models.base import BaseModel, SmilesEncoder



class RegressionTransformer(BaseModel, DistributionMatchingGenerator, SmilesEncoder):
    def __init__(self) -> None:
        super().__init__()
        self._model: RegressionTransformerGeneratorWrapper = None
        self._interface: ConditionalGenerator = None
        self._dataset = None
        self._tokenizer = None
        self._collator = None
        self._batch_size = None
        self._temperature = None
        self._top_k = None
        self._device = None
        self._tolerence  = None
        self._algorithm_version = None
        self._search_strategy = None 
        self._target = None
        self._fraction_to_mask = None
        self._tolerance = 100

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

        rets = []
        for i in tqdm(range(0, len(smiles), self._batch_size), "Encoding samples"):
             #batch = smiles[i:i+self._batch_size]
             batch = smiles
             enc = self._encode_from_smiles(self._model,batch,self._device)
             rets.extend(enc)
        return rets 
        #return np.stack(rets, axis=0)

    def load_pretrained(self, filename, *args, **kwargs):
        if filename == 'qed':
            self._model = self._load_regression_transformer_tokenizer_and_model(filename, self._device)
        else:
            self._tokenizer, self._model = self._load_regression_transformer_tokenizer_and_model(filename, self._device)
    
    
    def _encode_from_smiles(self,model,smiles,device):
        assert self._tokenizer is not None," Use load_pretrained to initialize the tokenizer"
        tokens = self._tokenizer(smiles)
        collator = collator = MaskedTextCollator(self._tokenizer)
        inputs = collator([tokens] * self.batch_size)
        output = model(map_tensor_dict(inputs,device),output_hidden_states=True)
        final_hidden_state = output.hidden_states[-1]
        return final_hidden_state.detach().cpu().numpy()



    @staticmethod
    def _load_regression_transformer_tokenizer_and_model(filename, device):
        if filename == 'qed':
            return 'qed'
        
        else:
            _interface = ConditionalGenerator(resources_path=filename, device=device, tolerance=100)
            xlnet_model, config = _interface.load_model(resources_path=filename)
            tokenizer = InferenceBertTokenizer.from_pretrained(
            resources_path =filename, pad_even=False)
            
            return tokenizer, InferenceRTWrapper(xlnet_model, tokenizer, config)
        
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
    

class InferenceRTWrapper(InferenceRT):
    def __call__(self, inputs: Dict[str, Union[torch.Tensor, Any]], output_hidden_states: bool = True) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Forward pass with additional handling for `output_hidden_states`.

        Args:
            inputs (Dict[str, Union[torch.Tensor, Any]]): Input dictionary.
            output_hidden_states (bool): Whether to return hidden states from the model.

        Returns:
            Dict[str, Union[torch.Tensor, Any]]: Model outputs.
        """
        # Remove 'output_hidden_states' from inputs if it's included, as it will be handled separately
        inputs.pop('output_hidden_states', None)

        if self.use_ne:
            embeddings = self.model_embed(inputs["input_ids"])
            numerical_embeddings = self.numerical_encoder(inputs["input_ids"])
            embeddings = self.combine_embed(embeddings, numerical_embeddings)
            inputs.pop("input_ids", None)
            outputs = self.model(inputs_embeds=embeddings, output_hidden_states=output_hidden_states, **inputs)
        else:
            outputs = self.model(output_hidden_states=output_hidden_states, **inputs)

        return outputs



