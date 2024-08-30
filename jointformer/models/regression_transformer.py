""" Implements Regression Transformer, as in: https://github.com/GT4SD/gt4sd-core/blob/main/notebooks/regression-transformer-demo.ipynb
"""

import torch
import random

import numpy as np
import pandas as pd 

from typing import Any, Dict, Union
from tqdm import tqdm
from transformers import XLNetTokenizer
from terminator.inference import InferenceRT
from terminator.tokenization import InferenceBertTokenizer
from terminator.collators import MaskedTextCollator, PropertyCollator
from terminator.tokenization import ExpressionBertTokenizer
from terminator.property_predictors import predict_qed
from terminator.search import Search
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from guacamol.assess_distribution_learning import DistributionMatchingGenerator
from gt4sd.algorithms.conditional_generation.regression_transformer import  RegressionTransformer as RegressionTransformerGeneratorWrapper
from gt4sd.algorithms.conditional_generation.regression_transformer import RegressionTransformerMolecules
from gt4sd.algorithms.conditional_generation.regression_transformer.implementation import ConditionalGenerator
from gt4sd.frameworks.torch import map_tensor_dict
 
from jointformer.models.base import BaseModel, SmilesEncoder



class RegressionTransformer(BaseModel, DistributionMatchingGenerator, SmilesEncoder):
    search: Search
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
        self._interface = None
        

    def to_guacamole_generator(self, temperature, fraction_to_mask, device, seed_dataset_file, *args, **kwargs) -> DistributionMatchingGenerator:
        self._dataset = self._load_regression_transformer_dataset(seed_dataset_file)
        self._temperature = temperature
        self._fraction_to_mask = fraction_to_mask
        self._seed_dataset_file = seed_dataset_file
        self._device = device
        return self
    
    def to_smiles_encoder(self, tokenizer, batch_size, device) -> SmilesEncoder:
        self._batch_size = 1  # supports only batch size = 1
        self._device = device
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
        assert self._model is not None, " Use load_pretrained to initialize the model"
        generated = []
        for _ in tqdm(range(0, number_samples), "Generating samples"):
            sample = self._generate_single_example()
            generated.append(sample)
        return generated[:number_samples]
    
    @torch.no_grad()
    def encode(self, smiles: list[str]) -> np.ndarray:
        self._model.model.eval()
        self._model.model.to(self._device)
        rets = []
        for i in tqdm(range(0, len(smiles)), "Encoding samples"):
             batch = smiles[i]
             enc = self._encode_from_smiles(batch)
             rets.extend(enc)
        return np.vstack(rets)
    
    @torch.no_grad()
    def predict(self, smiles: list[str]) -> np.ndarray:
        self._model.model.eval()
        self._model.model.to(self._device)
        rets = []
        for i in tqdm(range(0, len(smiles)), "Encoding samples"):
             batch = smiles[i]
             enc = self._predict_from_smiles(batch)
             rets.extend(enc)
        return np.vstack(rets)

    def load_pretrained(self, filename, *args, **kwargs):
            self._tokenizer, self._collator, self._interface, self._model = self._load_regression_transformer(filename, self._device)

    def _predict_from_smiles(self, batch):
        assert self._tokenizer is not None," Use load_pretrained to initialize the tokenizer"
        qed, _ = predict_qed(batch)
        assert qed != -1, f"qed cannot be calculated for the following smiles : {batch}"
        smiles = f'<qed>{qed}|{batch}'
        sequence = self._interface.sample_sequence(smiles)
        sequence = self._interface.normalize_sequence(sequence)   
        tokens = self._tokenizer(sequence)
        inputs = self._collator([tokens] * self._batch_size)
        input_ids = inputs["input_ids"].cpu()
        output = self._model(map_tensor_dict(inputs, self._device), output_hidden_states=True)
        prediction = self.search(output["logits"].detach())
        return self.compile_regression_result(input_ids, prediction)
    
    def _encode_from_smiles(self, batch):
        assert self._tokenizer is not None," Use load_pretrained to initialize the tokenizer"
        qed, _ = predict_qed(batch)
        assert qed != -1, f"qed cannot be calculated for the following smiles : {batch}"
        smiles = f'<qed>{qed}|{batch}'
        sequence = self._interface.sample_sequence(smiles)
        sequence = self._interface.normalize_sequence(sequence)   
        tokens = self._tokenizer(sequence)
        inputs = self._collator([tokens] * self._batch_size)
        output = self._model(map_tensor_dict(inputs, self._device), output_hidden_states=True)
        final_hidden_state = output.hidden_states[-1].mean(1) #taking the mean of the embeddings
        return final_hidden_state.detach().cpu().numpy()


    @staticmethod
    def _load_regression_transformer(filename, device):
        if filename == 'qed':
            return  None, None, None, 'qed' 
        else:
            interface = ConditionalGenerator(resources_path=filename, device=device, tolerance=100.0)
            xlnet_model, config = interface.load_model(resources_path=filename)
            tokenizer = InferenceBertTokenizer.from_pretrained('/Users/pankhilgawade/.gt4sd/algorithms/conditional_generation/RegressionTransformer/RegressionTransformerMolecules/qed')
            collator = MaskedTextCollator(tokenizer)
            return tokenizer, collator, interface, InferenceRTWrapper(xlnet_model, tokenizer, config)
        
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
