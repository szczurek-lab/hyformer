import importlib

from hyformer.configs.model import ModelConfig
from hyformer.models.base import BaseModel

class AutoModel:

    @classmethod
    def from_config(cls, config: ModelConfig, **kwargs) -> BaseModel:
        
        if config.model_type == 'GPT':
            return getattr(importlib.import_module(
                "hyformer.models.gpt"),
                "GPT").from_config(config)

        elif config.model_type == 'GPTForDownstreamPrediction':
            return getattr(importlib.import_module(
                "hyformer.models.gpt"),
                "GPTForDownstreamPrediction").from_config(config, **kwargs)
        
        elif config.model_type == 'Hyformer':
            return getattr(importlib.import_module(
                "hyformer.models.hyformer"),
                "Hyformer").from_config(config)
        
        elif config.model_type == 'HyformerForDownstreamPrediction':
            return getattr(importlib.import_module(
                "hyformer.models.hyformer"),
                "HyformerForDownstreamPredictionDeep").from_config(config, **kwargs)
        
        elif config.model_type == 'HyformerLlama':
            return getattr(importlib.import_module(
                "hyformer.models.llama_wrapper"),
                "HyformerLlama").from_config(config)
        
        elif config.model_type == 'HyformerLlamaForDownstreamPrediction':
            return getattr(importlib.import_module(
                "hyformer.models.llama_wrapper"),
                "HyformerLlamaForDownstreamPrediction").from_config(config, **kwargs)
        
        elif config.model_type == 'HyformerLlamaForDownstreamPredictionDeep':
            return getattr(importlib.import_module(
                "hyformer.models.llama_wrapper"),
                "HyformerLlamaForDownstreamPredictionDeep").from_config(config, **kwargs)
        
        elif config.model_type == 'ChemBERTa' and config.prediction_task_type == 'classification':
            return getattr(importlib.import_module(
                "hyformer.models.chemberta"),
                "RobertaForSequenceClassification").from_config(config)
        
        elif config.model_type == 'ChemBERTa' and config.prediction_task_type == 'regression':
            return getattr(importlib.import_module(
                "hyformer.models.chemberta"),
                "RobertaForRegression").from_config(config)

        if config.model_type == "Moler":
            return getattr(importlib.import_module(
                "hyformer.models.moler"),
                "Moler").from_config(config)
        
        if config.model_type == "RegressionTransformer":
            return getattr(importlib.import_module(
                "hyformer.models.regression_transformer"),
                "RegressionTransformer").from_config(config)
        
        if config.model_type == "UniMol":
            return getattr(importlib.import_module(
                "hyformer.models.unimol"),
                "UniMol").from_config(config)
        
        if config.model_type == "MolGPT":
            return getattr(importlib.import_module(
                "hyformer.models.molgpt"),
                "MolGPT").from_config(config)
        
        else:
            raise ValueError(f"Model {config.model_type} not supported.")
