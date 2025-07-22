import importlib

from hyformer.configs.model import ModelConfig
from hyformer.models.base import BaseModel

class AutoModel:

    @classmethod
    def from_config(cls, config: ModelConfig, **kwargs) -> BaseModel:
        
        if config.model_name == 'FancyModel':
            return getattr(importlib.import_module(
                "hyformer.models.fancy_model"),
                "FancyModel").from_config(config)
        
        elif config.model_name == 'DummyClassifierConstant':
            return getattr(importlib.import_module(
                "hyformer.models.dummy_classifier"),
                "DummyClassifierConstant").from_config(config, **kwargs)
        
        elif config.model_name == 'FancyModelForDownstreamPrediction':
            return getattr(importlib.import_module(
                "hyformer.models.fancy_model"),
                "FancyModelForDownstreamPrediction").from_config(config, **kwargs)
        
        elif config.model_name == 'GPT':
            return getattr(importlib.import_module(
                "hyformer.models.gpt"),
                "GPT").from_config(config)

        elif config.model_name == 'GPTForDownstreamPrediction':
            return getattr(importlib.import_module(
                "hyformer.models.gpt"),
                "GPTForDownstreamPrediction").from_config(config, **kwargs)
        
        elif config.model_name == 'JointGPTForDownstreamPrediction':
            return getattr(importlib.import_module(
                "hyformer.models.gpt"),
                "JointGPTForDownstreamPrediction").from_config(config, **kwargs)

        elif config.model_name == 'Hyformer':
            return getattr(importlib.import_module(
                "hyformer.models.hyformer"),
                "Hyformer").from_config(config)
        
        elif config.model_name == 'HyformerV2':
            return getattr(importlib.import_module(
                "hyformer.models.hyformer_v2"),
                "Hyformer").from_config(config)
        
        elif config.model_name == 'HyformerForDownstreamPrediction':
            return getattr(importlib.import_module(
                "hyformer.models.hyformer_v2"),
                "HyformerForDownstreamPrediction").from_config(config, **kwargs)
        
        elif config.model_name == 'HyformerLlamaForDownstreamPrediction':
            return getattr(importlib.import_module(
                "hyformer.models.hyformer"),
                "HyformerForDownstreamPrediction").from_config(config, **kwargs)
        
        elif config.model_name == 'HyformerLlamaForDownstreamPredictionDeep':
            return getattr(importlib.import_module(
                "hyformer.models.hyformer"),
                "HyformerForDownstreamPredictionDeep").from_config(config, **kwargs)

        elif config.model_name == 'HyformerWithPrefix':
            return getattr(importlib.import_module(
                "hyformer.models.hyformer"),
                "HyformerWithPrefix").from_config(config)
        
        elif config.model_name == 'HyformerWithMaxEmbeddings':
            return getattr(importlib.import_module(
                "hyformer.models.hyformer"),
                "HyformerWithMaxEmbeddings").from_config(config)

        elif config.model_name == 'GPT':
            return getattr(importlib.import_module(
                "hyformer.models.gpt"),
                "GPT").from_config(config)
        
        elif config.model_name == 'ChemBERTa' and config.prediction_task_type == 'classification':
            return getattr(importlib.import_module(
                "hyformer.models.chemberta"),
                "RobertaForSequenceClassification").from_config(config)
        
        elif config.model_name == 'ChemBERTa' and config.prediction_task_type == 'regression':
            return getattr(importlib.import_module(
                "hyformer.models.chemberta"),
                "RobertaForRegression").from_config(config)

        if config.model_name == "Moler":
            return getattr(importlib.import_module(
                "hyformer.models.moler"),
                "Moler").from_config(config)
        
        if config.model_name == "RegressionTransformer":
            return getattr(importlib.import_module(
                "hyformer.models.regression_transformer"),
                "RegressionTransformer").from_config(config)
        
        if config.model_name == "UniMol":
            return getattr(importlib.import_module(
                "hyformer.models.unimol"),
                "UniMol").from_config(config)
        
        if config.model_name == "MolGPT":
            return getattr(importlib.import_module(
                "hyformer.models.molgpt"),
                "MolGPT").from_config(config)
        
        if config.model_name == "HyformerWithContext":
            return getattr(importlib.import_module(
                "hyformer.models.hyformer_encoder_context"),
                "HyformerWithContext").from_config(config)
        
        else:
            raise ValueError(f"Model {config.model_name} not supported.")
