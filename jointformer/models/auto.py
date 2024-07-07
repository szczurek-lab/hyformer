import importlib

from jointformer.configs.model import ModelConfig
from jointformer.models.base import BaseModel

class AutoModel:

    @classmethod
    def from_config(cls, config: ModelConfig) -> BaseModel:

        if config.model_name == 'Jointformer':
            return getattr(importlib.import_module(
                "jointformer.models.jointformer"),
                "Jointformer").from_config(config)

        elif config.model_name == 'GPT':
            return getattr(importlib.import_module(
                "jointformer.models.gpt"),
                "GPT").from_config(config)
        
        elif config.model_name == 'ChemBERTa':
            return getattr(importlib.import_module(
                "jointformer.models.chemberta"),
                "ChemBERTa").from_config(config)

        else:
            raise ValueError(f"Model {config.model_name} not supported.")
