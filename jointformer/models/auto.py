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

        if config.model_name == 'GPT':
            return getattr(importlib.import_module(
                "jointformer.models.gpt"),
                "GPT").from_config(config)

        if config.model_name == "Moler":
            return getattr(importlib.import_module(
                "jointformer.models.moler"),
                "Moler").from_config(config)
            

        else:
            raise ValueError(f"Model {config.model_name} not supported.")
