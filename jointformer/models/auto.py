import importlib

from jointformer.configs.model import ModelConfig


class AutoModel:

    @classmethod
    def from_config(cls, config: ModelConfig):

        if config.model_name == 'Jointformer':
            return getattr(importlib.import_module(
                "jointformer.models.jointformer"),
                "Jointformer").from_config(config)

        if config.model_name == 'GPT':
            return getattr(importlib.import_module(
                "jointformer.models.gpt"),
                "GPT").from_config(config)

        else:
            raise ValueError(f"Model {config.model_name} not supported.")
