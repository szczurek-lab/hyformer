import importlib

from hybrid_transformer.configs.model import ModelConfig


class AutoModel:

    @classmethod
    def from_config(cls, config: ModelConfig):

        if config.model_type not in ["GPT"]:
            raise ValueError(f"`model_type` must be 'GPT', got {config.model_type}.")

        if config.model_type == 'GPT':
            return getattr(importlib.import_module(
                "hybrid_transformer.models.gpt"),
                "GPT").from_config(config)
