import importlib

from typing import Union

from hyformer.configs.model import ModelConfig
from hyformer.models.encoder import EncoderWrapper
from hyformer.models.base import TrainableModel


class AutoModel:

    @classmethod
    def from_config(cls, config: ModelConfig, **kwargs) -> Union[EncoderWrapper, TrainableModel]:
        
        if config.model_type == 'Hyformer':
            return getattr(importlib.import_module(
                "hyformer.models.hyformer"),
                "Hyformer").from_config(config, **kwargs)
        elif config.model_type == 'MolGPT':
            return getattr(importlib.import_module(
                "hyformer.models.baselines.molgpt"),
                "MolGPT").from_config(config, **kwargs)
        
        else:
            raise ValueError(f"Model {config.model_type} not supported.")
