import importlib

from jointformer.configs.model import ModelConfig

AVAILABLE_MODELS = [
            "GPTPreTrained", "BERTPreTrained", "HybridTransformerPreTrained", "GPTForPrediction", "BERTForPrediction",
            "JointGPT", "JointGPTNonLikelihood", "HybridTransformer", "HybridTransformerWithPenalty"
        ]


class AutoModel:

    @classmethod
    def from_config(cls, config: ModelConfig):

        if config.model_name not in AVAILABLE_MODELS:
            raise ValueError(f"`model_name` must be {AVAILABLE_MODELS}, got {config.model_name}.")

        if config.model_name == 'GPTPreTrained':
            return getattr(importlib.import_module(
                "jointformer.models.pre_train"),
                "GPTPreTrained").from_config(config)

        if config.model_name == 'BERTPreTrained':
            return getattr(importlib.import_module(
                "jointformer.models.pre_train"),
                "BERTPreTrained").from_config(config)

        if config.model_name == 'HybridTransformerPreTrained':
            return getattr(importlib.import_module(
                "jointformer.models.pre_train"),
                "HybridTransformerPreTrained").from_config(config)

        if config.model_name == 'GPTForPrediction':
            return getattr(importlib.import_module(
                "jointformer.models.prediction"),
                "GPTForPrediction").from_config(config)

        if config.model_name == 'BERTForPrediction':
            return getattr(importlib.import_module(
                "jointformer.models.prediction"),
                "JointGPT").from_config(config)

        if config.model_name == 'JointGPT':
            return getattr(importlib.import_module(
                "jointformer.models.prediction"),
                "JointGPT").from_config(config)

        if config.model_name == 'JointGPTNonLikelihood':
            return getattr(importlib.import_module(
                "jointformer.models.prediction"),
                "JointGPTNonLikelihood").from_config(config)

        if config.model_name == 'HybridTransformer':
            return getattr(importlib.import_module(
                "jointformer.models.prediction"),
                "HybridTransformer").from_config(config)

        if config.model_name == 'HybridTransformerWithPenalty':
            return getattr(importlib.import_module(
                "jointformer.models.prediction"),
                "HybridTransformerWithPenalty").from_config(config)

        raise ValueError("None of the available models were selected!")
