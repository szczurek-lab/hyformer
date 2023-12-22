import importlib

from hybrid_transformer.configs.tasks.distribution_learning import DistributionLearningConfig


class AutoModel:

    @classmethod
    def from_config(cls, config: DistributionLearningConfig):

        if config.model_type == 'GPT':
            return getattr(importlib.import_module(
                "hybrid_transformer.models.gpt"),
                "GPT").from_config(config)
