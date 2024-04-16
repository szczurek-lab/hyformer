import importlib

from jointformer.configs.task import TaskConfig


class AutoTokenizer:
    """Base Tokenizer. """

    @classmethod
    def from_config(cls, config: TaskConfig) -> "AutoTokenizer":

        if config.tokenizer not in ["SMILESTokenizer"]:
            raise ValueError(f"`config.tokenizer` must be 'SMILESTokenizer', got {config.tokenizer}.")

        if config.tokenizer == 'SMILESTokenizer':
            return getattr(importlib.import_module(
                "jointformer.utils.tokenizers.smiles"),
                "SMILESTokenizer").from_config(config)
        else:
            raise ValueError(f"Invalid `tokenizer`.")
