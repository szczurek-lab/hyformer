import importlib

from jointformer.configs.task import TaskConfig
from jointformer.utils.tokenizers.smiles.smiles import SmilesTokenizer


class AutoTokenizer:

    @classmethod
    def from_config(cls, config: TaskConfig) -> SmilesTokenizer:

        if config.tokenizer == 'SmilesTokenizer':
            return getattr(importlib.import_module(
                "jointformer.utils.tokenizers.smiles.smiles"),
                "SmilesTokenizer").from_config(config)
        else:
            raise ValueError(f"Tokenizer {config.tokenizer} not available.")
