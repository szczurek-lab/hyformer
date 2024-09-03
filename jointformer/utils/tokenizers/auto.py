import importlib

from jointformer.configs.task import TaskConfig
from jointformer.utils.tokenizers.smiles import SmilesTokenizer


class AutoTokenizer:

    @classmethod
    def from_config(cls, config: TaskConfig) -> SmilesTokenizer:

        if config.tokenizer == 'SmilesTokenizer':
            return getattr(importlib.import_module(
                "jointformer.utils.tokenizers.smiles"),
                "SmilesTokenizer").from_config(config)
        elif config.tokenizer == 'SmilesTokenizerWithPrefix':
            return getattr(importlib.import_module(
                "jointformer.utils.tokenizers.smiles_with_prefix"),
                "SmilesTokenizerWithPrefix").from_config(config)
        elif config.tokenizer == 'SmilesTokenizerSeparateTaskToken':
            return getattr(importlib.import_module(
                "jointformer.utils.tokenizers.smiles_separate_task_token"),
                "SmilesTokenizerSeparateTaskToken").from_config(config)
        elif config.tokenizer == "ESMTokenizer":
            return getattr(importlib.import_module(
                "jointformer.utils.tokenizers.amp"),
                "AMPTokenizer").from_config(config)
        elif config.tokenizer == "HFTokenizer":
            return getattr(importlib.import_module(
                "jointformer.utils.tokenizers.hf"),
                "HFTokenizer").from_config(config)
        else:
            raise ValueError(f"Tokenizer {config.tokenizer} not available.")
