import importlib

from hyformer.configs.tokenizer import TokenizerConfig
from hyformer.utils.tokenizers.smiles import SmilesTokenizer


class AutoTokenizer:
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, local_dir=None):
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import RepositoryNotFoundError
        except ImportError:
            print("HuggingFace Hub is not installed. Loading models from HuggingFace not available.")

        _tokenizer_config_path_hf = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename="tokenizer_config.json",
                    local_dir=local_dir
                )
        _tokenizer_config = TokenizerConfig.from_config_file(_tokenizer_config_path_hf)
        return cls.from_config(_tokenizer_config)
        
    @classmethod
    def from_config(cls, config: TokenizerConfig) -> SmilesTokenizer:

        if config.tokenizer == 'SmilesTokenizer':
            return getattr(importlib.import_module(
                "hyformer.utils.tokenizers.smiles"),
                "SmilesTokenizer").from_config(config)
        if config.tokenizer == 'SmilesBPETokenizer':
            return getattr(importlib.import_module(
                "hyformer.utils.tokenizers.smiles_bpe"),
                "SmilesBPETokenizer").from_config(config)
        elif config.tokenizer == 'SmilesTokenizerWithPrefix':
            return getattr(importlib.import_module(
                "hyformer.utils.tokenizers.smiles_with_prefix"),
                "SmilesTokenizerWithPrefix").from_config(config)
        elif config.tokenizer == 'SmilesTokenizerSeparateTaskToken':
            return getattr(importlib.import_module(
                "hyformer.utils.tokenizers.smiles_separate_task_token"),
                "SmilesTokenizerSeparateTaskToken").from_config(config)
        elif config.tokenizer == 'SmilesTokenizerSeparateTaskTokenFuture':
            return getattr(importlib.import_module(
                "hyformer.utils.tokenizers.smiles_separate_task_token_future"),
                "SmilesTokenizerSeparateTaskTokenFuture").from_config(config)
        elif config.tokenizer == "ESMTokenizer":
            return getattr(importlib.import_module(
                "hyformer.utils.tokenizers.amp"),
                "AMPTokenizer").from_config(config)
        elif config.tokenizer == "HFTokenizer":
            return getattr(importlib.import_module(
                "hyformer.utils.tokenizers.hf"),
                "HFTokenizer").from_config(config)
        elif config.tokenizer == "FancyTokenizer":
            return getattr(importlib.import_module(
                "hyformer.utils.tokenizers.fancy_tokenizer"),
                "FancyTokenizer").from_config(config)
        else:
            raise ValueError(f"Tokenizer {config.tokenizer} not available.")
