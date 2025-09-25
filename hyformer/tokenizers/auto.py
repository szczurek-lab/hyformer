import importlib
import os
import warnings
from typing import Dict, Any, Optional

from hyformer.configs.tokenizer import TokenizerConfig
from hyformer.tokenizers.base import BaseTokenizer, TOKENIZER_CONFIG_FILENAME

try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError
except ImportError:
    hf_hub_download = None
    RepositoryNotFoundError = Exception
    warnings.warn("HuggingFace Hub is not installed. Loading tokenizers from HuggingFace not available.")


class AutoTokenizer:
    """Factory class for creating tokenizers.
    
    This class provides a unified interface for creating tokenizers based on configuration,
    using a simple if-else approach to select the appropriate implementation.
    
    Currently supported tokenizers:
    - SMILESTokenizer: For SMILES molecular representations
    - HFTokenizer: For using Hugging Face tokenizers
    """

    @classmethod
    def from_config(cls, config: TokenizerConfig) -> BaseTokenizer:
        """Create a tokenizer from configuration.
        
        Parameters
        ----------
        config : TokenizerConfig
            Tokenizer configuration
            
        Returns
        -------
        BaseTokenizer
            Configured tokenizer instance
            
        Raises
        ------
        ValueError
            If the specified tokenizer type is not supported
        """
        # Simple if-else factory pattern
        if config.tokenizer_type == 'SMILESTokenizer':
            from hyformer.tokenizers.smiles import SMILESTokenizer
            return SMILESTokenizer.from_config(config)
        elif config.tokenizer_type == "AATokenizer":
            from hyformer.tokenizers.amino_acid import AATokenizer
            return AATokenizer.from_config(config)
        elif config.tokenizer_type == "HFTokenizer":
            from hyformer.tokenizers.hf import HFTokenizer
            return HFTokenizer.from_config(config)
        else:
            raise ValueError(f"Tokenizer type '{config.tokenizer_type}' is not supported. "
                             f"Supported types: 'SMILESTokenizer', 'AATokenizer', 'HFTokenizer'")
    
    @classmethod
    def get_supported_tokenizer_types(cls) -> Dict[str, str]:
        """Get a mapping of supported tokenizer types to their descriptions.
        
        Returns
        -------
        dict
            Dictionary mapping tokenizer types to descriptions
        """
        return {
            'SMILESTokenizer': 'Tokenizer for SMILES molecular representations',
            'AATokenizer': 'Tokenizer for amino acid sequences', 
            'HFTokenizer': 'Wrapper for Hugging Face tokenizers'
        }

    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_path: str,
        revision: str = "main",
        tokenizer_config: Optional[Dict[str, Any]] = None,
        local_dir: Optional[str] = None,
        local_dir_use_symlinks: str = "auto",
        subfolder: Optional[str] = None,
        repo_type: Optional[str] = None,
        **kwargs
    ) -> BaseTokenizer:
        """Load a pretrained tokenizer from HuggingFace Hub or a local path.

        Parameters
        ----------
        repo_id_or_path : str
            Path to local directory containing tokenizer files or HuggingFace Hub repository ID.
        revision : str, optional
            Git revision for HuggingFace Hub repositories, by default "main".
        tokenizer_config : dict, optional
            Tokenizer configuration dictionary. If None, will attempt to load from
            tokenizer_config.json, by default None.
        local_dir : str, optional
            Local directory to download the tokenizer files from HuggingFace Hub,
            by default None.
        local_dir_use_symlinks : str, optional
            Whether to use symlinks for local directory, by default "auto".
        **kwargs
            Additional keyword arguments passed to the tokenizer constructor.

        Returns
        -------
        BaseTokenizer
            Loaded tokenizer instance.

        Raises
        ------
        ValueError
            If tokenizer config is not found or tokenizer type is not supported.

        Examples
        --------
        Load from HuggingFace Hub:
        ```
        tokenizer = AutoTokenizer.from_pretrained("SzczurekLab/hyformer-tokenizer")
        ```
        
        Load from local directory:
        ```
        tokenizer = AutoTokenizer.from_pretrained("./my_tokenizer")
        ```
        """
        # Normalize repo_id and subfolder if a Hub path like "org/repo/sub/dir" was provided
        if not os.path.isdir(repo_id_or_path) and "/" in repo_id_or_path:
            parts = repo_id_or_path.split("/")
            if len(parts) > 2:
                repo_id_or_path = "/".join(parts[:2])
                inferred_subfolder = "/".join(parts[2:])
                if subfolder is None:
                    subfolder = inferred_subfolder

        # Load tokenizer config to determine tokenizer type
        if tokenizer_config is None:
            base_local_path = os.path.join(repo_id_or_path, subfolder) if (subfolder and os.path.isdir(repo_id_or_path)) else repo_id_or_path
            config_path_local = os.path.join(base_local_path, TOKENIZER_CONFIG_FILENAME)
            if os.path.exists(config_path_local):
                import json
                with open(config_path_local, 'r') as f:
                    tokenizer_config = json.load(f)
            else:
                try:
                    if hf_hub_download is None:
                        raise ValueError("HuggingFace Hub is not available and no local config found")
                    config_path_hf = hf_hub_download(
                        repo_id=repo_id_or_path, 
                        filename=TOKENIZER_CONFIG_FILENAME, 
                        revision=revision,
                        local_dir=local_dir, 
                        local_dir_use_symlinks=local_dir_use_symlinks,
                        subfolder=subfolder,
                        repo_type=repo_type
                    )
                    import json
                    with open(config_path_hf, 'r') as f:
                        tokenizer_config = json.load(f)
                except (Exception, RepositoryNotFoundError) as e:
                    raise ValueError(f"Tokenizer config not found in {repo_id_or_path}")

        # Get tokenizer type from config
        tokenizer_type = tokenizer_config.get('tokenizer_type')
        if not tokenizer_type:
            raise ValueError("Tokenizer config missing 'tokenizer_type' field")

        # Load the appropriate tokenizer class and call its from_pretrained method
        if tokenizer_type == 'SMILESTokenizer':
            from hyformer.tokenizers.smiles import SMILESTokenizer
            return SMILESTokenizer.from_pretrained(
                repo_id_or_path, revision, tokenizer_config, local_dir, local_dir_use_symlinks, subfolder=subfolder, repo_type=repo_type, **kwargs
            )
        elif tokenizer_type == "AATokenizer":
            from hyformer.tokenizers.amino_acid import AATokenizer
            return AATokenizer.from_pretrained(
                repo_id_or_path, revision, tokenizer_config, local_dir, local_dir_use_symlinks, subfolder=subfolder, repo_type=repo_type, **kwargs
            )
        elif tokenizer_type == "HFTokenizer":
            from hyformer.tokenizers.hf import HFTokenizer
            return HFTokenizer.from_pretrained(
                repo_id_or_path, revision, tokenizer_config, local_dir, local_dir_use_symlinks, subfolder=subfolder, repo_type=repo_type, **kwargs
            )
        else:
            raise ValueError(f"Tokenizer type '{tokenizer_type}' is not supported. "
                             f"Supported types: 'SMILESTokenizer', 'AATokenizer', 'HFTokenizer'")
