import os
from typing import Dict, Type, Optional

from hyformer.configs.model import ModelConfig
from hyformer.models.base import PreTrainedModel

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

class AutoModel:
    registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, model_type: str):
        """
        Decorator to register a model class with a given model_type string.

        Parameters
        ----------
        model_type : str
            The string identifier for the model type.

        Returns
        -------
        Callable
            A decorator that registers the model class.
        """
        def decorator(model_cls: Type) -> Type:
            cls.registry[model_type] = model_cls
            return model_cls
        return decorator

    @classmethod
    def from_config(cls, config: ModelConfig) -> PreTrainedModel:
        """
        Instantiate the correct model from a ModelConfig instance or dict.

        Parameters
        ----------
        config : ModelConfig or dict
            Configuration object with a 'model_type' attribute or key.

        Returns
        -------
        Any
            An instance of the registered model class, initialized with filtered config.

        Raises
        ------
        ValueError
            If model_type is missing or not registered.
        """
        model_type = getattr(config, 'model_type', None) or config.get('model_type', None)
        if not model_type or model_type not in cls.registry:
            raise ValueError(f"Unknown or missing model_type: {model_type}")
        model_cls = cls.registry[model_type]
        return model_cls.from_config(config)
    
    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_path: str,
        revision: str = "main",
        device: str = "cpu",
        model_config: Optional[ModelConfig] = None,
        local_dir: Optional[str] = None,
        local_dir_use_symlinks: str = "auto"
        ) -> PreTrainedModel:
        """
        Load a pretrained model from HuggingFace Hub or a local path.
        """
        if model_config is None:
            try:
                from hyformer.models.core.base import MODEL_CONFIG_FILENAME
                model_config = ModelConfig.from_config_filepath(os.path.join(repo_id_or_path, MODEL_CONFIG_FILENAME))
            except FileNotFoundError:
                if hf_hub_download is not None:
                    model_config_filepath = hf_hub_download(
                        repo_id=repo_id_or_path, filename=MODEL_CONFIG_FILENAME, revision=revision,
                        local_dir=local_dir, local_dir_use_symlinks=local_dir_use_symlinks)
                    model_config = ModelConfig.from_config_filepath(model_config_filepath)
                else:
                    raise FileNotFoundError(f"Model config not found in {repo_id_or_path}")
        model_type = getattr(model_config, 'model_type', None) or model_config.get('model_type', None)
        if not model_type or model_type not in cls.registry:
            raise ValueError(f"Unknown or missing model_type: {model_type}")
        model_cls = cls.registry[model_type]
        return model_cls.from_pretrained(repo_id_or_path, revision, device, model_config, local_dir, local_dir_use_symlinks)
