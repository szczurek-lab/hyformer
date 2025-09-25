"""Factory helpers for constructing Hyformer models. """

import os
from typing import Optional, Type

from hyformer.configs.model import ModelConfig
from hyformer.models.base import PreTrainedModel

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None


def _load_model_class(model_type: str) -> Type[PreTrainedModel]:
    """Return the concrete model class for the requested ``model_type``."""

    if model_type == "Hyformer":
        from hyformer.models.hyformer import Hyformer
        return Hyformer
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


class AutoModel:
    """Factory conveniences for building models from configs or checkpoints."""

    @staticmethod
    def from_config(config: ModelConfig) -> PreTrainedModel:
        """Instantiate a model directly from a ``ModelConfig``."""
        model_type = getattr(config, "model_type", None) or (
            config.get("model_type", None) if hasattr(config, "get") else None
        )
        if not model_type:
            raise ValueError("ModelConfig is missing 'model_type'.")

        model_cls = _load_model_class(model_type)
        return model_cls.from_config(config)

    @staticmethod
    def from_pretrained(
        pretrained_model_name_or_path: str,
        revision: str = "main",
        device: str = "cpu",
        model_config: Optional[ModelConfig] = None,
        local_dir: Optional[str] = None,
        local_dir_use_symlinks: str = "auto",
    ) -> PreTrainedModel:
        """Instantiate a model and load weights from the specified source."""
        if model_config is None:
            from hyformer.models.base import _MODEL_CONFIG_FILENAME

            config_dir: Optional[str] = None
            if os.path.isdir(pretrained_model_name_or_path):
                config_dir = pretrained_model_name_or_path
            elif os.path.isfile(pretrained_model_name_or_path):
                config_dir = os.path.dirname(pretrained_model_name_or_path)

            loaded_path: Optional[str] = None
            if config_dir is not None:
                candidate = os.path.join(config_dir, _MODEL_CONFIG_FILENAME)
                if os.path.exists(candidate):
                    loaded_path = candidate
            if loaded_path is None and hf_hub_download is not None:
                loaded_path = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename=_MODEL_CONFIG_FILENAME,
                    revision=revision,
                    local_dir=local_dir,
                    local_dir_use_symlinks=local_dir_use_symlinks,
                )
            if loaded_path is None:
                raise FileNotFoundError(
                    f"Model config not found in {pretrained_model_name_or_path}"
                )
            model_config = ModelConfig.from_config_filepath(loaded_path)

        model_type = getattr(model_config, "model_type", None) or (
            model_config.get("model_type", None) if hasattr(model_config, "get") else None
        )
        if not model_type:
            raise ValueError("Model config is missing 'model_type'.")

        model_cls = _load_model_class(model_type)
        return model_cls.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            revision=revision,
            device=device,
            model_config=model_config,
            local_dir=local_dir,
            local_dir_use_symlinks=local_dir_use_symlinks,
        )
