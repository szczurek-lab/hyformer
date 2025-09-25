"""Base model classes for Hyformer.

This module defines abstract base classes that implement shared functionality:

- PreTrainedModel: core save/load utilities with compile-aware state dict handling.
- EncoderModel: interface for encoder-like models (e.g., feature extraction).
- DecoderModel: interface for decoder-like models (e.g., generation).
- TrainableModel: trainer-compatible models with optimizer configuration.
"""

import os
import inspect
import warnings
import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from requests.exceptions import HTTPError

from hyformer.configs.model import ModelConfig
from hyformer.models.utils import ModelOutput
from hyformer.models.utils import (
    _adapt_state_dict_for_compiled_model,
    _remove_compile_artifacts_from_state_dict,
)

try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError
except ImportError:
    hf_hub_download = None
    RepositoryNotFoundError = Exception
    warnings.warn(
        "HuggingFace Hub is not installed. Loading models from HuggingFace not available."
    )

_TOKENIZER_CONFIG_FILENAME = "tokenizer_config.json"
_MODEL_CONFIG_FILENAME = "model_config.json"
_MODEL_WEIGHTS_FILENAME = "ckpt.pt"
_MODEL_WEIGHTS_KEY = "model"


class PreTrainedModel(nn.Module, ABC):
    """Base class for pre-trained models."""

    def __init__(self) -> None:
        super().__init__()

    def _state_dict_safe(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Return a state dict with compile artifacts removed.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model state dict without compilation artifacts.
        """
        state_dict = super().state_dict(*args, **kwargs)
        state_dict = _remove_compile_artifacts_from_state_dict(state_dict)
        return state_dict

    def _load_state_dict_safe(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> "PreTrainedModel":
        """Load a state dict with automatic adaptation of compile artifacts.

        Parameters
        ----------
        state_dict : Dict[str, torch.Tensor]
            State dict to load.

        Returns
        -------
        PreTrainedModel
            Self for convenience.
        """
        state_dict = _adapt_state_dict_for_compiled_model(state_dict, self)
        try:
            self.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=False
            )
            print(
                f"Model state_dict loaded with `strict=False`. Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}"
            )
            return self
        return self

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        revision: str = "main",
        device: str = "cpu",
        model_config: Optional[ModelConfig] = None,
        local_dir: Optional[str] = None,
        local_dir_use_symlinks: str = "auto",
    ) -> "PreTrainedModel":
        """Load pretrained model weights from HuggingFace Hub or a local path.

        Parameters
        ----------
        pretrained_model_name_or_path : str
            Path to a local checkpoint file (``.pt``), a local directory containing
            the checkpoint, or a HuggingFace Hub repository ID.
        revision : str, optional
            Git revision for HuggingFace Hub repositories, by default ``"main"``.
        device : str, optional
            Device to load the model on, by default ``"cpu"``.
        model_config : ModelConfig, optional
            Model configuration object. Required for local checkpoint loading.
        local_dir : str, optional
            Local directory for HuggingFace Hub downloads.
        local_dir_use_symlinks : str, optional
            Whether to use symlinks for the local directory, by default ``"auto"``.

        Returns
        -------
        PreTrainedModel
            Loaded model instance with pretrained weights.

        Raises
        ------
        ValueError
            If model weights or configuration are not found.

        Examples
        --------
        Load from HuggingFace Hub:

        >>> model = Hyformer.from_pretrained("SzczurekLab/hyformer_peptides")
        """

        ###
        # Load model weights and model config
        ###

        _state_dict_path_local = os.path.join(
            pretrained_model_name_or_path, _MODEL_WEIGHTS_FILENAME
        )

        if os.path.exists(
            _state_dict_path_local
        ):  # if local path exists, load state dict from local path
            print(f"Model weights loaded from {_state_dict_path_local}")
            state_dict = torch.load(_state_dict_path_local, map_location=device)[
                _MODEL_WEIGHTS_KEY
            ]
        else:
            try:  # if local path does not exist, load state dict from HuggingFace Hub
                _state_dict_path_hf = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename=_MODEL_WEIGHTS_FILENAME,
                    revision=revision,
                    local_dir=local_dir,
                    local_dir_use_symlinks=local_dir_use_symlinks,
                )
                state_dict = torch.load(_state_dict_path_hf, map_location=device)[
                    "model"
                ]
                print(f"Model weights loaded from {pretrained_model_name_or_path}")
            except (HTTPError, RepositoryNotFoundError) as e:
                raise ValueError(
                    f"Model weights not found in {pretrained_model_name_or_path}"
                )

        if (
            model_config is None
        ):  # if model config is not provided, load model config from local path
            _model_config_path_local = os.path.join(
                pretrained_model_name_or_path, _MODEL_CONFIG_FILENAME
            )
            if os.path.exists(_model_config_path_local):
                model_config = ModelConfig.from_config_filepath(
                    _model_config_path_local
                )
                print(f"Model config loaded from {pretrained_model_name_or_path}")
            else:
                try:  # if local path does not exist, load model config from HuggingFace Hub
                    _model_config_path_hf = hf_hub_download(
                        repo_id=pretrained_model_name_or_path,
                        filename=_MODEL_CONFIG_FILENAME,
                        revision=revision,
                        local_dir=local_dir,
                        local_dir_use_symlinks=local_dir_use_symlinks,
                    )
                    model_config = ModelConfig.from_config_filepath(
                        _model_config_path_hf
                    )
                    print(f"Model config loaded from {pretrained_model_name_or_path}")
                except (HTTPError, RepositoryNotFoundError) as e:
                    raise ValueError(
                        f"Model config not found in {pretrained_model_name_or_path}"
                    )

        ###
        # Initialize model
        ###

        model = cls.from_config(model_config)
        model._load_state_dict_safe(state_dict)
        return model

    @classmethod
    @abstractmethod
    def from_config(cls, config: ModelConfig) -> "PreTrainedModel":
        """Create a model instance from a configuration object.

        Parameters
        ----------
        config : ModelConfig
            Model configuration containing all required parameters.

        Returns
        -------
        PreTrainedModel
            Instantiated model instance.
        """
        pass

    def get_num_params(self) -> int:
        """Return the total number of parameters in the model.

        Returns
        -------
        int
            Total number of parameters.
        """
        return sum(p.numel() for p in self.parameters())


class EncoderModel(PreTrainedModel, ABC):
    """Abstract base class for encoder-like models."""

    @abstractmethod
    def encode(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Encode inputs into embeddings.

        Parameters
        ----------
        *args
            Positional arguments for encoding.
        **kwargs
            Keyword arguments for encoding.

        Returns
        -------
        torch.Tensor
            Model embeddings.
        """
        pass


class DecoderModel(PreTrainedModel, ABC):
    """Abstract base class for decoder-like models."""

    @abstractmethod
    def generate(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Generate sequences from inputs.

        Parameters
        ----------
        *args
            Positional arguments for generation.
        **kwargs
            Keyword arguments for generation.

        Returns
        -------
        torch.Tensor
            Generated token sequences.
        """
        pass


class TrainableModel(PreTrainedModel, ABC):
    """Abstract base class for models that can be trained with the Hyformer Trainer."""

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        task: str,
        attention_mask: Optional[torch.Tensor] = None,
        next_token_only: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        input_labels: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        loss_fn_reduction: str = "mean",
        nan_target_idx: int = -1,
        **kwargs,
    ) -> ModelOutput:
        """Forward pass of a trainable model with optional loss computation.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs of shape ``(batch_size, sequence_length)``.
        task : str
            Task type. One of ``{"lm", "mlm", "prediction"}``.
        attention_mask : torch.Tensor, optional
            Attention mask of shape ``(batch_size, sequence_length)`` indicating
            which tokens should be attended to.
        next_token_only : bool, optional
            Whether to return only the next token's output.
        use_cache : bool, optional
            Whether to use caching for faster inference.
        input_labels : torch.Tensor, optional
            Labels for language modeling tasks of shape ``(batch_size, sequence_length)``.
        target : torch.Tensor, optional
            Target values for prediction tasks of shape ``(batch_size, num_tasks)``.
        loss_fn_reduction : str, optional
            Reduction method for loss calculation (``"mean"``, ``"sum"``, ``"none"``).
        nan_target_idx : int, optional
            Index used to mark NaN targets in prediction tasks.
        **kwargs
            Additional keyword arguments passed to the model.

        Returns
        -------
        ModelOutput
            Model outputs including loss and/or logits depending on ``task``.

        Raises
        ------
        ValueError
            If ``task`` is not supported.

        Examples
        --------
        Language modeling task:

        >>> outputs = model.forward(input_ids=input_ids, task='lm', input_labels=labels)
        >>> loss = outputs['loss']

        Prediction task:

        >>> outputs = model.forward(input_ids=input_ids, task='prediction', target=target_values)
        >>> predictions = outputs['logits']
        """
        pass

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        """Configure an optimizer with weight-decay parameter grouping.

        Parameters
        ----------
        weight_decay : float
            Weight decay coefficient.
        learning_rate : float
            Learning rate for the optimizer.
        betas : tuple[float, float]
            Beta parameters for AdamW optimizer ``(beta1, beta2)``.
        device_type : str
            Device type (``"cuda"`` or ``"cpu"``).

        Returns
        -------
        torch.optim.Optimizer
            Configured optimizer instance.
        """
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = []
        no_decay_params = []
        for name, param in param_dict.items():
            if name.endswith(".bias") or "norm" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        return optimizer

    @abstractmethod
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for a specific module."""
        pass

    def initialize_parameters(self) -> None:
        """Initialize parameters of the model."""
        self.apply(self._init_weights)
