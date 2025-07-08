""" Implements base model classes.

The module implements:
- PreTrainedModel: Core save/load functionality with compile-aware state dict handling
- EncoderModel: Encoder-like models (e.g., feature extraction)
- DecoderModel: Decoder-like models (e.g., generation)
- TrainableModel: Compatible with Trainer, full training capabilities with optimizer configuration

Examples
--------
Basic usage with a custom model:

```
from hyformer.models.base import PreTrainedModel
from hyformer.models.utils import ModelOutput

@AutoModel.register("my_model")
class MyModel(TrainableModel):
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, input_ids: torch.Tensor, **kwargs) -> ModelOutput:
        embeddings = self.embedding(input_ids)
        return ModelOutput(embeddings=embeddings, task=task)

# Load and use the model
model_config = ModelConfig(model_type="my_model", vocab_size=1000, embedding_dim=256)
model = MyModel.from_config(model_config)
state_dict = ...
model.load_state_dict_safe(state_dict)
...
torch.save(model.state_dict_safe(), "checkpoint.pt")
"""

#TODO: implement logging module

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
from hyformer.models.utils import _adapt_state_dict_for_compiled_model, _remove_compile_artifacts_from_state_dict

try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError
except ImportError:
    hf_hub_download = None
    RepositoryNotFoundError = Exception
    warnings.warn("HuggingFace Hub is not installed. Loading models from HuggingFace not available.")

MODEL_CONFIG_FILENAME = "model_config.json"
MODEL_WEIGHTS_FILENAME = "ckpt.pt"


class PreTrainedModel(nn.Module, ABC):
    """ Base class for pre-trained models with unified save/load functionality.

    Methods
    -------
    from_pretrained : classmethod, abstract
        Load a pretrained model from HuggingFace Hub or local path.
    from_config : classmethod, abstract
        Create a model instance from a configuration object.
    state_dict_safe : method
        Get model state dict with compile artifacts removed.
    load_state_dict_safe : method
        Safely load state dict with automatic compile artifact adaptation.
    get_num_params : method
        Calculate the total number of parameters.
    forward : method, abstract
        Forward pass of the model.
    """
    def __init__(self) -> None:
        super().__init__()

    def state_dict_safe(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """ Get model state dict with compile artifacts removed.
        """
        state_dict = super().state_dict(*args, **kwargs)
        state_dict = _remove_compile_artifacts_from_state_dict(state_dict)
        return state_dict
    
    def load_state_dict_safe(self, state_dict: Dict[str, torch.Tensor]) -> "PreTrainedModel":
        """ Safely load state dict with automatic compile artifact adaptation.
        """
        state_dict = _adapt_state_dict_for_compiled_model(state_dict, self)
        try:
            self.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            warnings.warn(f"Model state_dict loaded with `strict=False`. Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")
            return self

    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_path: str,
        revision: str = "main",
        device: str = "cpu",
        model_config: Optional[ModelConfig] = None,
        local_dir: Optional[str] = None,
        local_dir_use_symlinks: str = "auto"
    ) -> "PreTrainedModel":
        """ Load pretrained model weights from HuggingFace Hub or a local path.

        Parameters
        ----------
        repo_id_or_path : str
            Path to local checkpoint file (.pt) or HuggingFace Hub repository ID.
            Local paths should end with '.pt' extension.
        revision : str, optional
            Git revision for HuggingFace Hub repositories, by default "main".
        device : str, optional
            Device to load the model on, by default "cpu".
        model_config : ModelConfig, optional
            Model configuration object. Required for local checkpoint loading,
            by default None.
        local_dir : str, optional
            Local directory to download the model weights and config from HuggingFace Hub,
            by default None.
        local_dir_use_symlinks : str, optional
            Whether to use symlinks for local directory, by default "auto".

        Returns
        -------
        PreTrainedModel
            Loaded model instance with pretrained weights.

        Raises
        ------
        ValueError
            If model weights or config are not found.

        Examples
        --------
        Load from HuggingFace Hub:
        ```
        model = Hyformer.from_pretrained("SzczurekLab/hyformer")
        ```
        
        """

        # load state dict
        _state_dict_path_local = os.path.join(repo_id_or_path, MODEL_WEIGHTS_FILENAME)
        if os.path.exists(_state_dict_path_local):
            state_dict = torch.load(_state_dict_path_local, map_location=device)
        else:
            try:
                _state_dict_path_hf = hf_hub_download(
                    repo_id=repo_id_or_path, filename=MODEL_WEIGHTS_FILENAME, revision=revision,
                    local_dir=local_dir, local_dir_use_symlinks=local_dir_use_symlinks)
                state_dict = torch.load(_state_dict_path_hf, map_location=device)
            except (HTTPError, RepositoryNotFoundError) as e:
                raise ValueError(f"Model weights not found in {repo_id_or_path}")
        print(f"Model weights loaded from {repo_id_or_path}")
        
        # load model config
        if model_config is None:
            _model_config_path_local = os.path.join(repo_id_or_path, MODEL_CONFIG_FILENAME)
            if os.path.exists(_model_config_path_local):
                model_config = ModelConfig.from_config_filepath(_model_config_path_local)
            else:
                try:
                    _model_config_path_hf = hf_hub_download(
                        repo_id=repo_id_or_path, filename=MODEL_CONFIG_FILENAME, revision=revision,
                        local_dir=local_dir, local_dir_use_symlinks=local_dir_use_symlinks)
                    model_config = ModelConfig.from_config_filepath(_model_config_path_hf)
                except (HTTPError, RepositoryNotFoundError) as e:
                    raise ValueError(f"Model config not found in {repo_id_or_path}")
        print(f"Model config loaded from {repo_id_or_path}")
        
        # initialize
        model = cls.from_config(model_config)
        model.load_state_dict_safe(state_dict)
        return model
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: ModelConfig) -> "PreTrainedModel":
        """ Create a model instance from a configuration object.

        Parameters
        ----------
        config : ModelConfig
            Configuration object containing model parameters.

        Returns
        -------
        PreTrainedModel
            Instantiated model instance.
        """
        pass

    def get_num_params(self) -> int:
        """ Calculate the total number of parameters in the model.

        Returns
        -------
        int
            Total number of parameters in the model.

        """
        return sum(p.numel() for p in self.parameters())


class EncoderModel(PreTrainedModel, ABC):
    """ Abstract base class for encoder-like models.
    """

    @abstractmethod
    def encode(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Encode input into embeddings.

        Parameters
        ----------
        *args : tuple
            Positional arguments for encoding.
        **kwargs : dict
            Keyword arguments for encoding.

        Returns
        -------
        torch.Tensor
            Encoded representations.

        """
        pass


class DecoderModel(PreTrainedModel, ABC):
    """Abstract base class for decoder-like models.
    """

    @abstractmethod
    def generate(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Generate sequences from input.

        Parameters
        ----------
        *args : tuple
            Positional arguments for generation.
        **kwargs : dict
            Keyword arguments for generation.

        Returns
        -------
        torch.Tensor
            Generated token sequences.

        """
        pass


class TrainableModel(PreTrainedModel, ABC):
    """Abstract base class for models that can be trained with the Hyformer Trainer.
    """
    
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
        loss_fn_reduction: str = 'mean',
        nan_target_idx: int = -1,
        **kwargs
    ) -> ModelOutput:
        """Forward pass of the trainable model with loss calculation.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs of shape (batch_size, sequence_length).
        task : str
            Task type. Must be one of 'lm', 'mlm', or 'prediction'.
        attention_mask : torch.Tensor, optional
            Attention mask of shape (batch_size, sequence_length) indicating
            which tokens should be attended to, by default None.
        next_token_only : bool, optional
            Whether to return only the next token's output, by default False.
        use_cache : bool, optional
            Whether to use caching for faster inference, by default False.
        input_labels : torch.Tensor, optional
            Labels for language modeling tasks of shape (batch_size, sequence_length),
            by default None.
        target : torch.Tensor, optional
            Target values for prediction tasks of shape (batch_size, num_tasks),
            by default None.
        loss_fn_reduction : str, optional
            Reduction method for loss calculation ('mean', 'sum', 'none'),
            by default 'mean'.
        nan_target_idx : int, optional
            Index for NaN targets in prediction tasks, by default -1.
        **kwargs : dict
            Additional keyword arguments passed to the model.

        Returns
        -------
        ModelOutput
            Model outputs.

        Raises
        ------
        ValueError
            If task is not one of the supported task types.

        Examples
        --------
        Language modeling task:

            outputs = model.forward(
                input_ids=input_ids,
                task='lm',
                input_labels=labels
            )
            loss = outputs['loss']

        Prediction task:

            outputs = model.forward(
                input_ids=input_ids,
                task='prediction',
                target=target_values
            )
            predictions = outputs['logits']
        """
        pass

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str
    ) -> torch.optim.Optimizer:
        """Configure optimizer with weight decay grouping.

        Parameters
        ----------
        weight_decay : float
            Weight decay coefficient.
        learning_rate : float
            Learning rate for the optimizer.
        betas : tuple
            Beta parameters for AdamW optimizer (beta1, beta2).
        device_type : str
            Device type ('cuda' or 'cpu').

        Returns
        -------
        torch.optim.AdamW
            Configured AdamW optimizer.
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
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    @abstractmethod
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for specific module.
        """
        pass

    def initialize_parameters(self) -> None:
        """Initialize parameters of the model.
        """
        self.apply(self._init_weights)
