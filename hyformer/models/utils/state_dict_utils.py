"""
State dict utilities for handling PyTorch compile artifacts and prefix adaptation.
"""
import torch
from typing import Dict
import warnings

UNWANTED_PREFIX = '_orig_mod.'


def _remove_compile_artifacts_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Clean a state dict to handle compiled model artifacts. """
    for k, _ in list(state_dict.items()):
        if k.startswith(UNWANTED_PREFIX):
            state_dict[k[len(UNWANTED_PREFIX):]] = state_dict.pop(k)
    return state_dict


def _adapt_state_dict_for_compiled_model(state_dict: Dict[str, torch.Tensor], model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Clean a state dict to handle compiled model artifacts. """
    is_model_compiled = any(key.startswith(UNWANTED_PREFIX) for key in model.state_dict().keys())
    is_state_dict_compiled = any(key.startswith(UNWANTED_PREFIX) for key in state_dict.keys())

    if is_model_compiled and not is_state_dict_compiled:
        warnings.warn("Model is compiled but state dict is not. Adding prefix to state dict.")
        new_state_dict = {f"{UNWANTED_PREFIX}{k}": v for k, v in state_dict.items()}
        return new_state_dict
    elif not is_model_compiled and is_state_dict_compiled:
        warnings.warn("Model is not compiled but state dict is. Removing prefix from state dict.")
        new_state_dict = {k[len(UNWANTED_PREFIX):] if k.startswith(UNWANTED_PREFIX) else k: v for k, v in state_dict.items()}
        return new_state_dict
    else:
        return state_dict 
    