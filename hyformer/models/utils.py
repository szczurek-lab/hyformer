import torch

from typing import Any, Optional, Dict


def clean_checkpoint_for_compiled_model(state_dict: Dict[str, torch.Tensor], model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Clean a state dict to handle compiled model artifacts.
    
    This function handles the mismatch between compiled and uncompiled model states
    by adjusting the parameter names in the state dict.
    
    Args:
        state_dict: The state dict to clean
        model: The model to load the state dict into
        
    Returns:
        The cleaned state dict
    """
    # Check if current model is compiled
    is_current_model_compiled = any(key.startswith("_orig_mod") for key in model.state_dict().keys())
    
    # Check if loaded state dict is from a compiled model
    is_loaded_state_compiled = any(key.startswith("_orig_mod") for key in state_dict.keys())
    
    # If there's a mismatch between compiled states, we need to adjust the keys
    if is_current_model_compiled != is_loaded_state_compiled:
        unwanted_prefix = '_orig_mod.'
        for k, _ in list(state_dict.items()):
            if is_loaded_state_compiled and k.startswith(unwanted_prefix):
                # Remove prefix if loading compiled weights into uncompiled model
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            elif is_current_model_compiled and not k.startswith(unwanted_prefix):
                # Add prefix if loading uncompiled weights into compiled model
                state_dict[f"{unwanted_prefix}{k}"] = state_dict.pop(k)
    
    return state_dict


class ModelInput(dict):

    def __init__(self, input_ids: torch.Tensor, task: str, attention_mask: Optional[torch.Tensor] = None, input_labels: Optional[torch.Tensor] = None, target: Optional[torch.Tensor] = None):
        """Initialize the ModelInput object.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input IDs of type torch.Tensor with dtype torch.int in shape (batch_size, sequence_length).
        task : str
            Task of type str.
        attention_mask : Optional[torch.Tensor], optional
            Attention mask of type torch.Tensor with dtype torch.bool in shape (batch_size, sequence_length).
        input_labels : Optional[torch.Tensor], optional
            Input labels of type torch.Tensor with dtype torch.int in shape (batch_size, sequence_length).
        target : Optional[torch.Tensor], optional
            target of type torch.Tensor with dtype torch.float in shape (batch_size, num_prediction_tasks).
        """
        super().__init__(input_ids=input_ids, task=task, attention_mask=attention_mask, input_labels=input_labels, target=target)

    def to_device(self, device: torch.device) -> 'ModelInput':
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                self[key] = value.to(device, non_blocking=True)
        return self


class ModelOutput(dict):
    
    def __init__(self, embeddings: torch.Tensor, logits: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, task: str = None, loss: Optional[torch.Tensor] = None):
        """Initialize the ModelOutput object.

        Parameters
        ----------
        embeddings : torch.Tensor
            Embeddings of type torch.Tensor with dtype torch.float in shape (batch_size, sequence_length, embedding_dim).
        logits : Optional[torch.Tensor], optional
            Logits of type torch.Tensor with dtype torch.float in shape (batch_size, sequence_length, vocab_size).
        attention_mask : Optional[torch.Tensor], optional
            Attention mask of type torch.Tensor with dtype torch.bool in shape (batch_size, sequence_length).
        task : str, optional
            Task of type str.
        loss : Optional[torch.Tensor], optional
            Loss of type torch.Tensor with dtype torch.float in shape (batch_size, num_prediction_tasks).
        """
        super().__init__(embeddings=embeddings, logits=logits, attention_mask=attention_mask, task=task, loss=loss)
        self.masked_logits = None
        self.masked_embeddings = None
    
    # @property
    # def masked_embeddings(self):
    #     _is_cls_token_embedding = True if self['cls_embeddings'] is not None else False
    #     if _is_cls_token_embedding:
    #         return self['cls_embeddings']
    #     elif self['attention_mask'] is not None:
    #         attn_mask = self["attention_mask"]
    #         if _is_cls_token_embedding:
    #             attn_mask = attn_mask[:, 1:]
    #         w = attn_mask / attn_mask.sum(dim=-1, keepdim=True)
    #         w = w.unsqueeze(-2)
    #         global_embedding = w @ self['lm_embeddings']
    #         return global_embedding.squeeze(-2)
    #     else:
    #         assert False
    
    @property
    def logits_masked(self):
        if self['logits'] is not None:
            return self['logits'].where(self['attention_mask'].unsqueeze(-1).repeat(1, 1, self['logits'].shape[-1]) == 0, -torch.inf)
        else:
            assert False
