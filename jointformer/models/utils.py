import torch
from typing import Any


class ModelInput(dict):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def to(self, device: str, pin_memory: bool = True) -> 'ModelInput':
        if device != 'cpu':
            for key, value in self.items():
                if isinstance(value, torch.Tensor):
                    if pin_memory:
                        self[key] = value.pin_memory().to(device, non_blocking=True)
                    else:
                        self[key] = value.to(device, non_blocking=True)
        return self


class ModelOutput(dict):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def global_embeddings(self):
        _is_cls_token_embedding = True if self['cls_embeddings'] is not None else False
        if _is_cls_token_embedding and False:
            return self['cls_embeddings']
        elif self['attention_mask'] is not None:
            attn_mask = self["attention_mask"]
            if _is_cls_token_embedding:
                attn_mask = attn_mask[:, 1:]
            w = attn_mask / attn_mask.sum(dim=-1, keepdim=True)
            w = w.unsqueeze(-2)
            global_embedding = w @ self['lm_embeddings']
            return global_embedding.squeeze(-2)
        else:
            assert False
    
    def __getitem__(self, key: Any) -> Any:
        if key == 'global_embeddings':
            return self.global_embeddings
        else:
            return super().__getitem__(key)
    
