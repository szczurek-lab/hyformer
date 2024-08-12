import torch

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
    
    def __init__(self, embeddings, loss=None, logits_generation=None, logits_prediction=None, logits_physchem=None, attention_mask=None):
        super().__init__(
            embeddings=embeddings,
            loss=loss,
            logits_generation=logits_generation,
            logits_prediction=logits_prediction,
            logits_physchem=logits_physchem,
            attention_mask=attention_mask
            )
     
    @property
    def global_embedding(self):
        if self.attention_mask is None:
            return self.embeddings.mean(dim=-1)
        else:
            w = self.attention_mask / self.attention_mask.sum(dim=-1, keepdim=True)
            w = w.unsqueeze(-2)
            global_embedding = w @ self.embeddings
            return global_embedding.squeeze(-2)
    
    