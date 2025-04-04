import torch

from typing import Any, Optional


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
    
    def __init__(self, embeddings: torch.Tensor, logits: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, task: str = None, loss: Optional[torch.Tensor] = None, past_key_values: Optional[torch.Tensor] = None):
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
        past_key_values : Optional[torch.Tensor], optional
            Past key values of type torch.Tensor with dtype torch.float in shape (batch_size, sequence_length, embedding_dim).
        """
        super().__init__(embeddings=embeddings, logits=logits, attention_mask=attention_mask, task=task, loss=loss, past_key_values=past_key_values)
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
    
    # @property
    # def masked_logits(self):
    #     if self['logits'] is not None:
    #         return self['logits'][self['attention_mask'] == 1]
    #     else:
    #         assert False
