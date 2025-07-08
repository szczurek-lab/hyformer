import inspect
import torch

import torch.nn as nn

from typing import Optional

from hyformer.models.layers.layer_norm import RMSNorm
from hyformer.models.layers.transformer_layer import TransformerLayer
from hyformer.models.utils import ModelOutput, clean_checkpoint_for_compiled_model
from hyformer.configs.model import ModelConfig
from hyformer.models.base import TrainableModel


class LLAMABackbone(TrainableModel):

    def __init__(
            self, vocab_size: int, embedding_dim: int, hidden_embedding_dim: int, attention_dropout_p: float,
            num_transformer_layers: int, num_attention_heads: int, layer_norm_eps: float, init_weights: bool = True) -> None:
        """
        Parameters
        ----------
        vocab_size : int
            The size of the vocabulary.
        embedding_dim : int
            The dimension of the embedding.
        hidden_embedding_dim : int
            The dimension of the hidden embedding.
        attention_dropout_p : float
            The dropout rate for the attention.
        num_transformer_layers : int
            The number of transformer layers.
        num_attention_heads : int
            The number of attention heads.
        layer_norm_eps : float
            The epsilon for the layer normalization.
        init_weights : bool, optional
            Whether to initialize the model weights, by default True.
        """
        super().__init__()
        
        # Required parameters
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_embedding_dim = hidden_embedding_dim
        self.attention_dropout_p = attention_dropout_p
        self.num_transformer_layers = num_transformer_layers
        self.num_attention_heads = num_attention_heads
        self.layer_norm_eps = layer_norm_eps

        # Token embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                embedding_dim=self.embedding_dim, hidden_embedding_dim=self.hidden_embedding_dim,
                attention_dropout_p=self.attention_dropout_p, num_attention_heads=self.num_attention_heads,
                layer_norm_eps=self.layer_norm_eps)
              for _ in range(self.num_transformer_layers)])
        self.layer_norm = RMSNorm(self.embedding_dim, self.layer_norm_eps)
        
        if init_weights:
            self.initialize_parameters()

    def _initialize_kv_cache(self):
        return [None] * self.num_transformer_layers
        
    def forward(
            self,
            input_ids: torch.Tensor,
            is_causal: bool,
            attention_mask: Optional[torch.Tensor] = None,
            cls_context: Optional[torch.Tensor] = None,
            next_token_only: bool = False,
            use_cache: bool = False
    ):

        x = self.token_embedding(input_ids)

        # add the context vector if present to cls token
        if cls_context is not None:
            x[:, 0] += cls_context
        
        # apply the transformer layers
        for _, layer in enumerate(self.layers):
            x = layer(
                x,
                is_causal=is_causal,
                attention_mask=attention_mask,
                use_cache=use_cache
            )

        # apply the layer normalization
        x = self.layer_norm(x)

        # Only return the last token's embedding if next_token_only is True
        if next_token_only:
            # Get the last token for each sequence in the batch
            if attention_mask is not None:
                # Find the last non-padding token position
                last_token_indices = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(x.size(0), device=x.device)
                x = x[batch_indices, last_token_indices]
            else:
                # If no attention mask, just take the last token position
                x = x[:, -1]
            
        # return the output
        return ModelOutput(embeddings=x, attention_mask=attention_mask)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def resize_token_embeddings(self, new_vocab_size: int):
        """ Resize the token embedding matrix to accommodate a new vocabulary size. """
        old_vocab_size, embedding_dim = self.token_embedding.weight.shape

        if new_vocab_size <= old_vocab_size:
            print(f"New vocab size {new_vocab_size} is not greater than current size {old_vocab_size}. No resizing needed.")
            return

        # Create new embedding layer
        new_embeddings = nn.Embedding(new_vocab_size, embedding_dim)
        new_embeddings.to(self.token_embedding.weight.device)

        # Initialize the new embeddings
        torch.nn.init.normal_(new_embeddings.weight, mean=0.0, std=0.02)

        # Copy old weights
        with torch.no_grad():
            new_embeddings.weight[:old_vocab_size] = self.token_embedding.weight

        # Replace the embedding layer
        self.token_embedding = new_embeddings
        self.vocab_size = new_vocab_size

        print(f"Resized token embeddings from {old_vocab_size} to {new_vocab_size}...")
    
    def init_cache(self, batch_size: int, max_seq_length: int):
        for layer in self.layers:
            layer.attention_layer.init_cache(batch_size, max_seq_length)
    
    def clear_cache(self):
        for layer in self.layers:
            layer.attention_layer.clear_cache()
    
    @classmethod
    def from_config(cls, config: ModelConfig):
        return cls(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_embedding_dim=config.hidden_embedding_dim,
            attention_dropout_p=config.attention_dropout,
            num_transformer_layers=config.num_layers,
            num_attention_heads=config.num_attention_heads,
            layer_norm_eps=config.layer_norm_eps
        )
