import torch
import torch.nn as nn

from hyformer.models.layers.layer_norm import RMSNorm
from hyformer.models.layers.attention import Attention
from hyformer.models.layers.feed_forward import FeedForward


class TransformerLayer(nn.Module):
    """A single transformer layer with attention and feed-forward components.

    Parameters
    ----------
    embedding_dim : int
        The dimension of the embedding.
    hidden_embedding_dim : int
        The dimension of the hidden embedding.
    num_attention_heads : int
        The number of attention heads.
    attention_dropout_p : float
        The dropout rate for the attention.
    layer_norm_eps : float
        The epsilon for the layer normalization.
    """

    def __init__(
        self, embedding_dim: int, hidden_embedding_dim: int, num_attention_heads: int,
        attention_dropout_p: float, layer_norm_eps: float
    ) -> None:
        super().__init__()
        self.attention_layer = Attention(
            embedding_dim=embedding_dim,
            num_attention_heads=num_attention_heads,
            attention_dropout_p=attention_dropout_p
        )
        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            hidden_embedding_dim=hidden_embedding_dim
        )
        self.attention_layer_normalization = RMSNorm(embedding_dim, eps=layer_norm_eps)
        self.feed_forward_normalization = RMSNorm(embedding_dim, eps=layer_norm_eps)
        
    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool,
        attention_mask: torch.Tensor = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] = None,
        use_cache: bool = False
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        _residual = x
        x = self.attention_layer(
            x=self.attention_layer_normalization(x),
            is_causal=is_causal,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache
            )
        x = _residual + self.feed_forward(self.feed_forward_normalization(x))
        return x
