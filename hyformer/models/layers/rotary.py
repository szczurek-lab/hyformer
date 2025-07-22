import torch
import torch.nn as nn

BASE = 10000
DTYPE = torch.float32


class RotaryPositionalEmbedding(nn.Module):
    """
    Applies Rotary Positional Encoding to a tensor of shape [BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM].
    Offset allows to apply rotary to sequnce part by part by telling how much tokens preecede the input in the sequence.
    """

    def __init__(self, head_dim) -> None:
        super().__init__()

        assert head_dim % 2 == 0
        self.hidden_dim = head_dim

        num_features = self.hidden_dim // 2
        thetas = BASE ** (-torch.arange(0, num_features, dtype=DTYPE) / num_features).reshape(1, 1, 1, num_features, 1)
        self.register_buffer("thetas", thetas, persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0):
        assert (
            len(x.shape) == 4
        ) 
        assert offset >= 0

        batch_size, seq_len, num_head, head_dim = x.shape
        device = x.device

        ms = torch.arange(offset, offset + seq_len, device=device).reshape(1, seq_len, 1, 1, 1)
        angles = ms * self.thetas
        cosines = torch.cos(angles).to(DTYPE)
        sines = torch.sin(angles).to(DTYPE)
        x_grp = x.reshape(batch_size, seq_len, num_head, head_dim//2, 2)
        x_cos = x_grp * cosines
        x_sin = x_grp * sines
        result = x_cos + torch.stack([-x_sin[..., 1], x_sin[...,0]], dim=-1)
        result = result.flatten(-2)

        return result
