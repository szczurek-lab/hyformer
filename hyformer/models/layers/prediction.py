import torch
import torch.nn as nn


class PredictionHead(nn.Module):

    def __init__(self, embedding_dim: int, num_tasks: int, dropout_p: float, activation_fn: str = 'gelu') -> None:
        super().__init__()
        self.dense = nn.Linear(embedding_dim, embedding_dim)
        
        # Initialize activation function based on parameter
        if activation_fn == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation_fn == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation_fn == 'gelu':
            self.activation_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}. Choose from 'tanh', 'relu', or 'gelu'")
            
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p is not None and dropout_p > 0 else nn.Identity()
        self.out_proj = nn.Linear(embedding_dim, num_tasks)
        
        # Apply initialization
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    