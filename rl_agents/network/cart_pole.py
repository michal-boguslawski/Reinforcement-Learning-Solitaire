import torch as T
import torch.nn as nn
import torch.nn.functional as F


class MLPNetwork(nn.Module):
    def __init__(self, in_features: int = 4, out_features: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, input_tensor: T.Tensor) -> T.Tensor:
        return self.layers(input_tensor)
