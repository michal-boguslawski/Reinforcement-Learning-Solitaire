import torch as T
import torch.nn as nn

from ..models.models import Features
from ..torch_registry import ACTIVATION_FUNCTIONS


class SimpleCNN(nn.Module):
    def __init__(self, input_shape: tuple, num_features: int = 64, activation_fn: str = "relu"):
        super().__init__()
        self.input_shape = input_shape
        self.num_features = num_features
        flattened_dim = 400
        self.activation = ACTIVATION_FUNCTIONS[activation_fn]

        self.network = nn.Sequential(
            nn.Conv2d(input_shape[2], 6, 8, 4),
            self.activation(),
            nn.Conv2d(6, 16, 4, 4),
            self.activation(),
            nn.Flatten(),
            nn.LayerNorm(flattened_dim),
            nn.Linear(flattened_dim, num_features),
        )

    def forward(self, input_tensor: T.Tensor) -> Features:
        features = self.network(input_tensor)
        return Features(features=features)
