import torch as T
import torch.nn as nn

from ..torch_registry import ACTIVATION_FUNCTIONS


class SimpleCNN(nn.Module):
    def __init__(self, input_shape: tuple, num_features: int = 64, activation_fn: str = "relu"):
        super().__init__()
        self.input_shape = input_shape
        self.num_features = num_features
        self.activation = ACTIVATION_FUNCTIONS[activation_fn]

        self.network = nn.Sequential(
            nn.Conv2d(input_shape[2], 32, 8, 4),
            self.activation(),
            nn.Conv2d(32, 64, 4, 2),
            self.activation(),
            nn.Conv2d(64, 64, 3, 1),
            self.activation(),
            nn.Flatten(),
            nn.LazyLinear(num_features),
        )

    def forward(self, input_tensor: T.Tensor) -> T.Tensor:
        features = self.network(input_tensor)
        return features
