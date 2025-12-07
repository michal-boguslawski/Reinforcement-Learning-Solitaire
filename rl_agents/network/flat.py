import torch as T
import torch.nn as nn

from .utils import activation_fns_dict


class MLPNetwork(nn.Module):
    def __init__(
        self,
        input_shape: int | tuple,
        channels: int = 64,
        hidden_dims: int = 64,
        num_layers: int = 2,
        activation_fn: str = "tanh"
    ):
        super().__init__()
        assert isinstance(input_shape, int)
        self.input_shape = input_shape
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.channels = channels
        self.activation_fn = activation_fns_dict[activation_fn]

        self._build_network()

    def _build_network(self):
        modules = []
        for i in range(self.num_layers):
            modules.append(
                nn.Linear(
                    in_features=self.input_shape if i == 0 else self.hidden_dims,
                    out_features=self.channels if i == (self.num_layers - 1) else self.hidden_dims
                )
            )
            modules.append(self.activation_fn())
        self.network = nn.Sequential(*modules)

    def forward(self, input_tensor: T.Tensor) -> T.Tensor:
        output_tensor = self.network(input_tensor)
        return output_tensor
