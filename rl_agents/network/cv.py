import torch as T
from torch import nn


class SimpleConvNetwork(nn.Module):
    def __init__(self, input_shape: tuple = (), channels: int = 256, *args, **kwargs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, 7, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 5, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 5, 2),
            # nn.BatchNorm2d(256),
            # nn.ReLU(True),
            # nn.Conv2d(256, 256, 3, 1),
            nn.MaxPool2d(5),
            nn.Flatten(),
            nn.Tanh()
        )

    def forward(self, input_tensor: T.Tensor) -> T.Tensor:
        assert input_tensor.ndim == 4
        if input_tensor.shape[-1] == 3:
            input_tensor = input_tensor.permute(0, 3, 1, 2)

        # rescaling to (-1, 1)
        input_tensor = 2 * input_tensor / 255. - 1.
            
        output_tensor = self.network(input_tensor)
        return output_tensor
