from torch import nn
import torch as T

from .base import FeatureExtractor


class SharedFeatureExtractor(FeatureExtractor):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, input_tensor: T.Tensor) -> T.Tensor:
        return self.backbone(input_tensor)
