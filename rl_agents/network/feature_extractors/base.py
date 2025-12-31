from abc import ABC, abstractmethod
import torch as T
from torch import nn


class FeatureExtractor(nn.Module, ABC):
    @abstractmethod
    def forward(self, input_tensor: T.Tensor, *args, **kwargs) -> T.Tensor:
        pass
