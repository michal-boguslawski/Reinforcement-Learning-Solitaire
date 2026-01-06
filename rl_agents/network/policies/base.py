from abc import ABC, abstractmethod
from torch import nn
import torch as T

from ..model_outputs import ActorCriticOutput, ActorOutput


class Policy(nn.Module, ABC):
    @abstractmethod
    def forward(self, features: T.Tensor, temperature: float) -> ActorCriticOutput | ActorOutput:
        pass
