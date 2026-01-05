from torch import nn
import torch as T

from .base import Policy
from ..distributions.base import ActionDistribution
from ..model_outputs import ActorOutput


class ActorPolicy(Policy):
    def __init__(self, head: nn.Module, dist: ActionDistribution):
        super().__init__()
        self.head = head
        self.dist = dist

    def forward(self, features: T.Tensor) -> ActorOutput:
        logits = self.head(features)
        dist = self.dist(logits)
        return ActorOutput(logits, dist)
