from torch import nn
import torch as T

from .base import Policy
from ..distributions.base import ActionDistribution
from ..model_outputs import ActorCriticOutput


class ActorCriticPolicy(Policy):
    def __init__(self, head: nn.Module, dist: ActionDistribution):
        super().__init__()
        self.head = head
        self.dist = dist

    def forward(self, features: T.Tensor) -> ActorCriticOutput:
        actor_logits, critic_value = self.head(features)
        dist = self.dist(logits=actor_logits)
        return ActorCriticOutput(actor_logits, critic_value, dist)
