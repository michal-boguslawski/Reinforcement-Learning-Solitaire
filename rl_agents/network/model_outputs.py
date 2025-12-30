from dataclasses import dataclass
import torch as T
from torch.distributions import Distribution


@dataclass
class ActorCriticOutput:
    logits: T.Tensor
    value: T.Tensor
    dist: Distribution


@dataclass
class ActorOutput:
    logits: T.Tensor
    dist: Distribution
