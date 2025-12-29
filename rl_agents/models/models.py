import gymnasium as gym
from typing import NamedTuple, Literal, Tuple
import numpy as np
import torch as T
from torch.distributions import Distribution

ActionSpaceType = Literal["discrete", "continuous"]

class Observation(NamedTuple):
    state: T.Tensor
    logits: T.Tensor
    action: T.Tensor
    reward: T.Tensor
    done: T.Tensor
    log_probs: T.Tensor
    value: T.Tensor | None = None

class QOutput(NamedTuple):
    logits: T.Tensor
    dist: Distribution

class A2COutput(NamedTuple):
    logits: T.Tensor
    value: T.Tensor
    dist: Distribution
    
class ActionOutput(NamedTuple):
    action: T.Tensor
    logits: T.Tensor
    log_probs: T.Tensor
    value: T.Tensor | None = None
    dist: Distribution | None = None

class OnPolicyMinibatch(NamedTuple):
    states: T.Tensor
    returns: T.Tensor
    actions: T.Tensor
    advantages: T.Tensor
    log_probs: T.Tensor

class EnvDetails(NamedTuple):
    action_dim: int
    state_dim: Tuple[int, ...]
    action_space_type: ActionSpaceType
    action_low: T.Tensor | None
    action_high: T.Tensor | None
