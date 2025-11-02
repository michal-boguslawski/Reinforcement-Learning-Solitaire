from typing import NamedTuple
import numpy as np
import torch as T
from torch.distributions import Distribution


class Observation(NamedTuple):
    state: T.Tensor
    next_state: T.Tensor
    logits: T.Tensor
    action: T.Tensor
    reward: T.Tensor
    done: T.Tensor
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
