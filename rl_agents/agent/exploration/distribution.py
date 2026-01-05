from .base import BaseExploration
import torch as T
from torch.distributions import Distribution


class DistributionExploration(BaseExploration):
    def __call__(self, dist: Distribution, *args, **kwargs) -> T.Tensor:
        action = dist.sample()
        
        return action
