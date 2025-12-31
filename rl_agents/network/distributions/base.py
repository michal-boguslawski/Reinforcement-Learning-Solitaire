from abc import ABC, abstractmethod
import torch as T
from torch.distributions import Distribution


class ActionBaseDistribution(ABC):
    @abstractmethod
    def __call__(self, logits: T.Tensor) -> Distribution:
        pass


class DistributionTransform(ABC):
    @abstractmethod
    def apply(self, dist: Distribution) -> Distribution:
        pass


class IdentityTransform(DistributionTransform):
    def __init__(self, *args, **kwargs):
        pass

    def apply(self, dist: Distribution) -> Distribution:
        return dist


class ActionDistribution:
    def __init__(self, base_dist, transform=None):
        self.base_dist = base_dist
        self.transform = transform or IdentityTransform()

    def __call__(self, logits) -> Distribution:
        dist = self.base_dist(logits)
        return self.transform.apply(dist)
