from torch.distributions import Distribution, Categorical

from .base import ActionBaseDistribution


class CategoricalDistribution(ActionBaseDistribution):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, logits) -> Distribution:
        return Categorical(logits=logits)
