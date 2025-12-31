import torch as T
import torch.nn.functional as F
from torch.distributions import Distribution, Normal, Independent, MultivariateNormal

from .base import ActionBaseDistribution


class NormalDistribution(ActionBaseDistribution):
    def __init__(self, log_std, *args, **kwargs):
        self.log_std = log_std

    def __call__(self, logits) -> Distribution:
        std = T.exp(self.log_std).expand_as(logits)
        return Independent(Normal(logits, std), 1)


class MultivariateNormalDistribution(ActionBaseDistribution):
    def __init__(self, raw_scale_tril, *args, **kwargs):
        self.raw_scale_tril = raw_scale_tril

    def _build_scale_tril(self) -> T.Tensor:
        L = T.tril(self.raw_scale_tril)
        diag = F.softplus(T.diagonal(L)) + 1e-5
        L = L.clone()
        L[range(len(diag)), range(len(diag))] = diag
        return L

    def __call__(self, logits) -> Distribution:
        return MultivariateNormal(
            loc=logits,
            scale_tril=self._build_scale_tril()
        )
