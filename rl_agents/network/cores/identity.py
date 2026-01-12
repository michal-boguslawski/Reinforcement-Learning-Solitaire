import torch as T
import torch.nn as nn

from ..models.models import CoreOutput


class IdentityCore(nn.Module):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__()

    def forward(self, features: T.Tensor, *args, **kwargs) -> CoreOutput:
        return CoreOutput(core_out=features)
