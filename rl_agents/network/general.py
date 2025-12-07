import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Distribution, TransformedDistribution, AffineTransform, TanhTransform, Independent, MultivariateNormal

from models.models import A2COutput
from .cv import SimpleConvNetwork
from .flat import MLPNetwork
from .head import ActorCriticHead


class ActorCriticNetwork(nn.Module):
    backbone_type_dict = {
        "mlp": MLPNetwork,
        "simple_cv": SimpleConvNetwork
    }
    
    head_type_dict = {
        "a2c": ActorCriticHead
    }
    
    def __init__(
        self,
        input_shape: int | tuple,
        num_actions: int,
        channels: int = 64,

        backbone_type: str = "mlp",
        backbone_kwargs: dict = {},
        head_type: str = "a2c",
        head_kwargs: dict = {},
        
        distribution: str = "categorical",
        low: T.Tensor | None = None,
        high: T.Tensor | None = None,
        initial_log_std: float = 0.0,
        device: T.device = T.device("cpu"),
        *args,
        **kwargs
    ):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.channels = channels
        self.backbone_type = backbone_type
        self.backbone_kwargs = backbone_kwargs
        self.head_type = head_type
        self.head_kwargs = head_kwargs
        
        self.distribution = distribution
        self.initial_log_std = initial_log_std
        self.high = high
        self.low = low

        self._setup()
        
        self.to(device)

    def _setup(self):
        self.log_std = nn.Parameter(T.full((self.num_actions, self.num_actions), self.initial_log_std))
        self.raw_scale_tril = nn.Parameter(T.eye(self.num_actions))
        # Precompute transform parameters for efficiency
        if self.low is not None and self.high is not None:
            self.transform_loc = (self.high + self.low) / 2
            self.transform_scale = (self.high - self.low) / 2
        self._build_network()

    def _build_network(self):
        self.backbone = self.backbone_type_dict[self.backbone_type](
            input_shape=self.input_shape,
            channels=self.channels,
            **self.backbone_kwargs
        )

        self.head = self.head_type_dict[self.head_type](
            channels=self.channels,
            num_actions=self.num_actions
        )

    def _build_scale_tril(self):
        L = T.tril(self.raw_scale_tril)
        diag = T.diagonal(L)
        diag = F.softplus(diag) + 1e-5
        
        L = L.clone()
        L[range(len(diag)), range(len(diag))] = diag
        return L

    def _set_distribution(self, logits: T.Tensor) -> Distribution:
        if T.isnan(logits).any():
            print("NaN detected in logits")
            print("logits:", logits)
            raise ValueError("NaN in actor output before creating distribution")

        log_std = T.clamp(self.log_std, min=-20, max=5)
        if T.isnan(log_std).any() or T.isinf(log_std).any():
            print("NaN/Inf in log_std!")
            print("Raw log_std:", self.log_std)
            print("Clamped log_std:", log_std)
            raise ValueError("NaN/Inf in log_std")
        if self.distribution == "normal":
            std = T.exp(log_std).expand_as(logits)
            dist = Normal(loc=logits, scale=std)
            dist = Independent(dist, 1)

        elif self.distribution == "multivariatenormal":
            scale_tril = self._build_scale_tril()
            dist = MultivariateNormal(loc=logits, scale_tril=scale_tril)

        else:
            dist = Categorical(logits=logits)

        if self.low is not None and self.high is not None:
            transforms = [
                TanhTransform(),
                AffineTransform(
                    loc=self.transform_loc,
                    scale=self.transform_scale
                )
            ]
            dist = TransformedDistribution(
                dist,
                transforms=transforms
            )
        return dist

    def forward(self, input_tensor: T.Tensor) -> A2COutput:
        x = self.backbone(input_tensor)

        actor_out, critic_out = self.head(x)

        dist = self._set_distribution(logits=actor_out)
        return A2COutput(actor_out, critic_out, dist)
