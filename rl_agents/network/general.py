import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Distribution, TransformedDistribution, AffineTransform, TanhTransform

from models.models import QOutput, A2COutput


class MLPNetwork(nn.Module):
    act_fn_dict = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
    }
    def __init__(
        self,
        in_features: int = 4,
        out_features: int = 2,
        hidden_dim: int = 64,
        *args,
        **kwargs
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_features),
        )
        self.distribution = Categorical # if out_activation == "identity" else Normal
        self.log_std = nn.Parameter(T.zeros(out_features))

    def forward(self, input_tensor: T.Tensor) -> QOutput:
        logits = self.layers(input_tensor)
        
        dist = self.distribution(logits=logits)
        return QOutput(logits, dist)


class ActorCriticNetwork(nn.Module):
    def __init__(
        self,
        in_features: int = 4,
        out_features: int = 2,
        hidden_dim: int = 64,
        distribution: str = "categorical",
        low: T.Tensor | None = None,
        high: T.Tensor | None = None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.low = low
        self.high = high

        self.backbone = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_features),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.distribution = distribution
        # self.distribution_fn = self.distribution_dict[distribution]
        self.log_std = nn.Parameter(T.zeros(out_features,))  # Start with smaller std

    def _set_distribution(self, logits: T.Tensor) -> Distribution:
        if T.isnan(logits).any():
            print("NaN detected in logits")
            print("logits:", logits)
            raise ValueError("NaN in actor output before creating distribution")

        log_std = T.clamp(self.log_std, min=-5, max=2)
        if T.isnan(log_std).any() or T.isinf(log_std).any():
            print("NaN/Inf in log_std!")
            print("Raw log_std:", self.log_std)
            print("Clamped log_std:", log_std)
            raise ValueError("NaN/Inf in log_std")
        
        std = T.exp(log_std).expand_as(logits)
        if T.isnan(std).any() or (std <= 0).any():
            print("Invalid std values:", std)
            raise ValueError("Invalid standard deviation")
        if self.distribution == "normal":
            # if self.low is not None and self.high is not None:
            logits = logits.clamp(-3, 3)
            dist = Normal(loc=logits, scale=std)
            if self.low is not None and self.high is not None:
                transforms = [
                    TanhTransform(),
                    AffineTransform(
                        loc = (self.high + self.low) / 2,
                        scale = (self.high - self.low) / 2
                    )
                ]
                dist = TransformedDistribution(dist, transforms=transforms)
        else:
            dist = Categorical(logits=logits)
        return dist

    def forward(self, input_tensor: T.Tensor) -> A2COutput:
        x = self.backbone(input_tensor)
        if T.isnan(x).any() or T.isinf(x).any():
            print("NaN in backbone output")
            print(x)
            raise ValueError("NaN in backbone output")
        actor_out = self.actor(x)
        critic_out = self.critic(x)

        if T.isnan(actor_out).any():
            print("NaN in actor_out")
            print(actor_out)
            raise ValueError("NaN in actor_out")
        dist = self._set_distribution(logits=actor_out)
        return A2COutput(actor_out, critic_out, dist)
