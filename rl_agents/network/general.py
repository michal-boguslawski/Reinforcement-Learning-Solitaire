import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Distribution, TransformedDistribution, AffineTransform, TanhTransform, Independent, MultivariateNormal

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
        initial_log_std: float = 0.0,
        device: T.device = T.device("cpu"),
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
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_features),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.distribution = distribution
        self.out_features = out_features
        self.log_std = nn.Parameter(T.full((out_features, out_features), initial_log_std))
        self.raw_scale_tril = nn.Parameter(T.eye(out_features))
        
        # Precompute transform parameters for efficiency
        if self.low is not None and self.high is not None:
            self.transform_loc = (self.high + self.low) / 2
            self.transform_scale = (self.high - self.low) / 2
        
        self.to(device)

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

        actor_out = self.actor(x)
        critic_out = self.critic(x)

        dist = self._set_distribution(logits=actor_out)
        return A2COutput(actor_out, critic_out, dist)
