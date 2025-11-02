import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple

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
        out_activation: str | None = None,
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
        if out_activation is None:
            out_activation = "identity"
        out_activation_fn = self.act_fn_dict.get(out_activation, nn.Identity)
        self.out_activation_fn = out_activation_fn()

    def forward(self, input_tensor: T.Tensor) -> QOutput:
        x = self.layers(input_tensor)
        logits = self.out_activation_fn(x)
        
        dist = Categorical(logits=logits)
        return QOutput(logits, dist)


class ActorCriticNetwork(nn.Module):
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
        out_activation: str | None = None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.backbone = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_features),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        if out_activation is None:
            out_activation = "identity"
        actor_activation = self.act_fn_dict.get(out_activation, nn.Identity)
        self.actor_activation = actor_activation()

    def forward(self, input_tensor: T.Tensor) -> A2COutput:
        x = self.backbone(input_tensor)
        actor_out = self.actor_activation(self.actor(x))
        critic_out = self.critic(x)
        
        dist = Categorical(logits=actor_out)
        return A2COutput(actor_out, critic_out, dist)
