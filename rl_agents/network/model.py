import torch as T
from torch import nn

from .factories import make_action_distribution, make_backbone, make_head, make_core
from .models.models import ModelOutput


class RLModel(nn.Module):
    def __init__(
        self,
        input_shape: tuple,                     # automatically derived
        num_actions: int,                       # automatically derived
        num_features: int = 64,

        backbone_name: str = "mlp",
        backbone_kwargs: dict = {},

        core_name: str = "identity",
        core_kwargs: dict = {},

        head_name: str = "actor_critic",
        head_kwargs: dict = {},
        
        distribution: str = "categorical",
        low: T.Tensor | None = None,            # automatically derived
        high: T.Tensor | None = None,           # automatically derived
        initial_log_std: float = 0.0,
        device: T.device = T.device("cpu"),
        *args,
        **kwargs
    ):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_features = num_features

        self.backbone_name = backbone_name
        self.backbone_kwargs = backbone_kwargs

        self.core_name = core_name
        self.core_kwargs = core_kwargs

        self.head_name = head_name
        self.head_kwargs = head_kwargs
        
        self.distribution = distribution
        self.initial_log_std = initial_log_std
        self.high = T.as_tensor(high, device=device) if high is not None else high
        self.low = T.as_tensor(low, device=device) if low is not None else low

        self._setup()
        
        self.to(device)

    def _setup(self):
        self._setup_dist()
        self._setup_backbone()
        self._setup_core()
        self._setup_head()

    def _setup_backbone(self):
        self.backbone = make_backbone(
            backbone_name=self.backbone_name,
            input_shape=self.input_shape,
            num_features=self.num_features
        )

    def _setup_core(self):
        self.core = make_core(
            core_name=self.core_name,
            num_features=self.num_features,
            **self.core_kwargs
        )

    def _setup_head(self):
        self.head = make_head(
            head_name=self.head_name,
            num_actions=self.num_actions,
            num_features=self.num_features,
            **self.head_kwargs
        )

    def _setup_dist(self):
        self.log_std = nn.Parameter(T.full((self.num_actions, ), self.initial_log_std))
        self.raw_scale_tril = nn.Parameter(T.eye(self.num_actions))
        self.dist = make_action_distribution(
            dist_name=self.distribution,
            log_std=self.log_std,
            raw_scale_tril=self.raw_scale_tril,
            high=self.high,
            low=self.low,
        )

    def forward(self, input_tensor: T.Tensor, core_state: T.Tensor | None = None, temperature: float = 1.) -> ModelOutput:
        features = self.backbone(input_tensor=input_tensor)
        core = self.core(features=features.features, core_state=core_state)
        head_output = self.head(features=core.core_out)
        dist = self.dist(logits = head_output.actor_logits / temperature)
        return ModelOutput(
            actor_logits=head_output.actor_logits,
            critic_value=head_output.critic_value,
            dist=dist,
            core_state=core.core_state
        )
