from abc import ABC, abstractmethod
import torch as T
from torch import nn
from torch.optim import Adam, Optimizer
from typing import Tuple, Any, Dict

from utils.utils import step_return_discounting
from memory.replay_buffer import ReplayBuffer
from models.models import Observation, ActionOutput, ActionSpaceType
from network.model import RLModel
from network.heads.actor_critic import ActorCriticHead
from agent.exploration.factory import get_exploration


class PolicyMixin(ABC):
    buffer: ReplayBuffer
    optimizer: Optimizer

    def __init__(
        self,
        action_space_type: ActionSpaceType,
        exploration_method: Dict[str, Any],
        gamma_: float = 0.99,
        lambda_: float = 1.,
        optimizer_kwargs: dict = {"lr": 3e-4},
        loss_fn: nn.modules.loss._Loss = nn.HuberLoss(),
        *args,
        **kwargs
    ):
        self.action_space_type = action_space_type
        self.exploration_method_params = exploration_method
        self._exploration_method = get_exploration(
            exploration_method_name=exploration_method["name"],
            exploration_kwargs=exploration_method.get("kwargs", {})
        )
        self.gamma_ = gamma_
        self.lambda_ = lambda_
        self.loss_fn = loss_fn

        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = Adam(self._setup_parameter_groups(optimizer_kwargs, self.action_network))

    @staticmethod
    def _setup_parameter_groups(optimizer_kwargs: dict[str, float], network: RLModel) -> list[dict]:
        lr = optimizer_kwargs.get("lr")
        actor_lr = optimizer_kwargs.get("actor_lr")
        critic_lr = optimizer_kwargs.get("critic_lr")

        if lr is not None:
            return [{"params": network.parameters(), "lr": lr}]

        if not (
            isinstance(network.head, ActorCriticHead)
            and isinstance(actor_lr, float)
            and isinstance(critic_lr, float)
        ):
            raise ValueError(
                f"Missing or invalid lr parameters: "
                f"actor_lr={actor_lr}, critic_lr={critic_lr}"
            )

        return [
            {"params": network.head.actor.parameters(), "lr": actor_lr},
            {"params": network.head.critic.parameters(), "lr": critic_lr},
            {"params": network.backbone.parameters(), "lr": critic_lr},
            {"params": network.core.parameters(), "lr": critic_lr},
            {"params": [network.log_std], "lr": actor_lr},
            {"params": [network.raw_scale_tril], "lr": actor_lr},
        ]
    
    def update_buffer(self, item: dict[str, Any], *args, **kwargs) -> None:
        self.buffer.push(item)

    @property
    @abstractmethod
    def action_network(self) -> RLModel:
        """Return the RLModel used to generate actions."""
        pass

    def eval_mode(self) -> None:
        """Changing action network to eval mode"""
        self.action_network.eval()

    def train_mode(self) -> None:
        """Changing action network to train mode"""
        self.action_network.train()

    def action(
        self,
        state: T.Tensor,
        core_state: T.Tensor | None = None,
        training: bool = True,
        temperature: float = 1.,
    ) -> ActionOutput:
        net = self.action_network
        assert isinstance(net, RLModel), f"Expected RLModel, got {type(net)}"
        with T.no_grad():
            output = net(input_tensor=state, core_state=core_state, temperature=temperature)

        logits = output.actor_logits
        value = output.critic_value
        dist = output.dist
        core_state = output.core_state
        
        action = self._exploration_method(
            logits = logits,
            dist = dist,
            training = training,
            low = net.low,
            high = net.high,
        )

        log_prob = dist.log_prob(action)
        action_output = ActionOutput(
            action=action,
            logits=logits,
            log_probs=log_prob,
            value=value,
            dist=dist,
            core_state=core_state
        )
        return action_output

    @staticmethod
    def _compute_advantage_and_results(
        rewards: T.Tensor, 
        dones: T.Tensor,
        state_values: T.Tensor,
        next_state_values: T.Tensor,
        gamma_: float = 1,
        lambda_: float = 1
    ) -> Tuple[T.Tensor, T.Tensor]:
        q_target = rewards + gamma_ * (1 - dones) * next_state_values
        td_errors = q_target - state_values
        advantages = step_return_discounting(
            values=td_errors, dones=dones, discount=(gamma_ * lambda_)
        )
        returns = advantages + state_values
        return returns, advantages

    def _preprocess_batch(self, batch: Observation) -> Observation:
        state = T.as_tensor(batch.state, dtype=T.float32)
        logits = T.as_tensor(batch.logits, dtype=T.float32)
        action = T.as_tensor(
            batch.action,
            dtype=T.int64 if self.action_space_type == "discrete" else T.float32
        )
        if self.action_space_type == "discrete":
            action = action.unsqueeze(-1)
        reward = T.as_tensor(batch.reward, dtype=T.float32)
        done = T.as_tensor(batch.done, dtype=T.float32)
        value = T.as_tensor(batch.value, dtype=T.float32) if batch.value is not None else None
        log_probs = T.as_tensor(batch.log_probs, dtype=T.float32)
        core_state = T.as_tensor(batch.core_state, dtype=T.float32) if batch.core_state is not None else None
        preprocessed_batch = type(batch)(
            state=state,
            logits=logits,
            action=action,
            reward=reward,
            done=done,
            value=value,
            log_probs=log_probs,
            core_state=core_state,
        )
        return preprocessed_batch

    def save_weights(self, folder_path: str):
        self.action_network.save_weights(folder_path)

    def load_weights(self, file_path: str, param_groups: list[str] | None = None):
        self.action_network.load_weights(file_path, param_groups)
