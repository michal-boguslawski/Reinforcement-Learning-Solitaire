from abc import ABC, abstractmethod
import random
import torch as T
from torch import nn
from torch.optim import Adam
from torch.distributions import Distribution, Categorical
from typing import Tuple, Any

from utils.utils import step_return_discounting
from memory.replay_buffer import ReplayBuffer
from models.models import Observation, ActionOutput, ActionSpaceType


class PolicyMixin(ABC):
    buffer: ReplayBuffer

    def __init__(
        self,
        action_space_type: ActionSpaceType,
        exploration_method: str,
        gamma_: float = 0.99,
        lambda_: float = 1.,
        lr: float = 1e-3,
        loss_fn: nn.modules.loss._Loss = nn.HuberLoss(),
        *args,
        **kwargs
    ):
        self.action_space_type = action_space_type
        self.exploration_method = exploration_method
        self.gamma_ = gamma_
        self.lambda_ = lambda_
        self.optimizer = Adam(self.action_network.parameters(), lr=lr)
        self.loss_fn = loss_fn

        if self.action_network is None:
            raise ValueError("action_network must be implemented")
    
    def update_buffer(self, item: dict[str, Any], *args, **kwargs) -> None:
        self.buffer.push(item)

    @property
    @abstractmethod
    def action_network(self) -> nn.Module:
        """Return the torch.nn.Module used to generate actions."""
        pass

    def eval_mode(self) -> None:
        """Changing action network to eval mode"""
        self.action_network.eval()

    def train_mode(self) -> None:
        """Changing action network to train mode"""
        self.action_network.train()

    def _action_sample_from_distribution(self, dist: Distribution) -> T.Tensor:
        action = dist.sample()
        
        return action

    def _action_best_from_distribution(self, dist: Distribution) -> T.Tensor:
        if isinstance(dist, Categorical):
            return dist.logits.argmax(keepdim=True)
        base_dist = getattr(dist, "base_dist", dist)
        transforms = getattr(dist, "transforms", None)
        
        mean = base_dist.mean
        if transforms:
            for transform in transforms:
                mean = transform(mean)

        return mean

    def _action_egreedy(self, epsilon_: float, logits: T.Tensor, dist: Distribution) -> T.Tensor:
        if random.random() > epsilon_:
            action = logits.argmax(-1, keepdim=True)
        else:
            # Handle vectorized environments by sampling for each environment
            if isinstance(dist, Categorical):
                batch_size, n = dist.param_shape
                action = T.randint(0, n, (batch_size, 1))
            pass
        
        return action

    def action(
        self,
        state: T.Tensor,
        epsilon_: float = 0.5,
        training: bool = True,
    ) -> ActionOutput:
        net = self.action_network
        assert isinstance(net, nn.Module), f"Expected nn.Module, got {type(net)}"
        with T.no_grad():
            output = net(state)

        logits = output.logits
        value = getattr(output, "value", None)
        if training and ( self.exploration_method == "egreedy" ):
            action = self._action_egreedy(epsilon_=epsilon_, logits=logits, dist=output.dist)
        elif training and ( self.exploration_method == "distribution" ):
            action = self._action_sample_from_distribution(dist=output.dist)
        elif ( not training ) or ( self.exploration_method == "best") :
            action = self._action_best_from_distribution(dist=output.dist)
        else:
            raise ValueError(f"Unknown method: {self.exploration_method}")
        logprob = output.dist.log_prob(action)
        action_output = ActionOutput(
            action=action,
            logits=logits,
            log_probs=logprob,
            value=value,
            dist=output.dist
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
        reward = T.as_tensor(batch.reward, dtype=T.float32)
        done = T.as_tensor(batch.done, dtype=T.float32)
        value = T.as_tensor(batch.value, dtype=T.float32) if batch.value is not None else None
        log_probs = T.as_tensor(batch.log_probs, dtype=T.float32)
        preprocessed_batch = type(batch)(
            state=state,
            logits=logits,
            action=action,
            reward=reward,
            done=done,
            value=value,
            log_probs=log_probs,
        )
        return preprocessed_batch
