from abc import ABC, abstractmethod
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces.space import Space
import numpy as np
import random
import torch as T
from torch import nn
from torch.optim import Adam
from torch.distributions import Distribution, Categorical
from typing import Tuple, Generator, NamedTuple, Any

from utils.utils import step_return_discounting
from memory.replay_buffer import ReplayBuffer
from models.models import Observation, ActionOutput


class PolicyMixin(ABC):
    def __init__(
        self,
        action_space: Space,
        num_actions: int,
        device: T.device = T.device('cpu'),
        gamma_: float = 0.99,
        lambda_: float = 1,
        loss_fn: nn.modules.loss._Loss = nn.HuberLoss(),
        lr: float = 1e-3,
        *args,
        **kwargs
    ):
        self.gamma_ = gamma_
        self.lambda_ = lambda_
        self.num_actions = num_actions
        self.device = device
        self.action_space = action_space
        self.loss_fn = loss_fn
        self.buffer: ReplayBuffer
        if self.action_network is None:
            raise ValueError("action_network must be implemented")
        self.optimizer = Adam(self.action_network.parameters(), lr=lr)

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
        high = getattr(self.action_space, "high", None)
        low = getattr(self.action_space, "low", None)
        if high is not None and low is not None:
            high = T.tensor(high, device=self.device)
            low = T.tensor(low, device=self.device)
            action = action.clamp(
                low + 1e-6,
                high - 1e-6
            )
        
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
            if isinstance(dist, Categorical):
                action = logits.argmax(keepdim=True)
            else:
                action = self._action_sample_from_distribution(dist)
        else:
            # Handle vectorized environments by sampling for each environment
            batch_size = logits.shape[0] if logits.ndim > 1 else 1
            if batch_size > 1:
                # Sample actions for each environment in the batch
                random_actions = [self.action_space.sample() for _ in range(batch_size)]
                action = T.tensor(random_actions, device=self.device, dtype=T.float32)
            else:
                action = T.tensor(self.action_space.sample(), device=self.device, dtype=T.float32)
        
        return action

    def action(
        self,
        state: T.Tensor,
        method: str,
        epsilon_: float = 0.5,
    ) -> ActionOutput:
        net = self.action_network
        assert isinstance(net, nn.Module), f"Expected nn.Module, got {type(net)}"
        with T.no_grad():
            output = net(state)

        logits = output.logits
        value = getattr(output, "value", None)
        if method == "egreedy":
            action = self._action_egreedy(epsilon_=epsilon_, logits=logits, dist=output.dist)
        elif method == "best":
            action = self._action_best_from_distribution(dist=output.dist)
        else:
            action = self._action_sample_from_distribution(dist=output.dist)
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
        # return q_target, td_errors

    def _preprocess_batch(self, batch: Observation) -> Observation:
        state = T.as_tensor(batch.state, dtype=T.float32)
        next_state = T.as_tensor(batch.next_state, dtype=T.float32)
        logits = T.as_tensor(batch.logits, dtype=T.float32)
        action = T.as_tensor(
            batch.action,
            dtype=T.int64 if isinstance(self.action_space, Discrete) else T.float32
        )
        reward = T.as_tensor(batch.reward, dtype=T.float32)
        done = T.as_tensor(batch.done, dtype=T.float32)
        value = T.as_tensor(batch.value, dtype=T.float32) if batch.value is not None else None
        log_probs = T.as_tensor(batch.log_probs, dtype=T.float32)
        preprocessed_batch = type(batch)(
            state=state,
            next_state=next_state,
            logits=logits,
            action=action,
            reward=reward,
            done=done,
            value=value,
            log_probs=log_probs,
        )
        return preprocessed_batch
    
    def update_buffer(self, item: dict[str, Any], *args, **kwargs) -> None:
        self.buffer.push(item)
