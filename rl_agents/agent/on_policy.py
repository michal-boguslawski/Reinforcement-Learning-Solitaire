from gymnasium.spaces.space import Space
from gymnasium.spaces.discrete import Discrete
import numpy as np
import torch as T
import torch.nn as nn
from torch.optim import Optimizer
import random
from typing import Generator, Tuple

from .base import BasePolicy
from .mixins import PolicyMixin
from memory.replay_buffer import ReplayBuffer
from models.models import Observation, OnPolicyMinibatch, ActionSpaceType


class OnPolicy(PolicyMixin, BasePolicy):
    def __init__(self, num_epochs: int = 1, device: T.device = T.device('cpu'), *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
            self.buffer = ReplayBuffer(device=device)
            self.device = device
            self.num_epochs = num_epochs
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OnPolicy: {e}")
        
    def train(self, minibatch_size: int = 64, **kwargs) -> list[float] | None:
        batch = self.buffer.get_all()
        if not batch:
            return
        losses = []
        for _ in range(self.num_epochs):
            minibatches = self._generate_minibatches(batch=batch, minibatch_size=minibatch_size)
            for minibatch in minibatches:
                loss = self.calculate_loss(minibatch)
                losses.append(loss)

        return [np.mean(column).item() for column in zip(*losses)]

    def _generate_minibatches(
        self,
        batch: Observation,
        minibatch_size: int = 64
    ) -> Generator[OnPolicyMinibatch, None, None]:
        batch = self._preprocess_batch(batch=batch)
        
        try:
            if batch.value is not None:
                if batch.value.shape[-1] != 1:
                    raise ValueError(f"Expected value tensor with last dim=1, got {batch.value.shape}")
                state_values = batch.value.squeeze(-1)
                next_state_values = state_values[:, 1:]
            else:
                if batch.action.max() >= batch.logits.shape[-1]:
                    raise ValueError(f"Action index {batch.action.max()} out of bounds for logits shape {batch.logits.shape}")
                state_values = batch.logits.gather(dim=-1, index=batch.action).squeeze(-1)
                next_state_values = state_values[:, 1:]

            returns, advantages = self._compute_advantage_and_results(
                rewards=batch.reward[:, :-1],
                dones=batch.done[:, :-1],
                state_values=state_values[:, :-1],
                next_state_values=next_state_values,
                gamma_=self.gamma_,
                lambda_=self.lambda_
            )
        except (IndexError, RuntimeError, ValueError) as e:
            raise RuntimeError(f"Failed to compute advantages: {e}")

        batch_size = int(np.prod(returns.shape))
        indices = T.arange(0, batch_size, device=self.device)
        minibatch_size = min(batch_size, minibatch_size)
        for start in range(0, batch_size, minibatch_size):
            end = min(start + minibatch_size, batch_size - 1)
            mb_idx = indices[start:end]
            batch_advantages = advantages.flatten(0, 1)[mb_idx]
            batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-6)
            yield OnPolicyMinibatch(
                states=batch.state[:, :-1].flatten(0, 1)[mb_idx],
                returns=returns.flatten(0, 1)[mb_idx],
                actions=batch.action[:, :-1].flatten(0, 1)[mb_idx],
                advantages=batch_advantages,
                log_probs=batch.log_probs[:, :-1].flatten(0, 1)[mb_idx]
            )


class SarsaPolicy(OnPolicy):
    def __init__(
        self, 
        network: nn.Module,
        num_actions: int,
        optimizer: Optimizer,
        action_space: Space,
        gamma_: float = 0.99,
        lambda_: float = 1,
        loss_fn: nn.modules.loss._Loss = nn.HuberLoss(),
        device: T.device = T.device('cpu'),
        *args,
        **kwargs
    ):
        super().__init__(
            device=device,
            num_actions=num_actions,
            gamma_=gamma_,
            lambda_=lambda_,
            action_space=action_space,
            loss_fn=loss_fn
        )
        
        self.network = network
        self.optimizer = optimizer
        self.gamma_ = gamma_
        self.lambda_ = lambda_
    
    @property
    def action_network(self) -> nn.Module:
        return self.network
        
    def calculate_loss(self, batch: Tuple[T.Tensor, ...]) -> Tuple[float, ...]:
        states, results, actions, _ = batch
        
        output = self.network(states)
        logits = output.logits
        assert isinstance(logits, T.Tensor)

        q_values = logits.gather(dim=-1, index=actions).squeeze(-1)
        
        loss = self.loss_fn(results, q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1)
        self.optimizer.step()
        
        return loss.item()


class A2CPolicy(OnPolicy):
    def __init__(
        self, 
        network: nn.Module,
        num_actions: int,
        action_space: Space,
        gamma_: float = 0.99,
        lambda_: float = 1,
        entropy_beta_: float = 0.01,
        loss_fn: nn.modules.loss._Loss = nn.HuberLoss(),
        device: T.device = T.device('cpu'),
        lr: float = 1e-3,
        *args,
        **kwargs
    ):
        self.network = network
        super().__init__(
            device=device,
            num_actions=num_actions,
            gamma_=gamma_,
            lambda_=lambda_,
            action_space=action_space,
            loss_fn=loss_fn,
            lr=lr,
        )
        self.gamma_ = gamma_
        self.lambda_ = lambda_
        self.entropy_beta_ = entropy_beta_
    
    @property
    def action_network(self) -> nn.Module:
        return self.network

    def calculate_loss(self, batch: OnPolicyMinibatch) -> Tuple[float, ...]:
        states, returns, actions, advantages = (
            batch.states,
            batch.returns,
            batch.actions,
            batch.advantages,
        )

        output = self.network(states)

        # calculation policy loss
        log_probs = output.dist.log_prob(actions)
        if isinstance(self.action_space, Discrete):
            sum_log_probs = log_probs
        else:
            # TransformedDistribution.log_prob() already sums across action dimensions
            # Regular Normal distribution returns [batch_size, action_dim], needs summing
            if hasattr(output.dist, 'transforms'):
                # TransformedDistribution already returns summed log_prob
                sum_log_probs = log_probs
            else:
                # Regular distribution, need to sum across action dimensions
                sum_log_probs = log_probs.sum(-1)
        assert sum_log_probs.shape == returns.shape, "Wrong shapes of value"
        actor_loss = -(sum_log_probs * advantages.detach()).mean()
        
        if T.isnan(log_probs.cpu()).any():
            print("actor_loss:", actor_loss.cpu().item())
            print("Actions:", actions.cpu().mean())
            print("log_probs:", log_probs.cpu())
            print("advantages:", advantages.cpu())
            print("results:", returns.cpu())
            raise ValueError("actions causing problems")

        # calculating critic loss
        value = output.value.squeeze(-1)
        assert value.shape == returns.shape, "Wrong shapes of value"
        critic_loss = self.loss_fn(value, returns.detach())

        # calculating entropy
        try:
            entropy = output.dist.entropy()
        except NotImplementedError:
            entropy = output.dist.base_dist.entropy()
        except Exception as e:
            raise RuntimeError(f"Failed to compute entropy: {e}")
        if entropy.numel() == 0:
            raise ValueError("Empty entropy tensor")
        entropy = entropy.mean()

        loss = actor_loss + 0.5 * critic_loss - self.entropy_beta_ * entropy

        # backpropagation of the error
        self.optimizer.zero_grad()
        loss.backward()
        
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        return loss.item(), actor_loss.item(), critic_loss.item(), entropy.item()


class PPOPolicy(OnPolicy):
    def __init__(
        self, 
        network: nn.Module,
        num_actions: int,
        action_space_type: ActionSpaceType,
        gamma_: float = 0.99,
        lambda_: float = 1,
        critic_coef_: float = 0.5,
        entropy_beta_: float = 0.01,
        entropy_decay: float = 1.,
        num_epochs: int = 10,
        clip_epsilon: float = 0.2,
        loss_fn: nn.modules.loss._Loss = nn.HuberLoss(),
        device: T.device = T.device('cpu'),
        lr: float = 1e-3,
        *args,
        **kwargs
    ):
        self.network = network
        super().__init__(
            device=device,
            num_actions=num_actions,
            gamma_=gamma_,
            lambda_=lambda_,
            action_space_type=action_space_type,
            loss_fn=loss_fn,
            lr=lr,
            num_epochs=num_epochs,
        )
        self.gamma_ = gamma_
        self.lambda_ = lambda_
        self.critic_coef_ = critic_coef_
        self.entropy_beta_ = entropy_beta_
        self.entropy_decay = entropy_decay
        self.clip_epsilon = clip_epsilon
    
    @property
    def action_network(self) -> nn.Module:
        return self.network

    def step_entropy_decay(self) -> None:
        self.entropy_beta_ *= self.entropy_decay

    def calculate_loss(self, batch: OnPolicyMinibatch) -> Tuple[float, ...]:
        states, returns, actions, advantages, old_log_probs = (
            batch.states,
            batch.returns,
            batch.actions,
            batch.advantages,
            batch.log_probs,
        )

        output = self.network(states)

        # calculation policy loss
        log_probs = output.dist.log_prob(actions)
        
        r_t = T.exp(log_probs - old_log_probs)
        
        if isinstance(self.action_space, Discrete):
            sum_r_t = r_t
        else:
            # TransformedDistribution.log_prob() already sums across action dimensions
            # Regular Normal distribution returns [batch_size, action_dim], needs summing
            if hasattr(output.dist, 'transforms'):
                # TransformedDistribution already returns summed log_prob
                sum_r_t = r_t
            else:
                # Regular distribution, need to sum across action dimensions
                sum_r_t = r_t.sum(-1)
        assert r_t.shape == returns.shape, "Wrong shapes of value"
        actor_loss = -(T.min(
            sum_r_t * advantages.detach(),
            T.clamp(sum_r_t, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages.detach()
        )).mean()
        
        if T.isnan(log_probs.cpu()).any():
            print("actor_loss:", actor_loss.cpu().item())
            print("Actions:", actions.cpu().mean())
            print("log_probs:", log_probs.cpu())
            print("advantages:", advantages.cpu())
            print("results:", returns.cpu())
            raise ValueError("actions causing problems")

        # calculating critic loss
        value = output.value.squeeze(-1)
        assert value.shape == returns.shape, "Wrong shapes of value"
        critic_loss = self.loss_fn(value, returns.detach())

        # calculating entropy
        try:
            entropy = output.dist.entropy()
        except NotImplementedError:
            entropy = output.dist.base_dist.entropy()
        except Exception as e:
            raise RuntimeError(f"Failed to compute entropy: {e}")
        if entropy.numel() == 0:
            raise ValueError("Empty entropy tensor")
        entropy = entropy.mean()

        loss = actor_loss + self.critic_coef_ * critic_loss - self.entropy_beta_ * entropy

        # backpropagation of the error
        self.optimizer.zero_grad()
        loss.backward()
        
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        return loss.item(), actor_loss.item(), critic_loss.item(), entropy.item()
