from gymnasium.spaces.space import Space
import numpy as np
import torch as T
import torch.nn as nn
from torch.optim import Optimizer
import random
from typing import Generator, Tuple

from .base import BasePolicy
from .mixins import PolicyMixin
from memory.replay_buffer import ReplayBuffer
from models.models import Observation


class OnPolicy(PolicyMixin, BasePolicy):
    def __init__(self, device: T.device = T.device('cpu'), *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
            self.buffer = ReplayBuffer(device=device)
            self.device = device
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OnPolicy: {e}")
        
    def train(self, minibatch_size: int = 64, **kwargs) -> list[float] | None:
        batch = self.buffer.get_all()
        if not batch:
            return
        losses = []
        for _ in range(4):
            minibatches = self._generate_minibatches(batch=batch, minibatch_size=minibatch_size)
            for minibatch in minibatches:
                loss = self.calculate_loss(minibatch)
                losses.append(loss)
        self.buffer.clear()
        return [np.mean(column).item() for column in zip(*losses)]

    def _generate_minibatches(
        self,
        batch: Observation,
        minibatch_size: int = 64
    ) -> Generator[Tuple[T.Tensor, ...], None, None]:
        batch = self._preprocess_batch(batch=batch)
        
        if batch.value is not None:
            state_values = batch.value.squeeze(-1)
            next_state_values = state_values[1:]
        else:
            state_values = batch.logits.gather(dim=-1, index=batch.action).squeeze(-1)
            next_state_values = state_values[1:]
        
        returns, advantages = self._compute_advantage_and_results(
            rewards=batch.reward[:-1],
            dones=batch.done[:-1],
            state_values=state_values[:-1],
            next_state_values=next_state_values,
            gamma_=self.gamma_,
            lambda_=self.lambda_
        )

        # More robust advantage normalization
        # adv_mean = advantages.mean()
        # adv_std = advantages.std()
        # advantages = (advantages - adv_mean) / (adv_std if adv_std > 1e-6 else 1)
        
        batch_size = batch.state.shape[0]
        minibatch_ids = np.random.permutation(batch_size // minibatch_size)
        for minibatch_id in minibatch_ids:
            start = minibatch_id * minibatch_size
            end = min((minibatch_id + 1) * minibatch_size, batch_size-1)
            batch_advantages = advantages[start:end]
            batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-6)
            # batch_advantages = T.clamp(batch_advantages, -10.0, 10.0)
            yield (batch.state[start:end], returns[start:end], batch.action[start:end], batch_advantages)


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
        )
        
        self.network = network
        self.optimizer = optimizer
        self.loss_fn = loss_fn
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
        optimizer: Optimizer,
        action_space: Space,
        gamma_: float = 0.99,
        lambda_: float = 1,
        entropy_beta_: float = 0.01,
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
        )
        
        self.network = network
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.gamma_ = gamma_
        self.lambda_ = lambda_
        self.entropy_beta_ = entropy_beta_
    
    @property
    def action_network(self) -> nn.Module:
        return self.network

    def calculate_loss(self, batch: Tuple[T.Tensor, ...]) -> Tuple[float, ...]:
        states, results, actions, advantages = batch
        
        output = self.network(states)
        log_probs = output.dist.log_prob(actions.squeeze())
        # log_probs = log_probs.clamp(-20, 20)
        # log_probs = T.nan_to_num(log_probs, nan=10.0, posinf=20.0, neginf=-20.0)
        actor_loss = -(log_probs.sum(-1) * advantages).mean()
        
        if T.isnan(log_probs.cpu()).any():
            print("actor_loss:", actor_loss.cpu().item())
            print("Actions:", actions.cpu().squeeze())
            print("log_probs:", log_probs.cpu())
            print("advantages:", advantages.cpu())
            print("results:", results.cpu())
            raise ValueError("actions causing problems")
        
        critic_loss = self.loss_fn(output.value.squeeze(-1), results)
        try:
            entropy = output.dist.entropy()
        except NotImplementedError:
            entropy = output.dist.base_dist.entropy().sum(-1)
        entropy = entropy.mean()

        loss = actor_loss + 0.5 * critic_loss - self.entropy_beta_ * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        return loss.item(), actor_loss.item(), critic_loss.item(), entropy.item()
