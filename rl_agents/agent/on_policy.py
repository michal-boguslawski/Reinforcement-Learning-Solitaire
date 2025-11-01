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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = ReplayBuffer()
        
    def train(self, minibatch_size: int = 64, **kwargs) -> np.floating | None:
        batch = self.buffer.get_all()
        if not batch:
            return
        minibatches = self._generate_minibatches(batch=batch, minibatch_size=minibatch_size)
        losses = []
        for minibatch in minibatches:
            loss = self.calculate_loss(minibatch)
            losses.append(loss)
        self.buffer.clear()
        return np.mean(losses)

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
        
        batch_size = batch.state.shape[0]
        minibatch_ids = np.random.permutation(batch_size // minibatch_size)
        for minibatch_id in minibatch_ids:
            start = minibatch_id * minibatch_size
            end = min((minibatch_id + 1) * minibatch_size, batch_size-1)
            yield (batch.state[start:end], returns[start:end], batch.action[start:end], advantages[start:end])


class SarsaPolicy(OnPolicy):
    def __init__(
        self, 
        network: nn.Module,
        num_actions: int,
        optimizer: Optimizer,
        gamma_: float = 0.99,
        lambda_: float = 1,
        loss_fn: nn.modules.loss._Loss = nn.HuberLoss(),
        *args,
        **kwargs
    ):
        super().__init__(
            num_actions=num_actions,
            gamma_=gamma_,
            lambda_=lambda_
        )
        
        self.network = network
        self.num_actions = num_actions
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.gamma_ = gamma_
        self.lambda_ = lambda_
    
    @property
    def action_network(self) -> nn.Module:
        return self.network
        
    def calculate_loss(self, batch: Tuple[T.Tensor, ...]) -> float:
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
        gamma_: float = 0.99,
        lambda_: float = 1,
        loss_fn: nn.modules.loss._Loss = nn.HuberLoss(),
        *args,
        **kwargs
    ):
        super().__init__(
            num_actions=num_actions,
            gamma_=gamma_,
            lambda_=lambda_
        )
        
        self.network = network
        self.num_actions = num_actions
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.gamma_ = gamma_
        self.lambda_ = lambda_
    
    @property
    def action_network(self) -> nn.Module:
        return self.network
        
    def calculate_loss(self, batch: Tuple[T.Tensor, ...]) -> float:
        states, results, actions, advantages = batch
        
        output = self.network(states)
        log_probs = output.dist.log_prob(actions.squeeze(-1))
        actor_loss = -(log_probs * advantages).mean()
        
        critic_loss = self.loss_fn(output.value.squeeze(-1), results)
        entropy = 0
        
        loss = actor_loss + critic_loss + entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1)
        self.optimizer.step()
        
        return loss.item()
