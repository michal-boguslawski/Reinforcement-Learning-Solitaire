from abc import ABC, abstractmethod
import torch as T
import torch.nn as nn
from torch.optim import Optimizer, Adam
import random
import numpy as np
from typing import Generator, Tuple

from memory.replay_buffer import ReplayBuffer
from utils.utils import step_return_discounting


class BasePolicy(ABC):
    @abstractmethod
    def action(self, state: T.Tensor):
        pass
    
    @abstractmethod
    def update_buffer(self, item: tuple):
        pass
    
    @abstractmethod
    def calculate_loss(self):
        pass
    
    @abstractmethod
    def train(self):
        pass


class OnPolicy(BasePolicy):
    def __init__(self):
        super().__init__()
    

class Sarsa(OnPolicy):
    def __init__(
        self, 
        network: nn.Module,
        buffer: ReplayBuffer,
        num_actions: int,
        optimizer: Optimizer,
        gamma_: float = 0.99,
        lambda_: float = 1,
        tau_: float = 0.005,
        loss_fn: nn.modules.loss._Loss = nn.HuberLoss,
    ):
        super().__init__()
        self.network = network
        self.num_actions = num_actions
        self.buffer = buffer
        self.optimizer = optimizer
        self.loss_fn = loss_fn()
        self.gamma_ = gamma_
        self.lambda_ = lambda_
        self.tau_ = tau_
    
    def action(self, state: T.Tensor, epsilon_: float = 0.5, method: str = "egreedy") -> int:
        if method == "egreedy":
            with T.no_grad():
                logits = self.network(state)
            if random.random() > epsilon_:
                action = logits.argmax().item()
            else:
                action = random.randint(0, self.num_actions - 1)
        return action, logits
    
    def update_buffer(self, items: tuple) -> None:
        self.buffer.push(items)
    
    @staticmethod
    def _compute_advantage_and_results(
        rewards: T.Tensor, 
        dones: T.Tensor,
        q_values: T.Tensor,
        next_state_values: T.Tensor,
        gamma_: float = 1,
        lambda_: float = 1
    ) -> T.Tensor:
        q_target = rewards + gamma_ * (1 - dones) * next_state_values
        td_errors = q_target - q_values
        discounted_td_errors = step_return_discounting(
            values=td_errors, dones=dones, discount=gamma_*lambda_
        )
        result = discounted_td_errors + q_values
        return result
        
    
    @staticmethod
    def _preprocess_batch(batch: tuple):
        state, logits, action, reward, done = batch
        state = T.tensor(state, dtype=T.float32)
        logits = T.tensor(logits, dtype=T.float32)
        action = T.tensor(action, dtype=T.int64)
        reward = T.tensor(reward, dtype=T.float32)
        done = T.tensor(done, dtype=T.float32)
        action = action.unsqueeze(-1)
        return state, logits, action, reward, done

    def _generate_minibatches(self, batch: tuple, minibatch_size: int = 64) -> Generator[Tuple[T.Tensor, ...]]:
        state, logits, action, reward, done = self._preprocess_batch(batch=batch)
        
        q_values = logits.gather(dim=-1, index=action).squeeze(-1)
        next_state_values = q_values[1:]
        
        results = self._compute_advantage_and_results(
            rewards=reward[:-1],
            dones=done[:-1],
            q_values=q_values[:-1],
            next_state_values=next_state_values,
            gamma_=self.gamma_,
            lambda_=self.lambda_
        )
        
        batch_size = state.shape[0]
        minibatch_ids = np.random.permutation(batch_size // minibatch_size)
        for minibatch_id in minibatch_ids:
            start = minibatch_id * minibatch_size
            end = (minibatch_id + 1) * minibatch_size
            if end == batch_size:
                end -= 1
            yield (state[start:end], results[start:end], action[start:end])
        
    def train(self, minibatch_size: int = 64, **kwargs):
        batch = self.buffer.get_all()
        minibatches = self._generate_minibatches(batch=batch, minibatch_size=minibatch_size)
        losses = []
        for minibatch in minibatches:
            loss = self.calculate_loss(minibatch)
            losses.append(loss)
        self.buffer.clear()
        return np.mean(losses)
        
    def calculate_loss(self, batch: Tuple[T.Tensor, ...]) -> float:
        states, results, actions = batch
        
        logits = self.network(states)
        q_values = logits.gather(dim=-1, index=actions).squeeze(-1)
        
        loss = self.loss_fn(results, q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1)
        self.optimizer.step()
        
        return loss.item()