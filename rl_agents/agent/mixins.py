from abc import ABC, abstractmethod
import numpy as np
import random
import torch as T
from torch import nn
from typing import Tuple, Generator, NamedTuple, Any

from utils.utils import step_return_discounting
from memory.replay_buffer import ReplayBuffer
from models.models import Observation, ActionOutput


class PolicyMixin(ABC):
    def __init__(
        self,
        num_actions: int,
        gamma_: float = 0.99,
        lambda_: float = 1,
        *args,
        **kwargs
    ):
        self.gamma_ = gamma_
        self.lambda_ = lambda_
        self.num_actions = num_actions
        self.buffer: ReplayBuffer

    @property
    @abstractmethod
    def action_network(self) -> nn.Module:
        """Return the torch.nn.Module used to generate actions."""
        pass

    def action(
        self,
        state: T.Tensor,
        epsilon_: float = 0.5,
        method: str = "egreedy"
    ) -> ActionOutput:
        net = self.action_network  # use the enforced getter
        assert isinstance(net, nn.Module), f"Expected nn.Module, got {type(net)}"

        if method == "egreedy":
            with T.no_grad():
                output = net(state)
            logits = output.logits
            value = getattr(output, "value", None)
            if random.random() > epsilon_:
                action = logits.argmax(keepdim=True)
            else:
                action = T.randint(0, self.num_actions, size=(1,))
            logprob = output.dist.log_prob(action)
            action_output = ActionOutput(
                action=action,
                logits=logits,
                log_probs=logprob,
                value=value
            )
            return action_output
        
        raise ValueError("Selected method is not valid")

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
    
    @staticmethod
    def _preprocess_batch(batch: Observation) -> Observation:
        state = T.as_tensor(batch.state, dtype=T.float32)
        next_state = T.as_tensor(batch.next_state, dtype=T.float32)
        logits = T.as_tensor(batch.logits, dtype=T.float32)
        action = T.as_tensor(batch.action, dtype=T.int64)
        reward = T.as_tensor(batch.reward, dtype=T.float32)
        done = T.as_tensor(batch.done, dtype=T.float32)
        value = T.as_tensor(batch.value, dtype=T.float32) if batch.value is not None else None
        preprocessed_batch = type(batch)(
            state=state,
            next_state=next_state,
            logits=logits,
            action=action,
            reward=reward,
            done=done,
            value=value,
        )
        return preprocessed_batch
    
    def update_buffer(self, item: dict[str, Any], *args, **kwargs) -> None:
        self.buffer.push(item)
