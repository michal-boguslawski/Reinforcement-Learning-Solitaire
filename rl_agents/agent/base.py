from abc import ABC, abstractmethod
import numpy as np
import torch as T
from torch import nn
from typing import Tuple, Any

from memory.replay_buffer import ReplayBuffer
from models.models import ActionOutput, OnPolicyMinibatch
from network.model import RLModel


class BasePolicy(ABC):
    network: RLModel
    optimizer: T.optim.Optimizer

    @abstractmethod
    def action(self, state: T.Tensor, training: bool = True, *args, **kwargs) -> ActionOutput:
        pass

    @abstractmethod
    def update_buffer(self, item: dict[str, Any], *args, **kwargs) -> None:
        pass

    @abstractmethod
    def calculate_loss(self, batch: OnPolicyMinibatch) -> Tuple[float, ...]:
        pass

    @abstractmethod
    def train(self, minibatch_size: int, *args, **kwargs) -> list[float] | None:
        pass

    @property
    @abstractmethod
    def action_network(self) -> nn.Module:
        """Return the torch.nn.Module used to generate actions."""
        pass

    @abstractmethod
    def eval_mode(self) -> None:
        """Changing action network to eval mode"""
        pass

    @abstractmethod
    def train_mode(self) -> None:
        """Changing action network to train mode"""
        pass

    @abstractmethod
    def load_weights(self, file_path: str, param_groups: list[str] | None = None) -> None:
        pass

    @abstractmethod
    def save_weights(self, folder_path: str) -> None:
        pass

    def _backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
