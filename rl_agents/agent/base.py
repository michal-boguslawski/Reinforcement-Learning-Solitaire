from abc import ABC, abstractmethod
import numpy as np
import torch as T
from torch import nn
from typing import Tuple, Any

from memory.replay_buffer import ReplayBuffer
from models.models import ActionOutput, OnPolicyMinibatch


class BasePolicy(ABC):
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
