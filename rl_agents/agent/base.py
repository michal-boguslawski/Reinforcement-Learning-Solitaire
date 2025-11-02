from abc import ABC, abstractmethod
import numpy as np
import torch as T
from torch import nn
from typing import Tuple, Any

from models.models import ActionOutput


class BasePolicy(ABC):
    @abstractmethod
    def action(self, state: T.Tensor, *args, **kwargs) -> ActionOutput:
        pass

    @abstractmethod
    def update_buffer(self, item: dict[str, Any], *args, **kwargs) -> None:
        pass

    @abstractmethod
    def calculate_loss(self, batch: Tuple[T.Tensor, ...]) -> float:
        pass

    @abstractmethod
    def train(self, minibatch_size: int, *args, **kwargs) -> np.floating | None:
        pass
