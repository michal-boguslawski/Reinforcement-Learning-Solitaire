from abc import ABC, abstractmethod
import numpy as np
import torch as T
from torch import nn
from typing import Tuple


class BasePolicy(ABC):
    @abstractmethod
    def action(self, state: T.Tensor, *args, **kwargs) -> Tuple[T.Tensor, T.Tensor | None, T.Tensor]:
        pass
    
    @abstractmethod
    def update_buffer(self, item: Tuple[T.Tensor | None, ...], *args, **kwargs) -> None:
        pass
    
    @abstractmethod
    def calculate_loss(self, batch: Tuple[T.Tensor, ...]) -> float:
        pass
    
    @abstractmethod
    def train(self, minibatch_size: int, *args, **kwargs) -> np.floating | None:
        pass
