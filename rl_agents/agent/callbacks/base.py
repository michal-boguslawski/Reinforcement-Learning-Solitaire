import torch as T
from typing import Protocol


class PolicyCallback(Protocol):
    def on_loss(self, loss: T.Tensor, name: str):
        pass

    def on_train_end(self):
        pass
