from torch import nn
import torch as T

from .on_policy import OnPolicy


class FramedStatesLSTMWrapper:
    def __init__(
        self,
        agent: OnPolicy,
        frame_size: int = 3,
        lstm_hidden_dims: int = 64,
    ):
        pass
