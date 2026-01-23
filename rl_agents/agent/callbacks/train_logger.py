import logging
import numpy as np
import torch as T

from .base import PolicyCallback


logger = logging.getLogger(__name__)


class TrainPolicyLogger(PolicyCallback):
    def __init__(self):
        super().__init__()
        self.logs = {}

    def on_log(self, log: T.Tensor, name: str):
        loss_list = self.logs.get(name, [])
        loss_list.append(log.detach().cpu().mean().item())
        self.logs[name] = loss_list

    def flush(self):
        log = {key: np.mean(value) for key, value in self.logs.items()}
        logger.debug(log)
        self.logs.clear()
