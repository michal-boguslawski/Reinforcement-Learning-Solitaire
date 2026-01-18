import logging
import numpy as np
import torch as T

from .base import PolicyCallback


logger = logging.getLogger(__name__)


class TrainPolicyLogger(PolicyCallback):
    def __init__(self):
        super().__init__()
        self.logs = {}

    def on_loss(self, loss: T.Tensor, name: str):
        loss_list = self.logs.get(name, [])
        loss_list.append(loss.detach().cpu().item())
        self.logs[name] = loss_list

    def on_train_end(self):
        log = ""
        for key, value in self.logs.items():
            if isinstance(value, list):
                value = np.mean(value)
            log += f"{key} {value:.4f} "
        logger.debug(log)
        self.logs.clear()
