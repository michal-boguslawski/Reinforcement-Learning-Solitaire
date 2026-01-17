from abc import ABC, abstractmethod
import numpy as np
import torch as T
from torch import nn
from typing import Any, Generator, Dict

from agent.exploration.factory import get_exploration
from memory.replay_buffer import ReplayBuffer
from models.models import ActionOutput, ActionSpaceType, OnPolicyMinibatch
from network.model import RLModel


class BasePolicy(ABC):
    network: RLModel
    optimizer: T.optim.Optimizer

    def __init__(
        self,
        network: RLModel,
        action_space_type: ActionSpaceType,
        exploration_method: Dict[str, Any],
        gamma_: float = 0.99,
        lambda_: float = 1.,
        buffer_size: int | None = None,
        device: T.device = T.device("cpu"),
        optimizer_kwargs: dict | None = None,
        loss_fn: nn.modules.loss._Loss = nn.HuberLoss(),
        *args,
        **kwargs
    ):
        self.network = network
        self.gamma_ = gamma_
        self.lambda_ = lambda_
        self.buffer = ReplayBuffer(buffer_size=buffer_size, device=device)
        self.action_space_type = action_space_type
        self.device = device
        self.optimizer = T.optim.Adam(self._build_param_groups(optimizer_kwargs), lr=lr)
        self.max_grad_norm = 0.5
        self.loss_fn = loss_fn

        self._exploration_method = get_exploration(
            exploration_method_name=exploration_method["name"],
            exploration_kwargs=exploration_method.get("kwargs", {})
        )

    @abstractmethod
    def _calculate_loss(self, batch: OnPolicyMinibatch) -> T.Tensor:
        pass

    @abstractmethod
    def _get_batch_for_training(self, *args, **kwargs) -> Dict[str, T.Tensor | None]:
        pass

    @abstractmethod
    def _generate_minibatches(self, minibatch_size: int, *args, **kwargs) -> Generator[OnPolicyMinibatch, None, None]:
        pass

    @abstractmethod
    def train(self, minibatch_size: int, *args, **kwargs) -> None:
        batch = self._get_batch_for_training(*args, **kwargs)
        self._train_step(minibatch_size=minibatch_size, batch=batch, *args, **kwargs)

    @property
    def has_critic(self) -> bool:
        return False

    def action(
        self,
        state: T.Tensor,
        core_state: T.Tensor | None = None,
        training: bool = True,
        temperature: float = 1.,
    ) -> ActionOutput:
        net = self.network
        assert isinstance(net, RLModel), f"Expected RLModel, got {type(net)}"

        with T.no_grad():
            output = net(input_tensor=state, core_state=core_state, temperature=temperature)
        
        action = self._exploration_method(
            logits = output.actor_logits,
            dist = output.dist,
            training = training,
            low = net.low,
            high = net.high,
        )

        log_prob = output.dist.log_prob(action)
        action_output = ActionOutput(
            action=action,
            logits=output.actor_logits,
            log_probs=log_prob,
            value=output.critic_value,
            dist=output.dist,
            core_state=output.core_state
        )
        return action_output

    def _train_step(self, minibatch_size: int, batch: Dict[str, T.Tensor | None], *args, **kwargs) -> None:
        minibatch_generator = self._generate_minibatches(minibatch_size, **batch)
        for minibatch in minibatch_generator:
            loss = self._calculate_loss(minibatch)
            self._backward(loss)

    def eval_mode(self) -> None:
        """Changing action network to eval mode"""
        self.network.eval()

    def train_mode(self) -> None:
        """Changing action network to train mode"""
        self.network.train()

    def _backward(self, loss: T.Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def save_weights(self, folder_path: str):
        self.network.save_weights(folder_path)

    def load_weights(self, file_path: str, param_groups: list[str] | None = None):
        self.network.load_weights(file_path, param_groups)
    
    def update_buffer(self, item: dict[str, Any], *args, **kwargs) -> None:
        self.buffer.push(item)

    def _build_param_groups(self, optimizer_kwargs: dict | None = None) -> list[dict]:
        optimizer_kwargs = optimizer_kwargs or {"lr": 3e-4}
        return [{"params": self.network.parameters(), "lr": optimizer_kwargs.get("lr", 3e-4)}]
