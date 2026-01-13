import torch as T
from torch import nn

from .base import BasePolicy
from .registry import POLICIES


def get_policy(
    policy_type: str,
    network: nn.Module,
    action_space_type: str,
    policy_kwargs: dict,
    device: T.device = T.device("cpu"),
) -> BasePolicy:
    agent = POLICIES.get(policy_type)
    if agent is None:
        raise ValueError(f"Agent {policy_type} does not exist")
    return agent(
        network=network,
        action_space_type=action_space_type,
        device=device,
        **policy_kwargs
    )
