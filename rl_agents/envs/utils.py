from gymnasium.spaces import Discrete
from gymnasium.vector import VectorEnv
import numpy as np
import torch as T
from .env_setup import make_env
from models.models import EnvDetails


def get_env_vec_details(env: VectorEnv):
    action_space = getattr(env, "action_space")
    observation_space = getattr(env, "observation_space")
    
    action_space_type = "discrete" if isinstance(env.action_space, Discrete) else "continuous"

    low = getattr(action_space, "low")[0]
    high = getattr(action_space, "high")[0]
    
    return EnvDetails(
        action_dim=int(action_space.shape[-1]),
        state_dim=observation_space.shape[1:],
        action_space_type=action_space_type,
        action_low=low,
        action_high=high,
    )
