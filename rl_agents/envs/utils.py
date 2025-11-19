import numpy as np
import torch as T
from .env_setup import make_env
from models.models import EnvDetails


def get_env_details(env_name: str):
    
    try:
        with make_env(env_name) as env:
            # env = make_vec(config["env_name"], num_envs=4, device=config["device"])
            
            action_n = getattr(env.action_space, "n", None)
            action_shape = getattr(env.action_space, "shape", None)
            if action_n is not None:
                num_actions = action_n
            elif action_shape is not None:
                num_actions = np.prod(action_shape)
            else:
                num_actions = -1
            if num_actions == -1:
                raise ValueError("Invalid number of actions")
            obs_space_shape = getattr(env.observation_space, "shape")
            if obs_space_shape is None:
                raise ValueError("Invalid observation space shape")
            in_features = np.prod(obs_space_shape)
        
            low = getattr(env.action_space, "low", None)
            low = None if low is None else T.tensor(low)
            high = getattr(env.action_space, "high", None)
            high = None if high is None else T.tensor(high)
        
    except Exception as e:
        print(f"Error initializing environment: {e}")
        raise e

    return EnvDetails(
        action_dim=int(num_actions),
        state_dim=int(in_features),
        action_low=low,
        action_high=high,
        action_space=env.action_space
    )
