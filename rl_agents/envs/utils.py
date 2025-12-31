import inspect

from gymnasium.spaces import Discrete, MultiDiscrete
from gymnasium.vector import VectorEnv
from models.models import EnvDetails


def get_env_vec_details(env: VectorEnv):
    action_space = getattr(env, "action_space")
    observation_space = getattr(env, "observation_space")
    
    action_space_type = "discrete" if isinstance(env.action_space, Discrete) \
        or isinstance(env.action_space, MultiDiscrete) else "continuous"

    if isinstance(env.action_space, Discrete):
        action_dim = env.action_space.n
    elif isinstance (env.action_space, MultiDiscrete):
        action_dim = env.action_space.nvec[0]
    else:
        action_dim = action_space.shape[-1]

    low = getattr(action_space, "low", None)
    if low:
        low = low[0]

    high = getattr(action_space, "high", None)
    if high:
        high = high[0]
    
    return EnvDetails(
        action_dim=int(action_dim),
        state_dim=observation_space.shape[1:],
        action_space_type=action_space_type,
        action_low=low,
        action_high=high,
    )


def clean_kwargs(func, kwargs: dict):
    """
    Return a new dict containing only keys that appear
    in the function's signature.
    """
    sig = inspect.signature(func)
    valid = set(sig.parameters.keys())

    return {k: v for k, v in kwargs.items() if k in valid}
