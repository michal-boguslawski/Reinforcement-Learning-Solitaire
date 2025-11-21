import gymnasium as gym
from gymnasium.wrappers import NumpyToTorch, RecordVideo, TransformReward
from gymnasium.vector import SyncVectorEnv
import gymnasium.wrappers.vector as vec_wrappers
import numpy as np
import torch as T
from typing import Tuple, Callable, Any, Dict

from .wrappers import TerminalBonusWrapper, PowerObsRewardWrapper, NoMovementInvPunishmentRewardWrapper
from config.config import EnvConfig, Config
import inspect


wrappers_dict: Dict[str, Callable] = {
    "scale_reward": lambda env, scale_factor, loc_factor: 
        TransformReward(env, lambda r: scale_factor * r + loc_factor),
    "clip_action": gym.wrappers.ClipAction,
    "record_video": RecordVideo,
}


def clean_kwargs(func, kwargs: dict):
    """
    Return a new dict containing only keys that appear
    in the function's signature.
    """
    sig = inspect.signature(func)
    valid = set(sig.parameters.keys())

    return {k: v for k, v in kwargs.items() if k in valid}


def make_env(
    experiment_name: str,
    device: T.device = T.device('cpu'),
    record: bool = False,
    video_folder: str = "logs/videos",
    name_prefix: str = "eval"
) -> gym.Env:
    config = Config(experiment_name).get_config()
    env_details = config.get("env", {})
    env = gym.make(
        # id=env_details.get("env_name"),
        render_mode="rgb_array" if record else None,
        **{k: v for k, v in env_details.items() if k != "vectorization_mode"}
        # **clean_kwargs(gym.make, env_details)
        # disable_env_checker=True,
        # apply_api_compatibility=True,
    )

    if record:
        env = RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix=name_prefix,
            episode_trigger=lambda x: True
        )
        return env

    return env


def make_vec(
    experiment_name: str,
    num_envs: int,
    device: T.device = T.device('cpu'),
):
    config = Config(experiment_name).get_config()
    env_details = config.get("env", {})
    env_config = EnvConfig(env_details.get("env_name")).get_config()
    wrappers = []
    wrappers_config = env_config.get("wrappers")
    if wrappers_config:
        for wrapper_name, wrapper_kwargs in wrappers_config.items():
            wrapper = wrappers_dict.get(wrapper_name, None)
            if wrapper is None:
                raise ValueError(f"Unknown wrapper '{wrapper_name}'")
            wrappers.append(lambda env, w=wrapper, kw=wrapper_kwargs: w(env=env, **kw))

    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}")
    
    try:
        # Fix lambda closure issue by using default parameter
        envs = gym.make_vec(
            # id=env_details.get("env_name"),
            num_envs=num_envs,
            # vectorization_mode=env_details.get("vectorization_mode", "sync"),
            vector_kwargs=None,
            wrappers=wrappers,
            **env_details
        )
        envs = vec_wrappers.DtypeObservation(envs, np.float32)
        envs = vec_wrappers.NumpyToTorch(envs, device=device)
    except Exception as e:
        raise RuntimeError(f"Failed to create vectorized environment '{experiment_name}' with {num_envs} envs: {e}")
    
    return envs
