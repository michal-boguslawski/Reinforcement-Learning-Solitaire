import gymnasium as gym
from gymnasium.wrappers import NumpyToTorch, RecordVideo, TransformReward, DtypeObservation
from gymnasium.vector import SyncVectorEnv
import gymnasium.wrappers.vector as vec_wrappers
import numpy as np
import torch as T
from typing import Tuple, Callable, Any, Dict

from .wrappers import TerminalBonusWrapper, PowerObsRewardWrapper, NoMovementInvPunishmentRewardWrapper
from config.config import EnvConfig


wrappers_dict: Dict[str, Callable] = {
    "scale_reward": lambda env, scale_factor, loc_factor: 
        TransformReward(env, lambda r: scale_factor * r + loc_factor),
    "terminal_bonus": TerminalBonusWrapper,
    "power_obs_reward": PowerObsRewardWrapper,
    "clip_action": gym.wrappers.ClipAction,
    "record_video": RecordVideo,
}


def make_env(
    env_config: dict[str, Any],
    record: bool = False,
    video_folder: str = "logs/videos",
    name_prefix: str = "eval"
) -> gym.Env:
    env = gym.make(
        # id=env_details.get("env_name"),
        render_mode="rgb_array" if record else None,
        **{k: v for k, v in env_config.items() if k not in  ["vectorization_mode", "num_envs"]}
        # **clean_kwargs(gym.make, env_details)
        # disable_env_checker=True,
        # apply_api_compatibility=True,
    )
    env = DtypeObservation(env, np.float32)

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
    id: str,
    num_envs: int,
    *args,
    **kwargs
):
    env_config = EnvConfig(id).get_config()
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
            id=id,
            num_envs=num_envs,
            # vectorization_mode=env_details.get("vectorization_mode", "sync"),
            vector_kwargs=None,
            wrappers=wrappers,
            *args,
            **kwargs
        )
        envs = vec_wrappers.DtypeObservation(envs, np.float32)
        envs = vec_wrappers.NumpyToTorch(envs)
    except Exception as e:
        raise RuntimeError(f"Failed to create vectorized environment '{id}' with {num_envs} envs: {e}")
    
    return envs
