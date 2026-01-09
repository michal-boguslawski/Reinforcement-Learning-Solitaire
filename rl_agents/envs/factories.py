import gymnasium as gym
from gymnasium.wrappers import NumpyToTorch, RecordVideo, DtypeObservation, RescaleObservation
import gymnasium.wrappers.vector as vec_wrappers
import numpy as np
import torch as T
from typing import Any, Dict

from .registry import WRAPPERS
from .wrappers import VecTransposeObservationWrapper, TransposeObservationWrapper
from config.config import EnvConfig


def make_env(
    env_config: Dict[str, Any],
    record: bool = False,
    video_folder: str = "logs/videos",
    name_prefix: str = "eval",
) -> gym.Env:
    env = gym.make(
        # id=env_details.get("env_name"),
        render_mode="rgb_array" if record else None,
        **{k: v for k, v in env_config.items() if k not in  ["vectorization_mode", "num_envs", "permute_observations"]}
        # **clean_kwargs(gym.make, env_details)
        # disable_env_checker=True,
        # apply_api_compatibility=True,
    )
    env = DtypeObservation(env, np.float32)
    if env_config.get("permute_observations"):
        env = TransposeObservationWrapper(env)

    rescale_observation = env_config.get("rescale_observation")
    if rescale_observation:
        env = RescaleObservation(
            env,
            min_obs=np.float32(rescale_observation["min_obs"]),
            max_obs=np.float32(rescale_observation["max_obs"])
        )

    env_config = EnvConfig(env_config["id"]).get_config()

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
    permute_observations: bool = False,
    *args,
    **kwargs
):
    env_config = EnvConfig(id).get_config()
    wrappers = []
    wrappers.append(lambda env: DtypeObservation(env, np.float32))
    wrappers_config = env_config.get("wrappers")
    if wrappers_config:
        for wrapper_name, wrapper_kwargs in wrappers_config.items():
            wrapper = WRAPPERS.get(wrapper_name, None)
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
        envs = vec_wrappers.NumpyToTorch(envs)
        if permute_observations:
            envs = VecTransposeObservationWrapper(envs)
    except Exception as e:
        raise RuntimeError(f"Failed to create vectorized environment '{id}' with {num_envs} envs: {e}")
    
    return envs
