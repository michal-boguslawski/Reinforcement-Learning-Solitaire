import gymnasium as gym
from gymnasium.wrappers import NumpyToTorch, RecordVideo
from gymnasium.vector import SyncVectorEnv
import gymnasium.wrappers.vector as vec_wrappers
import numpy as np
import torch as T

from .wrappers import TerminalBonusWrapper, PowerObsRewardWrapper, NoMovementInvPunishmentRewardWrapper
from config.config import EnvConfig


def make_env(
    env_name: str,
    device: T.device = T.device('cpu'),
    record: bool = False,
    video_folder: str = "logs/videos",
    name_prefix: str = "eval"
) -> gym.Env:
    config = EnvConfig(env_name).get_config()
    env = gym.make(
        env_name,
        render_mode="rgb_array" if record else None,
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

    env = NumpyToTorch(env, device=device)

    terminated_bonus = config.get("terminated_bonus")
    truncated_bonus = config.get("truncated_bonus")
    if terminated_bonus or truncated_bonus:
        env = TerminalBonusWrapper(env, terminated_bonus=terminated_bonus, truncated_bonus=truncated_bonus)
    
    pow_factors = config.get("pow_factors")
    abs_factors = config.get("abs_factors")
    decay_factor = config.get("decay_factor")
    if pow_factors or abs_factors:
        env = PowerObsRewardWrapper(
            env,
            pow_factors=T.tensor(pow_factors, device=device) if pow_factors else None,
            abs_factors=T.tensor(abs_factors, device=device) if abs_factors else None,
            decay_factor=decay_factor
        )
    
    no_movement_inv_punishment = config.get("no_movement_inv_punishment")
    if no_movement_inv_punishment:
        env = NoMovementInvPunishmentRewardWrapper(env, T.tensor(no_movement_inv_punishment, device=device))

    scale_reward = config.get("scale_reward", 1.)
    loc_reward = config.get("loc_reward", 0.)
    if scale_reward != 1. or loc_reward != 0.:
        env = gym.wrappers.TransformReward(env, lambda x: x * scale_reward + loc_reward)

    return env


def make_vec(
    env_name: str,
    num_envs: int,
    device: T.device = T.device('cpu'),
):
    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}")
    
    try:
        # Fix lambda closure issue by using default parameter
        envs = SyncVectorEnv(
            [lambda name=env_name, dev=device: make_env(env_name=name, device=dev) for _ in range(num_envs)],
        )
        envs = vec_wrappers.NumpyToTorch(envs)
        # envs = vec_wrappers.ArrayConversion(envs, env_xp=np, target_xp=np)
    except Exception as e:
        raise RuntimeError(f"Failed to create vectorized environment '{env_name}' with {num_envs} envs: {e}")
    
    return envs
