import gymnasium as gym
from gymnasium.wrappers import NumpyToTorch, RecordVideo
from gymnasium.vector import SyncVectorEnv
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
        render_mode="rgb_array" if record else None
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

    terminal_bonus = config.get("terminal_bonus")
    truncated_bonus = config.get("truncated_bonus")
    if terminal_bonus or truncated_bonus:
        env = TerminalBonusWrapper(env, terminal_bonus=terminal_bonus, truncated_bonus=truncated_bonus)
    
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

    scale_reward = config.get("scale_reward")
    if scale_reward:
        env = gym.wrappers.TransformReward(env, lambda x: x * scale_reward)

    return env


def make_vec(
    env_name: str,
    num_envs: int,
    device: T.device = T.device('cpu'),
):
    envs = SyncVectorEnv(
        num_envs * [lambda: make_env(env_name=env_name, device=device), ]
    )
    return envs
