import gymnasium as gym
from gymnasium.wrappers import NumpyToTorch, RecordVideo
import torch as T

from .wrappers import TerminalBonusWrapper, PowerObsRewardWrapper
from config.config import EnvConfig


def make_env(env_name: str, record: bool = False, video_folder: str = "logs/videos", name_prefix: str = "eval") -> gym.Env:
    config = EnvConfig().get_config(env_name)
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

    env = NumpyToTorch(env)

    terminated_reward = config.get("terminated_reward")
    if terminated_reward:
        env = TerminalBonusWrapper(env, terminated_reward)
    
    pow_factors = config.get("pow_factors")
    if pow_factors:
        env = PowerObsRewardWrapper(env, T.tensor(pow_factors))
    return env