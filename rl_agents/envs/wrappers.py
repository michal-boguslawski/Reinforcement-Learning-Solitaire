import gymnasium as gym
from gymnasium.core import ObsType
import numpy as np
import torch as T


class TerminalBonusWrapper(gym.RewardWrapper):
    def __init__(self, env, terminated_bonus: float | None = 0., truncated_bonus: float | None = 0.):
        super().__init__(env)
        self.terminated_bonus = terminated_bonus or 0
        self.truncated_bonus = truncated_bonus or 0
        print(f"TerminalBonusWrapper attached with params {terminated_bonus} {truncated_bonus}")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add extra reward when the episode terminates
        reward = float(reward)
        if terminated:
            reward += self.terminated_bonus
        if truncated:
            reward += self.truncated_bonus

        return obs, reward, terminated, truncated, info


class PowerObsRewardWrapper(gym.RewardWrapper):
    def __init__(
        self,
        env,
        pow_factors: T.Tensor | None = None,
        abs_factors: T.Tensor | None = None,
        decay_factor: float | None = 1
    ):
        super().__init__(env)
        self.pow_factors = pow_factors
        self.abs_factors = abs_factors
        self.decay = 1
        self.decay_factor = decay_factor or 1
        print(f"PowerObsRewardWrapper attached with params {pow_factors} {abs_factors} {decay_factor}")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)
        if self.pow_factors is not None:
            reward += (obs ** 2 * self.pow_factors).sum().item() * self.decay
        if self.abs_factors is not None:
            reward += (np.abs(obs) * self.abs_factors).sum().item() * self.decay
        if terminated:
            self.decay *= self.decay_factor
        
        return obs, reward, terminated, truncated, info


class NoMovementInvPunishmentRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, punishment: T.Tensor):
        super().__init__(env)
        self.punishment = punishment

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)
        inv_obs = 1 / (np.abs(obs) + 1e-6)
        reward -= (inv_obs.clip(0, 100) * self.punishment).sum().item()

        return obs, reward, terminated, truncated, info


class VecTransposeObservationWrapper(gym.vector.VectorObservationWrapper):
    def observations(self, observations):
        return observations.permute(0, 3, 1, 2).contiguous()


class TransposeObservationWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class ActionPowerRewardWrapper(gym.RewardWrapper):
    def __init__(
        self,
        env,
        pow_factors: T.Tensor | None = None,
        abs_factors: T.Tensor | None = None,
        decay_factor: float | None = 1
    ):
        super().__init__(env)
        self.pow_factors = pow_factors
        self.abs_factors = abs_factors
        self.decay = 1
        self.decay_factor = decay_factor or 1
        print(f"PowerObsRewardWrapper attached with params {pow_factors} {abs_factors} {decay_factor}")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)
        if self.pow_factors is not None:
            reward += (action ** 2 * self.pow_factors).sum().item() * self.decay
        if self.abs_factors is not None:
            reward += (np.abs(action) * self.abs_factors).sum().item() * self.decay
        if terminated:
            self.decay *= self.decay_factor
        
        return obs, reward, terminated, truncated, info
