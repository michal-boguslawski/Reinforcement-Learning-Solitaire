import gymnasium as gym
import torch as T

class TerminalBonusWrapper(gym.RewardWrapper):
    def __init__(self, env, terminal_bonus: float = 0.):
        super().__init__(env)
        self.terminal_bonus = terminal_bonus

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add extra reward when the episode terminates
        reward = float(reward)
        if terminated:
            reward += self.terminal_bonus

        return obs, reward, terminated, truncated, info


class PowerObsRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, pow_factors: T.Tensor):
        super().__init__(env)
        self.pow_factors = pow_factors

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)
        reward -= T.sum(obs ** 2 * self.pow_factors).item()
        
        return obs, reward, terminated, truncated, info

