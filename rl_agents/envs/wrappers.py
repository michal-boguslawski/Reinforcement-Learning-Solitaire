import gymnasium as gym
import torch as T

class TerminalBonusWrapper(gym.RewardWrapper):
    def __init__(self, env, terminal_bonus: float | None = 0., truncated_bonus: float | None = 0.):
        super().__init__(env)
        self.terminal_bonus = terminal_bonus or 0
        self.truncated_bonus = truncated_bonus or 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add extra reward when the episode terminates
        reward = float(reward)
        if terminated:
            reward += self.terminal_bonus
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

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)
        if self.pow_factors is not None:
            reward += T.sum(obs ** 2 * self.pow_factors).item() * self.decay
        if self.abs_factors is not None:
            reward += T.sum(obs.abs() * self.abs_factors).item() * self.decay
        if terminated:
            self.decay *= self.decay_factor
        elif truncated:
            self.decay /= (self.decay_factor ** (1/10))
            self.decay = min(self.decay, 1)
        
        return obs, reward, terminated, truncated, info


class NoMovementInvPunishmentRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, punishment: T.Tensor):
        super().__init__(env)
        self.punishment = punishment

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)
        inv_obs = 1 / (obs.abs() + 1e-6)
        reward += T.sum(inv_obs.clip(0, 100) * self.punishment).item()

        return obs, reward, terminated, truncated, info
