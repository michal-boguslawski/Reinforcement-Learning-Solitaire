import numpy as np
import torch as T

from agent.base import BasePolicy
from agent.on_policy import OnPolicy
from envs.env_setup import make_env

class Worker:
    def __init__(
        self,
        env_name: str,
        agent: BasePolicy,
        epsilon_start_: float = 0.5,
        epsilon_decay_factor_: float = 0.9999,
        temperature_start_: float = 1,
        *args,
        **kwargs
    ):
        self.agent = agent
        self.env_name = env_name
        self.env = make_env(self.env_name)
        
        self.epsilon_start_ = epsilon_start_
        self.epsilon_decay_factor_ = epsilon_decay_factor_
        self.temperature_start_ = temperature_start_
        
        # training vars
        self.i: int = 0
        self.losses_list = []
        self.train_step: int = 1
        self.batch_size: int = 32
        self.minibatch_size: int = 32
        self.epsilon: float = 1.
        self.method = "egreedy"
    
    def _reset_training_vars(
        self,
        train_step: int,
        batch_size: int,
        minibatch_size: int,
        epsilon_start: float,
        timesteps: int | None = None
    ):
        self.i = 0
        self.losses_list = []
        self.train_step = batch_size if isinstance(self.agent, OnPolicy) else train_step
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.epsilon = epsilon_start
        self.timesteps = timesteps

    def _train_one_episode(self):
        total_reward = 0
        done = False
        try:
            state, _ = self.env.reset()
        except Exception as e:
            print(f"Error resetting environment: {e}")
            return 0
        
        while not done:
            try:
                action, value, logits = self.agent.action(state, epsilon_=self.epsilon, method=self.method)
                next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            except Exception as e:
                print(f"Error during episode step: {e}")
                break
            
            done = terminated or truncated
            reward = T.as_tensor(reward)
            done = T.as_tensor(done)
            self.agent.update_buffer((state, next_state, logits, action, reward, done, value))
            
            if self.i % self.train_step == 0 and self.i > 0:
                loss = self.agent.train(
                    batch_size=self.batch_size,
                    minibatch_size=self.minibatch_size,
                    timesteps = self.timesteps
                )
                if loss:
                    self.losses_list.append(loss)
            
            state = next_state
            
            total_reward += reward
            self.i += 1

        self.epsilon *= self.epsilon_decay_factor_
        
        return total_reward

    def train(
        self,
        episodes: int,
        batch_size: int,
        timesteps: int | None = None,
        minibatch_size: int = 64,
        train_step: int = 16,
        *args,
        **kwargs
    ):
        rewards_list = []
        self._reset_training_vars(
            train_step=train_step,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            epsilon_start=self.epsilon_start_,
            timesteps=timesteps
        )
        
        for episode in range(int(episodes)):
            total_reward = self._train_one_episode()
            rewards_list.append(total_reward)
            
            if episode % 100 == 0:
                self._print_results(episode, rewards_list)

        self._print_results(episodes, rewards_list)
        self.env.close()

    def _print_results(self, episode: int, rewards_list: list[float]) -> None:
        try:
            recent_rewards = rewards_list[-100:] if rewards_list else [0]
            recent_losses = self.losses_list[-100:] if self.losses_list else [0]
            print(f"Episode {episode}, i {self.i}, Avg Reward {np.mean(recent_rewards):.4f},",
                f"Max Reward {max(recent_rewards):4f}, Loss {np.mean(recent_losses):.6f}, epsilon {self.epsilon:.4f}")
            if episode % 1000 == 0:
                self._eval_record_video(episode=episode)
        except Exception as e:
            print(f"Error printing results: {e}")

    def _eval_record_video(self, episode: int) -> None:
        env = make_env(
            self.env_name,
            record=True,
            video_folder = f"logs/{self.env_name}/videos/episode_{episode}",
        )
        total_reward = 0.
        step_count = 0
        state, _ = env.reset()
        done = False
        truncated = False
        while not done:
            state = T.as_tensor(state, dtype=T.float32)
            action, _, _ = self.agent.action(state, epsilon_=0.0, method=self.method)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += float(reward)
            state = next_state
            step_count += 1
        result = "Win" if truncated else "Lose"
        print(f"Episode {episode + 1}: {step_count} steps, reward = {total_reward}, result: {result}")
        env.close()
