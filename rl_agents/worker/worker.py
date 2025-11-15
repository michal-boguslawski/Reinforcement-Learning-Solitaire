from gymnasium.spaces import Box
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
        action_exploration_method: str = "egreedy",
        device: T.device = T.device("cpu"),
        epsilon_start_: float = 0.5,
        epsilon_decay_factor_: float = 0.9999,
        temperature_start_: float = 1,
        *args,
        **kwargs
    ):
        self.agent = agent
        self.env_name = env_name
        self.device = device
        self.env = make_env(self.env_name, device=device)
        self.env_box_type = isinstance(self.env.action_space, Box)

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
        self.action_exploration_method = action_exploration_method
    
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
                action_output = self.agent.action(method=self.action_exploration_method, state=state, epsilon_=self.epsilon)
                action = action_output.action
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action.detach().cpu().numpy() if self.env_box_type else action.item()
                )
            except Exception as e:
                print(f"Error during episode step: {e}")
                raise e
            
            done = terminated or truncated
            reward = T.as_tensor(reward)
            done = T.as_tensor(done)
            action = T.as_tensor(action if self.env_box_type else action.item())
            
            record = {
                "state": state,
                "next_state": next_state,
                "logits": action_output.logits,
                "action": action,
                "reward": reward,
                "done": done,
                "value": action_output.value
            }
            
            self.agent.update_buffer(record)
            self.i += 1

            if (
                (self.i % self.train_step == 0 and self.i > 0) or 
                (isinstance(self.agent, OnPolicy) and (len(self.agent.buffer) == self.batch_size))
            ):
                loss = self.agent.train(
                    batch_size=self.batch_size,
                    minibatch_size=self.minibatch_size,
                    timesteps = self.timesteps
                )
                if loss:
                    self.losses_list.append(loss)
            
            state = next_state
            
            total_reward += reward
            

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
        print(20 * "=", "Start training", 20 * "=")
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
            decay = getattr(self.env, "decay", 1)
            recent_rewards = rewards_list if rewards_list else [0,]
            recent_losses = self.losses_list if self.losses_list else [(0,)]
            mean_losses = [np.mean(column).round(8).item() for column in zip(*recent_losses)]
            print(
                f"Episode {episode}, i {self.i}, Avg Reward {np.mean(recent_rewards):.4f},",
                f"Max Reward {max(recent_rewards):.4f}, Loss {mean_losses}, epsilon {self.epsilon:.4f}",
                f"Decay {decay:.6f}"
                )
            recent_rewards.clear()
            recent_losses.clear()
            if episode % 1000 == 0:
                self._eval_record_video(episode=episode)
        except Exception as e:
            print(f"Error printing results: {e}")
            raise e

    def _eval_record_video(self, episode: int) -> None:
        env = make_env(
            self.env_name,
            device=self.device,
            record=True,
            video_folder = f"logs/{self.env_name}/videos/episode_{episode}",
        )
        total_reward = 0.
        step_count = 0
        state, _ = env.reset()
        done = False
        truncated = False
        terminated = False
        while not done:
            state = T.as_tensor(state, dtype=T.float32, device=self.device)
            action_output = self.agent.action(method="best", state=state)
            action = action_output.action
            next_state, reward, terminated, truncated, _ = env.step(
                    action.detach().cpu().numpy() if self.env_box_type else action.item()
                )
            done = terminated or truncated
            total_reward += float(reward)
            state = next_state
            step_count += 1
        print(
            f"Episode {episode}: {step_count} steps, reward = {total_reward:.2f}, truncated = {truncated}, terminated = {terminated}"
        )
        env.close()
