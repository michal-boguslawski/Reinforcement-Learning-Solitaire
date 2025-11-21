from gymnasium.spaces import Discrete
import numpy as np
import torch as T
from typing import Any

from agent.base import BasePolicy
from agent.on_policy import OnPolicy, A2CPolicy, PPOPolicy, SarsaPolicy
from envs.env_setup import make_env, make_vec
import torch.nn as nn


np.set_printoptions(linewidth=1000)


class Worker:
    policy_dict = {
        "a2c": A2CPolicy,
        "ppo": PPOPolicy,
        "sarsa": SarsaPolicy,
    }

    def __init__(
        self,
        experiment_name: str,
        num_envs: int,
        network: nn.Module,
        policy_name: str,
        policy_kwargs: dict[str, Any],
        action_exploration_method: str = "egreedy",
        device: T.device = T.device("cpu"),
        epsilon_start_: float = 0.5,
        epsilon_decay_factor_: float = 0.9999,
        temperature_start_: float = 1,
        *args,
        **kwargs
    ):
        if policy_name not in self.policy_dict:
            raise ValueError(f"Policy {policy_name} not supported")

        if not isinstance(network, nn.Module):
            raise TypeError(f"Expected nn.Module, got {type(network)}")

        try:
            self.agent: BasePolicy = self.policy_dict[policy_name](
                network=network,
                **policy_kwargs
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize {policy_name} policy: {e}")
        self.experiment_name = experiment_name
        self.device = device
        self.num_envs = num_envs
        self.env = make_vec(self.experiment_name, num_envs=num_envs, device=device)
        self.env_discrete_type = isinstance(self.env.action_space, Discrete)

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
        
        self.state, _ = self.env.reset()
        self.total_reward = T.zeros(self.num_envs, device=self.device)

    def _train_one_step(self):
        try:
            state = self.state  # .to(T.float)
            action_output = self.agent.action(method=self.action_exploration_method, state=state, epsilon_=self.epsilon)
            action = action_output.action
            if self.env_discrete_type:
                # For discrete actions, convert to scalar for single env or keep tensor for multiple envs
                action = action.item() if action.numel() == 1 else action.detach().cpu().numpy()
            else:
                # For continuous actions, always convert to numpy array
                action = action.detach().cpu().numpy()
            next_state, reward, terminated, truncated, _ = self.env.step(
                action
            )
        except Exception as e:
            print(f"Error during episode step: {e}")
            raise e
        
        done = T.logical_or(terminated, truncated)
        reward = T.as_tensor(reward)
        done = T.as_tensor(done, dtype=T.bool)
        action = T.as_tensor(action)
        
        record = {
            "state": state,
            "next_state": next_state,
            "logits": action_output.logits,
            "action": action,
            "reward": reward,
            "done": done,
            "value": action_output.value,
            "log_probs": action_output.log_probs
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
        
        self.state = next_state
        
        self.total_reward += reward

        self.epsilon *= self.epsilon_decay_factor_
        return done

    def train(
        self,
        num_steps: int,
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
            timesteps=timesteps,
        )
        
        for num_step in range(int(num_steps)):
            done = self._train_one_step()
            if sum(done):
                total_rewards = self.total_reward * done
                rewards_list.append(total_rewards.sum() / done.sum())
                self.total_reward *= T.logical_not(done)
            
            if num_step % 10000 == 0:
                self._print_results(num_step, rewards_list)

        self._print_results(num_steps, rewards_list)
        self.env.close()

    def _print_results(self, num_step: int, rewards_list: list[float]) -> None:
        try:
            # decay = np.mean([getattr(e, "decay", 1) for e in self.env.env.envs])
            recent_rewards = rewards_list if rewards_list else [0,]
            recent_losses = self.losses_list if self.losses_list else [(0,)]
            mean_losses = [np.mean(column).round(8).item() for column in zip(*recent_losses)]
            print(
                f"Step {num_step}, i {self.i}, Avg Reward {np.mean(recent_rewards):.4f},",
                f"Max Reward {max(recent_rewards):.4f}, Loss {mean_losses}, epsilon {self.epsilon:.4f}",
                # f"Decay {decay:.6f}"
                )
            recent_rewards.clear()
            recent_losses.clear()
            if num_step % 100000 == 0:
                self._eval_record_video(num_step=num_step)
        except Exception as e:
            print(f"Error printing results: {e}")
            # raise e

    def _eval_record_video(self, num_step: int) -> None:
        env = make_env(
            self.experiment_name,
            device=self.device,
            record=True,
            video_folder = f"logs/{self.experiment_name}/videos/num_step_{num_step}",
        )
        total_reward = 0.
        step_count = 0
        state, _ = env.reset()
        done = False
        truncated = False
        terminated = False
        env_discrete_type = isinstance(env.action_space, Discrete)
        while not done:
            state = T.as_tensor(state, dtype=T.float32, device=self.device)
            action_output = self.agent.action(method="best", state=state)
            action = action_output.action
            action = action.item() if env_discrete_type else action.detach().cpu().numpy()
            next_state, reward, terminated, truncated, _ = env.step(
                    action
                )
            done = terminated or truncated
            total_reward += float(reward)
            state = next_state
            step_count += 1
        print(
            f"Step {num_step}: {step_count} steps, reward = {total_reward:.2f}, truncated = {truncated}, terminated = {terminated}"
        )
        print(state.round(2))
        env.close()
