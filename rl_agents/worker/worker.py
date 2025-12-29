from gymnasium.vector import VectorEnv
import numpy as np
import torch as T
import torch.nn as nn
from tqdm import tqdm
from typing import Any

from agent.base import BasePolicy
from agent.on_policy import OnPolicy, A2CPolicy, PPOPolicy, SarsaPolicy
from envs.env_setup import make_env, make_vec
from envs.utils import get_env_vec_details
from models.models import ActionSpaceType, EnvDetails
from network.general import ActorCriticNetwork, MLPNetwork


np.set_printoptions(linewidth=1000)


class Worker:
    env: VectorEnv
    action_space_type: ActionSpaceType
    env_details: EnvDetails
    agent: BasePolicy
    i: int
    losses_list: list[float]
    train_step: int
    batch_size: int
    minibatch_size: int
    epsilon: float

    policy_dict = {
        "a2c": A2CPolicy,
        "ppo": PPOPolicy,
        "sarsa": SarsaPolicy,
    }
    
    network_dict = {
        "ac_network": ActorCriticNetwork
    }

    def __init__(
        self,
        experiment_name: str,
        env_config: dict[str, Any],
        policy_config: dict[str, Any],
        network_config: dict[str, Any] = {},
        action_exploration_method: str = "egreedy",
        device: T.device = T.device("cpu"),
        epsilon_start_: float = 1.,
        epsilon_decay_factor_: float = 1.,
        temperature_start_: float = 1.,
        *args,
        **kwargs
    ):
        self.experiment_name = experiment_name
        self.device = device
        self.epsilon_start_ = epsilon_start_
        self.epsilon_decay_factor_ = epsilon_decay_factor_
        self.temperature_start_ = temperature_start_
        self.action_exploration_method = action_exploration_method

        self._setup_env(env_config)
        self._setup_network(network_config, device)
        self._setup_policy(policy_config)

    def _setup_env(self, env_config: dict[str, Any]) -> None:
        self.env = make_vec(**env_config)
        self.env_details = get_env_vec_details(self.env)
        self.action_space_type = self.env_details.action_space_type

    def _setup_network(self, network_config: dict[str, Any], device: T.device = T.device("cpu")) -> None:
        network_type = network_config.get("type", "ac_network")
        if network_type is None or network_type not in self.network_dict:
            raise ValueError(f"Network {network_type} not supported")

        network_kwargs = network_config.get("kwargs", {})
        self.network = self.network_dict[network_type](
            input_shape=self.env_details.state_dim,
            num_actions=self.env_details.action_dim,
            low=self.env_details.action_low,
            high=self.env_details.action_high,
            device=device,
            **network_kwargs
        )

    def _setup_policy(self, policy_config: dict[str, Any]) -> None:
        policy_type = policy_config.get("type", "a2c")
        if policy_type is None or policy_type not in self.policy_dict:
            raise ValueError(f"Policy {policy_type} not supported")

        policy_kwargs = policy_config.get("kwargs", {})
        self.agent = self.policy_dict[policy_type](
            network=self.network,
            num_actions=self.env_details.action_dim,
            action_space_type=self.action_space_type,
            device=self.device,
            **policy_kwargs
        )

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
        self.total_reward = T.zeros(len(self.state))

    def _train_one_step(self):
        try:
            state = self.state.to(self.device)
            action_output = self.agent.action(method=self.action_exploration_method, state=state, epsilon_=self.epsilon)
            action = action_output.action
            if self.action_space_type == "discrete":
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
        
        tq_iter = tqdm(range(int(num_steps)), desc=f"Training {self.experiment_name}", unit="steps")
        
        for num_step in tq_iter:
            done = self._train_one_step()
            if sum(done):
                total_rewards = self.total_reward * done
                rewards_list.append(total_rewards.sum() / done.sum())
                self.total_reward *= T.logical_not(done)

                temp_reward_mean = np.mean(rewards_list)
                tq_iter.set_postfix_str(f"temp mean rewards {temp_reward_mean:.2f}")
            
            if num_step % 10_000 == 0:
                try:
                    self.agent.step_entropy_decay()  # type: ignore
                except AttributeError:
                    pass
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
            # if num_step % 100_000 == 0:
            #     self._eval_record_video(num_step=num_step)
        except Exception as e:
            print(f"Error printing results: {e}")
            raise e

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
        action_output = None
        env_discrete_type = isinstance(env.action_space, Discrete)
        self.agent.eval_mode()
        while not done:
            state = T.as_tensor(state, dtype=T.float32, device=self.device).unsqueeze(0)
            action_output = self.agent.action(method="best", state=state)
            action = action_output.action.squeeze(0)
            action = action.item() if env_discrete_type else action.detach().cpu().numpy()
            next_state, reward, terminated, truncated, _ = env.step(
                    action
                )
            done = terminated or truncated
            total_reward += float(reward)
            state = next_state
            step_count += 1
        env.close()
        self.agent.train_mode()
        print(
            f"Step {num_step}: {step_count} steps, reward = {total_reward:.2f}, truncated = {truncated}, terminated = {terminated}"
        )
        if state.ndim < 3:
            print("Evaluation stopped in state", state.round(2))
        if action_output and action_output.dist:
            try:
                if "covariance_matrix" in dir(action_output.dist.base_dist):  # type: ignore
                    print("Covariance matrix \n", action_output.dist.base_dist.covariance_matrix[0].numpy())  # type: ignore
                else:
                    print("Standard deviation \n", action_output.dist.base_dist.stddev[0].numpy())  # type: ignore
            except AttributeError:
                pass
