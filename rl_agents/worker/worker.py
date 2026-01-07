from gymnasium.vector import VectorEnv
import logging
import numpy as np
import torch as T
import torch.nn as nn
from tqdm import tqdm
from typing import Any, Literal

from agent.base import BasePolicy
from agent.on_policy import OnPolicy, A2CPolicy, PPOPolicy, SarsaPolicy
from envs.factories import make_env, make_vec
from envs.utils import get_env_vec_details
from models.models import ActionSpaceType, EnvDetails
from network.model import RLModel
from .utils import get_device


np.set_printoptions(linewidth=1000)
logger = logging.getLogger(__name__)


class Worker:
    env: VectorEnv
    action_space_type: ActionSpaceType
    env_details: EnvDetails
    agent: BasePolicy
    i: int
    losses_list: list[list[float] | float]
    train_step: int
    batch_size: int
    minibatch_size: int
    epsilon: float

    policy_dict = {
        "a2c": A2CPolicy,
        "ppo": PPOPolicy,
        "sarsa": SarsaPolicy,
    }

    def __init__(
        self,
        experiment_name: str,
        env_config: dict[str, Any],
        policy_config: dict[str, Any],
        network_config: dict[str, Any] = {},
        device: T.device | Literal["auto", "cpu", "cuda"] = T.device("cpu"),
        record_step: int = 100_000,
        *args,
        **kwargs
    ):
        self.experiment_name = experiment_name
        self.device = get_device(device)
        self.record_step = record_step

        self._setup_env(env_config)
        self._setup_network(network_config, self.device)
        self._setup_policy(policy_config, self.device)

    def _setup_env(self, env_config: dict[str, Any]) -> None:
        self.env_config = env_config
        self.env = make_vec(**env_config)
        self.env_details = get_env_vec_details(self.env)
        self.action_space_type = self.env_details.action_space_type

    def _setup_network(self, network_config: dict[str, Any], device: T.device = T.device("cpu")) -> None:
        network_kwargs = network_config.get("kwargs", {})
        self.network = RLModel(
            input_shape=self.env_details.state_dim,
            num_actions=self.env_details.action_dim,
            low=self.env_details.action_low,
            high=self.env_details.action_high,
            device=device,
            **network_kwargs
        )

    def _setup_policy(self, policy_config: dict[str, Any], device: T.device = T.device("cpu")) -> None:
        policy_type = policy_config.get("type", "a2c")
        if policy_type is None or policy_type not in self.policy_dict:
            raise ValueError(f"Policy {policy_type} not supported")

        policy_kwargs = policy_config.get("kwargs", {})
        self.agent = self.policy_dict[policy_type](
            network=self.network,
            action_space_type=self.action_space_type,
            device=device,
            **policy_kwargs
        )

    def _reset_training_vars(
        self,
        train_step: int,
        batch_size: int,
        minibatch_size: int,
        timesteps: int | None = None
    ):
        self.losses_list = []
        self.train_step = batch_size if isinstance(self.agent, OnPolicy) else train_step
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.timesteps = timesteps
        
        self.state, _ = self.env.reset()
        self.total_reward = T.zeros(len(self.state), device=self.device)

    def _step(self):
        try:
            state = self.state.to(self.device)
            action_output = self.agent.action(state=state)
            action = action_output.action
            if self.action_space_type == "discrete":
                # For discrete actions, convert to scalar for single env or keep tensor for multiple envs
                env_action = action.item() if action.numel() == 1 else action.detach().squeeze(-1).cpu().numpy()
                
            else:
                # For continuous actions, always convert to numpy array
                env_action = action.detach().cpu().numpy()
            next_state, reward, terminated, truncated, _ = self.env.step(
                env_action
            )
        except Exception as e:
            logger.error(f"Error during episode step: {e}")
            raise e
        
        done = T.logical_or(terminated, truncated)
        reward = T.as_tensor(reward, device=self.device)
        done = T.as_tensor(done, dtype=T.bool, device=self.device)
        action = T.as_tensor(action, device=self.device)
        
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
        
        self.state = next_state
        
        self.total_reward += reward

        return done

    def _train_agent(self):
        loss = self.agent.train(
            batch_size=self.batch_size,
            minibatch_size=self.minibatch_size,
            timesteps = self.timesteps
        )
        if loss:
            self.losses_list.append(loss)

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
        logger.info(f"{20 * '='} Start training {20 * '='}")
        rewards_list = []
        self._reset_training_vars(
            train_step=train_step,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            timesteps=timesteps,
        )
        
        tq_iter = tqdm(range(int(num_steps)), desc=f"Training {self.experiment_name}", unit="steps")
        
        for num_step in tq_iter:
            done = self._step()

            if (num_step % self.train_step == 0 and num_step > 0):
                self._train_agent()

            if sum(done):
                total_rewards = self.total_reward * done
                rewards_list.append(total_rewards.sum().cpu() / done.sum().cpu())
                self.total_reward *= T.logical_not(done)

                temp_reward_mean = np.mean(rewards_list)
                tq_iter.set_postfix_str(f"temp mean rewards {temp_reward_mean:.2f}")
                logger.debug(f"Step {num_step}, temp mean rewards {temp_reward_mean:.2f}")
            
            if num_step % 5_000 == 0:
                self._print_results(num_step, rewards_list)
            # Record video
            if num_step % self.record_step == 0:
                self._eval_record_video(num_step=num_step)

        self._print_results(num_steps, rewards_list)
        self._eval_record_video(num_step=num_steps)
        
        self.env.close()
        logger.info(f"{20 * '='} End training {20 * '='}")

    def _print_results(self, num_step: int, rewards_list: list[float]) -> None:
        try:
            # decay = np.mean([getattr(e, "decay", 1) for e in self.env.env.envs])
            recent_rewards = rewards_list if rewards_list else [0,]
            recent_losses = self.losses_list if self.losses_list else [(0,)]
            mean_losses = [np.mean(column).round(8).item() for column in zip(*recent_losses)]
            logger.info(
                f"Step {num_step}, Avg Reward {np.mean(recent_rewards):.4f}, "
                f"Max Reward {max(recent_rewards):.4f}, Loss {mean_losses}"
            )
            recent_rewards.clear()
            recent_losses.clear()

        except Exception as e:
            logger.error(f"Error printing results: {e}")
            raise e

    def _eval_record_video(self, num_step: int) -> None:
        env = make_env(
            env_config=self.env_config,
            record=True,
            video_folder = f"logs/{self.experiment_name}/videos/num_step_{num_step}",
        )
        
        # Initialize evaluation metrics
        total_reward = 0.
        step_count = 0
        state, _ = env.reset()
        done = False
        truncated = False
        terminated = False
        action_output = None

        # Start env inference
        self.agent.eval_mode()
        while not done:
            state = T.as_tensor(state, dtype=T.float32, device=self.device).unsqueeze(0)
            action_output = self.agent.action(state=state, training=False, temperature=0.1)
            action = action_output.action.squeeze(0)
            action = action.item() if self.action_space_type == "discrete" else action.detach().cpu().numpy()
            next_state, reward, terminated, truncated, _ = env.step(
                    action
                )
            done = terminated or truncated
            total_reward += float(reward)
            state = next_state
            step_count += 1
        env.close()
        self.agent.train_mode()
        logger.info(
            f"Step {num_step}: {step_count} steps, reward = {total_reward:.2f}, truncated = {truncated}, terminated = {terminated}"
        )
        if state.ndim < 3:
            logger.info("Evaluation stopped in state %s", state.round(2))
        if action_output and action_output.dist:
            try:
                if "covariance_matrix" in dir(action_output.dist.base_dist):  # type: ignore
                    logger.info("Covariance matrix \n", action_output.dist.base_dist.covariance_matrix[0].cpu().numpy())  # type: ignore
                else:
                    logger.info("Standard deviation \n", action_output.dist.base_dist.stddev[0].cpu().numpy())  # type: ignore
            except AttributeError:
                pass
