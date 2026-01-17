from gymnasium.vector import VectorEnv
import logging
import os
import torch as T

from agent.base import BasePolicy
from agent.factories import get_policy
from envs.factories import make_vec
from envs.utils import get_env_vec_details
from config.config import ExperimentConfig
from network.model import RLModel
from worker.utils import prepare_action_for_env, get_device


logger = logging.getLogger(__name__)


class Evaluator:
    envs: VectorEnv

    def __init__(
        self,
        id: str,
        vectorization_mode: str = "sync",
        num_envs: int = 1,
        record: bool = True,
        video_folder: str | None = None,
        training_wrappers: dict | None = None,
        general_wrappers: dict | None = None,
        *args,
        **kwargs
    ):
        self.id = id
        self.vectorization_mode = vectorization_mode
        self.num_envs = num_envs
        self.record = record
        self.video_folder = video_folder
        self.training_wrappers = training_wrappers
        self.general_wrappers = general_wrappers

        self._setup_envs()

    def _setup_envs(self):
        self.envs = make_vec(
            id=self.id,
            num_envs=self.num_envs,
            training=False,
            record=self.record,
            video_folder=self.video_folder,
            name_prefix="eval",
            training_wrappers=self.training_wrappers,
            general_wrappers=self.general_wrappers,
            vectorization_mode=self.vectorization_mode,
        )

    def _print_evaluation_results(self, info):
        print(info)

    def change_env_config(self, num_envs: int | None = None, record: bool = True):
        if num_envs:
            self.num_envs = num_envs
        if record:
            self.record = record

        self._setup_envs()
        

    def evaluate(self, agent: BasePolicy, min_episodes: int = 1, action_space_type: str = "discrete"):
        device = agent.action_network.device
        state, info = self.envs.reset()
        finished_envs = 0
        
        while finished_envs < min_episodes:
            state = state.to(device)
            with T.no_grad():
                action_output = agent.action(state=state, training=False)

            action = action_output.action
            env_action = prepare_action_for_env(action, action_space_type)
            state, _, terminated, truncated, info = self.envs.step(env_action)
            done = T.logical_or(terminated, truncated)
            finished_envs += done.sum().cpu().item()

        self.envs.close()
        self._print_evaluation_results(info)
        


def record_episode(num_step: int):
    # policy_name = "ppo"
    config_instance = ExperimentConfig()
    config = config_instance.get_config()
    experiment_name = config["experiment_name"]
    
    envs = make_vec(
        id=config["env_kwargs"]["id"],
        num_envs=1,
        training=False,
        record=True,
        video_folder=f"logs/{experiment_name}/videos/num_step_{num_step}",
    )

    env_details = get_env_vec_details(envs)
    device = get_device("auto")

    network_config = config["network"]
    network_kwargs = network_config.get("kwargs", {})
    network = RLModel(
        input_shape=env_details.state_dim,
        num_actions=env_details.action_dim,
        low=env_details.action_low,
        high=env_details.action_high,
        device=device,
        **network_kwargs
    )

    policy_config = config["policy"]
    policy_type = policy_config.get("type", "a2c")
    policy_kwargs = policy_config.get("kwargs", {})

    agent = get_policy(
        policy_type=policy_type,
        network=network,
        action_space_type=env_details.action_space_type,
        policy_kwargs=policy_kwargs,
        device=device
    )

    agent.load_weights("/app/rl_agents/logs/CarRacing-PPO-GRU/model.pt")
    agent.eval_mode()

    action_output = None
    total_reward = 0.
    step_count = 0
    done = False
    truncated = False
    terminated = False
    state, _ = envs.reset()
    core_state = None
    

    while not done:
        state = state.to(device)
        with T.no_grad():
            action_output = agent.action(state=state, core_state=core_state, temperature=0.1)

        action = action_output.action

        env_action = prepare_action_for_env(action, env_details.action_space_type)
        state, reward, terminated, truncated, _ = envs.step(env_action)
        core_state = action_output.core_state
        total_reward += reward
        done = terminated or truncated
        step_count += 1

    logger.info(
        f"Step {num_step}: {step_count} steps, reward = {float(total_reward):.2f}, truncated = {bool(truncated)}, terminated = {bool(terminated)}"
    )

    if state.ndim < 3:
        logger.info("Evaluation stopped in state %s", state.round(2))

    dist = getattr(action_output, "dist")
    if dist:
        try:
            if "covariance_matrix" in dir(dist.base_dist):  # type: ignore
                cov = dist.base_dist.covariance_matrix[0].cpu().numpy()  # type: ignore
            else:
                cov = dist.base_dist.stddev[0].cpu().numpy()  # type: ignore
            logger.info(f"Covariance matrix:\n{cov}")
        except AttributeError:
            pass

    envs.close()
