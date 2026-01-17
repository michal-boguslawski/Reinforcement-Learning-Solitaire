from gymnasium.vector import VectorEnv
import logging
import numpy as np
import torch as T

from agent.base import BasePolicy
from agent.factories import get_policy
from envs.factories import make_vec
from envs.utils import get_env_vec_details
from config.config import ExperimentConfig
from models.models import ActionOutput
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
        self._closed = False

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

    def _print_evaluation_results(self, rewards: list[float], action_output: ActionOutput | None = None):
        rewards_array = np.array(rewards)
        logger.info(
            f"Evaluation results: mean = {rewards_array.mean():.2f}, "
            f"std = {rewards_array.std():.2f}, "
            f"min = {rewards_array.min():.2f}, "
            f"max = {rewards_array.max():.2f}, "
            f"count = {len(rewards)}"
        )

        if action_output is not None:
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

    def evaluate(self, agent: BasePolicy, min_episodes: int = 1, action_space_type: str = "discrete"):
        device = agent.action_network.device
        state, info = self.envs.reset()
        finished_envs = 0
        rewards = []
        action_output = None
        
        while finished_envs < min_episodes:
            state = state.to(device)
            with T.no_grad():
                action_output = agent.action(state=state, training=False)

            action = action_output.action
            env_action = prepare_action_for_env(action, action_space_type)
            state, _, terminated, truncated, info = self.envs.step(env_action)
            done = T.logical_or(terminated, truncated)
            tmp_finished_envs = done.sum().cpu().item()
            finished_envs += tmp_finished_envs

            if tmp_finished_envs:
                tmp_rewards = info.get("episode", {}).get("r")
                rewards.extend(tmp_rewards[done].cpu().numpy().tolist())

        self._print_evaluation_results(rewards, action_output)

    def close(self):
        if not self._closed and hasattr(self, "envs"):
            self.envs.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
