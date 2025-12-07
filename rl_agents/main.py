import os
import shutil
import os
import sys

from config.config import Config
from envs.utils import get_env_details
from network.general import ActorCriticNetwork, MLPNetwork
from worker.worker import Worker


os.environ["MUJOCO_GL"] = "osmesa"


if __name__ == "__main__":
    experiment_name = sys.argv[1] if len(sys.argv) > 1 else "Pendulum-v1"
    # experiment_name = "MountainCarContinuous-v0"
    policy_name = sys.argv[2] if len(sys.argv) > 2 else "a2c"
    # policy_name = "ppo"
    config = Config(experiment_name=experiment_name).get_config()

    if os.path.exists(f"logs/{experiment_name}"):
        shutil.rmtree(f"logs/{experiment_name}")

    env_details = get_env_details(experiment_name=experiment_name)

    # ActorCriticNetwork
    network = ActorCriticNetwork(
        input_shape=env_details.state_dim,
        num_actions=env_details.action_dim,
        low=env_details.action_low,
        high=env_details.action_high,
        **config["network_kwargs"]
    )

    policy_kwargs = config[policy_name].copy()
    policy_kwargs["action_space"] = env_details.action_space
    policy_kwargs["num_actions"] = env_details.action_dim

    worker = Worker(
        experiment_name=experiment_name,
        network=network,
        policy_name=policy_name,
        policy_kwargs=policy_kwargs,
        **config["worker_kwargs"]
    )
    worker.train(**config["train_kwargs"])
