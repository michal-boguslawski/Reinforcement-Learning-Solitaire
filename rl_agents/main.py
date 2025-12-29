import os
import shutil
import os
import sys

from config.config import Config
from network.general import ActorCriticNetwork, MLPNetwork
from worker.worker import Worker


os.environ["MUJOCO_GL"] = "osmesa"


if __name__ == "__main__":
    # policy_name = "ppo"
    config = Config().get_config
    experiment_name = config["experiment_name"]

    if os.path.exists(f"logs/{experiment_name}"):
        shutil.rmtree(f"logs/{experiment_name}")

    # env_details = get_env_details(experiment_name=experiment_name)

    # # ActorCriticNetwork
    # network = ActorCriticNetwork(
    #     input_shape=env_details.state_dim,
    #     num_actions=env_details.action_dim,
    #     low=env_details.action_low,
    #     high=env_details.action_high,
    #     **config["network_kwargs"]
    # )

    policy_kwargs = config.copy()
    # policy_kwargs["action_space"] = env_details.action_space
    # policy_kwargs["num_actions"] = env_details.action_dim

    worker = Worker(
        experiment_name=experiment_name,
        env_config=config["env_kwargs"],
        policy_config=config["policy"],
        network_config=config.get("network", {}),
        **config["worker_kwargs"]
    )
    worker.train(**config["train_kwargs"])
