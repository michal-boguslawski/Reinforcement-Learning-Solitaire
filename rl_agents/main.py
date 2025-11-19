import os
import shutil
import sys

from config.config import Config
from envs.utils import get_env_details
from network.general import ActorCriticNetwork, MLPNetwork
from worker.worker import Worker


if __name__ == "__main__":
    env_name = sys.argv[1] if len(sys.argv) > 1 else "Pendulum-v1"
    policy_name = sys.argv[2] if len(sys.argv) > 2 else "a2c"
    config = Config(env_name=env_name).get_config()

    if os.path.exists(f"logs/{env_name}"):
        shutil.rmtree(f"logs/{env_name}")

    env_details = get_env_details(env_name=env_name)

    # ActorCriticNetwork
    network = ActorCriticNetwork(
        in_features=env_details.state_dim,
        out_features=env_details.action_dim,
        low=env_details.action_low,
        high=env_details.action_high,
        **config["network_kwargs"]
    )

    policy_kwargs = config[policy_name]["policy_config_kwargs"].copy()
    policy_kwargs["action_space"] = env_details.action_space
    policy_kwargs["num_actions"] = env_details.action_dim

    worker = Worker(
        env_name=env_name,
        network=network,
        policy_name=policy_name,
        policy_kwargs=policy_kwargs,
        **config[policy_name]["worker_kwargs"]
    )
    worker.train(**config[policy_name]["train_kwargs"])
