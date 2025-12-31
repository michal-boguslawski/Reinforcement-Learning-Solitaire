import os
import shutil
import os

from config.config import ExperimentConfig
from worker.worker import Worker


os.environ["MUJOCO_GL"] = "osmesa"


if __name__ == "__main__":
    # policy_name = "ppo"
    config_instance = ExperimentConfig()
    config = config_instance.get_config()
    experiment_name = config["experiment_name"]

    logs_path = f"logs/{experiment_name}"
    if os.path.exists(logs_path):
        shutil.rmtree(logs_path)
    os.makedirs(logs_path, exist_ok=True)
    
    config_instance.save_config(os.path.join(logs_path, "config.yaml"))

    worker = Worker(
        experiment_name=experiment_name,
        env_config=config["env_kwargs"],
        policy_config=config["policy"],
        network_config=config.get("network", {}),
        **config["worker_kwargs"]
    )
    worker.train(**config["train_kwargs"])
