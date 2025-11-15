import torch as T
from typing import Any


class EnvConfig:
    env_config = {
        "CartPole-v1":
            {
                "terminated_bonus": -10,
                "pow_factors": [0, 0, -100, 0]
            },
        "MountainCarContinuous-v0":
            {
                "truncated_bonus": -10,
                "pow_factors": [0, 100],
                "decay_factor": 0.999,
                "abs_factors": [0, 5],
            },
        "Acrobot-v1":
            {
                "pow_factors": [0, 0, 0, 0, 0.01, 0.01]
            }
    }

    def __init__(self, env_name: str):
        self.env_name = env_name
    
    def get_config(self) -> dict[str, Any]:
        return self.env_config.get(self.env_name, {})


class Config:
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    policy_config = {
        "a2c": {
            "gamma_": 0.99,
            "lambda_": 0.95,
            "entropy_beta": 0.001,
            "device": device,
            "lr": 3e-4,
            "worker_kwargs": {
                "action_exploration_method": "distribution",
                "batch_size": 2048,
                "minibatch_size": 256,
                "device": device
            }
        },
    }
    config = {
        "CartPole-v1":
            {
                "network_kwargs": {
                    "hidden_dim": 32,
                    "distribution": "categorical",
                    "device": device
                }
            }
    }

    def __init__(self, env_name: str):
        self.env_name = env_name
    
    def get_config(self) -> dict[str, Any]:
        return self.config.get(self.env_name, {})
