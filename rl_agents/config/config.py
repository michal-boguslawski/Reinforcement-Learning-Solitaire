import torch as T
from typing import Any


class EnvConfig:
    env_config = {
        "CartPole-v1":
            {
                # "terminated_bonus": -100,
                # "pow_factors": [-1, 0, -1000, 0],
                # "abs_factors": [-0.1, 0, -1, 0],
                # "loc_reward": -0.9,
                # "scale_reward": 0.01,
            },
        "MountainCarContinuous-v0":
            {
                "wrappers":
                    {
                        "terminal_bonus": {
                            "truncated_bonus": -10
                        },
                        "power_obs_reward": {
                            "pow_factors": [0, 50],
                            "abs_factors": [0, 0.5],
                            "decay_factor": 0.999,
                        }
                    }
            },
        "Acrobot-v1":
            {
                # "terminated_bonus": 100,
                # "pow_factors": [0, 0, 0, 0, 0.01, 0.01],
            },
        "Pendulum-v1":
            {
                # "scale_reward": 2/16.2736044,
                # "loc_reward": -1,
            },
        "BipedalWalker-v3":
            {
                "wrappers":
                    {
                        "scale_reward":
                            {
                                "scale_factor": 1,
                                "loc_factor": -0.1
                            }
                    }                
            },
        "HalfCheetah-v5":
            {
                "wrappers":
                    {
                        "scale_reward":
                            {
                                "scale_factor": 1,
                                "loc_factor": 0
                            }
                    }                
            },
        "RacingCar-v3":
            {
                
            }
    }

    def __init__(self, env_name: str):
        self.env_name = env_name
    
    def get_config(self) -> dict[str, Any]:
        return self.env_config.get(self.env_name, {})


class Config:
    config = {
        "experiment_name": "MountainCarContinuous-PPO",
        "env_kwargs": {
            "id": "MountainCarContinuous-v0",
            "vectorization_mode": "async",
            "num_envs": 8,
        },
        "policy": {
            "type": "ppo",
            "kwargs": {
                "gamma_": 0.99,
                "lambda_": 0.95,
                "entropy_beta_": 0.02,
                "entropy_decay": 0.95,
                "lr": 3e-4,
                "num_epochs": 10,
                "clip_epsilon": 0.2,
            },
        },
        "worker_kwargs": {
            "action_exploration_method": "distribution",
            "device": T.device("cuda" if T.cuda.is_available() else "cpu"),
        },
        "train_kwargs": {
            "num_steps": int(5e5),
            "batch_size": 2048,
            "minibatch_size": 256,
        },
        "network": {
            "type": "ac_network",
            "kwargs": {
                "distribution": "normal",
                "initial_log_std": 0.,
            },
        }
    }

    @property
    def get_config(self) -> dict[str, Any]:
        return self.config
