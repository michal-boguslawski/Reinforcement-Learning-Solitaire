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
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    config = {
        "MountainCarContinuous-v0":
            {
                "env": {
                    "id": "MountainCarContinuous-v0",
                    "vectorization_mode": "async",
                },
                "ppo": {
                    "gamma_": 0.99,
                    "lambda_": 0.95,
                    "entropy_beta_": 0.02,
                    "entropy_decay": 0.95,
                    "device": device,
                    "lr": 3e-4,
                    "num_epochs": 10,
                    "clip_epsilon": 0.2,
                },
                "worker_kwargs": {
                    "num_envs": 8,
                    "action_exploration_method": "distribution",
                    "device": device
                },
                "train_kwargs": {
                    "num_steps": int(5e5),
                    "batch_size": 2048,
                    "minibatch_size": 256,
                },
                "network_kwargs": {
                    "hidden_dim": 32,
                    "distribution": "normal",
                    "device": device,
                    "initial_log_std": 0.,
                },
            },
        "BipedalWalker-v3":
            {
                "ppo": {
                    "gamma_": 0.99,
                    "lambda_": 0.95,
                    "entropy_beta_": 0.05,
                    "entropy_decay": 0.95,
                    "device": device,
                    "lr": 3e-4,
                    "num_epochs": 10,
                    "clip_epsilon": 0.2,
                },
                "worker_kwargs": {
                    "num_envs": 32,
                    "action_exploration_method": "distribution",
                    "device": device
                },
                "train_kwargs": {
                    "num_steps": int(5e5),
                    "batch_size": 2048,
                    "minibatch_size": 256,
                },
                "env": {
                    "id": "BipedalWalker-v3",
                    "vectorization_mode": "async",
                },
                "network_kwargs": {
                    "hidden_dim": 64,
                    "distribution": "multivariatenormal",
                    "device": device,
                    "initial_log_std": 0.,
                },
            },
        "HalfCheetah-v5":
            {
                "ppo": {
                    "gamma_": 0.99,
                    "lambda_": 0.95,
                    "entropy_beta_": 0.01,
                    "entropy_decay": 0.95,
                    "device": device,
                    "lr": 3e-4,
                    "num_epochs": 10,
                    "clip_epsilon": 0.2,
                },
                "worker_kwargs": {
                    "num_envs": 32,
                    "action_exploration_method": "distribution",
                    "device": device,
                    # "env_experiment_name": "HalfCheetah-v5",
                },
                "train_kwargs": {
                    "num_steps": int(1e6),
                    "batch_size": 2048,
                    "minibatch_size": 256,
                },
                "env": {
                    "id": "HalfCheetah-v5",
                    "vectorization_mode": "sync",
                },
                "network_kwargs": {
                    "hidden_dim": 32,
                    "distribution": "normal",
                    "device": device,
                    "initial_log_std": 0.,
                },
            },
        "BipedalWalker-v3-hardcore":
            {
                "ppo": {
                    "gamma_": 0.99,
                    "lambda_": 0.95,
                    "critic_coef_": 0.5,
                    "entropy_beta_": 0.01,
                    "entropy_decay": 0.95,
                    "device": device,
                    "lr": 3e-4,
                    "num_epochs": 10,
                    "clip_epsilon": 0.2,
                },
                "worker_kwargs": {
                    "num_envs": 8,
                    "action_exploration_method": "distribution",
                    "device": device
                },
                "train_kwargs": {
                    "num_steps": int(1e7),
                    "batch_size": 2048,
                    "minibatch_size": 256,
                },
                "env": {
                    "id": "BipedalWalker-v3",
                    "hardcore": True,
                    "vectorization_mode": "async",
                },
                "network_kwargs": {
                    "hidden_dim": 64,
                    "distribution": "multivariatenormal",
                    "device": device,
                    "initial_log_std": 0.,
                },
            },
        "CarRacing-v3":
            {
                "ppo": {
                    "gamma_": 0.99,
                    "lambda_": 0.95,
                    "critic_coef_": 0.5,
                    "entropy_beta_": 0.01,
                    "num_epochs": 10,
                    "clip_epsilon": 0.2,
                    "device": device,
                    "lr": 3e-4,
                },

                "worker_kwargs": {
                    "num_envs": 1,
                    "action_exploration_method": "distribution",
                    "device": device
                },
                
                "train_kwargs": {
                    "num_steps": int(1e6),
                    "batch_size": 256,
                    "minibatch_size": 64,
                },
                "env": {
                    "id": "CarRacing-v3",
                    "vectorization_mode": "async",
                    "domain_randomize": True,
                    "continuous": True,
                },
                "network_kwargs": {
                    "channels": 256,
                    "backbone_type": "simple_cv",
                    "distribution": "multivariatenormal",
                    "device": device,
                    "initial_log_std": 0.,
                },
            },
    }

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
    
    def get_config(self) -> dict[str, Any]:
        return self.config.get(self.experiment_name, {})
