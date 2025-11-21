import torch as T
from typing import Any


class EnvConfig:
    env_config = {
        "CartPole-v1":
            {
                "terminated_bonus": -100,
                "pow_factors": [-1, 0, -1000, 0],
                "abs_factors": [-0.1, 0, -1, 0],
                # "loc_reward": -0.9,
                # "scale_reward": 0.01,
            },
        "MountainCarContinuous-v0":
            {
                "truncated_bonus": -10,
                "pow_factors": [0, 100],
                "decay_factor": 0.999,
                "abs_factors": [0, 1],
            },
        "Acrobot-v1":
            {
                # "terminated_bonus": 100,
                "pow_factors": [0, 0, 0, 0, 0.01, 0.01],
            },
        "Pendulum-v1":
            {
                "scale_reward": 2/16.2736044,
                "loc_reward": -1,
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
    }

    def __init__(self, env_name: str):
        self.env_name = env_name
    
    def get_config(self) -> dict[str, Any]:
        return self.env_config.get(self.env_name, {})


class Config:
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    config = {
        "CartPole-v1":
            {
                "a2c": {
                    "policy_config_kwargs": {
                        "gamma_": 0.95,
                        "lambda_": 0.9,
                        "entropy_beta": 0.01,
                        "device": device,
                        "lr": 3e-4,
                    },
                    "worker_kwargs": {
                        "num_envs": 8,
                        "action_exploration_method": "distribution",
                        "device": device
                    },
                    "train_kwargs": {
                        "num_steps": int(5e5),
                        "batch_size": 32,
                        "minibatch_size": 64,
                    }
                },
                "network_kwargs": {
                    "hidden_dim": 32,
                    "distribution": "categorical",
                    "device": device
                },
            },
        "Acrobot-v1":
            {
                "a2c": {
                    "policy_config_kwargs": {
                        "gamma_": 0.99,
                        "lambda_": 0.95,
                        "entropy_beta": 0.01,
                        "device": device,
                        "lr": 5e-4,
                    },
                    "worker_kwargs": {
                        "num_envs": 8,
                        "action_exploration_method": "distribution",
                        "device": device
                    },
                    "train_kwargs": {
                        "num_steps": int(5e5),
                        "batch_size": 32,
                        "minibatch_size": 256,
                    }
                },
                "network_kwargs": {
                    "hidden_dim": 32,
                    "distribution": "categorical",
                    "device": device
                },
            },
        "MountainCarContinuous-v0":
            {
                "a2c": {
                    "policy_config_kwargs": {
                        "gamma_": 0.99,
                        "lambda_": 0.95,
                        "entropy_beta": 0.01,
                        "device": device,
                        "lr": 1e-4,
                    },
                    "worker_kwargs": {
                        "num_envs": 32,
                        "action_exploration_method": "distribution",
                        "device": device
                    },
                    "train_kwargs": {
                        "num_steps": int(5e5),
                        "batch_size": 64,
                        "minibatch_size": 256,
                    }
                },
                "network_kwargs": {
                    "hidden_dim": 32,
                    "distribution": "normal",
                    "device": device,
                    "initial_log_std": 0.,
                },
                "ppo": {
                    "policy_config_kwargs": {
                        "gamma_": 0.99,
                        "lambda_": 0.95,
                        "entropy_beta": 0.01,
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
                    }
                },
                "network_kwargs": {
                    "hidden_dim": 32,
                    "distribution": "normal",
                    "device": device,
                    "initial_log_std": 0.,
                },
            },
        "Pendulum-v1":
            {
                "a2c": {
                    "policy_config_kwargs": {
                        "gamma_": 0.99,
                        "lambda_": 0.95,
                        "entropy_beta": 0.01,
                        "device": device,
                        "lr": 3e-4,
                    },
                    "worker_kwargs": {
                        "num_envs": 16,
                        "action_exploration_method": "distribution",
                        "device": device
                    },
                    "train_kwargs": {
                        "num_steps": int(1e6),
                        "batch_size": 64,
                        "minibatch_size": 256,
                    }
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
                    "policy_config_kwargs": {
                        "gamma_": 0.99,
                        "lambda_": 0.95,
                        "entropy_beta": 0.01,
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
                    }
                },
                "network_kwargs": {
                    "hidden_dim": 32,
                    "distribution": "normal",
                    "device": device,
                    "initial_log_std": 0.,
                },
            },
        "HalfCheetah-v5":
            {
                "ppo": {
                    "policy_config_kwargs": {
                        "gamma_": 0.99,
                        "lambda_": 0.95,
                        "entropy_beta": 0.01,
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
                    }
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
                    "policy_config_kwargs": {
                        "gamma_": 0.99,
                        "lambda_": 0.95,
                        "entropy_beta": 0.05,
                        "device": device,
                        "lr": 3e-4,
                        "num_epochs": 10,
                        "clip_epsilon": 0.2,
                    },
                    "worker_kwargs": {
                        "num_envs": 64,
                        "action_exploration_method": "distribution",
                        "device": device
                    },
                    "train_kwargs": {
                        "num_steps": int(1e6),
                        "batch_size": 2048,
                        "minibatch_size": 256,
                    }
                },
                "env": {
                    "id": "BipedalWalker-v3",
                    "hardcore": True,
                    "vectorization_mode": "sync",
                },
                "network_kwargs": {
                    "hidden_dim": 64,
                    "distribution": "normal",
                    "device": device,
                    "initial_log_std": 0.,
                },
            },
    }

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
    
    def get_config(self) -> dict[str, Any]:
        return self.config.get(self.experiment_name, {})
