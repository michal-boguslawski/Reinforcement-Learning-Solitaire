import os
from pathlib import Path
import torch as T
from typing import Any
import yaml

from .models import ExperimentConfigModel


class EnvConfig:
    env_config = {
        "CartPole-v1":
            {
                "wrappers": {
                    "terminal_bonus": {
                        "terminated_bonus": -100
                    },
                    "power_obs_reward": {
                        "pow_factors": [-0.01, 0, -10, 0],
                        "abs_factors": [-1, 0, -10, 0],
                    },
                }
            },
        "MountainCarContinuous-v0":
            {
                "wrappers":
                    {
                        "power_obs_reward": {
                            "pow_factors": [0, 50],
                            "abs_factors": [0, 0.5],
                            "decay_factor": 0.5,
                        },
                        "scale_reward": {
                            "scale_factor": 0.1,
                            "loc_factor": 0.
                        },
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
        "LunarLander-v3":
            {
                
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
        "CarRacing-v3":
            {
                "wrappers": {
                    "actions_interactions":
                        {
                            "factors":
                                {(0, 1): -1., (1, 2): -1.}
                        }
                    # "action_reward": {
                    #         "abs_factors": [-0.1, 0.01, -0.01],
                    #         "decay_factor": 0.99,
                    #     },
                    # "terminal_bonus": {
                    #     "truncated_bonus": -100
                    # },
                }
            }
    }

    def __init__(self, env_name: str):
        self.env_name = env_name
    
    def get_config(self) -> dict[str, Any]:
        return self.env_config.get(self.env_name, {})


class ExperimentConfig:
    def __init__(self, config_path: str = os.path.join(Path(__file__).parent.absolute(), "config.yaml")):
        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f)
        self.config = ExperimentConfigModel.model_validate(data)

    def get_config(self) -> dict[str, Any]:
        return self.config.model_dump()

    def save_config(self, write_path: str):
        with open(write_path, "w") as f:
            yaml.dump(self.config.model_dump(), f)
