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
    
    def get_config(self, env_name: str) -> dict[str, Any]:
        return self.env_config.get(env_name, {})


