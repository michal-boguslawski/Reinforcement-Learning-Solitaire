from typing import Any


class EnvConfig:
    env_config = {
        "CartPole-v1":
            {
                "terminated_reward": -10,
                "pow_factors": [0, 0, 100, 0]
            },
        "MountainCarContinuous-v0":
            {
                "terminated_reward": -100,
            }
    }
    
    def get_config(self, env_name: str) -> dict[str, Any]:
        return self.env_config[env_name]


