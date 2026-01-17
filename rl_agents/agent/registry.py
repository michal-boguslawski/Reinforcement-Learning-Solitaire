# Policies
from .on_policy import A2CPolicy, PPOPolicy, SarsaPolicy


POLICIES = {
    "a2c": A2CPolicy,
    "ppo": PPOPolicy,
    "sarsa": SarsaPolicy,
}
