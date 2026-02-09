# Policies
from .on_policy.sarsa import SarsaPolicy
from .on_policy.a2c import A2CPolicy
from .on_policy.ppo import PPOPolicy

#Schedulers
from .schedulers.entropy import LinearSchedule


POLICIES = {
    "a2c": A2CPolicy,
    "ppo": PPOPolicy,
    "sarsa": SarsaPolicy,
}


SCHEDULERS = {
    "linear_entropy": LinearSchedule
}
