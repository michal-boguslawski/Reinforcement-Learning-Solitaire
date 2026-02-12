# Policies
from .on_policy.sarsa import SarsaPolicy
from .on_policy.a2c import A2CPolicy
from .on_policy.ppo import PPOPolicy

#Schedulers
from .schedulers.entropy import LinearSchedule
from torch.optim.lr_scheduler import LinearLR


POLICIES = {
    "a2c": A2CPolicy,
    "ppo": PPOPolicy,
    "sarsa": SarsaPolicy,
}


SCHEDULERS = {
    "linear_entropy": LinearSchedule,
    "linear_lr": LinearLR,
}
