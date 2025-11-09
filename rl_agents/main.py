import numpy as np
import os
import shutil
import torch as T
from torch.optim import Adam
import torch.nn as nn

from envs.env_setup import make_env
from agent.on_policy import A2CPolicy, SarsaPolicy
from network.general import ActorCriticNetwork, MLPNetwork
from worker.worker import Worker


if __name__ == "__main__":
    # T.autograd.set_detect_anomaly(True)
    config = {
        "env_name": "MountainCarContinuous-v0",
        "hidden_dim": 32,
        "buffer_size": 100000,
        "batch_size": 2048,
        "minibatch_size": 256,
        "gamma_": 0.99,
        "lambda_": 0.95,
        "tau_": 0.005,
        "entropy_beta_": 0.01,
        "epsilon_start_": 1,
        "epsilon_decay_factor_": 0.9995,
        "episodes": 10000,
        "distribution": "normal",
        "method": "entropy",
        "loss_fn": nn.HuberLoss(),
        "device": T.device("cuda" if T.cuda.is_available() else "cpu")
    }
    
    if os.path.exists(f"logs/{config['env_name']}"):
        shutil.rmtree(f"logs/{config['env_name']}")
    
    try:
        env = make_env(config["env_name"], device=config["device"])
        action_n = getattr(env.action_space, "n", None)
        action_shape = getattr(env.action_space, "shape", None)
        if action_n is not None:
            num_actions = action_n
            action_state_type = "discrete"
        elif action_shape is not None:
            num_actions = np.prod(action_shape)
            action_state_type = "continuous"
        else:
            num_actions = -1
            action_state_type = ""
        if num_actions == -1:
            raise ValueError("Invalid number of actions")
        obs_space_shape = getattr(env.observation_space, "shape")
        if obs_space_shape is None:
            raise ValueError("Invalid observation space shape")
        in_features = np.prod(obs_space_shape)
    except Exception as e:
        print(f"Error initializing environment: {e}")
        exit(1)
    
    
    low = getattr(env.action_space, "low", None)
    low = None if low is None else T.tensor(low, device=config["device"])
    high = getattr(env.action_space, "high", None)
    high = None if high is None else T.tensor(high, device=config["device"])

    # ActorCriticNetwork
    network = ActorCriticNetwork(
        in_features=in_features,
        out_features=int(num_actions),
        low = low,
        high = high,
        **config
    ).to(config["device"])
    
    # A2CPolicy
    agent = A2CPolicy(
        network=network,
        num_actions=int(num_actions),
        action_space=env.action_space,
        optimizer=Adam(network.parameters(), lr=3e-4),
        **config
    )
    
    worker = Worker(env=env, agent=agent, **config)
    worker.train(**config)
