import numpy as np
import os
import shutil
from torch.optim import Adam
import torch.nn as nn

from envs.env_setup import make_env
from agent.on_policy import A2CPolicy
from network.general import ActorCriticNetwork
from worker.worker import Worker


if __name__ == "__main__":
    config = {
        "env_name": "MountainCarContinuous-v0",
        "hidden_dim": 32,
        "out_activation": "tanh",
        "buffer_size": 100000,
        "batch_size": 256,
        "minibatch_size": 64,
        "gamma_": 0.99,
        "lambda_": 0.95,
        "tau_": 0.005,
        "epsilon_start_": 1,
        "epsilon_decay_factor_": 0.9995,
        "episodes": 5000,
        "loss_fn": nn.HuberLoss()
    }
    
    if os.path.exists(f"logs/{config['env_name']}"):
        shutil.rmtree(f"logs/{config['env_name']}")
    
    try:
        env = make_env(config["env_name"])
        action_n = getattr(env.action_space, "n", None)
        action_shape = getattr(env.action_space, "shape", None)
        if action_n is not None:
            num_actions = action_n
        elif action_shape is not None:
            num_actions = np.prod(action_shape)
        else:
            num_actions = -1
        if num_actions == -1:
            raise ValueError("Invalid number of actions")
        obs_space_shape = getattr(env.observation_space, "shape")
        if obs_space_shape is None:
            raise ValueError("Invalid observation space shape")
        in_features = np.prod(obs_space_shape)
    except Exception as e:
        print(f"Error initializing environment: {e}")
        exit(1)
    
    network = ActorCriticNetwork(
        in_features=in_features,
        out_features=int(num_actions),
        **config
    )
    
    agent = A2CPolicy(
        network=network,
        num_actions=int(num_actions),
        optimizer=Adam(network.parameters(), lr=2e-4),
        **config
    )
    
    worker = Worker(env=env, agent=agent, **config)
    worker.train(**config)
