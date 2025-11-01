import numpy as np
import os
import shutil
from torch.optim import Adam


from envs.env_setup import make_env
from agent.off_policy import DQNetworkPolicy
from agent.on_policy import SarsaPolicy, A2CPolicy
from network.general import MLPNetwork, ActorCriticNetwork
from worker.worker import Worker
import torch.nn as nn


if __name__ == "__main__":
    config = {
        "env_name": "CartPole-v1",
        "hidden_dim": 32,
        "buffer_size": 100000,
        "batch_size": 256,
        "minibatch_size": 64,
        "gamma_": 0.99,
        "lambda_": 0.9,
        "tau_": 0.005,
        "epsilon_start_": 1,
        "epsilon_decay_factor_": 0.9995,
        "episodes": 5000,
        "loss_fn": nn.HuberLoss()
    }
    if os.path.exists("logs/" + config["env_name"]):
        shutil.rmtree("logs/" + config["env_name"])
    try:
        env = make_env(config["env_name"])
        num_actions = np.prod(getattr(env.action_space, "n"))
        in_features = np.prod(getattr(env.observation_space, "shape"))
    except Exception as e:
        print(f"Error initializing environment: {e}")
        exit(1)
    
    network = ActorCriticNetwork(
        in_features=in_features,
        out_features=num_actions,
        hidden_dim=config["hidden_dim"]
    )
    # network = MLPNetwork(
    #     in_features=in_features,
    #     out_features=num_actions,
    #     hidden_dim=config["hidden_dim"]
    # )
    optimizer = Adam(network.parameters(), lr=2e-4)
    agent = A2CPolicy(
        network=network,
        num_actions=num_actions,
        optimizer=optimizer,
        **config
    )
    
    # agent = SarsaPolicy(
    #     network=network,
    #     num_actions=num_actions,
    #     optimizer=optimizer,
    #     **config
    # )
    
    worker = Worker(
        env=env,
        agent=agent,
        **config
        # epsilon_start_=epsilon_start_,
        # epsilon_decay_factor_=epsilon_decay_factor_
    )
    
    worker.train(
        **config
        # episodes=episodes,
        # batch_size=batch_size,
        # minibatch_size=minibatch_size,
    )
