import gymnasium as gym
from gymnasium.wrappers import NumpyToTorch
from torch.optim import Adam


from worker.worker import Worker
from agent.agent import Sarsa
from network.cart_pole import DQNNetwork
from memory.replay_buffer import ReplayBuffer


if __name__ == "__main__":
    batch_size = 256
    minibatch_size = 64
    
    env = gym.make("CartPole-v1")
    env = NumpyToTorch(env)
    network = DQNNetwork(
        in_features=4,
        out_features=2,
        hidden_dim=32
    )
    optimizer = Adam(network.parameters(), lr=2e-4)
    buffer = ReplayBuffer(batch_size)
    agent = Sarsa(
        network=network,
        buffer=buffer,
        num_actions=2,
        optimizer=optimizer,
        lambda_=0.95
    )
    
    worker = Worker(
        env=env,
        agent=agent,
        epsilon_start_=0.99
    )
    
    worker.train(
        episodes=1e5,
        batch_size=batch_size,
        minibatch_size=minibatch_size,
        train_step=batch_size
    )
