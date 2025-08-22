from models.network import SolitaireNetwork1
from envs.model_game_wrapper import GymLikeGameWrapper
from agents.dqn import DQN
from utils.replay_buffer import ReplayBuffer
import torch as T

if __name__ == "__main__":
    model = SolitaireNetwork1(104, 100)
    env = GymLikeGameWrapper(verbose=False, max_iter=1000, move_penalty=0.5, truncation_penalty=0.)
    replay_buffer = ReplayBuffer(10**5)
    agent = DQN(
        network=SolitaireNetwork1,
        in_features=104,
        n_classes=100
    )
    agent.train(env=env, replay_buffer=replay_buffer, episodes=10000, batch_size=16)
