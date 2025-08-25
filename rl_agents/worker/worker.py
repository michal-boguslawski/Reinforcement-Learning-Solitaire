import torch as T
import torch.nn as nn
import numpy as np

from agent.agent import BasePolicy

class Worker:
    def __init__(
        self,
        env,
        agent: BasePolicy,
        epsilon_start_: float = 0.5,
        epsilon_decay_factor_: float = 0.9999,
        temperature_start_: float = 1,
    ):
        self.agent = agent
        self.env = env
        
        self.epsilon_start_ = epsilon_start_
        self.epsilon_decay_factor_ = epsilon_decay_factor_
        self.temperature_start_ = temperature_start_

    def train(
        self,
        episodes: int,
        batch_size: int,
        timesteps: int = 1,
        minibatch_size: int = 64,
        train_step: int = 16
    ):
        losses_list = []
        rewards_list = []
        i = 0
        epsilon = self.epsilon_start_
        
        for episode in range(int(episodes)):
            total_reward = 0
            done = False
            state, _ = self.env.reset()
            
            while not done:
                action, logits = self.agent.action(state, epsilon_=epsilon, method="egreedy")
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                done = terminated or truncated
                self.agent.update_buffer((state, logits, action, reward - 10 * int(terminated), done))
                
                if i % train_step == 0 and i > 0:
                    loss = self.agent.train(
                        batch_size=batch_size,
                        minibatch_size=minibatch_size
                    )
                    losses_list.append(loss)
                
                state = next_state
                
                total_reward += reward
                i += 1

            epsilon *= self.epsilon_decay_factor_
            
            rewards_list.append(total_reward)
            if episode % 10 == 0:
                print(f"Episode {episode}, i {i}, Reward {np.mean(rewards_list[-100:]):.4f},",
                    f"Max Reward {max(rewards_list[-100:])}, Loss {np.mean(losses_list[-100:]):.6f}, epsilon {epsilon:.4f}")
