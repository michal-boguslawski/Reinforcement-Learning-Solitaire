from pathlib import Path
import sys
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim

from .agent import Agent

parent_path = Path(__file__).parent.parent
sys.path.append(str(parent_path))

from utils.replay_buffer import ReplayBuffer


class DQN(Agent):
    def __init__(
        self,
        network,
        in_features: int,
        n_classes: int,
        policy: str = "egreedy",
        train_freq: int = 16,
        epsilon_: float = 0.5,
        tau_: float = 0.05,
        gamma_: float = 0.99,
        lambda_: float = 0.95,
        learning_rate: float = 1e-3,
        temperature: float = 2.,
        temperature_decay: float = 0.999,
        optimizer: optim.Optimizer = optim.AdamW,
        loss_fn: nn.modules.loss._Loss = nn.HuberLoss, 
        device: T.device = None
    ):
        self.in_features = in_features
        self.n_classes = n_classes
        
        self.policy_net = network(in_features=in_features, n_classes=n_classes)
        self.target_net = network(in_features=in_features, n_classes=n_classes)
        
        self.policy = policy
        self.train_freq = train_freq
        
        self.epsilon_ = epsilon_
        self.tau_ = tau_
        self.gamma_ = gamma_
        self.lambda_ = lambda_
        self.discounting_triangle = None
        
        self.learning_rate = learning_rate
        self.optimizer = optimizer(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = loss_fn()
        
        self.device = device
        self.replay_buffer = None
        
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        
        self.update_target_net(if_full_update=True)
    
    @T.no_grad()
    def update_target_net(self, if_full_update: bool = False):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        used_tau = (1 - if_full_update) * self.tau_ + if_full_update
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * used_tau + target_net_state_dict[key] * (1 - used_tau)
        self.target_net.load_state_dict(target_net_state_dict)
        
    def _set_discounting_triangle(self, window_size: int, discount: float):
        number_range = T.arange(0, window_size)
        square = number_range.unsqueeze(0) - number_range.unsqueeze(1)
        self.discounting_triangle = (discount ** square).triu()
            
    def action(self, state: np.ndarray | T.Tensor, inference: bool = False, temperature: float = 1, exploration_strategy: str = "softmax"):
        if not isinstance(state, T.Tensor):
            state = [T.tensor(state_x, dtype=T.int32) for state_x in state]
        with T.no_grad():
            _, probs = self.policy_net(state, temperature)
        if inference or exploration_strategy == "softmax":
                action = T.multinomial(probs, num_samples=1)
                action = action.squeeze(0)
        elif exploration_strategy == "egreedy":
            random_number = np.random.random()
            if random_number > self.epsilon_:
                action = probs.argmax(dim=-1)
            else:
                action = np.random.randint(0, self.n_classes, size=(1,))
        return action
    
    def move_step(self, state, env):
        action = self.action(state, temperature=self.temperature)
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        item = {
            "state": state,
            "action": action,
            "reward": reward,
            "done": done
        }
        self.replay_buffer.push(item)
        return next_state, done, reward
    
    def _calculate_loss(self, sample: dict):
        states = sample["state"]
        next_states = sample["next_state"]
        actions = sample["action"]
        rewards = sample["reward"]
        dones = sample["done"]
        
        shapes = actions.shape[:2]
        
        states = [x.reshape(-1, *x.shape[2:]) for x in states]
        q_values, _ = self.policy_net(states)
        q_values = q_values.reshape(*shapes, -1)
        q_action_values = q_values.gather(-1, actions)
        
        next_states = [x.reshape(-1, *x.shape[2:]) for x in next_states]
        with T.no_grad():
            next_state_logits, _ = self.target_net(next_states)
        next_state_logits = next_state_logits.reshape(*shapes, -1)
        next_state_values = next_state_logits.max(dim=-1, keepdim=True)[0]
        
        q_target = rewards + (1 - dones) * self.gamma_ * next_state_values
        # diff = (q_action_values - q_target).squeeze(-1) @ self.discounting_triangle
        # loss = self.loss_fn(diff, T.zeros_like(diff))
        loss = self.loss_fn(q_action_values, q_target)
        return loss
    
    def train_step(self, batch_size: int, window_size: int):
        sample = self.replay_buffer.sample(size=batch_size, window_size=window_size)
        self.optimizer.zero_grad()
        loss = self._calculate_loss(sample)
        loss.backward()
        T.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()
        self.update_target_net()
        return loss.item()
    
    def train(self, env, replay_buffer: ReplayBuffer, episodes: int, batch_size: int, window_size: int = 5):
        i = 0
        loss_list = []
        self.replay_buffer = replay_buffer
        self._set_discounting_triangle(window_size=window_size, discount=self.gamma_ * self.lambda_)
        
        for episode in range(episodes):
            total_reward = 0
            state, reward, terminated, truncated, _ = env.reset()
            done = terminated or truncated
            while not done:
                state, done, reward = self.move_step(state=state, env=env)
                total_reward += reward
                if i % self.train_freq == 0 and len(replay_buffer) > batch_size + window_size + 1:
                    loss = self.train_step(batch_size=batch_size, window_size=window_size)
                    loss_list.append(loss)
                    
            self.temperature *= self.temperature_decay
            self.temperature = max(self.temperature, 0.5)
            
            with T.no_grad():
                temp = self.policy_net(
                    [T.tensor(state_x, dtype=T.int32) for state_x in state]
                )[0].topk(k=3, dim=-1)
            i += 1
            print(env)
            print(f"Max value: {temp.values.numpy(force=True)[0]}, pile from {temp.indices.numpy(force=True)[0] // 10}, pile to {temp.indices.numpy(force=True)[0] % 10}")
            print(f"Episode: {episode} with loss mean {np.mean(loss_list[-100:]):.6f} with reward {total_reward.item():.2f}")
        