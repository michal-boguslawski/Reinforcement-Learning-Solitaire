from collections import deque
import numpy as np
from random import choices

class ReplayBuffer:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self._sample_method_choice = {
            1: self._sample_one_timestep
        }
        
    def __len__(self):
        return len(self.buffer)
        
    def push(self, item: tuple | list):
        self.buffer.append(item)
        
    def clear(self):
        self.buffer.clear()

    def _sample_one_timestep(self, batch_size: int, **kwargs) -> tuple[np.ndarray, ...]:
        
        sample_list = choices(self.buffer, k=min(batch_size, len(self.buffer)))
        sample = tuple(np.stack(column, axis = 0) for column in zip(*sample_list))
        return sample
    
    def _sample_multiple_timesteps(self, batch_size: int, timesteps: int, **kwargs) -> tuple[np.ndarray, ...]:
        pass
    
    def sample(self, batch_size: int, timesteps: int = 1, **kwargs) -> tuple[np.ndarray, ...]:
        sample_method = self._sample_method_choice.get(timesteps, self._sample_multiple_timesteps)
        sample = sample_method(batch_size=batch_size, timesteps=timesteps)
        return sample
    
    def get_all(self):
        sample = tuple(np.stack(column, axis=0) for column in zip(*self.buffer))
        return sample
