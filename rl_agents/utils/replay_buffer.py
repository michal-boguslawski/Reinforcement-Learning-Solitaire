from collections import deque
from random import choices
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size: int, dtypes_schema: list | None = None):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.dtypes_schema = dtypes_schema
        
    def __len__(self):
        return len(self.buffer)
        
    def clear(self):
        self.buffer.clear()
        
    def push(self, item):
        self.buffer.append(item)
        
    @staticmethod
    def __stack(data: tuple) -> np.ndarray:
        try:
            return np.stack(data, axis=0)
        except ValueError:
            return [np.stack(column) for column in zip(*data)]
        except Exception as e:
            print(e)
            
    def _sample_simple(self, size: int):
        buffer_sample = choices(list(self.buffer), k=size)
        output = [self.__stack(column) for column in zip(*buffer_sample)]
        return output
    
    def _sample_window(self, size: int, window_size: int):
        pass
        
    def sample(self, size: int, sample_type: str = "simple", window_size: int = 5) -> tuple:
        if sample_type == "simple":
            return self._sample_simple(size)
        elif sample_type == "window":
            return self._sample_window(size, window_size)
        return ()