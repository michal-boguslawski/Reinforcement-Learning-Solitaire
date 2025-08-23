from collections import deque
from random import sample
import numpy as np
import torch as T


class ReplayBuffer:
    def __init__(self, buffer_size: int, dtypes_schema: dict | None = None):
        self.buffer_size = buffer_size
        self.buffers = {key: [] for key in ["state", "action", "reward", "done"]}
        self.dtypes_schema = dtypes_schema
        self.len_ = 0
        self.stack_ = np.stack
        
    def __len__(self):
        return self.len_
        
    def clear(self):
        self.buffer.clear()
        
    def push(self, item):
        self.len_ += 1
        for key, value in item.items():
            key_list = self.buffers[key]
            key_list.append(value)
            if len(key_list) > self.buffer_size + (key == "state"):
                key_list.pop(0)
        self.len_ = min(self.len_, self.buffer_size)
        
    def __stack(self, list_: list):
        try:
            return self.stack_(list_)
        except ValueError:
            return [self.stack_(column) for column in zip(*list_)]
        except Exception as e:
            raise e
    
    def _sample_window(self, size: int, window_size: int):
        indices = sample(range(self.__len__() - window_size - 1), size)
        buffer_sample = {}
        for key, value in self.buffers.items():
            key_sample = []
            for i in indices:
                key_sample.append(self.__stack(value[i:(i + window_size + (key == "state"))]))
            buffer_sample[key] = self.__stack(key_sample)
        return buffer_sample
    
    def _cast_types(self, dict_: dict):
        if self.dtypes_schema:
            for key, dtype_ in self.dtypes_schema.items():
                value = dict_[key]
                if isinstance(value, tuple) or isinstance(value, list):
                    dict_[key] = [T.tensor(x, dtype=dtype_) for x in value]
                else:
                    dict_[key] = T.tensor(value, dtype=dtype_)
        return dict_
    
    def _split_states(self, dict_: dict):
        state_value = dict_["state"]
        dict_["next_state"] = [x[:, 1:, ...] for x in state_value]
        dict_["state"] = [x[:, :-1, ...] for x in state_value]
        return dict_
        
    def sample(self, size: int, window_size: int = 5) -> tuple:
        sample = self._sample_window(size, window_size)
        casted_sample = self._cast_types(sample)
        final_sample = self._split_states(casted_sample)
        return final_sample
