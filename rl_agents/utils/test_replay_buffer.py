import pytest
import numpy as np
import torch as T

from replay_buffer import ReplayBuffer


def generate_item():
    item = {
        "state": (
            np.random.randint(low=0, high=10, size=(14, 7)),
            np.random.randint(low=0, high=10, size=(4, )),
            np.random.randint(low=0, high=10, size=(1, )),
            np.random.randint(low=0, high=10, size=(1, )),
        ),
        "action": np.random.randint(low=0, high=10, size=(1, )),
        "reward": np.random.rand(1, ),
        "done": np.random.rand(1, ) > 0.5
    }
    return item

def test_replay_buffer_push():
    replay_buffer = ReplayBuffer(10)
    item = generate_item()
    replay_buffer.push(item)
    for key, value in item.items():
        assert value in replay_buffer.buffers[key]

    assert len(replay_buffer) == 1
    assert all([len(value) == 1 for value in replay_buffer.buffers.values()])

def test_replay_buffer_length_n_lower_than_size():
    replay_buffer = ReplayBuffer(10)
    first_item = generate_item()
    replay_buffer.push(first_item)
    
    for _ in range(5):
        item = generate_item()
        replay_buffer.push(item)
    
    for key, value in first_item.items():
        assert id(value) in [id(value) for value in replay_buffer.buffers[key]]
    
    assert len(replay_buffer) == 6
    assert all([len(value) == 6 for value in replay_buffer.buffers.values()])
        
def test_replay_buffer_length_n_equal_to_size_plus_one():
    replay_buffer = ReplayBuffer(10)
    first_item = generate_item()
    replay_buffer.push(first_item)
    
    for _ in range(10):
        item = generate_item()
        replay_buffer.push(item)
    
    assert len(replay_buffer) == 10
    assert all([len(value) == (10 + (key == "state")) for key, value in replay_buffer.buffers.items()])
    for key, value in first_item.items():
        replay_buffers_key_id = [id(buffer_value) for buffer_value in replay_buffer.buffers[key]]
        if key == "state":
            assert id(value) in replay_buffers_key_id
        else:
            assert id(value) not in replay_buffers_key_id
    
    item = generate_item()
    replay_buffer.push(item)
    
    assert len(replay_buffer) == 10
    assert all([len(value) == (10 + (key == "state")) for key, value in replay_buffer.buffers.items()])
    for key, value in first_item.items():
        replay_buffers_key_id = [id(buffer_value) for buffer_value in replay_buffer.buffers[key]]
        assert id(value) not in replay_buffers_key_id
        
def test_replay_buffer_length_n_equal_to_size_plus_two():
    replay_buffer = ReplayBuffer(10)
    first_item = generate_item()
    replay_buffer.push(first_item)
    
    for _ in range(11):
        item = generate_item()
        replay_buffer.push(item)
    
    assert len(replay_buffer) == 10
    assert all([len(value) == (10 + (key == "state")) for key, value in replay_buffer.buffers.items()])
    for key, value in first_item.items():
        replay_buffers_key_id = [id(buffer_value) for buffer_value in replay_buffer.buffers[key]]
        assert id(value) not in replay_buffers_key_id

def test_replay_buffer_window_sample_shapes():
    dtypes_schema = {
        "state": T.int32,
        "action": T.int32,
        "reward": T.float32,
        "done": T.float32
    }
    replay_buffer = ReplayBuffer(10, dtypes_schema=dtypes_schema)
    
    for _ in range(10):
        item = generate_item()
        replay_buffer.push(item)
    
    sample = replay_buffer.sample(4, 5)
    for key, value in sample.items():
        if key in ["state", "next_state"]:
            print(key, [(x.shape, type(x), x.dtype) for x in value])
        else:
            print(key, (value.shape, type(value), value.dtype))
    
        
test_replay_buffer_push()
test_replay_buffer_length_n_lower_than_size()
test_replay_buffer_length_n_equal_to_size_plus_one()
test_replay_buffer_length_n_equal_to_size_plus_two()
test_replay_buffer_window_sample_shapes()
