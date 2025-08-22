from replay_buffer import ReplayBuffer
import numpy as np

buffer = ReplayBuffer(100)

for _ in range(100):
    buffer.push(
        [
            (np.ones(shape=(14, 7)),
            np.ones(shape=(4,)),
            np.ones(shape=(1,)),
            np.ones(shape=(1,)),
            ),
            np.ones(shape=(1,))
        ]
    )

sample = buffer.sample(8)
print(sample)
print([x.shape for x in sample[0]])