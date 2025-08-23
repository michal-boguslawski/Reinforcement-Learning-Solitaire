import torch.nn as nn
import torch.nn.functional as F
import torch as T

ttensor = T.arange(0, 10)
temp = ttensor.unsqueeze(0) - ttensor.unsqueeze(1)
gamma = 0.9
temp2 = (gamma ** temp).triu()
print((gamma ** temp).triu())

print(ttensor.unsqueeze(0).to(T.float) @ temp2)

import numpy as np
print(np.array([True,]) or np.array([False,]))