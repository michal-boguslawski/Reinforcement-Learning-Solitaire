import os
from pathlib import Path
import torch as T


folder_path = Path(__file__).parent.absolute()
states_path = os.path.join(folder_path, "data", "states.pt")
states = T.load(states_path)
print(states)
