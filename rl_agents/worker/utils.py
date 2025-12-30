import torch as T
from typing import Literal


def get_device(device: Literal["auto", "cpu", "cuda"] | T.device = T.device("cpu")):
    if isinstance(device, str) and device == "auto":
        device = T.device("cuda" if T.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = T.device(device)
    return device
