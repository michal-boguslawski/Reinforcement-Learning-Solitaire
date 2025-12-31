import torch as T


def step_return_discounting(
    values: T.Tensor, 
    dones: T.Tensor, 
    discount: float, 
    next_values: T.Tensor | None = None
) -> T.Tensor:
    """
    Compute discounted returns from a sequence of rewards.

    Args:
        values (T.Tensor): shape [N, T], reward at each step
        dones (T.Tensor): shape [N, T], True if episode ended at this step, False otherwise
        discount (float): discount factor
        next_values (T.Tensor): shape [N], bootstrap value estimate for the next state

    Returns:
        T.Tensor: discounted returns, shape [N, T]
    """
    
    returns = T.zeros_like(values, dtype=T.float32)
    T_steps = values.shape[-1]
    if next_values is None:
        next_values = T.zeros_like(returns[:, 0], dtype=T.float32)
    g = next_values
    
    for t in reversed(range(T_steps)):
        g = values[..., t] + discount * g * (1 - dones[..., t].float())
        returns[..., t] = g
    return returns
