device = "cpu"

ppo = {
    "gamma_": 0.99,
    "lambda_": 0.95,
    "critic_coef_": 0.5,
    "entropy_beta_": 0.01,
    "num_epochs": 10,
    "clip_epsilon": 0.2,
    "device": device,
    "lr": 3e-4,
}

worker_kwargs = {
    "num_envs": 1,
    "action_exploration_method": "distribution",
    "device": device,
    "epsilon_start_": 1.,
    "epsilon_decay_factor_": 1.,
    "temperature_start_": 1.,
}

train_kwargs = {
    "num_steps": int(1e6),
    "batch_size": 2048,
    "timesteps": 2048,
    "minibatch_size": 256,
    "train_step": 16
}

env = {
    "id": "CarRacing-v3",
    "vectorization_mode": "async",
}

network_kwargs = {
    "channels": 64,
    "backbone_type": "mlp",
    "backbone_kwargs": {},
    "head_type": "a2c",
    "head_kwargs": {},
    "distribution": "categorical",
    "initial_log_std": 0.,
    "device": device
}

final_config = {
    "agent_type": ppo,
    "worker_kwargs": worker_kwargs,
    "train_kwargs": train_kwargs,
    "env": env,
    "network_kwargs": network_kwargs,
}