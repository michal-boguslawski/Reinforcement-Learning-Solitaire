from typing import Literal
from pydantic import BaseModel, Field


# ---------- Sub-configs ----------

class EnvConfig(BaseModel):
    id: str
    vectorization_mode: Literal["sync", "async"]
    num_envs: int


class PolicyPPOKwargs(BaseModel):
    gamma: float = Field(0.99, ge=0, le=1)
    lambda_: float = Field(0.95, ge=0, le=1)
    value_loss_coef: float = Field(0.5)
    entropy_coef: float = Field(0.01)
    entropy_decay: float = Field(1, ge=0, le=1)
    lr: float = Field(0.001)
    num_epochs: int = Field(10, ge=1)
    clip_epsilon: float = Field(0.2)
    exploration_method: Literal["distribution", "egreedy", "best"]
    advantage_normalize: Literal["batch", "global"] | None


class PolicyConfig(BaseModel):
    type: Literal["ppo"]
    kwargs: PolicyPPOKwargs


class WorkerConfig(BaseModel):
    device: Literal["auto", "cpu", "cuda"]
    record_step: int = Field(100_000, ge=10_000)


class TrainConfig(BaseModel):
    num_steps: int
    batch_size: int
    minibatch_size: int


class ACNetworkKwargs(BaseModel):
    distribution: Literal["normal"]
    initial_log_std: float


class NetworkConfig(BaseModel):
    type: Literal["ac_network"]
    kwargs: ACNetworkKwargs


# ---------- Root config ----------

class ExperimentConfigModel(BaseModel):
    experiment_name: str
    env_kwargs: EnvConfig
    policy: PolicyConfig
    worker_kwargs: WorkerConfig
    train_kwargs: TrainConfig
    network: NetworkConfig
