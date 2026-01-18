from typing import Literal
from pydantic import BaseModel, Field


# ---------- Sub-configs ----------

class EnvConfig(BaseModel):
    id: str
    vectorization_mode: Literal["sync", "async"]
    num_envs: int
    training_wrappers: dict = Field(default_factory=dict)
    general_wrappers: dict = Field(default_factory=dict)


class ExplorationMethod(BaseModel):
    name: Literal["distribution", "egreedy"]
    kwargs: dict = Field(default_factory=dict)


class OptimizerConfig(BaseModel):
    lr: float | None = None
    actor_lr: float | None = None
    critic_lr: float | None = None


class PolicyKwargs(BaseModel):
    gamma: float = Field(0.99, ge=0, le=1)
    lambda_: float = Field(0.95, ge=0, le=1)
    value_loss_coef: float | None = None
    entropy_coef: float | None = None
    entropy_decay: float | None = None
    num_epochs: int | None = None
    clip_epsilon: float | None = None
    advantage_normalize: Literal["batch", "global"] | None = None
    exploration_method: ExplorationMethod
    optimizer_kwargs: OptimizerConfig


class PolicyConfig(BaseModel):
    type: Literal["ppo", "sarsa", "a2c"]
    kwargs: PolicyKwargs


class WorkerConfig(BaseModel):
    device: Literal["auto", "cpu", "cuda"]
    record_step: int = Field(100_000, ge=5_000)


class TrainConfig(BaseModel):
    num_steps: int
    batch_size: int
    minibatch_size: int


class NetworkKwargs(BaseModel):
    num_features: int = 64

    backbone_name: Literal["mlp", "simple_cnn"] = "mlp"
    backbone_kwargs: dict = Field(default_factory=dict)

    core_name: Literal["identity", "lstm", "gru"] = "identity"
    core_kwargs: dict = Field(default_factory=dict)

    head_name: Literal["actor_critic", "actor"] = "actor_critic"
    head_kwargs: dict = Field(default_factory=dict)

    distribution: Literal["normal", "mvn", "categorical"] = "normal"
    initial_deviation: float = 1.0


class NetworkConfig(BaseModel):
    kwargs: NetworkKwargs


# ---------- Root config ----------

class ExperimentConfigModel(BaseModel):
    experiment_name: str
    env_kwargs: EnvConfig
    policy: PolicyConfig
    worker_kwargs: WorkerConfig
    train_kwargs: TrainConfig
    network: NetworkConfig
