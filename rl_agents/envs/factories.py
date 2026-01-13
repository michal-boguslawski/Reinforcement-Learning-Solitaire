import gymnasium as gym
from gymnasium.wrappers import NumpyToTorch, RecordVideo, DtypeObservation
import gymnasium.wrappers.vector as vec_wrappers
import numpy as np

from .env_utils import prepare_wrappers
from config.config import EnvConfig
from .wrappers import VecTransposeObservationWrapper


def make_vec(
    id: str,
    num_envs: int,
    training: bool = True,
    record: bool = False,
    video_folder: str = "logs/videos",
    name_prefix: str = "eval",
    *args,
    **kwargs
):
    env_config = EnvConfig(id).get_config()

    wrappers = []
    wrappers.append(lambda env: DtypeObservation(env, np.float32))

    if record:
        record_wrapper = lambda env: RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix=name_prefix,
            episode_trigger=lambda x: True
        )
        kwargs["render_mode"] = "rgb_array"
        wrappers.append(record_wrapper)

    if training:
        training_wrappers_config = env_config.get("training_wrappers")
        wrappers.extend(prepare_wrappers(training_wrappers_config))

    general_wrappers_config = env_config.get("general_wrappers")
    wrappers.extend(prepare_wrappers(general_wrappers_config))

    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}")
    
    try:
        envs = gym.make_vec(
            id=id,
            num_envs=num_envs,
            vector_kwargs=None,
            wrappers=wrappers,
            *args,
            **kwargs
        )
        envs = vec_wrappers.NumpyToTorch(envs)

        # if len(envs.observation_space.shape) == 4:
        #     envs = VecTransposeObservationWrapper(envs)
    except Exception as e:
        raise RuntimeError(f"Failed to create vectorized environment '{id}' with {num_envs} envs: {e}")
    
    return envs
