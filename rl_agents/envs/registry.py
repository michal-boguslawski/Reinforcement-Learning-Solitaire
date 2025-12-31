
from gymnasium.wrappers import RecordVideo, TransformReward, ClipAction, TransformObservation
from typing import Callable, Dict

from .wrappers import TerminalBonusWrapper, PowerObsRewardWrapper


WRAPPERS: Dict[str, Callable] = {
    "scale_reward": lambda env, scale_factor, loc_factor: 
        TransformReward(env, lambda r: scale_factor * r + loc_factor),
    "terminal_bonus": TerminalBonusWrapper,
    "power_obs_reward": PowerObsRewardWrapper,
    "clip_action": ClipAction,
    "record_video": RecordVideo,
    "transpose_image": lambda env: TransformObservation(env, lambda obs: obs.transpose(2, 0, 1), None)
}
