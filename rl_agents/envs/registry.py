
from gymnasium.wrappers import RecordVideo, TransformReward, ClipAction, RescaleObservation
from typing import Callable, Dict

from .wrappers import TerminalBonusWrapper, PowerObsRewardWrapper, ActionPowerRewardWrapper, ActionInteractionWrapper, \
    OutOfTrackPenaltyAndTerminationWrapper


WRAPPERS: Dict[str, Callable] = {
    "scale_reward": lambda env, scale_factor, loc_factor: 
        TransformReward(env, lambda r: scale_factor * r + loc_factor),
    "terminal_bonus": TerminalBonusWrapper,
    "power_obs_reward": PowerObsRewardWrapper,
    "clip_action": ClipAction,
    "record_video": RecordVideo,
    "action_reward": ActionPowerRewardWrapper,
    "actions_interactions": ActionInteractionWrapper,
    "rescale_observation": RescaleObservation,
    "out_of_track": OutOfTrackPenaltyAndTerminationWrapper,
}
