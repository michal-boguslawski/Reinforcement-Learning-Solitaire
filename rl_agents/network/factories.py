from torch import nn

from .registry import DISTRIBUTIONS, TRANSFORMS, BACKBONES, HEADS, CORES
from .distributions.base import ActionDistribution


def make_action_distribution(
    dist_name: str,
    *args,
    **kwargs,
) -> ActionDistribution:
    transform = None

    try:
        base_cls = DISTRIBUTIONS[dist_name]
    except KeyError:
        raise ValueError(f"Unknown distribution: {dist_name}")

    base = base_cls(**kwargs)

    transform_cls = TRANSFORMS.get(dist_name, None)
    if transform_cls:
        transform = transform_cls(**kwargs)

    return ActionDistribution(base, transform)


def make_backbone(
    backbone_name: str,
    input_shape: tuple,
    num_features: int,
    **kwargs
) -> nn.Module:
    backbone = BACKBONES[backbone_name](
        input_shape=input_shape,
        num_features=num_features,
        **kwargs
    )
    return backbone


def make_head(
    head_name: str,
    num_actions: int,
    num_features: int,
    *args,
    **kwargs
) -> nn.Module:
    head = HEADS[head_name](
        num_actions=num_actions,
        num_features=num_features,
        **kwargs
    )
    return head


def make_core(
    core_name: str,
    num_features: int,
    *args,
    **kwargs
) -> nn.Module:
    core = CORES[core_name](
        num_features=num_features,
        **kwargs
    )
    return core
