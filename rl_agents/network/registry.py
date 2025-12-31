from torch import nn

# Distributions
from .distributions.categorical import CategoricalDistribution
from .distributions.normal import NormalDistribution, MultivariateNormalDistribution

# Transforms
from .distributions.transforms import TanhAffineTransform

# Heads
from .heads.actor_critic import ActorCriticHead

# Policies
from .policies.a2c import ActorCriticPolicy

# Backbones
from .backbones.mlp import MLPNetwork

# Feature extractors
from .feature_extractors.shared import SharedFeatureExtractor


ACTIVATION_FUNCTIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "identity": nn.Identity,
    "gelu": nn.GELU
}


DISTRIBUTIONS = {
    "normal": NormalDistribution,
    "categorical": CategoricalDistribution,
    "mvn": MultivariateNormalDistribution,
}


TRANSFORMS = {
    "normal": TanhAffineTransform,
    "mvn": TanhAffineTransform,
}


HEADS = {
    "actor_critic": ActorCriticHead
}


POLICIES = {
    "actor_critic": ActorCriticPolicy
}


BACKBONES = {
    "mlp": MLPNetwork
}


FEATURE_EXTRACTORS = {
    "shared": SharedFeatureExtractor
}
