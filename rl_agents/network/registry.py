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
from .backbones.cnn import SimpleCNN

# Feature extractors
from .feature_extractors.shared import SharedFeatureExtractor


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
    "mlp": MLPNetwork,
    "simple_cnn": SimpleCNN
}


FEATURE_EXTRACTORS = {
    "shared": SharedFeatureExtractor
}
