# Exploration methods
from .exploration.egreedy import EGreedyExploration
from .exploration.distribution import DistributionExploration


EXPLORATIONS = {
    "egreedy": EGreedyExploration,
    "distribution": DistributionExploration,
}
