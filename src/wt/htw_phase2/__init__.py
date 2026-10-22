"""HTW Phase-2 hierarchical controller prototype (oracle TW, tracking aux)."""

from .tracking import DEFAULT_TRACKING_LAMBDA, auxiliary_reward
from .vector import HierarchicalTrackingVector

__all__ = ["DEFAULT_TRACKING_LAMBDA", "HierarchicalTrackingVector", "auxiliary_reward"]
