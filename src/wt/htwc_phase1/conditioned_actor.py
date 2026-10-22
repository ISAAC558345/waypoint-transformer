"""Goal-conditioned actor-critic with the Round-30 PPO algorithm unchanged.

The class mirrors DeepSetsActorCritic but reads adapter-expanded UAV features.
It does not copy or edit GAE, clip, or reward code.
"""
from __future__ import annotations

from wt.round30.models import CentralDeepSetsCritic, DeepSetsActorCritic, DeepSetsPPOShape, SharedDeepSetsActor

from .adapter import CONDITIONED_UAV_DIM


def conditioned_shape() -> DeepSetsPPOShape:
    return DeepSetsPPOShape(uav_dim=CONDITIONED_UAV_DIM)


class GoalConditionedActorCritic(DeepSetsActorCritic):
    """policy(state, TW_new) via concatenated UAV features."""

    def __init__(self) -> None:
        super().__init__(conditioned_shape())
        self.actor = SharedDeepSetsActor(self.shape)
        self.critic = CentralDeepSetsCritic(self.shape)

    @property
    def uses_task_waypoint(self) -> bool:
        return True

    @property
    def uses_transformer(self) -> bool:
        return False


__all__ = ["GoalConditionedActorCritic", "conditioned_shape"]
