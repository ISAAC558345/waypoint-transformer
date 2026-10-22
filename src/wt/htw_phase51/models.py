"""Low-level actor-critic conditioned on task option z. PPO update is unchanged."""
from __future__ import annotations

from wt.round30.models import CentralDeepSetsCritic, DeepSetsActorCritic, DeepSetsPPOShape, SharedDeepSetsActor

from .option import OPTION_UAV_DIM


class OptionActorCritic(DeepSetsActorCritic):
    def __init__(self) -> None:
        shape = DeepSetsPPOShape(uav_dim=OPTION_UAV_DIM)
        super().__init__(shape)
        self.actor = SharedDeepSetsActor(shape)
        self.critic = CentralDeepSetsCritic(shape)

    @property
    def uses_task_waypoint(self) -> bool:
        return False

    @property
    def uses_transformer(self) -> bool:
        return False


__all__ = ["OptionActorCritic"]
