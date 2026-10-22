"""L2-A actor-critics. Reuses Round-30 DeepSets modules; does not edit PPO."""
from __future__ import annotations

from wt.round30.models import CentralDeepSetsCritic, DeepSetsActorCritic, DeepSetsPPOShape, SharedDeepSetsActor

from .features import BUOY_DIM, GLOBAL_DIM, MAX_UAVS, OPTION_UAV_DIM, UAV_DIM


def _shape(uav_dim: int) -> DeepSetsPPOShape:
    return DeepSetsPPOShape(uav_dim=uav_dim, buoy_dim=BUOY_DIM, global_dim=GLOBAL_DIM, max_uavs=MAX_UAVS)


class L2AStateActorCritic(DeepSetsActorCritic):
    def __init__(self) -> None:
        shape = _shape(UAV_DIM)
        super().__init__(shape)
        self.actor = SharedDeepSetsActor(shape)
        self.critic = CentralDeepSetsCritic(shape)


class L2AOptionActorCritic(DeepSetsActorCritic):
    def __init__(self) -> None:
        shape = _shape(OPTION_UAV_DIM)
        super().__init__(shape)
        self.actor = SharedDeepSetsActor(shape)
        self.critic = CentralDeepSetsCritic(shape)

    @property
    def uses_task_waypoint(self) -> bool:
        return False

    @property
    def uses_transformer(self) -> bool:
        return False


__all__ = ["L2AOptionActorCritic", "L2AStateActorCritic"]
