"""Six-UAV DeepSets actor-critic. Shape comes from the feature schema, not (5, 21)."""
from __future__ import annotations

from typing import Any, Mapping

import torch

from wt.round30.models import CentralDeepSetsCritic, DeepSetsActorCritic, DeepSetsPPOShape, SharedDeepSetsActor


def shape_from_schema(schema: Mapping[str, Any]) -> DeepSetsPPOShape:
    return DeepSetsPPOShape(
        uav_dim=int(schema["uav_dim"]),
        buoy_dim=int(schema["buoy_dim"]),
        global_dim=int(schema["global_dim"]),
        max_uavs=int(schema["max_uavs"]),
        action_dim=int(schema["action_dim"]),
    )


class L3BGoalActorCritic(DeepSetsActorCritic):
    def __init__(self, schema: Mapping[str, Any]) -> None:
        if int(schema["max_uavs"]) != 6:
            raise ValueError("Phase-10.1 low-level model is the L3-B six-UAV adapter")
        shape = shape_from_schema(schema)
        super().__init__(shape)
        self.actor = SharedDeepSetsActor(shape)
        self.critic = CentralDeepSetsCritic(shape)
        self.schema = dict(schema)

    @property
    def uses_hgnn(self) -> bool:
        return False

    @property
    def uses_transformer(self) -> bool:
        return False


def assert_fresh_init(state: Mapping[str, torch.Tensor], forbidden: Mapping[str, torch.Tensor] | None = None) -> None:
    if forbidden is None:
        return
    overlap = [key for key in state if key in forbidden and tuple(state[key].shape) == tuple(forbidden[key].shape)]
    if overlap:
        raise RuntimeError("refused to copy M=5 tensors into the six-UAV low-level model")


__all__ = ["L3BGoalActorCritic", "assert_fresh_init", "shape_from_schema"]
