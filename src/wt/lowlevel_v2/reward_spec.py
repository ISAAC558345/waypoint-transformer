"""Reward protocol draft. Coefficients are not frozen. Do not train."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RewardDraft:
    primary: str = "actual payload increment / normalization scale"
    r_progress: str = "service progress increment"
    r_near: str = "soft near-collision cost"
    r_project: str = "SafetyProjector intervention penalty"
    r_residual: str = "residual action magnitude penalty"
    formula: str = "r = r_payload + lp*r_progress - ln*r_near - lproj*r_project - lres*r_residual"
    lambda_progress: None = None
    lambda_near: None = None
    lambda_project: None = None
    lambda_residual: None = None
    auto_search_forbidden: bool = True
    design_freeze_required_before_training: bool = True
    evaluation_uses_train_reward: bool = False
    evaluation_metrics: tuple[str, ...] = (
        "payload",
        "completed_blocks",
        "collision_count",
        "minimum_separation",
        "flight_distance",
        "queue",
        "drop",
    )
    extra_interference_penalty: bool = False
    reason_no_interference_penalty: str = "SINR->rate->payload already carries interference; do not add a second hand-designed penalty in v1"


def reward_draft_dict() -> dict:
    return asdict(RewardDraft())


__all__ = ["RewardDraft", "reward_draft_dict"]
