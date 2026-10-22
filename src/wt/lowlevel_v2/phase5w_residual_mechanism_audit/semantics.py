"""Service rules taken from L3BEnv.step and active_links, not from feature names."""
from __future__ import annotations


def service_semantics() -> dict:
    return {
        "progress_owner": "uav",
        "served_k_meaning": "buoy id whose accumulator is stored in progress[uav]",
        "buoy_does_not_own_progress": True,
        "progress_is_not_transferred_between_uavs": True,
        "switch_target": "reset",
        "switch_detail": "If the new target differs from served_k, progress[uav] is set to 0 and served_k becomes the new target before any increment.",
        "leave_service_radius": "pause",
        "leave_detail": "An out-of-range UAV is omitted from active_links, so the accumulator is not increased and is not cleared.",
        "conflict": "UAVs that share a buoy are blocked and do not accumulate.",
        "empty_queue": "A buoy with no queue does not accumulate.",
        "completion_rule": "accumulator",
        "completion_detail": "Each active step adds min((rate/reference_rate)/service_slots, 1). At or above 1 the accumulator returns to 0 and one block is delivered.",
        "service_slots": 3,
        "service_slots_meaning": "divisor of the rate-normalized increment, not a stored consecutive-step counter",
        "consecutive_steps_required": False,
        "same_target_preserves_existing_progress": True,
        "continue_progress_formula": "same_target * progress[uav]",
        "continue_progress_matches_reset": True,
        "continue_progress_limits": [
            "It does not model the pause while a UAV is outside the service radius.",
            "It does not convert progress into remaining bits.",
            "buoy_progress is the current server's accumulator and is not given to a newly assigned UAV.",
            "Phase-5U z_rel copies progress[uav] onto every selected edge, including a switched target that the environment will clear.",
        ],
    }


__all__ = ["service_semantics"]
