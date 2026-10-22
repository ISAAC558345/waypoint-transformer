"""Independent shadow controller. It does not call the deployment decision function."""
from __future__ import annotations

import numpy as np

from wt.htw_phase100.contract import TaskOption
from wt.l3b.env import L3BEnv, conflict_mask
from wt.lowlevel_v2.contracts import refuse_rematch
from wt.lowlevel_v2.phase5x_executor_lookahead.shadow import fingerprint, max_abs_diff

from .protocol import HOLD_STEPS


def select_index(scores: np.ndarray) -> int:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("non-finite scores")
    return int(np.argsort(-values, kind="stable")[0])


def assignment_ok(assignment: np.ndarray, num_uavs: int, num_buoys: int) -> tuple[bool, str]:
    chosen = np.asarray(assignment, dtype=np.int64).reshape(-1)
    if chosen.shape != (int(num_uavs),):
        return False, "shape"
    if np.any((chosen < -1) | (chosen >= int(num_buoys))):
        return False, "illegal_target"
    active = chosen[chosen >= 0]
    if active.size != np.unique(active).size:
        return False, "duplicate_target"
    if bool(np.any(conflict_mask(chosen, int(num_buoys)))):
        return False, "conflict"
    return True, "ok"


def build_task_option(env: L3BEnv, info: dict, assignment: np.ndarray) -> TaskOption:
    targets = np.asarray(assignment, dtype=np.int64).copy()
    option = TaskOption(
        targets=targets,
        decision_time=int(info["time_step"]),
        remaining_hold=int(HOLD_STEPS),
        uav_mask=np.ones(int(env.config.num_uavs), dtype=bool),
        buoy_mask=np.ones(int(env.config.num_buoys), dtype=bool),
        association=targets.copy(),
        urgency=None,
    )
    refuse_rematch(option)
    return option


def continue_or_switch(current: np.ndarray, selected: np.ndarray) -> list[str]:
    now = np.asarray(current, dtype=np.int64)
    nxt = np.asarray(selected, dtype=np.int64)
    return ["continue" if int(left) == int(right) else "switch" for left, right in zip(now, nxt)]


def caller_unchanged(before: dict, after: dict) -> bool:
    return max_abs_diff(before, after) == 0.0


def snapshot(env: L3BEnv) -> dict:
    return fingerprint(env)


__all__ = [
    "assignment_ok",
    "build_task_option",
    "caller_unchanged",
    "continue_or_switch",
    "select_index",
    "snapshot",
]
