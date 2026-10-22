"""Spawn-safe H=15 utility worker. Independent env copy per call; no shared mutable env."""
from __future__ import annotations

from typing import Any

import numpy as np

from wt.htw_phase9.const import HOLD
from wt.l3b.env import L3BEnv
from wt.l3b.policies import fly_toward, oracle_option_assignment

from .candidate_pool import assignment_hash, is_legal
from .utility_rollout import shadow_h15


def restore_hold_state(scenario: int, time_step: int) -> L3BEnv:
    env = L3BEnv()
    info = env.reset(seed=int(scenario))
    while int(env.t) < int(time_step) and not info.get("terminated", False):
        chosen = oracle_option_assignment(env)
        info = env.step(fly_toward(env, chosen), chosen)
    return env


def h15_payload_worker(item: dict[str, Any]) -> dict[str, Any]:
    """Top-level worker for Windows spawn. `item` must be pickleable."""
    env = restore_hold_state(int(item["scenario"]), int(item["t"]))
    assignment = np.asarray(item["assignment"], dtype=np.int64)
    stats = shadow_h15(env, assignment)
    return {
        "scenario": int(item["scenario"]),
        "t": int(item["t"]),
        "slot": int(item["slot"]),
        "assignment_hash": assignment_hash(assignment),
        "legal": bool(is_legal(assignment, env.config.num_buoys)),
        "hold": int(HOLD),
        **stats,
    }


__all__ = ["h15_payload_worker", "restore_hold_state"]
