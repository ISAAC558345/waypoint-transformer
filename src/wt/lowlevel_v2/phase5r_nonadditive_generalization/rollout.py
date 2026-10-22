"""Parallel frozen-MAPPO payloads. Workers do not score the high-level models."""
from __future__ import annotations

import numpy as np

from wt.lowlevel_v2.phase5g_closed_loop.collect import load_frozen_actor
from wt.lowlevel_v2.phase5o_executor_joint_utility_audit.rollout import execute_mappo

from .world import replay_snapshot


ACTOR = None


def _init_worker() -> None:
    global ACTOR
    ACTOR = load_frozen_actor()


def roll_assignments(payload: tuple[int, int, list[str]]) -> tuple[int, int, list[tuple[str, float]]]:
    scenario, moment, vectors = payload
    env, movement, _batch = replay_snapshot(int(scenario), int(moment))
    rows = []
    for vector in vectors:
        assignment = np.asarray([int(item) for item in str(vector).split(";")], dtype=np.int64)
        result = execute_mappo(env, assignment, ACTOR, movement, trace=False)
        rows.append((str(vector), float(result["j_mappo"])))
    return int(scenario), int(moment), rows


__all__ = ["_init_worker", "roll_assignments"]
