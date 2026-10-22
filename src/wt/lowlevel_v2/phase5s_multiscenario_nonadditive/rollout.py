"""Frozen MAPPO payloads. Confirmation workers refuse every training scenario."""
from __future__ import annotations

import numpy as np

from wt.lowlevel_v2.phase5g_closed_loop.collect import load_frozen_actor
from wt.lowlevel_v2.phase5o_executor_joint_utility_audit.rollout import execute_mappo

from .world import replay_confirmation


ACTOR = None


def _init_worker() -> None:
    global ACTOR
    ACTOR = load_frozen_actor()


def roll_confirmation(payload: tuple[int, int, list[str]]) -> tuple[int, int, list[tuple[str, float]]]:
    scenario, moment, vectors = payload
    env, movement, _batch, _diagnosis = replay_confirmation(int(scenario), int(moment))
    rows = []
    for vector in vectors:
        assignment = np.asarray([int(item) for item in str(vector).split(";")], dtype=np.int64)
        result = execute_mappo(env, assignment, ACTOR, movement, trace=False)
        rows.append((str(vector), float(result["j_mappo"])))
    return int(scenario), int(moment), rows


__all__ = ["_init_worker", "roll_confirmation"]
