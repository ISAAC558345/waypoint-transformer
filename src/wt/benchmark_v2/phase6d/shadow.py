"""Diagnostic macro-action returns. They do not change the live environment."""
from __future__ import annotations

import copy

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.l3b.policies import fly_toward

_STATE_FIELDS = (
    "uav_positions",
    "assignment",
    "progress",
    "served_k",
    "q_bits",
    "queue",
    "service_remainder_bits",
    "completed_block_equivalent",
    "cumulative_served_bits",
    "cumulative_dropped_bits",
    "cumulative_generated_arrival_bits",
    "cumulative_snap_bits",
    "last_sinr",
    "last_rate",
    "legacy_progress_completions",
)


def capture_state(env: L3BBitQueueEnvV2) -> dict:
    snap = {name: np.asarray(getattr(env, name)).copy() for name in _STATE_FIELDS}
    snap["t"] = int(env.t)
    snap["payload_bits"] = float(env.payload_bits)
    snap["dropped"] = float(env.dropped)
    return snap


def restore_state(env: L3BBitQueueEnvV2, snap: dict) -> None:
    for name in _STATE_FIELDS:
        getattr(env, name)[...] = snap[name]
    env.t = int(snap["t"])
    env.payload_bits = float(snap["payload_bits"])
    env.dropped = float(snap["dropped"])


def _zero_arrivals(env: L3BBitQueueEnvV2) -> None:
    zeros = np.zeros(env.config.num_buoys, dtype=np.float64)
    env.arrivals = lambda: zeros.copy()
    env.arrivals_at = lambda time_step: zeros.copy()


def macro_return(
    env: L3BBitQueueEnvV2,
    assignment: np.ndarray,
    horizon: int,
    *,
    use_realized_future_arrivals: bool,
) -> float:
    """Sum of served bits over H fly-toward steps on a clone.

    use_realized_future_arrivals=False is the causal label J_H_CURRENT.
    True is a non-causal upper bound and must not train a controller.
    """
    clone = copy.deepcopy(env)
    _quiet_info(clone)
    if not use_realized_future_arrivals:
        _zero_arrivals(clone)
    total = 0.0
    chosen = np.asarray(assignment, dtype=np.int64)
    steps = 0
    while steps < int(horizon) and int(clone.t) < int(clone.config.episode_horizon):
        info = clone.step(fly_toward(clone, chosen), chosen)
        total += float(info["payload_increment_bits"])
        steps += 1
    return float(total)


def _quiet_info(env: L3BBitQueueEnvV2) -> None:
    """Skip the diagnostic graph inside a shadow step. Accounting still runs."""
    env.info = lambda: {"sinr": env.last_sinr, "rate_bps": env.last_rate}


def score_pool(
    env: L3BBitQueueEnvV2,
    assignments: list[np.ndarray],
    horizon: int,
    *,
    use_realized_future_arrivals: bool,
    prepare_clone=None,
) -> tuple[np.ndarray, float]:
    """Return the lex-smallest assignment with the best shadowed return."""
    scratch = copy.deepcopy(env)
    _quiet_info(scratch)
    if prepare_clone is not None:
        prepare_clone(scratch)
    if not use_realized_future_arrivals:
        _zero_arrivals(scratch)
    snap = capture_state(scratch)
    best_value = -1.0
    best = np.asarray(assignments[0], dtype=np.int64).copy()
    for assignment in assignments:
        restore_state(scratch, snap)
        if not use_realized_future_arrivals:
            _zero_arrivals(scratch)
        total = 0.0
        steps = 0
        while steps < int(horizon) and int(scratch.t) < int(scratch.config.episode_horizon):
            info = scratch.step(fly_toward(scratch, assignment), assignment)
            total += float(info["payload_increment_bits"])
            steps += 1
        identity = tuple(int(item) for item in assignment)
        if total > best_value + 1.0 or (abs(total - best_value) <= 1.0 and identity < tuple(int(item) for item in best)):
            best_value = float(total)
            best = np.asarray(assignment, dtype=np.int64).copy()
    return best, float(best_value)


__all__ = ["capture_state", "macro_return", "restore_state", "score_pool"]
