"""Classify a decoded assignment against frozen generator families. Does not change L=24."""
from __future__ import annotations

import numpy as np

from wt.htw_phase9.const import MAX_UAVS
from wt.htw_phase945.candidate_pool import (
    _competition_alt,
    _fill_busy,
    _idle_variants,
    _interference_alt,
    _one_edge,
    _pad,
    _two_edge,
    assignment_hash,
    is_legal,
)
from wt.l3a.env import competition_pairs
from wt.l3b.env import L3BEnv
from wt.l3b.policies import exclusive_matching, myopic_assignment, oracle_option_assignment


def _key(assignment: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in np.asarray(assignment, dtype=np.int64).tolist())


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    x = np.asarray(a, dtype=np.int64)
    y = np.asarray(b, dtype=np.int64)
    n = min(x.size, y.size)
    return int(np.sum(x[:n] != y[:n]))


def edge_symdiff(a: np.ndarray, b: np.ndarray) -> int:
    sa = {(i, int(v)) for i, v in enumerate(np.asarray(a, dtype=np.int64).tolist()) if int(v) >= 0}
    sb = {(i, int(v)) for i, v in enumerate(np.asarray(b, dtype=np.int64).tolist()) if int(v) >= 0}
    return int(len(sa ^ sb))


def idle_count(assignment: np.ndarray) -> int:
    return int(np.sum(np.asarray(assignment, dtype=np.int64)[:MAX_UAVS] < 0))


def nearest_pool(assignment: np.ndarray, pool: np.ndarray) -> dict:
    best = None
    for slot, row in enumerate(np.asarray(pool, dtype=np.int64)):
        ham = hamming(assignment, row)
        esd = edge_symdiff(assignment, row)
        idle = abs(idle_count(assignment) - idle_count(row))
        rec = (ham, esd, idle, int(slot))
        if best is None or rec < best:
            best = rec
    assert best is not None
    slot = best[3]
    return {
        "nearest_slot": slot,
        "minimum_assignment_distance": best[0],
        "edge_symdiff": best[1],
        "idle_count_diff": best[2],
        "nearest_hash": assignment_hash(pool[slot]),
    }


def coverage_label(assignment: np.ndarray, env: L3BEnv) -> str:
    chosen = np.asarray(assignment, dtype=np.int64)
    if not is_legal(chosen, env.config.num_buoys):
        return "illegal"
    ref = _pad(oracle_option_assignment(env))
    families = {
        "heuristic_oracle": ref,
        "exclusive": _pad(exclusive_matching(env)),
        "myopic": _pad(myopic_assignment(env)),
        "competition_alt": _pad(_competition_alt(env)),
        "interference_2_1": _pad(_interference_alt(env, 2, 1)),
        "interference_3_1": _pad(_interference_alt(env, 3, 1)),
        "interference_1_2": _pad(_interference_alt(env, 1, 2)),
        "idle_variant_2": _pad(_idle_variants(ref, env, 2)),
        "idle_variant_4": _pad(_idle_variants(ref, env, 4)),
        "all_idle": np.full(MAX_UAVS, -1, dtype=np.int64),
        "fill_busy": _pad(_fill_busy(ref, env)),
    }
    key = _key(chosen)
    for name, row in families.items():
        if _key(row) == key:
            return name
    one = {_key(_pad(row)) for row in _one_edge(ref, env)}
    if key in one:
        return "1-edge_from_oracle"
    two = {_key(_pad(row)) for row in _two_edge(ref, env)}
    if key in two:
        return "2-edge_from_oracle"
    ham = hamming(chosen, ref)
    if ham == 1:
        return "1-edge_hamming"
    if ham == 2:
        return "2-edge_hamming"
    return "uncovered_structure"


def competition_count(assignment: np.ndarray, num_buoys: int = 12) -> int:
    return len(competition_pairs(np.asarray(assignment, dtype=np.int64), int(num_buoys)))


__all__ = ["competition_count", "coverage_label", "edge_symdiff", "hamming", "idle_count", "nearest_pool"]
