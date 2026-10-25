"""Assignment rules for the four pre-registered task distributions."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.phase6e.generators import build_pools, is_legal
from wt.benchmark_v2.phase6iaa_formal_planner.planner import FormalPlanner
from wt.benchmark_v2.phase6ib_highlevel_dataset.jobs import _key
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import GENERATOR, SELECTED_K
from wt.benchmark_v2.phase6i_urgency_stage_a.runner import ScoreTap
from wt.benchmark_v2.phase6iy_candidate_adapter.adapter import frozen_candidate_keys

from .protocol import pool_hash


def scheme_c_decision(planner: FormalPlanner, tap: ScoreTap):
    """Use the formal Scheme C selection already computed for this state."""

    def decision(env, position_prev, predecessor):
        tap.last = None
        selection = planner.select_assignment(env, position_prev, predecessor)
        if selection.status != "selected" or selection.assignment is None or tap.last is None:
            raise RuntimeError(f"scheme C selection failed at t={int(env.t)}: {selection.reason}")
        scores = np.asarray(tap.last.scores, dtype=np.float64)
        index = int(selection.score_index)
        return np.asarray(selection.assignment, dtype=np.int64), _meta(
            candidate_pool_hash=str(selection.pool_hash),
            candidate_index=index,
            candidate_count=int(selection.candidate_count),
            unique_candidate_count=int(selection.candidate_count),
            scheme_c_score=float(scores[index]),
            scheme_c_called=int(selection.scheme_c_called),
        )

    return decision


def g1_rate_decision(planner: FormalPlanner):
    """Use the formal G1_RATE k=6 additive score and its existing tie-break."""

    def decision(env, position_prev, predecessor):
        selection = planner.select_assignment(env, position_prev, predecessor)
        if selection.status != "selected" or selection.assignment is None:
            raise RuntimeError(f"G1_RATE selection failed at t={int(env.t)}: {selection.reason}")
        if int(selection.scheme_c_called) != 0:
            raise RuntimeError("G1_RATE baseline called scheme C")
        return np.asarray(selection.assignment, dtype=np.int64), _meta(
            candidate_pool_hash=str(selection.pool_hash),
            candidate_index=int(selection.score_index),
            candidate_count=int(selection.candidate_count),
            unique_candidate_count=int(selection.candidate_count),
            scheme_c_score="",
            scheme_c_called=0,
        )

    return decision


def training_pool_decision(rng: np.random.Generator):
    """Uniform sample from the k=5 pool used by the frozen low-level training worker."""

    def decision(env, _position_prev, _predecessor):
        if GENERATOR != "G1_RATE" or int(SELECTED_K) != 6:
            raise RuntimeError("frozen high-level generator changed; k=5 training pool is no longer the paired control")
        pools, _packed, illegal = build_pools(env, (("G1_RATE", 5),))
        if int(illegal) != 0:
            raise RuntimeError("ILLEGAL")
        pool = pools[("G1_RATE", 5)]
        if not pool:
            raise RuntimeError("G1_RATE k=5 pool is empty")
        index = int(rng.integers(0, len(pool)))
        assignment = np.asarray(pool[index], dtype=np.int64)
        if not is_legal(assignment, int(env.config.num_uavs), int(env.config.num_buoys)):
            raise RuntimeError("ILLEGAL")
        keys = sorted({_key(item) for item in pool})
        return assignment, _meta(
            candidate_pool_hash=pool_hash(keys),
            candidate_index=index,
            candidate_count=int(len(pool)),
            unique_candidate_count=int(len(keys)),
            scheme_c_score="",
            scheme_c_called=0,
        )

    return decision


def wide_pool_decision(rng: np.random.Generator):
    """Uniform sample from the legal k=6 keys returned for this state."""

    def decision(env, _position_prev, _predecessor):
        keys, raw_count = frozen_candidate_keys(env)
        if not keys:
            raise RuntimeError("frozen k=6 pool is empty")
        index = int(rng.integers(0, len(keys)))
        assignment = np.asarray(keys[index], dtype=np.int64)
        if not is_legal(assignment, int(env.config.num_uavs), int(env.config.num_buoys)):
            raise RuntimeError("ILLEGAL")
        return assignment, _meta(
            candidate_pool_hash=pool_hash(keys),
            candidate_index=index,
            candidate_count=int(len(keys)),
            unique_candidate_count=int(len(keys)),
            raw_candidate_count=int(raw_count),
            scheme_c_score="",
            scheme_c_called=0,
        )

    return decision


def _meta(**values) -> dict:
    values.setdefault("raw_candidate_count", values["candidate_count"])
    return values
