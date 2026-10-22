"""Isolated G1_RATE k=6 pool. It calls the frozen builder and does not edit the planner."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wt.benchmark_v2.phase6e.generators import build_pools, feature_matrices, is_legal, proposal_scores
from wt.benchmark_v2.phase6ib_highlevel_dataset.jobs import _key
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import GENERATOR, SELECTED_K
from wt.benchmark_v2.phase6ib_highlevel_dataset.schema import state_hash
from wt.benchmark_v2.phase6ix_candidate_alignment.audit import _restore
from wt.benchmark_v2.phase6iy_candidate_adapter.protocol import CANDIDATE_COUNT, config_sha256


@dataclass(frozen=True)
class CandidateBinding:
    """One assignment bound to the score row that scheme C consumes."""

    score_index: int
    assignment: tuple[int, ...]
    legal: bool
    contains_idle: bool


@dataclass(frozen=True)
class FrozenCandidatePool:
    scenario_id: int
    time_step: int
    state_hash: str
    config_sha256: str
    raw_count: int
    bindings: tuple[CandidateBinding, ...]
    by_assignment: dict[tuple[int, ...], int]

    def assignment_matrix(self) -> np.ndarray:
        rows = [binding.assignment for binding in self.bindings]
        return np.asarray(rows, dtype=np.int64)

    def assignment_at(self, score_index: int) -> tuple[int, ...]:
        binding = self.bindings[int(score_index)]
        if int(binding.score_index) != int(score_index):
            raise RuntimeError("score index is not bound to this assignment")
        mapped = self.by_assignment[binding.assignment]
        if int(mapped) != int(score_index):
            raise RuntimeError("assignment map does not point back to this score index")
        return binding.assignment

    def score_index_of(self, assignment) -> int:
        return int(self.by_assignment[_key(assignment)])


def frozen_candidate_keys(env) -> tuple[list[tuple[int, ...]], int]:
    """Same construction as Phase-6I-B before any J15 evaluation."""
    if GENERATOR != "G1_RATE" or int(SELECTED_K) != 6:
        raise RuntimeError("frozen candidate protocol is not G1_RATE k=6")
    pools, _packed, illegal = build_pools(env, ((GENERATOR, SELECTED_K),))
    if int(illegal) != 0:
        raise RuntimeError("ILLEGAL")
    raw = pools[(GENERATOR, SELECTED_K)]
    keys = sorted({_key(item) for item in raw})
    if any(not is_legal(np.asarray(key, dtype=np.int64), int(env.config.num_uavs), int(env.config.num_buoys)) for key in keys):
        raise RuntimeError("ILLEGAL")
    return keys, int(len(raw))


def bind_keys(keys: list[tuple[int, ...]], *, n_uavs: int = 6, n_buoys: int = 12) -> tuple[tuple[CandidateBinding, ...], dict[tuple[int, ...], int]]:
    bindings = []
    by_assignment: dict[tuple[int, ...], int] = {}
    for index, key in enumerate(keys):
        if key in by_assignment:
            raise RuntimeError("duplicate assignment survived the frozen unique sort")
        binding = CandidateBinding(
            score_index=int(index),
            assignment=key,
            legal=bool(is_legal(np.asarray(key, dtype=np.int64), int(n_uavs), int(n_buoys))),
            contains_idle=any(int(item) < 0 for item in key),
        )
        bindings.append(binding)
        by_assignment[key] = int(index)
    return tuple(bindings), by_assignment


def additive_assignment(keys: list[tuple[int, ...]], normalized: np.ndarray) -> tuple[int, ...]:
    """First maximum of the G1_RATE sum, in the already sorted key order."""
    totals = []
    for key in keys:
        total = 0.0
        for uav, buoy in enumerate(key):
            if int(buoy) >= 0:
                total += float(normalized[uav, int(buoy)])
        totals.append(total)
    return keys[int(np.argmax(np.asarray(totals, dtype=np.float64)))]


def generate_frozen_pool(scenario_id: int, time_step: int) -> FrozenCandidatePool:
    """Restore one saved snapshot and build its k=6 pool. The environment is not stepped."""
    env, saved = _restore(int(scenario_id), int(time_step))
    if int(env.t) != int(time_step):
        raise RuntimeError("snapshot time was not restored")
    keys, raw_count = frozen_candidate_keys(env)
    if int(len(keys)) != CANDIDATE_COUNT:
        raise RuntimeError(f"frozen pool size {len(keys)} is not {CANDIDATE_COUNT}")
    bindings, by_assignment = bind_keys(keys, n_uavs=int(env.config.num_uavs), n_buoys=int(env.config.num_buoys))
    if any(not binding.legal for binding in bindings):
        raise RuntimeError("ILLEGAL")
    return FrozenCandidatePool(
        scenario_id=int(scenario_id),
        time_step=int(time_step),
        state_hash=state_hash(int(scenario_id), saved),
        config_sha256=config_sha256(),
        raw_count=int(raw_count),
        bindings=bindings,
        by_assignment=by_assignment,
    )


def snapshot_rate_gap(scenario_id: int, time_step: int) -> float:
    env, saved = _restore(int(scenario_id), int(time_step))
    recomputed = np.asarray(feature_matrices(env)["rate"], dtype=np.float64)
    saved_rate = np.asarray(saved["uav_buoy_hypothetical_rate"], dtype=np.float64)
    return float(np.max(np.abs(recomputed - saved_rate)))


def g1_choice(scenario_id: int, time_step: int, pool: FrozenCandidatePool) -> tuple[int, ...]:
    env, _saved = _restore(int(scenario_id), int(time_step))
    normalized = proposal_scores(feature_matrices(env))[GENERATOR]
    return additive_assignment([binding.assignment for binding in pool.bindings], normalized)


__all__ = [
    "CandidateBinding",
    "FrozenCandidatePool",
    "additive_assignment",
    "bind_keys",
    "frozen_candidate_keys",
    "g1_choice",
    "generate_frozen_pool",
    "snapshot_rate_gap",
]
