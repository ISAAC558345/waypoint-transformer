"""Current-state legal pool and the fixed 128-candidate subset."""
from __future__ import annotations

import hashlib

import numpy as np

from wt.benchmark_v2.phase6e.generators import build_pools, feature_matrices, is_legal, proposal_scores
from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import DECISION_TIMES
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import GENERATOR, SELECTED_K
from wt.benchmark_v2.phase6ib_highlevel_dataset.schema import state_arrays, state_hash
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.sampling import select_candidates
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.shards import ShardError, load_manifest, load_npz

from .protocol import MAX_CANDIDATES, PILOT_SEED


def assignment_text(assignment) -> str:
    return "|".join(str(int(item)) for item in np.asarray(assignment, dtype=np.int64).reshape(-1))


def _hash_assignments(assignments: np.ndarray) -> str:
    raw = np.ascontiguousarray(assignments, dtype=np.int64).tobytes()
    return hashlib.sha256(raw).hexdigest()


def legal_pool(env) -> dict:
    pools, _packed, illegal = build_pools(env, ((GENERATOR, int(SELECTED_K)),))
    if int(illegal) != 0:
        raise RuntimeError("candidate pool builder reported an illegal assignment")
    raw = pools[(GENERATOR, int(SELECTED_K))]
    keys = sorted({tuple(int(item) for item in np.asarray(row, dtype=np.int64).reshape(-1)) for row in raw})
    assignment = np.asarray(keys, dtype=np.int64)
    if assignment.ndim != 2 or assignment.shape[0] < int(MAX_CANDIDATES):
        raise RuntimeError("legal pool is smaller than the 128-candidate subset")
    n_uavs = int(env.config.num_uavs)
    n_buoys = int(env.config.num_buoys)
    if any(not is_legal(row, n_uavs, n_buoys) for row in assignment):
        raise RuntimeError("sorted pool contains an illegal assignment")
    return {
        "candidate_id": np.arange(assignment.shape[0], dtype=np.int64),
        "assignment": assignment,
        "pool_hash": _hash_assignments(assignment),
        "pool_count": int(assignment.shape[0]),
        "illegal_count": 0,
    }


def _dense_rank(scores: np.ndarray, ids: np.ndarray) -> np.ndarray:
    order = np.argsort(ids, kind="mergesort")
    order = order[np.argsort(-np.asarray(scores, dtype=np.float64)[order], kind="mergesort")]
    ranks = np.zeros(scores.shape[0], dtype=np.int64)
    rank = 0
    previous = None
    for position in order.tolist():
        value = float(scores[int(position)])
        if previous is None or value < previous:
            rank += 1
            previous = value
        ranks[int(position)] = int(rank)
    return ranks


def _additive_rate(env, assignment: np.ndarray) -> np.ndarray:
    rates = np.asarray(proposal_scores(feature_matrices(env))[GENERATOR], dtype=np.float64)
    scores = np.zeros(assignment.shape[0], dtype=np.float64)
    for index, key in enumerate(assignment):
        total = 0.0
        for uav, buoy in enumerate(np.asarray(key, dtype=np.int64)):
            if int(buoy) >= 0:
                total += float(rates[uav, int(buoy)])
        scores[index] = total
    return scores


def _stored_labels(scenario_id: int, time_step: int, env, pool: dict) -> tuple[np.ndarray, np.ndarray] | None:
    if int(time_step) not in (0, 15, 30):
        return None
    try:
        manifest = load_manifest(int(scenario_id), int(time_step))
    except ShardError:
        return None
    live_hash = state_hash(int(scenario_id), state_arrays(env))
    if str(manifest.get("state_hash")) != live_hash:
        return None
    arrays, _digest = load_npz(int(scenario_id), int(time_step), "candidates")
    stored = np.asarray(arrays["assignment"], dtype=np.int64)
    stored_ids = np.asarray(arrays["candidate_id"], dtype=np.int64).reshape(-1)
    if stored.shape != pool["assignment"].shape or not np.array_equal(stored, pool["assignment"]):
        raise RuntimeError("frozen candidates do not match the live G1_RATE pool at a matching state hash")
    if stored_ids.shape != pool["candidate_id"].shape or not np.array_equal(stored_ids, pool["candidate_id"]):
        raise RuntimeError("frozen candidate ids do not match the sorted live pool")
    return (
        np.asarray(arrays["j15_bits"], dtype=np.float64).reshape(-1),
        np.asarray(arrays["utility_rank"], dtype=np.int64).reshape(-1),
    )


def sample_subset(env, scenario_id: int, time_step: int) -> dict:
    """Fix 128 candidates before any urgent-service label is read."""
    if int(time_step) not in DECISION_TIMES:
        raise RuntimeError(f"subset sampler refuses time {time_step}")
    pool = legal_pool(env)
    stored = _stored_labels(scenario_id, time_step, env, pool)
    if stored is None:
        scores = _additive_rate(env, pool["assignment"])
        ranks = _dense_rank(scores, pool["candidate_id"])
        source = "additive_g1_rate"
    else:
        scores, ranks = stored
        source = "stored_j15"
    if int(time_step) == 0 and source != "stored_j15":
        raise RuntimeError("t=0 does not match the frozen train shard")
    picked = select_candidates(
        pool["candidate_id"],
        scores,
        ranks,
        scenario_id=int(scenario_id),
        time_step=int(time_step),
        seed=int(PILOT_SEED),
    )
    if int(picked["count"]) != int(MAX_CANDIDATES):
        raise RuntimeError(f"subset size is {picked['count']}, not {MAX_CANDIDATES}")
    rows = []
    for item in picked["rows"]:
        position = int(item["row_index"])
        rows.append(
            {
                "scenario_id": int(scenario_id),
                "time_step": int(time_step),
                "candidate_id": int(item["candidate_id"]),
                "assignment": assignment_text(pool["assignment"][position]),
                "assignment_vector": np.asarray(pool["assignment"][position], dtype=np.int64).copy(),
                "sampling_score": float(scores[position]),
                "sampling_source": source,
                "utility_rank": int(ranks[position]),
                "selection": str(item["selection"]),
                "stratum": int(item["stratum"]),
                "filled": int(item["filled"]),
                "pool_hash": pool["pool_hash"],
                "pool_count": int(pool["pool_count"]),
            }
        )
    subset_hash = _hash_assignments(np.asarray([row["assignment_vector"] for row in rows], dtype=np.int64))
    for row in rows:
        row["subset_hash"] = subset_hash
    return {
        "rows": rows,
        "source": source,
        "pool_hash": pool["pool_hash"],
        "pool_count": int(pool["pool_count"]),
        "subset_hash": subset_hash,
        "fill_count": int(picked["fill_count"]),
        "state_hash": state_hash(int(scenario_id), state_arrays(env)),
    }
