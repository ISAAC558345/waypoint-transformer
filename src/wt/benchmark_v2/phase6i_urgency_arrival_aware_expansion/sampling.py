"""Candidate subset for one decision. Sampling scores are not the new labels."""
from __future__ import annotations

import hashlib

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6d.shadow import restore_state
from wt.benchmark_v2.phase6e.generators import feature_matrices, proposal_scores
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import GENERATOR, PRIMARY_TIMES
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.candidates import load_candidates
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.sampling import select_candidates
from wt.benchmark_v2.phase6i_urgency_stage_a.ledger import AdmissionAgeLedger
from wt.benchmark_v2.phase6i_urgency_tiebreak_closedloop.pool import legal_pool

from .protocol import LABEL_SEED, MAX_CANDIDATES, horizon_for


def _subset_hash(assignments: list[np.ndarray]) -> str:
    raw = np.ascontiguousarray(np.vstack(assignments), dtype=np.int64).tobytes()
    return hashlib.sha256(raw).hexdigest()


def _assignment_text(values) -> str:
    return "|".join(str(int(item)) for item in np.asarray(values, dtype=np.int64).reshape(-1))


def _env_at(scenario: int, env_snap: dict):
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=int(scenario))
    restore_state(env, env_snap)
    return env


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


def _urgency_start(ledger_snap: dict, assignment: np.ndarray, q_bits: np.ndarray, tol_bits: float, time_step: int) -> float:
    ledger = AdmissionAgeLedger(int(np.asarray(q_bits).size))
    ledger.restore(ledger_snap)
    view = ledger.view(int(time_step), np.asarray(q_bits, dtype=np.float64), float(tol_bits))
    urgency = np.asarray(view["urgency"], dtype=np.float64)
    total = 0.0
    for buoy in np.asarray(assignment, dtype=np.int64).reshape(-1):
        if int(buoy) >= 0:
            total += float(urgency[int(buoy)])
    return float(total)


def select_state(scenario: int, state: dict) -> tuple[list[dict], dict]:
    """Fix 128 candidates before either arrival arm is rolled out."""
    time_step = int(state["time_step"])
    if time_step in PRIMARY_TIMES:
        table = load_candidates(int(scenario), time_step, state["env_snap"], {
            "checksum_candidates": state["manifest_checksum_candidates"],
            "candidate_count": state["manifest_candidate_count"],
            "illegal": state["manifest_illegal"],
        })
        source = "stored_j15"
        scores = np.asarray(table["j15_bits"], dtype=np.float64)
        ranks = np.asarray(table["utility_rank"], dtype=np.int64)
        tie_group = np.asarray(table["tie_group_id"], dtype=np.int64)
        stored_j15 = scores
        pool_hash = str(table["pool_hash"])
        pool_count = int(table["candidate_count"])
        checksum = str(table["checksum_candidates"])
        assignment = np.asarray(table["assignment"], dtype=np.int64)
        candidate_id = np.asarray(table["candidate_id"], dtype=np.int64)
    else:
        env = _env_at(int(scenario), state["env_snap"])
        if int(env.t) != time_step:
            raise RuntimeError("restored snapshot is not at the requested decision time")
        table = legal_pool(env)
        source = "additive_g1_rate"
        scores = _additive_rate(env, table["assignment"])
        ranks = _dense_rank(scores, table["candidate_id"])
        tie_group = np.full(table["candidate_id"].shape, -1, dtype=np.int64)
        stored_j15 = None
        pool_hash = str(table["pool_hash"])
        pool_count = int(table["pool_count"])
        checksum = ""
        assignment = np.asarray(table["assignment"], dtype=np.int64)
        candidate_id = np.asarray(table["candidate_id"], dtype=np.int64)
    picked = select_candidates(
        candidate_id,
        scores,
        ranks,
        scenario_id=int(scenario),
        time_step=time_step,
        seed=int(LABEL_SEED),
    )
    if int(picked["count"]) != int(MAX_CANDIDATES):
        raise RuntimeError(f"subset size is {picked['count']}, not {MAX_CANDIDATES}")
    q_bits = np.asarray(state["env_snap"]["q_bits"], dtype=np.float64)
    rows = []
    assignments = []
    for item in picked["rows"]:
        position = int(item["row_index"])
        chosen = np.asarray(assignment[position], dtype=np.int64).copy()
        assignments.append(chosen)
        stored_value = None if stored_j15 is None else float(stored_j15[position])
        rows.append(
            {
                "scenario_id": int(scenario),
                "time_step": time_step,
                "candidate_id": int(item["candidate_id"]),
                "assignment": _assignment_text(chosen),
                "assignment_vector": chosen,
                "j15_bits": stored_value,
                "sampling_score": float(scores[position]),
                "sampling_source": source,
                "utility_rank": int(ranks[position]),
                "tie_group_id": int(tie_group[position]),
                "selection": str(item["selection"]),
                "stratum": int(item["stratum"]),
                "filled": int(item["filled"]),
                "idle_count": int(np.sum(chosen < 0)),
                "urgency_start_assigned": _urgency_start(
                    state["ledger_snap"], chosen, q_bits, float(state["tol_bits"]), time_step
                ),
                "state_hash": str(state["state_hash"]),
                "pool_hash": pool_hash,
                "row_index": position,
                "horizon": horizon_for(time_step),
            }
        )
    digest = _subset_hash(assignments)
    for row in rows:
        row["subset_hash"] = digest
    return rows, {
        "pool_hash": pool_hash,
        "pool_count": pool_count,
        "subset_hash": digest,
        "checksum_candidates": checksum,
        "sampling_source": source,
    }
