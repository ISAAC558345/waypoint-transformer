"""Static payload proxy plus a bounded tag term. This module does not step or draw."""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from wt.benchmark_v2.phase6e.generators import is_legal
from wt.benchmark_v2.phase6i_priority_tag_assignment_feasibility.candidates import assignment_key, pool_sha256

from .protocol import DUST, TAG_WEIGHT


def active_mask(assignment) -> tuple[int, ...]:
    return tuple(1 if int(item) >= 0 else 0 for item in assignment_key(assignment))


def _as_matrix(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise RuntimeError(f"{name} must be a UAV-by-buoy matrix")
    if not np.all(np.isfinite(array)):
        raise RuntimeError(f"{name} is not finite")
    return array


def _bucket_rows(buckets, decision_time: int) -> list[dict]:
    rows = []
    for bucket in buckets:
        if int(bucket["admission_step"]) > int(decision_time):
            continue
        rows.append(bucket)
    rows.sort(key=lambda bucket: (int(bucket["admission_step"]), int(bucket["insert_index"])))
    return rows


def link_components(rate, eta, queue, buckets_by_buoy, *, horizon: float, delta_t: float, decision_time: int) -> tuple[np.ndarray, np.ndarray]:
    """Finite-window service proxy and the FIFO-prefix tagged part of that proxy."""
    rate_matrix = _as_matrix(rate, "rate")
    eta_matrix = _as_matrix(eta, "eta")
    if rate_matrix.shape != eta_matrix.shape:
        raise RuntimeError("rate and eta shapes differ")
    n_uav, n_buoy = rate_matrix.shape
    queue_vector = np.asarray(queue, dtype=np.float64).reshape(-1)
    if queue_vector.size != n_buoy or len(buckets_by_buoy) != n_buoy:
        raise RuntimeError("queue width does not match the buoy count")
    if not np.all(np.isfinite(queue_vector)):
        raise RuntimeError("queue is not finite")
    if float(horizon) < 0.0 or float(delta_t) < 0.0:
        raise RuntimeError("horizon or time step is negative")
    capacity = np.zeros((n_uav, n_buoy), dtype=np.float64)
    tagged = np.zeros((n_uav, n_buoy), dtype=np.float64)
    for buoy in range(n_buoy):
        known = _bucket_rows(buckets_by_buoy[buoy], decision_time)
        queued = float(sum(float(bucket["bits_remaining"]) for bucket in known))
        if abs(queued - float(queue_vector[buoy])) > 1.0:
            raise RuntimeError("ledger does not match q_bits")
        for uav in range(n_uav):
            window = max(0.0, float(horizon) - float(eta_matrix[uav, buoy])) * float(delta_t) * max(float(rate_matrix[uav, buoy]), 0.0)
            amount = min(max(float(queue_vector[buoy]), 0.0), window)
            capacity[uav, buoy] = amount
            remaining = amount
            taken_tag = 0.0
            for bucket in known:
                if remaining <= float(DUST):
                    break
                available = float(bucket["bits_remaining"])
                if available <= float(DUST):
                    continue
                take = min(available, remaining)
                remaining -= take
                if int(bucket["priority_tag"]) == 1:
                    taken_tag += take
            tagged[uav, buoy] = taken_tag
            if tagged[uav, buoy] > capacity[uav, buoy] + 1.0e-6:
                raise RuntimeError("tagged proxy exceeded the link capacity")
    return capacity, tagged


def _dedup(pool):
    unique = []
    first_index = {}
    width = None
    for position, assignment in enumerate(pool):
        key = assignment_key(assignment)
        if width is None:
            width = len(key)
        if len(key) != width:
            raise RuntimeError("assignment width changed inside one pool")
        if key not in first_index:
            first_index[key] = int(position)
            unique.append(key)
    return unique, first_index


def _score_assignment(key, capacity, tagged) -> tuple[float, float]:
    payload = 0.0
    known_tag = 0.0
    for uav, buoy in enumerate(key):
        if int(buoy) < 0:
            continue
        payload += float(capacity[uav, int(buoy)])
        known_tag += float(tagged[uav, int(buoy)])
    if not math.isfinite(payload) or not math.isfinite(known_tag) or payload < 0.0 or known_tag < 0.0:
        raise RuntimeError("assignment proxy is outside the registered bounds")
    if known_tag > payload + 1.0e-6:
        raise RuntimeError("assignment tag proxy exceeded the payload proxy")
    return float(payload), float(known_tag)


def select_rule(pool, baseline_index: int, rate, eta, queue, buckets_by_buoy, *, horizon: float, delta_t: float, decision_time: int, weight: float) -> dict:
    """Choose N, M, and T from one same-mask pool. No scenario id and no RNG."""
    if not pool:
        raise RuntimeError("candidate pool is empty")
    index = int(baseline_index)
    if index < 0 or index >= len(pool):
        raise RuntimeError("baseline index is outside the candidate pool")
    if not math.isfinite(float(weight)) or float(weight) < 0.0:
        raise RuntimeError("tag weight is not a finite non-negative number")
    numpy_state = np.random.get_state()
    numpy_key = np.array(numpy_state[1], copy=True)
    capacity, tagged = link_components(
        rate, eta, queue, buckets_by_buoy, horizon=horizon, delta_t=delta_t, decision_time=decision_time,
    )
    n_uav, n_buoy = capacity.shape
    unique, first_index = _dedup(pool)
    baseline = assignment_key(pool[index])
    if len(baseline) != n_uav:
        raise RuntimeError("assignment width does not match the rate matrix")
    if not is_legal(np.asarray(baseline, dtype=np.int64), n_uav, n_buoy):
        raise RuntimeError("natural assignment is illegal")
    mask = active_mask(baseline)
    eligible = [key for key in unique if active_mask(key) == mask]
    if baseline not in eligible:
        raise RuntimeError("natural assignment left the same-mask set")
    scored = {}
    rows = []
    for key in eligible:
        if not is_legal(np.asarray(key, dtype=np.int64), n_uav, n_buoy):
            raise RuntimeError("same-mask candidate is illegal")
        payload, known_tag = _score_assignment(key, capacity, tagged)
        scored[key] = (payload, known_tag)
        rows.append(
            {
                "pool_index": int(first_index[key]),
                "assignment": [int(item) for item in key],
                "active_mask": [int(item) for item in mask],
                "P": repr(payload),
                "K": repr(known_tag),
                "score_m": repr(payload),
                "score_t": repr(payload + float(weight) * known_tag),
            }
        )
    rows.sort(key=lambda item: int(item["pool_index"]))

    def pick(score_of) -> tuple[int, ...]:
        return min(eligible, key=lambda key: (-float(score_of(key)), int(first_index[key])))

    masked = pick(lambda key: scored[key][0])
    tagged_choice = pick(lambda key: scored[key][0] + float(weight) * scored[key][1])
    numpy_after = np.random.get_state()
    if (
        numpy_after[0] != numpy_state[0]
        or numpy_after[2] != numpy_state[2]
        or numpy_after[3] != numpy_state[3]
        or numpy_after[4] != numpy_state[4]
        or not np.array_equal(numpy_after[1], numpy_key)
    ):
        raise RuntimeError("fixed rule consumed the global numpy stream")

    def choice(name: str, key: tuple[int, ...], pool_index: int) -> dict:
        payload, known_tag = scored[key]
        return {
            "role": name,
            "pool_index": int(pool_index),
            "assignment": [int(item) for item in key],
            "active_mask": [int(item) for item in active_mask(key)],
            "P": repr(payload),
            "K": repr(known_tag),
            "score_m": repr(payload),
            "score_t": repr(payload + float(weight) * known_tag),
        }

    return {
        "weight": float(weight),
        "pool_sha256": pool_sha256(pool),
        "pool_size": int(len(pool)),
        "unique_count": int(len(unique)),
        "eligible_count": int(len(eligible)),
        "baseline_pool_index": index,
        "baseline_assignment": [int(item) for item in baseline],
        "baseline_mask": [int(item) for item in mask],
        "natural": choice("N", baseline, index),
        "masked": choice("M", masked, int(first_index[masked])),
        "tagged": choice("T", tagged_choice, int(first_index[tagged_choice])),
        "eligible": rows,
        "tagged_differs_from_masked": bool(tagged_choice != masked),
    }


def select_registered(pool, baseline_index: int, rate, eta, queue, buckets_by_buoy, *, horizon: float, delta_t: float, decision_time: int) -> dict:
    """Production entry. The weight is the registered constant, not a searched value."""
    if float(TAG_WEIGHT) != 0.001:
        raise RuntimeError("registered tag weight changed")
    return select_rule(
        pool,
        baseline_index,
        rate,
        eta,
        queue,
        buckets_by_buoy,
        horizon=horizon,
        delta_t=delta_t,
        decision_time=decision_time,
        weight=float(TAG_WEIGHT),
    )


def choice_sha256(records: list[dict]) -> str:
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
