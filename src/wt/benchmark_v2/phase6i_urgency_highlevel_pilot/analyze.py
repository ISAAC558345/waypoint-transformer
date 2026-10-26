"""Scale and rank comparison. lambda_H is intentionally left unset."""
from __future__ import annotations

import numpy as np

from .protocol import EPS_BITS, LAMBDA_H, PAYLOAD_SCALE


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    start = 0
    while start < order.size:
        stop = start + 1
        while stop < order.size and float(values[order[stop]]) == float(values[order[start]]):
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    x = _average_ranks(np.asarray(left, dtype=np.float64))
    y = _average_ranks(np.asarray(right, dtype=np.float64))
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def reversal_rate(payload_bits: np.ndarray, urgent_bits: np.ndarray) -> dict:
    """Pair order among candidates whose payload labels differ by more than one bit."""
    payload = np.asarray(payload_bits, dtype=np.float64)
    urgent = np.asarray(urgent_bits, dtype=np.float64)
    compared = 0
    reversed_pairs = 0
    urgent_ties = 0
    for left in range(payload.size):
        for right in range(left + 1, payload.size):
            payload_gap = float(payload[left] - payload[right])
            if abs(payload_gap) <= float(EPS_BITS):
                continue
            compared += 1
            urgent_gap = float(urgent[left] - urgent[right])
            if abs(urgent_gap) <= float(EPS_BITS):
                urgent_ties += 1
                continue
            if (payload_gap > 0.0) != (urgent_gap > 0.0):
                reversed_pairs += 1
    return {
        "payload_separated_pairs": int(compared),
        "reversed_pairs": int(reversed_pairs),
        "urgent_tie_pairs": int(urgent_ties),
        "reversal_rate": 0.0 if compared == 0 else float(reversed_pairs / compared),
    }


def _distribution(values: np.ndarray) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p50": float(np.median(array)),
        "max": float(np.max(array)),
    }


def state_rank_summary(rows: list[dict]) -> dict:
    payload_bits = np.asarray([float(row["payload_bits"]) for row in rows], dtype=np.float64)
    urgent_bits = np.asarray([float(row["urgent_service_bits"]) for row in rows], dtype=np.float64)
    payload_term = payload_bits / float(PAYLOAD_SCALE)
    urgent_term = urgent_bits / float(PAYLOAD_SCALE)
    pairs = reversal_rate(payload_bits, urgent_bits)
    coefficient = spearman(payload_term, urgent_term)
    payload_order = np.argsort(-payload_bits, kind="mergesort")
    urgent_order = np.argsort(-urgent_bits, kind="mergesort")
    return {
        "count": int(len(rows)),
        "payload_term": _distribution(payload_term),
        "urgent_service_term": _distribution(urgent_term),
        "mean_urgent_over_mean_payload": None
        if float(np.mean(payload_term)) == 0.0
        else float(np.mean(urgent_term) / np.mean(payload_term)),
        "spearman": coefficient,
        "reversal_rate": pairs["reversal_rate"],
        "payload_separated_pairs": pairs["payload_separated_pairs"],
        "reversed_pairs": pairs["reversed_pairs"],
        "urgent_tie_pairs": pairs["urgent_tie_pairs"],
        "top1_payload_candidate_id": int(rows[int(payload_order[0])]["candidate_id"]) if rows else None,
        "top1_urgent_candidate_id": int(rows[int(urgent_order[0])]["candidate_id"]) if rows else None,
        "top1_changed": bool(rows) and int(payload_order[0]) != int(urgent_order[0]),
        "lambda_H": LAMBDA_H,
    }


def macro_rank_summary(per_state: list[dict]) -> dict:
    spearman_values = [float(row["spearman"]) for row in per_state if row["spearman"] is not None]
    reversal_values = [float(row["reversal_rate"]) for row in per_state]
    return {
        "states": int(len(per_state)),
        "mean_spearman": None if not spearman_values else float(np.mean(spearman_values)),
        "min_spearman": None if not spearman_values else float(np.min(spearman_values)),
        "mean_reversal_rate": None if not reversal_values else float(np.mean(reversal_values)),
        "max_reversal_rate": None if not reversal_values else float(np.max(reversal_values)),
        "states_with_top1_change": int(sum(bool(row["top1_changed"]) for row in per_state)),
        "lambda_H": LAMBDA_H,
    }
