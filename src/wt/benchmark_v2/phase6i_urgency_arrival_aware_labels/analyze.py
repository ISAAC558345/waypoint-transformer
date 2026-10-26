"""Rank the same 128 candidates under two arrival conventions."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.analyze import spearman

from .protocol import EPS_BITS


def _top(rows: list[dict], key: str) -> dict:
    best = max(float(row[key]) for row in rows)
    group = [row for row in rows if abs(float(row[key]) - best) <= float(EPS_BITS)]
    return min(group, key=lambda row: int(row["candidate_id"]))


def pairwise_inversion(left: np.ndarray, right: np.ndarray) -> dict:
    """Count sign disagreements after ignoring gaps inside 1 bit."""
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    compared = 0
    inverted = 0
    tied = 0
    for index in range(first.size):
        for other in range(index + 1, first.size):
            gap_left = float(first[index] - first[other])
            gap_right = float(second[index] - second[other])
            if abs(gap_left) <= float(EPS_BITS) or abs(gap_right) <= float(EPS_BITS):
                tied += 1
                continue
            compared += 1
            if (gap_left > 0.0) != (gap_right > 0.0):
                inverted += 1
    return {
        "compared_pairs": int(compared),
        "inverted_pairs": int(inverted),
        "tied_pairs": int(tied),
        "inversion_rate": 0.0 if compared == 0 else float(inverted / compared),
    }


def _within(rows: list[dict], left_id: int, right_id: int, key: str) -> bool:
    left = next(row for row in rows if int(row["candidate_id"]) == int(left_id))
    right = next(row for row in rows if int(row["candidate_id"]) == int(right_id))
    return abs(float(left[key]) - float(right[key])) <= float(EPS_BITS)


def rank_state(rows: list[dict]) -> dict:
    zero_payload = _top(rows, "zero_payload_bits")
    real_payload = _top(rows, "real_payload_bits")
    zero_urgent = _top(rows, "zero_urgent_service_bits")
    real_urgent = _top(rows, "real_urgent_service_bits")
    payload_pairs = pairwise_inversion(
        np.asarray([row["zero_payload_bits"] for row in rows]),
        np.asarray([row["real_payload_bits"] for row in rows]),
    )
    urgent_pairs = pairwise_inversion(
        np.asarray([row["zero_urgent_service_bits"] for row in rows]),
        np.asarray([row["real_urgent_service_bits"] for row in rows]),
    )
    payload_regret = float(real_payload["real_payload_bits"]) - float(zero_payload["real_payload_bits"])
    urgent_regret = float(real_urgent["real_urgent_service_bits"]) - float(zero_urgent["real_urgent_service_bits"])
    return {
        "count": int(len(rows)),
        "payload_spearman": spearman(
            np.asarray([row["zero_payload_bits"] for row in rows]),
            np.asarray([row["real_payload_bits"] for row in rows]),
        ),
        "urgent_spearman": spearman(
            np.asarray([row["zero_urgent_service_bits"] for row in rows]),
            np.asarray([row["real_urgent_service_bits"] for row in rows]),
        ),
        "payload_inversion": payload_pairs,
        "urgent_inversion": urgent_pairs,
        "zero_payload_top1": int(zero_payload["candidate_id"]),
        "real_payload_top1": int(real_payload["candidate_id"]),
        "payload_top1_changed": int(zero_payload["candidate_id"]) != int(real_payload["candidate_id"]),
        "payload_top1_within_one_bit": bool(
            _within(rows, int(zero_payload["candidate_id"]), int(real_payload["candidate_id"]), "zero_payload_bits")
            and _within(rows, int(zero_payload["candidate_id"]), int(real_payload["candidate_id"]), "real_payload_bits")
        ),
        "payload_top1_regret_bits": float(payload_regret),
        "zero_urgent_top1": int(zero_urgent["candidate_id"]),
        "real_urgent_top1": int(real_urgent["candidate_id"]),
        "urgent_top1_changed": int(zero_urgent["candidate_id"]) != int(real_urgent["candidate_id"]),
        "urgent_top1_within_one_bit": bool(
            _within(rows, int(zero_urgent["candidate_id"]), int(real_urgent["candidate_id"]), "zero_urgent_service_bits")
            and _within(rows, int(zero_urgent["candidate_id"]), int(real_urgent["candidate_id"]), "real_urgent_service_bits")
        ),
        "urgent_top1_regret_bits": float(urgent_regret),
        "payload_delta_mean_bits": float(np.mean([row["payload_delta_bits"] for row in rows])),
        "urgent_delta_mean_bits": float(np.mean([row["urgent_delta_bits"] for row in rows])),
        "dropped_delta_mean_bits": float(np.mean([row["dropped_delta_bits"] for row in rows])),
        "final_queue_delta_mean_bits": float(np.mean([row["final_queue_delta_bits"] for row in rows])),
        "max_abs_j15_gap_bits": float(max(abs(float(row["zero_payload_gap_vs_j15_bits"])) for row in rows)),
    }


def quartile_labels(values: list[float]) -> list[str]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return []
    edges = np.quantile(array, [0.25, 0.5, 0.75])
    labels = []
    for value in array:
        if float(value) <= float(edges[0]):
            labels.append("q1")
        elif float(value) <= float(edges[1]):
            labels.append("q2")
        elif float(value) <= float(edges[2]):
            labels.append("q3")
        else:
            labels.append("q4")
    return labels


def stratum_summary(rows: list[dict], key: str) -> list[dict]:
    """Describe slices. A slice is not a cause."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    output = []
    for name in sorted(groups):
        chosen = groups[name]
        urgent = pairwise_inversion(
            np.asarray([row["zero_urgent_service_bits"] for row in chosen]),
            np.asarray([row["real_urgent_service_bits"] for row in chosen]),
        )
        output.append(
            {
                "stratum": name,
                "count": int(len(chosen)),
                "payload_delta_mean_bits": float(np.mean([row["payload_delta_bits"] for row in chosen])),
                "urgent_delta_mean_bits": float(np.mean([row["urgent_delta_bits"] for row in chosen])),
                "dropped_delta_mean_bits": float(np.mean([row["dropped_delta_bits"] for row in chosen])),
                "urgent_inversion_rate": urgent["inversion_rate"],
                "urgent_compared_pairs": urgent["compared_pairs"],
            }
        )
    return output
