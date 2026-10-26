"""Rank one decision under two arrival conventions. Horizons are not pooled here."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.phase6i_urgency_arrival_aware_labels.analyze import quartile_labels
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
    kendall = None if compared == 0 else float((compared - (2 * inverted)) / compared)
    return {
        "compared_pairs": int(compared),
        "inverted_pairs": int(inverted),
        "tied_pairs": int(tied),
        "inversion_rate": 0.0 if compared == 0 else float(inverted / compared),
        "kendall_tau": kendall,
    }


def _within(rows: list[dict], left_id: int, right_id: int, key: str) -> bool:
    left = next(row for row in rows if int(row["candidate_id"]) == int(left_id))
    right = next(row for row in rows if int(row["candidate_id"]) == int(right_id))
    return abs(float(left[key]) - float(right[key])) <= float(EPS_BITS)


def _brief(row: dict, label_key: str) -> dict:
    return {
        "candidate_id": int(row["candidate_id"]),
        "assignment": str(row["assignment"]),
        "idle_count": int(row["idle_count"]),
        "zero_payload_bits": float(row["zero_payload_bits"]),
        "real_payload_bits": float(row["real_payload_bits"]),
        "zero_urgent_service_bits": float(row["zero_urgent_service_bits"]),
        "real_urgent_service_bits": float(row["real_urgent_service_bits"]),
        "reference_bits": float(row[label_key]),
    }


def pareto_frontier(rows: list[dict]) -> dict:
    """Real-arrival payload and urgent service. A 1-bit gap is not a strict improvement."""
    keep = []
    for row in rows:
        dominated = False
        for other in rows:
            if int(other["candidate_id"]) == int(row["candidate_id"]):
                continue
            payload_not_worse = float(other["real_payload_bits"]) >= float(row["real_payload_bits"]) - float(EPS_BITS)
            urgent_not_worse = float(other["real_urgent_service_bits"]) >= float(row["real_urgent_service_bits"]) - float(EPS_BITS)
            payload_better = float(other["real_payload_bits"]) > float(row["real_payload_bits"]) + float(EPS_BITS)
            urgent_better = float(other["real_urgent_service_bits"]) > float(row["real_urgent_service_bits"]) + float(EPS_BITS)
            if payload_not_worse and urgent_not_worse and (payload_better or urgent_better):
                dominated = True
                break
        if not dominated:
            keep.append(row)
    payloads = [float(row["real_payload_bits"]) for row in keep]
    urgents = [float(row["real_urgent_service_bits"]) for row in keep]
    return {
        "count": int(len(keep)),
        "payload_min_bits": None if not payloads else float(min(payloads)),
        "payload_max_bits": None if not payloads else float(max(payloads)),
        "urgent_min_bits": None if not urgents else float(min(urgents)),
        "urgent_max_bits": None if not urgents else float(max(urgents)),
        "candidate_ids": [int(row["candidate_id"]) for row in sorted(keep, key=lambda item: int(item["candidate_id"]))],
    }


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
    j15_gaps = [row["zero_payload_gap_vs_j15_bits"] for row in rows if row["zero_payload_gap_vs_j15_bits"] is not None]
    return {
        "count": int(len(rows)),
        "horizon": int(rows[0]["horizon"]),
        "payload_spearman": spearman(
            np.asarray([row["zero_payload_bits"] for row in rows]),
            np.asarray([row["real_payload_bits"] for row in rows]),
        ),
        "urgent_spearman": spearman(
            np.asarray([row["zero_urgent_service_bits"] for row in rows]),
            np.asarray([row["real_urgent_service_bits"] for row in rows]),
        ),
        "payload_urgent_spearman": spearman(
            np.asarray([row["real_payload_bits"] for row in rows]),
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
        "payload_top1_regret_bits": float(real_payload["real_payload_bits"]) - float(zero_payload["real_payload_bits"]),
        "zero_urgent_top1": int(zero_urgent["candidate_id"]),
        "real_urgent_top1": int(real_urgent["candidate_id"]),
        "urgent_top1_changed": int(zero_urgent["candidate_id"]) != int(real_urgent["candidate_id"]),
        "urgent_top1_within_one_bit": bool(
            _within(rows, int(zero_urgent["candidate_id"]), int(real_urgent["candidate_id"]), "zero_urgent_service_bits")
            and _within(rows, int(zero_urgent["candidate_id"]), int(real_urgent["candidate_id"]), "real_urgent_service_bits")
        ),
        "urgent_top1_regret_bits": float(real_urgent["real_urgent_service_bits"]) - float(zero_urgent["real_urgent_service_bits"]),
        "payload_delta_mean_bits": float(np.mean([row["payload_delta_bits"] for row in rows])),
        "urgent_delta_mean_bits": float(np.mean([row["urgent_delta_bits"] for row in rows])),
        "dropped_delta_mean_bits": float(np.mean([row["dropped_delta_bits"] for row in rows])),
        "final_queue_delta_mean_bits": float(np.mean([row["final_queue_delta_bits"] for row in rows])),
        "real_generated_mean_bits": float(np.mean([row["real_generated_bits"] for row in rows])),
        "real_admitted_mean_bits": float(np.mean([row["real_admitted_bits"] for row in rows])),
        "real_dropped_mean_bits": float(np.mean([row["real_dropped_bits"] for row in rows])),
        "max_abs_j15_gap_bits": None if not j15_gaps else float(max(abs(float(item)) for item in j15_gaps)),
        "max_abs_arrival_account_gap_bits": float(max(abs(float(row["arrival_account_gap_bits"])) for row in rows)),
        "zero_payload_top": _brief(zero_payload, "zero_payload_bits"),
        "real_payload_top": _brief(real_payload, "real_payload_bits"),
        "zero_urgent_top": _brief(zero_urgent, "zero_urgent_service_bits"),
        "real_urgent_top": _brief(real_urgent, "real_urgent_service_bits"),
        "pareto": pareto_frontier(rows),
    }


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


__all__ = ["pairwise_inversion", "pareto_frontier", "quartile_labels", "rank_state", "stratum_summary"]
