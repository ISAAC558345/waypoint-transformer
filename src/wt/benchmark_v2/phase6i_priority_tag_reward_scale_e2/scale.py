"""Offline lambda tables for frozen trajectories. No lambda is selected."""
from __future__ import annotations

import json

import numpy as np

from .protocol import (
    E1_RELATIVE_TOLERANCE,
    E1_REWARD_SCALE,
    LAMBDA_CANDIDATES,
    LAMBDA_REFERENCE,
    LAMBDA_SELECTED,
    MEAN_BONUS_OVER_PAYLOAD_MAX,
    P95_BONUS_OVER_PENALTY_MAX,
    PAYLOAD_TIE_BITS,
    PENALTY_DUST,
    RATIO_SPAN_MAX,
)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), float(quantile)))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def material_reorder(payload: list[float], bonus: list[float]) -> dict[str, bool]:
    """A material swap moves two steps whose payload terms differ by more than one bit."""
    base = np.asarray(payload, dtype=np.float64)
    extra = np.asarray(bonus, dtype=np.float64)
    total = base + extra
    order_base = np.argsort(base, kind="mergesort")
    order_total = np.argsort(total, kind="mergesort")
    material = False
    for left in range(int(base.size)):
        for right in range(left + 1, int(base.size)):
            gap = float(base[left] - base[right])
            moved = float(total[left] - total[right])
            if abs(gap) > float(PAYLOAD_TIE_BITS) and gap * moved < 0.0:
                material = True
                break
        if material:
            break
    return {
        "order_changed": bool(not np.array_equal(order_base, order_total)),
        "material_order_changed": bool(material),
    }


def _sign_flips(base: list[float], augmented: list[float]) -> int:
    flips = 0
    for left, right in zip(base, augmented):
        if (float(left) > 0.0) != (float(right) > 0.0):
            flips += 1
    return int(flips)


def _ratio_stats(bonus: list[float], denominator: list[float]) -> dict:
    finite: list[float] = []
    unbounded = 0
    for left, right in zip(bonus, denominator):
        if float(right) > float(PENALTY_DUST):
            finite.append(float(left) / float(right))
        elif float(left) > float(PENALTY_DUST):
            unbounded += 1
    return {
        "mean": _mean(finite),
        "p50": _percentile(finite, 50),
        "p95": _percentile(finite, 95),
        "p99": _percentile(finite, 99),
        "max": float(max(finite)) if finite else None,
        "finite_steps": int(len(finite)),
        "unbounded_steps": int(unbounded),
    }


def load_e1_fraction() -> dict:
    payload = json.loads(E1_REWARD_SCALE.read_text(encoding="utf-8"))
    communication = float(payload["smoke_payload_term_mean"])
    urgency = float(payload["smoke_urgency_term_mean"])
    fraction = 0.0 if communication == 0.0 else urgency / communication
    return {
        "source": "results/phase6i_priority_tag_prototype/reward_scale.json",
        "smoke_payload_term_mean": communication,
        "smoke_urgency_term_mean": urgency,
        "urgency_over_payload": float(fraction),
        "old_buoy_urgency_ratio_used": False,
    }


def _episode_fraction(rows: list[dict]) -> float | None:
    served = float(sum(float(row["served_bits"]) for row in rows))
    if served <= 1.0:
        return None
    urgency = float(sum(float(row["urgency_served_bits"]) for row in rows))
    return urgency / served


def summarize(episodes: list[dict]) -> dict:
    """Build step-free scale evidence from completed ledger episodes."""
    rows = [row for episode in episodes for row in episode["rows"]]
    e1 = load_e1_fraction()
    fractions = []
    fraction_by_scenario = []
    for episode in episodes:
        fraction = _episode_fraction(episode["rows"])
        fraction_by_scenario.append({"scenario_id": int(episode["scenario_id"]), "urgency_over_served": fraction})
        if fraction is not None:
            fractions.append(float(fraction))
    if len(fractions) == len(episodes) and fractions and min(fractions) > 0.0:
        ratio_span = float(max(fractions) / min(fractions))
    else:
        ratio_span = None
    frozen_fraction = None if not fractions else float(np.mean(np.asarray(fractions, dtype=np.float64)))
    if frozen_fraction is None or float(e1["urgency_over_payload"]) == 0.0:
        relative_gap = None
    else:
        relative_gap = abs(float(frozen_fraction) - float(e1["urgency_over_payload"])) / abs(float(e1["urgency_over_payload"]))
    by_lambda = []
    candidate_ok: dict[str, bool] = {}
    for lam in (*LAMBDA_CANDIDATES, LAMBDA_REFERENCE):
        bonus = [float(lam) * float(row["urgency_term"]) for row in rows]
        reward = [float(row["base_reward"]) + bonus[index] for index, row in enumerate(rows)]
        payload = [float(row["payload_term"]) for row in rows]
        penalty = [float(row["penalty"]) for row in rows]
        base = [float(row["base_reward"]) for row in rows]
        pooled = material_reorder(payload, bonus)
        episode_material = []
        episode_order = []
        episode_sign = []
        episode_bonus = []
        episode_reward = []
        episode_bounds_ok = []
        for episode in episodes:
            local_bonus = [float(lam) * float(row["urgency_term"]) for row in episode["rows"]]
            local_payload = [float(row["payload_term"]) for row in episode["rows"]]
            local_penalty = [float(row["penalty"]) for row in episode["rows"]]
            local_base = [float(row["base_reward"]) for row in episode["rows"]]
            flags = material_reorder(local_payload, local_bonus)
            local_penalty_stats = _ratio_stats(local_bonus, local_penalty)
            local_payload_stats = _ratio_stats(local_bonus, local_payload)
            episode_material.append(bool(flags["material_order_changed"]))
            episode_order.append(bool(flags["order_changed"]))
            episode_sign.append(_sign_flips(local_base, [local_base[index] + local_bonus[index] for index in range(len(local_base))]))
            episode_bonus.append(float(sum(local_bonus)))
            episode_reward.append(float(sum(local_base) + sum(local_bonus)))
            episode_bounds_ok.append(
                bool(
                    local_penalty_stats["p95"] is not None
                    and float(local_penalty_stats["p95"]) <= float(P95_BONUS_OVER_PENALTY_MAX)
                    and int(local_penalty_stats["unbounded_steps"]) == 0
                    and local_payload_stats["mean"] is not None
                    and float(local_payload_stats["mean"]) <= float(MEAN_BONUS_OVER_PAYLOAD_MAX)
                    and not flags["material_order_changed"]
                )
            )
        payload_totals = [float(sum(float(row["payload_term"]) for row in episode["rows"])) for episode in episodes]
        episode_rank = material_reorder(payload_totals, episode_bonus)
        penalty_stats = _ratio_stats(bonus, penalty)
        payload_stats = _ratio_stats(bonus, payload)
        supported = bool(
            float(lam) in LAMBDA_CANDIDATES
            and float(lam) > 0.0
            and bool(episode_bounds_ok)
            and all(episode_bounds_ok)
            and not pooled["material_order_changed"]
        )
        candidate_ok[f"{float(lam):.4f}"] = supported
        by_lambda.append(
            {
                "lambda_u": float(lam),
                "selected": False,
                "role": "high_risk_reference" if float(lam) == float(LAMBDA_REFERENCE) else "candidate",
                "supported_by_preregistered_bound": supported,
                "steps": int(len(rows)),
                "reward_mean": _mean(reward),
                "reward_p50": _percentile(reward, 50),
                "reward_p95": _percentile(reward, 95),
                "reward_p99": _percentile(reward, 99),
                "reward_max": float(max(reward)) if reward else None,
                "bonus_mean": _mean(bonus),
                "bonus_p50": _percentile(bonus, 50),
                "bonus_p95": _percentile(bonus, 95),
                "bonus_p99": _percentile(bonus, 99),
                "bonus_max": float(max(bonus)) if bonus else None,
                "bonus_over_payload": payload_stats,
                "bonus_over_penalty": penalty_stats,
                "pooled_order_changed": bool(pooled["order_changed"]),
                "pooled_material_order_changed": bool(pooled["material_order_changed"]),
                "episodes_with_tie_only_reorder": int(sum(order and not material for order, material in zip(episode_order, episode_material))),
                "episodes_with_material_reorder": int(sum(episode_material)),
                "episodes_within_preregistered_bound": int(sum(episode_bounds_ok)),
                "episode_rank_changed": bool(episode_rank["order_changed"]),
                "episode_rank_material_changed": bool(episode_rank["material_order_changed"]),
                "sign_flips": int(_sign_flips(base, reward)),
                "episode_sign_flips": episode_sign,
                "episode_bonus_sum": episode_bonus,
                "episode_reward_sum": episode_reward,
            }
        )
    positive = [float(lam) for lam in LAMBDA_CANDIDATES if float(lam) > 0.0 and candidate_ok[f"{float(lam):.4f}"]]
    return {
        "lambda_selected": LAMBDA_SELECTED,
        "e1_smoke": e1,
        "frozen_urgency_over_served_mean": frozen_fraction,
        "urgency_fraction_by_scenario": fraction_by_scenario,
        "urgency_fraction_span": ratio_span,
        "ratio_span_max": float(RATIO_SPAN_MAX),
        "ratio_span_ok": bool(ratio_span is not None and float(ratio_span) <= float(RATIO_SPAN_MAX)),
        "e1_relative_gap": relative_gap,
        "e1_relative_tolerance": float(E1_RELATIVE_TOLERANCE),
        "e1_scale_close": bool(relative_gap is not None and float(relative_gap) <= float(E1_RELATIVE_TOLERANCE)),
        "positive_candidates_within_bound": positive,
        "positive_candidate_supported": bool(positive),
        "suggested_upper_candidate": None if not positive else float(max(positive)),
        "reference_not_trainable": float(LAMBDA_REFERENCE),
        "bounds": {
            "p95_bonus_over_penalty_max": float(P95_BONUS_OVER_PENALTY_MAX),
            "mean_bonus_over_payload_max": float(MEAN_BONUS_OVER_PAYLOAD_MAX),
            "penalty_dust": float(PENALTY_DUST),
            "payload_tie": float(PAYLOAD_TIE_BITS),
        },
        "by_lambda": by_lambda,
        "old_buoy_urgency_ratio_used": False,
    }
