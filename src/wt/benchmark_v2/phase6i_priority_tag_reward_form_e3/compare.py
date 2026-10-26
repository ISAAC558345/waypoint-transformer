"""Offline comparison of additive and payload-bounded urgency bonuses."""
from __future__ import annotations

import numpy as np

from .protocol import (
    CROSSCHECK_ATOL,
    LAMBDA_FORM_A,
    LAMBDA_FORM_B,
    LAMBDA_REFERENCE,
    MEAN_BONUS_OVER_PAYLOAD_MAX,
    P95_BONUS_OVER_PENALTY_MAX,
    PAYLOAD_TIE,
    PENALTY_DUST,
    RHO_CANDIDATES,
    SIGNAL_RETAINED_MIN,
)


def uncapped_bonus(urgency_term: float, lambda_u: float) -> float:
    return float(lambda_u) * float(urgency_term)


def capped_bonus(urgency_term: float, payload_term: float, lambda_u: float, rho: float) -> float:
    """Cap the additive bonus by rho times this step's payload term. A zero payload term caps at zero."""
    raw = uncapped_bonus(urgency_term, lambda_u)
    if float(payload_term) == 0.0:
        return 0.0
    return float(min(raw, float(rho) * float(payload_term)))


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), float(quantile)))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def ratio_stats(bonus: list[float], denominator: list[float]) -> dict:
    """Ratios use only denominators above the dust. Tiny denominators stay as a count."""
    finite: list[float] = []
    near_zero = 0
    positive_on_near_zero = 0
    for left, right in zip(bonus, denominator):
        if float(right) > float(PENALTY_DUST):
            finite.append(float(left) / float(right))
        else:
            near_zero += 1
            if float(left) > float(PENALTY_DUST):
                positive_on_near_zero += 1
    return {
        "mean": _mean(finite),
        "p50": _percentile(finite, 50),
        "p95": _percentile(finite, 95),
        "p99": _percentile(finite, 99),
        "max": None if not finite else float(max(finite)),
        "finite_steps": int(len(finite)),
        "near_zero_denominator_steps": int(near_zero),
        "positive_bonus_on_near_zero_denominator": int(positive_on_near_zero),
    }


def payload_step_reorder(payload: list[float], bonus: list[float]) -> dict[str, bool]:
    """E2 definition: payload order versus payload plus bonus. A tie created from a strict gap is not material."""
    base = np.asarray(payload, dtype=np.float64)
    total = base + np.asarray(bonus, dtype=np.float64)
    material = False
    for left in range(int(base.size)):
        for right in range(left + 1, int(base.size)):
            gap = float(base[left] - base[right])
            moved = float(total[left] - total[right])
            if abs(gap) > float(PAYLOAD_TIE) and gap * moved < 0.0:
                material = True
                break
        if material:
            break
    return {
        "order_changed": bool(not np.array_equal(np.argsort(base, kind="mergesort"), np.argsort(total, kind="mergesort"))),
        "material_order_changed": bool(material),
    }


def reward_reorder(base: list[float], formed: list[float], payload: list[float]) -> dict[str, bool]:
    """Material means a base-reward order change between steps whose payload terms differ by more than one bit."""
    base_score = np.asarray(base, dtype=np.float64)
    form_score = np.asarray(formed, dtype=np.float64)
    payload_score = np.asarray(payload, dtype=np.float64)
    order_changed = bool(not np.array_equal(np.argsort(base_score, kind="mergesort"), np.argsort(form_score, kind="mergesort")))
    material = False
    for left in range(int(base_score.size)):
        for right in range(left + 1, int(base_score.size)):
            if abs(float(payload_score[left] - payload_score[right])) <= float(PAYLOAD_TIE):
                continue
            before = float(base_score[left] - base_score[right])
            after = float(form_score[left] - form_score[right])
            if (before > 0.0) != (after > 0.0):
                material = True
                break
        if material:
            break
    return {"order_changed": order_changed, "material_order_changed": material}


def _column(rows: list[dict], name: str) -> list[float]:
    return [float(row[name]) for row in rows]


def evaluate_bonus(rows: list[dict], bonus: list[float], reference_bonus: list[float] | None = None) -> dict:
    payload = _column(rows, "payload_term")
    penalty = _column(rows, "penalty")
    base = _column(rows, "base_reward")
    formed = [base[index] + bonus[index] for index in range(len(rows))]
    order = reward_reorder(base, formed, payload)
    payload_order = payload_step_reorder(payload, bonus)
    payload_ratio = ratio_stats(bonus, payload)
    penalty_ratio = ratio_stats(bonus, penalty)
    zero_payload = int(sum(value == 0.0 for value in payload))
    positive_urgency_on_zero_payload = int(
        sum(float(row["urgency_term"]) > float(PENALTY_DUST) and float(row["payload_term"]) == 0.0 for row in rows)
    )
    sign_flips = int(sum((float(base[index]) > 0.0) != (float(formed[index]) > 0.0) for index in range(len(base))))
    retained = None
    removed = None
    if reference_bonus is not None:
        raw_sum = float(sum(reference_bonus))
        kept_sum = float(sum(bonus))
        retained = 1.0 if raw_sum <= float(PENALTY_DUST) else kept_sum / raw_sum
        removed = float(raw_sum - kept_sum)
    mean_payload_ok = (
        float(sum(abs(value) for value in bonus)) <= float(PENALTY_DUST)
        if payload_ratio["mean"] is None
        else float(payload_ratio["mean"]) <= float(MEAN_BONUS_OVER_PAYLOAD_MAX)
    )
    penalty_ok = (
        penalty_ratio["p95"] is not None
        and float(penalty_ratio["p95"]) <= float(P95_BONUS_OVER_PENALTY_MAX)
        and int(penalty_ratio["positive_bonus_on_near_zero_denominator"]) == 0
    )
    return {
        "steps": int(len(rows)),
        "bonus_mean": _mean(bonus),
        "bonus_p50": _percentile(bonus, 50),
        "bonus_p95": _percentile(bonus, 95),
        "bonus_p99": _percentile(bonus, 99),
        "bonus_max": None if not bonus else float(max(bonus)),
        "bonus_sum": float(sum(bonus)),
        "payload_sum": float(sum(payload)),
        "urgency_served_bits_sum": float(sum(float(row["urgency_served_bits"]) for row in rows)),
        "penalty_sum": float(sum(penalty)),
        "projection_penalty_sum": float(sum(float(row["projection_penalty"]) for row in rows)),
        "residual_penalty_sum": float(sum(float(row["residual_penalty"]) for row in rows)),
        "base_reward_mean": _mean(base),
        "reward_mean": _mean(formed),
        "sign_flips": sign_flips,
        "bonus_over_payload": payload_ratio,
        "bonus_over_penalty": penalty_ratio,
        "order_changed_vs_base": bool(order["order_changed"]),
        "material_order_changed": bool(order["material_order_changed"]),
        "payload_order_changed": bool(payload_order["order_changed"]),
        "payload_material_order_changed": bool(payload_order["material_order_changed"]),
        "zero_payload_steps": zero_payload,
        "positive_urgency_on_zero_payload_steps": positive_urgency_on_zero_payload,
        "signal_retained_fraction": retained,
        "bonus_removed_by_cap": removed,
        "meets_mean_payload_bound": bool(mean_payload_ok),
        "meets_penalty_p95_bound": bool(penalty_ok),
        "meets_no_material_reorder": bool(not order["material_order_changed"]),
        "meets_conservative_standard": bool(mean_payload_ok and penalty_ok and not order["material_order_changed"]),
        "signal_too_weak": bool(retained is not None and float(retained) < float(SIGNAL_RETAINED_MIN)),
    }


def _scenarios(rows: list[dict]) -> list[tuple[int, list[dict]]]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["scenario_id"]), []).append(row)
    return [(scenario, grouped[scenario]) for scenario in sorted(grouped)]


def _public_row(form: str, lambda_u: float, rho: float | None, scenario_id: str, stats: dict, worst_scenario: int | None) -> dict:
    payload = stats["bonus_over_payload"]
    penalty = stats["bonus_over_penalty"]
    return {
        "form": form,
        "lambda_u": float(lambda_u),
        "rho": "" if rho is None else float(rho),
        "scenario_id": scenario_id,
        "worst_scenario_id": "" if worst_scenario is None else int(worst_scenario),
        "steps": stats["steps"],
        "bonus_mean": stats["bonus_mean"],
        "bonus_p50": stats["bonus_p50"],
        "bonus_p95": stats["bonus_p95"],
        "bonus_p99": stats["bonus_p99"],
        "bonus_max": stats["bonus_max"],
        "bonus_sum": stats["bonus_sum"],
        "bonus_over_payload_mean": payload["mean"],
        "bonus_over_payload_p95": payload["p95"],
        "bonus_over_penalty_p50": penalty["p50"],
        "bonus_over_penalty_p95": penalty["p95"],
        "bonus_over_penalty_p99": penalty["p99"],
        "near_zero_penalty_steps": penalty["near_zero_denominator_steps"],
        "positive_bonus_on_near_zero_penalty": penalty["positive_bonus_on_near_zero_denominator"],
        "order_changed_vs_base": int(stats["order_changed_vs_base"]),
        "material_order_changed": int(stats["material_order_changed"]),
        "payload_order_changed": int(stats["payload_order_changed"]),
        "payload_material_order_changed": int(stats["payload_material_order_changed"]),
        "payload_sum": stats["payload_sum"],
        "urgency_served_bits_sum": stats["urgency_served_bits_sum"],
        "penalty_sum": stats["penalty_sum"],
        "projection_penalty_sum": stats["projection_penalty_sum"],
        "residual_penalty_sum": stats["residual_penalty_sum"],
        "reward_mean": stats["reward_mean"],
        "sign_flips": stats["sign_flips"],
        "zero_payload_steps": stats["zero_payload_steps"],
        "positive_urgency_on_zero_payload_steps": stats["positive_urgency_on_zero_payload_steps"],
        "signal_retained_fraction": "" if stats["signal_retained_fraction"] is None else stats["signal_retained_fraction"],
        "bonus_removed_by_cap": "" if stats["bonus_removed_by_cap"] is None else stats["bonus_removed_by_cap"],
        "meets_conservative_standard": int(stats["meets_conservative_standard"]),
        "signal_too_weak": int(stats["signal_too_weak"]),
    }


def _worst_scenario(per_scenario: list[tuple[int, dict]]) -> int | None:
    ranked = []
    for scenario, stats in per_scenario:
        penalty = stats["bonus_over_penalty"]["p95"]
        score = float("inf") if int(stats["bonus_over_penalty"]["positive_bonus_on_near_zero_denominator"]) else (-1.0 if penalty is None else float(penalty))
        ranked.append((score, float(stats["bonus_max"] or 0.0), int(scenario)))
    if not ranked:
        return None
    ranked.sort()
    if all(item[0] == ranked[0][0] and item[1] == ranked[0][1] for item in ranked):
        return None
    return int(ranked[-1][2])


def _combo(rows: list[dict], form: str, lambda_u: float, rho: float | None) -> dict:
    per_scenario = []
    ordered_rows: list[dict] = []
    flat_bonus: list[float] = []
    flat_reference: list[float] = []
    for scenario, group in _scenarios(rows):
        reference = [uncapped_bonus(float(row["urgency_term"]), lambda_u) for row in group]
        if rho is None:
            bonus = list(reference)
        else:
            bonus = [capped_bonus(float(row["urgency_term"]), float(row["payload_term"]), lambda_u, rho) for row in group]
        per_scenario.append((scenario, evaluate_bonus(group, bonus, None if rho is None else reference)))
        ordered_rows.extend(group)
        flat_bonus.extend(bonus)
        flat_reference.extend(reference)
    pooled = evaluate_bonus(ordered_rows, flat_bonus, None if rho is None else flat_reference)
    maxima = [pooled["bonus_over_payload"]["max"]] + [stats["bonus_over_payload"]["max"] for _scenario, stats in per_scenario]
    cap_holds = True if rho is None else all(value is None or float(value) <= float(rho) + 1.0e-9 for value in maxima)
    worst = _worst_scenario(per_scenario)
    all_meet = bool(per_scenario) and all(stats["meets_conservative_standard"] for _scenario, stats in per_scenario)
    any_weak = any(stats["signal_too_weak"] for _scenario, stats in per_scenario) or bool(pooled["signal_too_weak"])
    usable = bool(all_meet and not any_weak and float(lambda_u) > 0.0 and float(lambda_u) != float(LAMBDA_REFERENCE))
    return {
        "form": form,
        "lambda_u": float(lambda_u),
        "rho": rho,
        "role": "high_risk_reference" if float(lambda_u) == float(LAMBDA_REFERENCE) else "candidate",
        "all_scenarios_meet_conservative_standard": all_meet,
        "signal_too_weak": bool(any_weak),
        "usable_candidate": usable,
        "cap_holds": bool(cap_holds),
        "worst_scenario_id": worst,
        "pooled": pooled,
        "per_scenario": [{"scenario_id": scenario, **stats} for scenario, stats in per_scenario],
        "rows": [_public_row(form, lambda_u, rho, str(scenario), stats, worst) for scenario, stats in per_scenario]
        + [_public_row(form, lambda_u, rho, "pooled", pooled, worst)],
    }


def compare_forms(rows: list[dict]) -> dict:
    combos = [_combo(rows, "A", float(lam), None) for lam in (*LAMBDA_FORM_A, LAMBDA_REFERENCE)]
    for lam in LAMBDA_FORM_B:
        for rho in RHO_CANDIDATES:
            combos.append(_combo(rows, "B", float(lam), float(rho)))
    payload_values = [float(combo["pooled"]["payload_sum"]) for combo in combos]
    same_payload = bool(payload_values) and max(payload_values) - min(payload_values) == 0.0
    caps_ok = all(bool(combo["cap_holds"]) for combo in combos)
    usable = [
        {"form": combo["form"], "lambda_u": combo["lambda_u"], "rho": combo["rho"]}
        for combo in combos
        if combo["usable_candidate"]
    ]
    return {
        "combinations": combos,
        "form_rows": [row for combo in combos for row in combo["rows"]],
        "payload_sums_identical_across_forms": bool(same_payload),
        "payload_caps_hold": bool(caps_ok),
        "usable_candidates": usable,
        "any_candidate_meets_standard": bool(usable),
        "standards": {
            "mean_bonus_over_payload_max": MEAN_BONUS_OVER_PAYLOAD_MAX,
            "p95_bonus_over_penalty_max": P95_BONUS_OVER_PENALTY_MAX,
            "payload_tie": PAYLOAD_TIE,
            "penalty_dust": PENALTY_DUST,
            "signal_retained_min": SIGNAL_RETAINED_MIN,
        },
    }


def crosscheck_form_a(comparison: dict, e2_summary: dict) -> dict:
    """Form A on the saved rows must reproduce the E2 lambda table."""
    expected = {float(item["lambda_u"]): item for item in e2_summary["scale"]["by_lambda"]}
    mismatches = []
    for combo in comparison["combinations"]:
        if combo["form"] != "A" or float(combo["lambda_u"]) not in expected:
            continue
        target = expected[float(combo["lambda_u"])]
        pooled = combo["pooled"]
        checks = (
            ("bonus_mean", pooled["bonus_mean"], target["bonus_mean"]),
            ("bonus_p95", pooled["bonus_p95"], target["bonus_p95"]),
            ("reward_mean", pooled["reward_mean"], target["reward_mean"]),
            ("bonus_over_payload_mean", pooled["bonus_over_payload"]["mean"], target["bonus_over_payload"]["mean"]),
            ("bonus_over_penalty_p95", pooled["bonus_over_penalty"]["p95"], target["bonus_over_penalty"]["p95"]),
        )
        for name, observed, wanted in checks:
            if observed is None or abs(float(observed) - float(wanted)) > float(CROSSCHECK_ATOL):
                mismatches.append({"lambda_u": combo["lambda_u"], "metric": name, "observed": observed, "e2": wanted})
        episode_bonus = [float(item["bonus_sum"]) for item in combo["per_scenario"]]
        episode_reward = [float(item["payload_sum"]) - float(item["penalty_sum"]) + float(item["bonus_sum"]) for item in combo["per_scenario"]]
        for name, observed_list, wanted_list in (
            ("episode_bonus_sum", episode_bonus, target["episode_bonus_sum"]),
            ("episode_reward_sum", episode_reward, target["episode_reward_sum"]),
        ):
            if len(observed_list) != len(wanted_list):
                mismatches.append({"lambda_u": combo["lambda_u"], "metric": name, "observed": len(observed_list), "e2": len(wanted_list)})
                continue
            for index, (observed, wanted) in enumerate(zip(observed_list, wanted_list)):
                if abs(float(observed) - float(wanted)) > float(CROSSCHECK_ATOL):
                    mismatches.append({"lambda_u": combo["lambda_u"], "metric": f"{name}[{index}]", "observed": observed, "e2": wanted})
    return {"ok": not mismatches, "mismatches": mismatches}
