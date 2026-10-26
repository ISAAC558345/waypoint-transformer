"""Compare candidate bonuses with payload and the stored penalty. No weight is selected."""
from __future__ import annotations

import json

import numpy as np

from .protocol import LAMBDA_CANDIDATES, LAMBDA_REFERENCE, LAMBDA_SELECTED, ROOT


def load_saved_probe() -> dict:
    path = ROOT / "results" / "phase6i_urgency_lowlevel" / "smoke_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    probe = payload["one_update_timing"]
    communication = float(probe["communication_reward"])
    reward = float(probe["mean_reward"])
    return {
        "source": "results/phase6i_urgency_lowlevel/smoke_summary.json",
        "variant": str(probe["variant"]),
        "model_seed": int(probe["model_seed"]),
        "updates": int(probe["updates"]),
        "communication_reward": communication,
        "mean_reward": reward,
        "penalty_reward": float(communication - reward),
        "note": "Stored L1 timing probe. It is not a tag-aware episode.",
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), float(quantile)))


def scale_table(rows: list[dict], probe: dict) -> dict:
    payload = [float(row["served_bits"]) / 1.0e6 for row in rows]
    urgency = [float(row["urgency_served_bits"]) / 1.0e6 for row in rows]
    penalty = float(probe["penalty_reward"])
    order_payload = [int(index) for index in np.argsort(np.asarray(payload, dtype=np.float64), kind="mergesort")]
    by_lambda = []
    for lam in (*LAMBDA_CANDIDATES, LAMBDA_REFERENCE):
        bonus = [float(lam) * value for value in urgency]
        total = [payload[index] + bonus[index] for index in range(len(rows))]
        ratios_payload = [bonus[index] / payload[index] if payload[index] > 0.0 else 0.0 for index in range(len(rows))]
        ratios_penalty = [] if penalty == 0.0 else [bonus[index] / penalty for index in range(len(rows))]
        order_total = [int(index) for index in np.argsort(np.asarray(total, dtype=np.float64), kind="mergesort")]
        sign_flip = 0
        penalty_sign_flip = 0
        material_order = False
        for index, base in enumerate(payload):
            augmented = total[index]
            if (base > 0.0 and augmented < 0.0) or (base < 0.0 and augmented > 0.0):
                sign_flip += 1
            if penalty > 0.0 and (base - penalty > 0.0) != (augmented - penalty > 0.0):
                penalty_sign_flip += 1
        for left in range(len(payload)):
            for right in range(left + 1, len(payload)):
                payload_gap = payload[left] - payload[right]
                total_gap = total[left] - total[right]
                if abs(payload_gap) > 1.0e-6 and payload_gap * total_gap < 0.0:
                    material_order = True
        by_lambda.append(
            {
                "lambda_u": float(lam),
                "selected": False,
                "role": "high_risk_reference" if float(lam) == float(LAMBDA_REFERENCE) else "candidate",
                "steps": int(len(rows)),
                "bonus_mean": float(np.mean(bonus)) if bonus else 0.0,
                "bonus_p50": _percentile(bonus, 50),
                "bonus_p95": _percentile(bonus, 95),
                "bonus_p99": _percentile(bonus, 99),
                "bonus_max": float(max(bonus) if bonus else 0.0),
                "bonus_over_payload_mean": float(np.mean(ratios_payload)) if ratios_payload else 0.0,
                "bonus_over_payload_max": float(max(ratios_payload) if ratios_payload else 0.0),
                "bonus_over_recorded_penalty_mean": None if not ratios_penalty else float(np.mean(ratios_penalty)),
                "bonus_over_recorded_penalty_max": None if not ratios_penalty else float(max(ratios_penalty)),
                "step_order_changed": bool(order_total != order_payload),
                "material_step_order_changed": bool(material_order),
                "payload_sign_flips": int(sign_flip),
                "sign_flips_after_subtracting_recorded_penalty": int(penalty_sign_flip),
                "ceiling_over_recorded_penalty": None if penalty == 0.0 else float(lam) * float(probe["communication_reward"]) / penalty,
            }
        )
    return {
        "lambda_selected": LAMBDA_SELECTED,
        "saved_probe": probe,
        "smoke_payload_term_mean": float(np.mean(payload)) if payload else 0.0,
        "smoke_urgency_term_mean": float(np.mean(urgency)) if urgency else 0.0,
        "smoke_steps": int(len(rows)),
        "analytical_bounds": _analytical_bounds(penalty),
        "by_lambda": by_lambda,
        "old_buoy_urgency_ratio_used": False,
    }


def _analytical_bounds(penalty: float) -> dict:
    """Per-block bounds. They do not use the old buoy-level urgency ratio."""
    fresh_u = 0.25
    aged_u = 1.0
    rows = []
    for lam in (*LAMBDA_CANDIDATES, LAMBDA_REFERENCE):
        fresh = float(lam) * fresh_u
        aged = float(lam) * aged_u
        rows.append(
            {
                "lambda_u": float(lam),
                "role": "high_risk_reference" if float(lam) == float(LAMBDA_REFERENCE) else "candidate",
                "fresh_tagged_block_bonus": fresh,
                "age_48_block_bonus": aged,
                "fresh_tagged_over_recorded_penalty": None if penalty == 0.0 else fresh / penalty,
                "age_48_over_recorded_penalty": None if penalty == 0.0 else aged / penalty,
                "bonus_over_payload_if_whole_block_is_served": float(lam) * fresh_u if fresh_u else float(lam),
            }
        )
    return {
        "bonus_upper_bound": "lambda_u * served_bits / 1e6",
        "fresh_tagged_block_bits": 1.0e6,
        "fresh_tagged_block_urgency": fresh_u,
        "fresh_tagged_block_meaning": "tag 1 admitted at the service step with age 0, so u = beta_tag = 0.25",
        "age_48_block_urgency": aged_u,
        "recorded_penalty": float(penalty),
        "penalty_is_combined_project_plus_residual": True,
        "project_and_residual_not_split_in_saved_probe": True,
        "by_lambda": rows,
    }
