"""Preregistered effect rule. The 0.01 screen is not a performance test."""
from __future__ import annotations

from .protocol import EFFECT_MEASURABLE, EFFECT_MIXED, EFFECT_NONE, EFFECT_UPDATES, R_THRESHOLD, TRAIN_SEEDS, VARIANTS


def _effect_rows(rows: list[dict]) -> list[dict]:
    selected = []
    for row in rows:
        if int(row["ppo_update"]) not in EFFECT_UPDATES:
            continue
        if row.get("source_variant") not in VARIANTS:
            continue
        if int(row["model_seed"]) not in TRAIN_SEEDS:
            continue
        selected.append(row)
    return selected


def assess_effect(rows: list[dict]) -> dict:
    selected = _effect_rows(rows)
    per_seed = []
    measurable_seeds = 0
    complete = True
    all_below = True
    valid_count = 0
    above_count = 0
    for seed in TRAIN_SEEDS:
        group = [row for row in selected if int(row["model_seed"]) == int(seed)]
        comparisons = []
        above = 0
        if len(group) != 4:
            complete = False
            all_below = False
        for row in group:
            ratio = row.get("r")
            usable = bool(row.get("repeat_ok")) and ratio is not None and not bool(row.get("j_norm_zero"))
            if not usable:
                complete = False
                all_below = False
                comparisons.append({"source_variant": row["source_variant"], "ppo_update": int(row["ppo_update"]), "r": None, "usable": False})
                continue
            valid_count += 1
            value = float(ratio)
            if value >= float(R_THRESHOLD):
                above += 1
                above_count += 1
                all_below = False
            comparisons.append({"source_variant": row["source_variant"], "ppo_update": int(row["ppo_update"]), "r": value, "usable": True})
        if above >= 3 and len(group) == 4:
            measurable_seeds += 1
        per_seed.append({"model_seed": int(seed), "comparison_count": len(group), "above_threshold": int(above), "comparisons": comparisons})
    if measurable_seeds >= 2:
        effect = EFFECT_MEASURABLE
    elif complete and all_below and len(selected) == 12:
        effect = EFFECT_NONE
    else:
        effect = EFFECT_MIXED
    return {
        "effect_assessment": effect,
        "measurable_seed_count": int(measurable_seeds),
        "valid_effect_comparisons": int(valid_count),
        "above_threshold_comparisons": int(above_count),
        "complete_coverage": bool(complete and len(selected) == 12),
        "per_seed": per_seed,
        "r_threshold": float(R_THRESHOLD),
        "threshold_role": "descriptive_direction_screen",
    }


def recommendation(effect: str, *, trial_complete: bool, constraint_failure_count: int) -> str:
    if trial_complete and effect == EFFECT_MEASURABLE and int(constraint_failure_count) == 0:
        return "register_one_full_budget_clip_group_comparison"
    return "do_not_extend_the_pilot"
