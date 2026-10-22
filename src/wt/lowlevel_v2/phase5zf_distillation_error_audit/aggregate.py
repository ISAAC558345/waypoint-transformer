"""Summaries for the locked substitution audit. Fitted slopes are not scored."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from wt.lowlevel_v2.phase5r_nonadditive_generalization.metrics import cluster_bootstrap

from .common import direction, mean
from .metrics import choose_gate, recovery_ratio
from .protocol import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    COMBINED_RECOVERY_BAR,
    GROUPS,
    RECOVERY_BAR,
    TIE_BITS,
)
from .ranking import diagnose


SINGLE = tuple(f"P1+TRUE_{name.upper()}" for name in GROUPS)
INJECTED = tuple(f"R1-PRED_{name.upper()}" for name in GROUPS)


def _selected(records: list[dict], model: str, **filters) -> list[float]:
    values = []
    for row in records:
        if row["model"] != model:
            continue
        if any(row[key] != value for key, value in filters.items()):
            continue
        values.append(float(row["selected_j15"]))
    return values


def recovery_rows(records: list[dict]) -> list[dict]:
    rows = []
    gap = mean(_selected(records, "R1")) - mean(_selected(records, "P1"))
    levels = [("overall", {})]
    for scenario in sorted({int(row["scenario_id"]) for row in records}):
        levels.append(("scenario", {"scenario_id": scenario}))
    for seed in sorted({int(row["seed"]) for row in records}):
        levels.append(("seed", {"seed": seed}))
    for level, filters in levels:
        base = mean(_selected(records, "P1", **filters))
        oracle = mean(_selected(records, "R1", **filters))
        local_gap = oracle - base
        for model in SINGLE + ("C1", "C2", "C3"):
            hybrid = mean(_selected(records, model, **filters))
            rows.append(
                {
                    "level": level,
                    "scenario_id": filters.get("scenario_id", ""),
                    "seed": filters.get("seed", ""),
                    "model": model,
                    "selected_j15": hybrid,
                    "recovery": recovery_ratio(hybrid - base, local_gap if level != "overall" else gap),
                    "denominator_positive": float((local_gap if level != "overall" else gap) > 0.0),
                }
            )
    return rows


def damage_rows(records: list[dict]) -> list[dict]:
    rows = []
    levels = [("overall", {})]
    for scenario in sorted({int(row["scenario_id"]) for row in records}):
        levels.append(("scenario", {"scenario_id": scenario}))
    for seed in sorted({int(row["seed"]) for row in records}):
        levels.append(("seed", {"seed": seed}))
    for level, filters in levels:
        oracle = mean(_selected(records, "R1", **filters))
        for model in INJECTED:
            hybrid = mean(_selected(records, model, **filters))
            rows.append(
                {
                    "level": level,
                    "scenario_id": filters.get("scenario_id", ""),
                    "seed": filters.get("seed", ""),
                    "model": model,
                    "selected_j15": hybrid,
                    "damage": oracle - hybrid,
                }
            )
    return rows


def stability_rows(records: list[dict]) -> list[dict]:
    rows = []
    models = SINGLE + INJECTED + ("C1", "C2", "C3", "R1")
    for model in models:
        reference = "P1" if model.startswith("P1") or model.startswith("C") else "R1"
        sign = 1.0 if reference == "P1" else -1.0
        for scenario in sorted({int(row["scenario_id"]) for row in records}):
            delta = sign * (
                mean(_selected(records, model, scenario_id=scenario))
                - mean(_selected(records, reference, scenario_id=scenario))
            )
            rows.append(
                {
                    "model": model,
                    "unit": "scenario",
                    "unit_id": scenario,
                    "delta_j15": delta,
                    "direction": direction(delta, TIE_BITS),
                }
            )
        for seed in sorted({int(row["seed"]) for row in records}):
            delta = sign * (
                mean(_selected(records, model, seed=seed)) - mean(_selected(records, reference, seed=seed))
            )
            rows.append(
                {
                    "model": model,
                    "unit": "seed",
                    "unit_id": seed,
                    "delta_j15": delta,
                    "direction": direction(delta, TIE_BITS),
                }
            )
    return rows


def bootstrap_table(records: list[dict]) -> dict:
    output = {}
    models = SINGLE + INJECTED + ("C1", "C2", "C3", "R1")
    for model in models:
        pairs = []
        for row in records:
            if row["model"] != model:
                continue
            base = next(
                item["selected_j15"]
                for item in records
                if item["model"] == "P1"
                and item["scenario_id"] == row["scenario_id"]
                and item["time_step"] == row["time_step"]
                and item["seed"] == row["seed"]
            )
            pairs.append(
                {
                    "scenario_id": int(row["scenario_id"]),
                    "delta": float(row["selected_j15"]) - float(base),
                }
            )
        output[model] = cluster_bootstrap(pairs, "delta", draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    return output


def _best_single(recovery: dict[str, float]) -> str:
    ranked = sorted(GROUPS, key=lambda name: recovery.get(name, float("-inf")), reverse=True)
    return ranked[0]


def gate_flags(recovery: dict[str, float], intervals: dict, case_rank: float, aligned: bool) -> dict:
    sinr_interval = intervals["P1+TRUE_SINR"]
    progress_interval = intervals["P1+TRUE_PROGRESS"]
    interference_interval = intervals["P1+TRUE_INTERFERENCE"]
    combined_interval = intervals["C3"]
    best = _best_single(recovery)
    singles = {name: float(recovery.get(name, float("nan"))) for name in GROUPS}

    def clears(name: str, interval: dict) -> bool:
        value = singles[name]
        return bool(np.isfinite(value) and value >= RECOVERY_BAR and float(interval["low"]) > 0.0)

    sinr_recovery_ok = clears("sinr", sinr_interval)
    case_ok = bool(np.isfinite(case_rank) and float(case_rank) <= 3.0)
    progress_unique = best == "progress" and all(
        singles["progress"] > singles[name] for name in GROUPS if name != "progress"
    )
    interference_unique = best == "interference" and all(
        singles["interference"] > singles[name] for name in GROUPS if name != "interference"
    )
    flags = {
        "aligned": bool(aligned),
        "sinr_ok": bool(sinr_recovery_ok and case_ok),
        "progress_ok": bool(progress_unique and clears("progress", progress_interval)),
        "interference_ok": bool(interference_unique and clears("interference", interference_interval)),
        "all_single_below": bool(all((not np.isfinite(value)) or value < RECOVERY_BAR for value in singles.values())),
        "combined_ok": bool(
            np.isfinite(recovery.get("C3", float("nan")))
            and float(recovery["C3"]) >= COMBINED_RECOVERY_BAR
            and float(combined_interval["low"]) > 0.0
        ),
        "sinr_recovery_without_case": bool(sinr_recovery_ok and not case_ok),
        "best_single": best,
        "case_median_rank": case_rank,
    }
    flags["gate"] = choose_gate(
        aligned=flags["aligned"],
        sinr_ok=flags["sinr_ok"],
        progress_ok=flags["progress_ok"],
        interference_ok=flags["interference_ok"],
        all_single_below=flags["all_single_below"],
        combined_ok=flags["combined_ok"],
    )
    return flags


def calibration_rows(feature_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in feature_rows:
        grouped[str(row["feature"])].append(row)
    output = []
    for feature, rows in grouped.items():
        slope = np.asarray([float(row["affine_slope"]) for row in rows], dtype=np.float64)
        positive = slope[np.isfinite(slope)] > 0.0
        diagnosis = diagnose(
            mean(row["spearman"] for row in rows),
            mean(row["pairwise"] for row in rows),
            mean(row["bias"] for row in rows),
            mean(row["scale_ratio"] for row in rows),
            mean(row["std_true"] for row in rows),
        )
        output.append(
            {
                "feature": feature,
                "mae": mean(row["mae"] for row in rows),
                "bias": mean(row["bias"] for row in rows),
                "scale_ratio": mean(row["scale_ratio"] for row in rows),
                "statewise_spearman": mean(row["spearman"] for row in rows),
                "statewise_kendall": mean(row["kendall"] for row in rows),
                "statewise_pairwise": mean(row["pairwise"] for row in rows),
                "affine_slope_mean": mean(slope),
                "affine_positive_fraction": float(np.mean(positive)) if positive.size else float("nan"),
                "positive_affine_leaves_ranking_unchanged": True,
                "used_for_gate": False,
                "diagnosis": diagnosis,
            }
        )
    return output


def direction_counts(rows: list[dict], model: str, unit: str) -> dict[str, int]:
    chosen = [row for row in rows if row["model"] == model and row["unit"] == unit]
    return {
        "positive": sum(row["direction"] == "positive" for row in chosen),
        "tie": sum(row["direction"] == "tie" for row in chosen),
        "negative": sum(row["direction"] == "negative" for row in chosen),
    }


__all__ = [
    "INJECTED",
    "SINGLE",
    "bootstrap_table",
    "calibration_rows",
    "damage_rows",
    "direction_counts",
    "gate_flags",
    "recovery_rows",
    "stability_rows",
]
