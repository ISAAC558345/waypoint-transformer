"""Label-gap census for Phase-6I-F. Reserved scenarios are never opened."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import (
    PRIMARY_TIMES,
    RESERVED_TEST,
    REVISION_DATASET,
    TRAIN_POOL,
    VALIDATION_POOL,
)

ROOT = Path(__file__).resolve().parents[4]
CENSUS_PATH = ROOT / "results/phase6if_ranking_loss" / "label_census.json"
GAPS = (0.0, 1.0, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7)


def _split_of(scenario: int) -> str:
    if int(scenario) in TRAIN_POOL:
        return "train"
    if int(scenario) in VALIDATION_POOL:
        return "validation"
    raise RuntimeError(f"phase-6if census refused scenario {scenario}")


def _labels(scenario: int, time_step: int) -> np.ndarray:
    if int(scenario) in RESERVED_TEST:
        raise RuntimeError(f"phase-6if census refused reserved scenario {scenario}")
    path = REVISION_DATASET / _split_of(scenario) / f"scenario_{int(scenario)}" / f"t{int(time_step):03d}_candidates.npz"
    with np.load(path) as payload:
        values = np.asarray(payload["j15_bits"], dtype=np.float64)
    if values.ndim != 1 or int(values.size) < 3 or not np.all(np.isfinite(values)):
        raise RuntimeError(f"invalid J15 vector at {scenario} t={time_step}")
    return values


def _pair_counts(sorted_values: np.ndarray) -> dict[str, float]:
    count = int(sorted_values.size)
    prefix = np.zeros(count + 1, dtype=np.float64)
    prefix[1:] = np.cumsum(sorted_values)
    total_pairs = count * (count - 1) / 2.0
    out = {"candidates": float(count), "unordered_pairs": float(total_pairs)}
    strict = 0.0
    weight = 0.0
    for gap in GAPS:
        lower = np.searchsorted(sorted_values, sorted_values - float(gap), side="left")
        pairs = float(np.sum(lower))
        weighted = float(np.sum(lower.astype(np.float64) * sorted_values - prefix[lower]))
        if gap == 0.0:
            strict = pairs
            weight = weighted
        out[f"pairs_gt_{gap:g}"] = pairs
        out[f"weight_gt_{gap:g}"] = weighted
    rounded = np.sort(sorted_values.astype(np.float32).astype(np.float64))
    maximum = float(sorted_values[-1])
    out.update(
        {
            "strict_pairs": float(strict),
            "equal_pairs": float(total_pairs - strict),
            "strict_fraction": float(strict / total_pairs) if total_pairs else float("nan"),
            "raw_weight": float(weight),
            "near_max_1": float(np.sum(sorted_values >= maximum - 1.0)),
            "near_max_1e4": float(np.sum(sorted_values >= maximum - 1.0e4)),
            "near_max_1e5": float(np.sum(sorted_values >= maximum - 1.0e5)),
            "near_max_1e6": float(np.sum(sorted_values >= maximum - 1.0e6)),
            "label_min": float(sorted_values[0]),
            "label_max": maximum,
            "label_std": float(np.std(sorted_values)),
            "float32_changed": float(np.sum(sorted_values.astype(np.float32).astype(np.float64) != sorted_values)),
            "float32_strict_pairs": float(np.sum(np.searchsorted(rounded, rounded, side="left"))),
        }
    )
    return out


def _mean_rows(rows: list[dict]) -> dict[str, float]:
    keys = [key for key in rows[0] if key not in {"scenario_id", "time_step"}]
    packed = {"n_states": float(len(rows))}
    for key in keys:
        packed[key] = float(np.mean([float(row[key]) for row in rows]))
    return packed


def run_census() -> dict:
    rows = {"train": [], "validation": []}
    for split, pool in (("train", TRAIN_POOL), ("validation", VALIDATION_POOL)):
        for scenario in pool:
            for time_step in PRIMARY_TIMES:
                values = np.sort(_labels(int(scenario), int(time_step)))
                row = _pair_counts(values)
                row["scenario_id"] = int(scenario)
                row["time_step"] = int(time_step)
                rows[split].append(row)
                print(f"census {split} {scenario} t={time_step} strict={row['strict_fraction']:.4f}", flush=True)
    summary = {split: _mean_rows(items) for split, items in rows.items()}
    # Weight share retained when small gaps are removed. This is the scheme-A/B supervision check.
    for split, items in rows.items():
        for gap in GAPS:
            if gap == 0.0:
                continue
            summary[split][f"weight_share_gt_{gap:g}"] = float(
                np.mean([row[f"weight_gt_{gap:g}"] / row["raw_weight"] if row["raw_weight"] else float("nan") for row in items])
            )
            summary[split][f"pair_share_gt_{gap:g}"] = float(
                np.mean([row[f"pairs_gt_{gap:g}"] / row["strict_pairs"] if row["strict_pairs"] else float("nan") for row in items])
            )
    payload = {"summary": summary, "states": rows}
    CENSUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CENSUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_census(), indent=2), flush=True)
