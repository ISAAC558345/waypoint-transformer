"""Reuse Phase-9.4.1 I(s)/C(s). New competition tertiles freeze on collect seeds only."""
from __future__ import annotations

from typing import Any

import numpy as np

from wt.htw_phase941.strata import freeze_quantiles, scenario_pressures


def freeze_competition_tertiles(train_seeds: tuple[int, ...]) -> dict[str, Any]:
    rows = [scenario_pressures(seed) for seed in train_seeds]
    comp = np.asarray([row["competition"] for row in rows], dtype=np.float64)
    return {
        "metric": "same-top-buoy fraction under myopic preference, episode-mean (Phase-9.4.1 C(s))",
        "source": "COLLECT_SEEDS only; not confirmation outcomes",
        "n": int(len(rows)),
        "competition_q33": float(np.quantile(comp, 1.0 / 3.0)),
        "competition_q67": float(np.quantile(comp, 2.0 / 3.0)),
    }


def label_three(value: float, q33: float, q67: float) -> str:
    if value <= q33:
        return "low"
    if value <= q67:
        return "medium"
    return "high"


def regime_of(i_label: str, c_label: str) -> str | None:
    if i_label == "low" and c_label == "low":
        return "R1"
    if i_label == "high" and c_label == "low":
        return "R2"
    if i_label == "low" and c_label == "high":
        return "R3"
    if i_label == "high" and c_label == "high":
        return "R4"
    return None


__all__ = ["freeze_competition_tertiles", "freeze_quantiles", "label_three", "regime_of", "scenario_pressures"]
