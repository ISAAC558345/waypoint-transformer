"""Scenario-paired bootstrap. One frozen high-level checkpoint: no fake seed CI."""
from __future__ import annotations

from typing import Any

import numpy as np


def paired_bootstrap(delta: np.ndarray, *, n_boot: int, seed: int) -> dict[str, Any]:
    values = np.asarray(delta, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(n_boot), dtype=np.float64)
    n = values.shape[0]
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        means[b] = float(values[idx].mean())
    lo = float(np.percentile(means, 2.5))
    hi = float(np.percentile(means, 97.5))
    base = float(np.mean(np.abs(values) * 0 + 1))
    return {
        "n": int(n),
        "n_boot": int(n_boot),
        "boot_seed": int(seed),
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "positive_scenario_frac": float(np.mean(values > 0.0)),
        "ci95": {"low": lo, "high": hi},
        "relative_gain_vs_abs_mean_ref": None,
        "model_seed_ci_invented": False,
        "note": "scenario-paired percentile bootstrap; single frozen high-level seed",
    }


def classify_sys(
    *,
    vs_m0: dict[str, Any],
    vs_m1: dict[str, Any],
    vs_m2: dict[str, Any] | None,
    blocks_m3: float,
    blocks_m0: float,
    blocks_m1: float,
    legal_m3: float,
    legal_m0: float,
    legal_m1: float,
    drop_m3: float,
    drop_m0: float,
    drop_m1: float,
    idle_m3: float,
    idle_ref: float,
) -> dict[str, Any]:
    ok0 = float(vs_m0["mean_delta"]) > 0.0 and float(vs_m0["ci95"]["low"]) > 0.0
    ok1 = float(vs_m1["mean_delta"]) > 0.0 and float(vs_m1["ci95"]["low"]) > 0.0
    ok2 = True
    m2_note = "M2 absent"
    if vs_m2 is not None:
        ok2 = float(vs_m2["mean_delta"]) > 0.0
        m2_note = "M2 present; require mean>0 and not clearly negative"
    blocks_ok = float(blocks_m3) + 1.0e-9 >= float(blocks_m0) and float(blocks_m3) + 1.0e-9 >= float(blocks_m1)
    legal_ok = float(legal_m3) + 1.0e-12 >= min(float(legal_m0), float(legal_m1)) - 1.0e-9
    not_idle_cheat = float(idle_m3) <= float(idle_ref) + 0.15
    not_drop_cheat = float(drop_m3) <= max(float(drop_m0), float(drop_m1)) + 1.0e-6
    a = ok0 and ok1 and ok2 and blocks_ok and legal_ok and not_idle_cheat and not_drop_cheat
    if a:
        gate, text = "A", "FINAL-HIER shows a confirmed system advantage on this pre-final set (not untouched)."
    elif (float(vs_m0["mean_delta"]) > 0.0 and float(vs_m1["mean_delta"]) > 0.0) and (not ok0 or not ok1):
        gate, text = "B", "Advantage trend vs primary baselines, but paired CI crosses 0."
    else:
        gate, text = "C", "FINAL-HIER does not show a confirmed advantage vs primary baselines."
    return {
        "SYS": gate,
        "system_advantage_confirmed": bool(a),
        "text": text,
        "ok_vs_m0": ok0,
        "ok_vs_m1": ok1,
        "ok_vs_m2": ok2,
        "m2_note": m2_note,
        "completed_blocks_not_worse": blocks_ok,
        "legality_not_worse": legal_ok,
        "not_idle_or_drop_cheat": bool(not_idle_cheat and not_drop_cheat),
    }


__all__ = ["classify_sys", "paired_bootstrap"]
