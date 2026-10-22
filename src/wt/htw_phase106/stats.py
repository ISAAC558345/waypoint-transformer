"""FINAL-A/B/C gate. Statistics reused from Phase-10.5 paired bootstrap."""
from __future__ import annotations

from typing import Any

from wt.htw_phase105.stats import paired_bootstrap


def classify_final(
    *,
    vs_m0: dict[str, Any],
    vs_m1: dict[str, Any],
    vs_m2: dict[str, Any],
    blocks_m3: float,
    blocks_m0: float,
    blocks_m2: float,
    legal_m3: float,
    legal_m0: float,
    legal_m2: float,
    drop_m3: float,
    drop_m0: float,
    drop_m2: float,
    idle_m3: float,
    idle_m0: float,
    idle_m2: float,
) -> dict[str, Any]:
    ok0_mean = float(vs_m0["mean_delta"]) > 0.0
    ok0_ci = float(vs_m0["ci95"]["low"]) > 0.0
    ok2_mean = float(vs_m2["mean_delta"]) > 0.0
    ok2_ci = float(vs_m2["ci95"]["low"]) > 0.0
    not_reverse_m1 = float(vs_m1["mean_delta"]) >= 0.0
    blocks_ok = float(blocks_m3) + 1.0e-9 >= float(blocks_m0) and float(blocks_m3) + 1.0e-9 >= float(blocks_m2)
    legal_ok = float(legal_m3) + 1.0e-12 >= min(float(legal_m0), float(legal_m2)) - 1.0e-9
    idle_ok = float(idle_m3) <= max(float(idle_m0), float(idle_m2)) + 0.15
    drop_ok = float(drop_m3) <= max(float(drop_m0), float(drop_m2)) + 1.0e-6
    a = ok0_mean and ok0_ci and ok2_mean and ok2_ci and not_reverse_m1 and blocks_ok and legal_ok and idle_ok and drop_ok
    if a:
        gate, text = "A", "Final untouched confirmation of FINAL-HIER vs primary frozen baselines."
    elif (ok0_mean and ok2_mean and not_reverse_m1) and (not ok0_ci or not ok2_ci):
        gate, text = "B", "Final test still shows an advantage trend, but at least one primary CI crosses 0."
    else:
        gate, text = "C", "Pre-final advantage did not reproduce on untouched, or a systematic regression appeared."
    return {
        "FINAL": gate,
        "final_system_confirmed": bool(a),
        "text": text,
        "ok_m3_m0_mean": ok0_mean,
        "ok_m3_m0_ci": ok0_ci,
        "ok_m3_m2_mean": ok2_mean,
        "ok_m3_m2_ci": ok2_ci,
        "m3_m1_not_reverse": not_reverse_m1,
        "completed_blocks_not_below_m0_m2": blocks_ok,
        "legality_not_worse": legal_ok,
        "not_drop_or_idle_cheat": bool(idle_ok and drop_ok),
    }


__all__ = ["classify_final", "paired_bootstrap"]
