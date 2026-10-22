"""Initial-geometry feasibility audit. Does not retune d_safe."""
from __future__ import annotations

from typing import Any

import numpy as np

from wt.l3b.config import L3BConfig
from wt.l3b.env import L3BEnv
from wt.lowlevel_v2.contracts import minimum_pairwise_distance

from .protocol import D_SAFE_M, SERVICE_SAFETY_MARGIN_M, load_pools, refuse_untouched


def audit_initial_geometry() -> dict[str, Any]:
    train, design = load_pools()
    cfg = L3BConfig()
    if float(cfg.service_radius_m) <= float(SERVICE_SAFETY_MARGIN_M):
        raise RuntimeError("service_radius is not greater than service_safety_margin")
    rows = []
    worst = float("inf")
    for split, seeds in (("train", train), ("design", design)):
        for seed in seeds:
            refuse_untouched(int(seed))
            env = L3BEnv()
            env.reset(seed=int(seed))
            mind = minimum_pairwise_distance(env.uav_positions)
            worst = min(worst, mind)
            rows.append({"split": split, "seed": int(seed), "min_pairwise_m": mind})
            if mind + 1.0e-9 < D_SAFE_M:
                return {
                    "ok": False,
                    "infeasible_initial_geometry": True,
                    "failing_seed": int(seed),
                    "failing_split": split,
                    "min_pairwise_m": mind,
                    "d_safe_m": D_SAFE_M,
                    "auto_lowered_d_safe": False,
                    "rows": rows,
                }
    return {
        "ok": True,
        "infeasible_initial_geometry": False,
        "worst_min_pairwise_m": worst,
        "d_safe_m": D_SAFE_M,
        "service_radius_m": float(cfg.service_radius_m),
        "service_safety_margin_m": SERVICE_SAFETY_MARGIN_M,
        "n_train": len(train),
        "n_design": len(design),
        "auto_lowered_d_safe": False,
        "rows": rows,
    }
