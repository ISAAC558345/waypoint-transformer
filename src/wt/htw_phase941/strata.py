"""Policy-independent interference/competition strata. Frozen on collect seeds only."""
from __future__ import annotations

from typing import Any

import numpy as np

from wt.htw_phase9.const import HOLD
from wt.l3b.env import L3BEnv
from wt.l3b.policies import exclusive_matching, fly_toward, myopic_assignment


def scenario_pressures(seed: int) -> dict[str, float]:
    env = L3BEnv()
    info = env.reset(seed=int(seed))
    pending = 0
    assignment = np.full(env.config.num_uavs, -1, dtype=np.int64)
    interfs = []
    comps = []
    while not info.get("terminated", False):
        if pending <= 0:
            requested = myopic_assignment(env)
            active = [int(b) for b in requested if int(b) >= 0]
            if active:
                uniq = len(set(active))
                comps.append(1.0 - float(uniq) / float(len(active)))
            else:
                comps.append(0.0)
            assignment = exclusive_matching(env)
            pending = int(HOLD)
        after = env.step(fly_toward(env, assignment), assignment)
        matrix = np.asarray(after["interference_uav_uav"], dtype=np.float64)
        interfs.append(float(np.mean(matrix)))
        pending -= 1
        info = after
    return {
        "scenario": int(seed),
        "interference": float(np.mean(interfs)) if interfs else 0.0,
        "competition": float(np.mean(comps)) if comps else 0.0,
    }


def freeze_quantiles(train_seeds: tuple[int, ...]) -> dict[str, Any]:
    rows = [scenario_pressures(seed) for seed in train_seeds]
    inter = np.asarray([row["interference"] for row in rows], dtype=np.float64)
    comp = np.asarray([row["competition"] for row in rows], dtype=np.float64)
    return {
        "source": "collect/train scenarios only; exclusive+myopic descriptors, not confirmation outcomes",
        "n": int(len(rows)),
        "interference_q33": float(np.quantile(inter, 1.0 / 3.0)),
        "interference_q66": float(np.quantile(inter, 2.0 / 3.0)),
        "competition_q50": float(np.quantile(comp, 0.5)),
        "train_rows": rows,
    }


def label_interference(value: float, q33: float, q66: float) -> str:
    if value <= q33:
        return "low"
    if value <= q66:
        return "medium"
    return "high"


def label_competition(value: float, q50: float) -> str:
    return "low" if value <= q50 else "high"


__all__ = ["freeze_quantiles", "label_competition", "label_interference", "scenario_pressures"]
