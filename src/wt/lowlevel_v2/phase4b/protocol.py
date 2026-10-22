"""LL-V2.4B protocol. Audit only. No training. Frozen Phase-3 Residual MAPPO."""
from __future__ import annotations

import json
from pathlib import Path

from wt.lowlevel_v2.phase1.protocol import ALPHA_RESIDUAL, D_SAFE_M, SERVICE_SAFETY_MARGIN_M
from wt.lowlevel_v2.phase3.protocol import EXPECTED_HIGH_SHA, PHASE3, high_sha, refuse_historical, refuse_phase2_validation
from wt.lowlevel_v2.phase4a.protocol import PHASE3_VAL_FROZEN, PHASE4A, load_phase3_gate

PHASE4B = Path(__file__).resolve().parent
OUTPUT = PHASE4B / "outputs"
PHASE3_CKPT = PHASE3 / "outputs"
PHASE4A_GATE = PHASE4A / "LLV2_PHASE4A_GATE.json"
LOW_SEEDS = (0, 1, 2, 3, 4)
# 50% of max residual amplitude alpha_residual * tanh<=1. Single frozen offset. No grid.
CANDIDATE_OFFSET = 0.5 * float(ALPHA_RESIDUAL)
H_SHADOW = 3
HIGH_FRAC = 0.25
LOW_FRAC = 0.25
HEADROOM_5PCT = 0.05
TRAINING = False


def load_phase4a_gate() -> dict:
    payload = json.loads(PHASE4A_GATE.read_text(encoding="utf-8"))
    if payload.get("COLREL") != "C":
        raise RuntimeError("Phase-4A COLREL is not C")
    if payload.get("collision_relation_internalized") is True:
        raise RuntimeError("Phase-4A unexpectedly internalized collision")
    return payload


def refuse_phase4b(seed: int) -> None:
    refuse_historical(int(seed), allow_phase3_val=False)
    refuse_phase2_validation(int(seed))
    s = int(seed)
    if s in PHASE3_VAL_FROZEN:
        raise RuntimeError(f"refused consumed Phase-3 validation seed {s}")


__all__ = [
    "ALPHA_RESIDUAL",
    "CANDIDATE_OFFSET",
    "D_SAFE_M",
    "EXPECTED_HIGH_SHA",
    "HEADROOM_5PCT",
    "HIGH_FRAC",
    "H_SHADOW",
    "LOW_FRAC",
    "LOW_SEEDS",
    "OUTPUT",
    "PHASE3_CKPT",
    "PHASE4A_GATE",
    "PHASE4B",
    "SERVICE_SAFETY_MARGIN_M",
    "TRAINING",
    "load_phase3_gate",
    "load_phase4a_gate",
    "refuse_phase4b",
    "high_sha",
]
