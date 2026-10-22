"""LL-V2.4A protocol. Frozen Phase-3 Residual MAPPO plus collision distillation only."""
from __future__ import annotations

import json
from pathlib import Path

from wt.lowlevel_v2.phase2d.protocol import (
    ALPHA_RESIDUAL,
    BETA_RESIDUAL,
    D_SAFE_M,
    EPSILON_BASE_FRACTION,
    EVAL_EVERY,
    EXPECTED_HIGH_SHA,
    HOLD_STEPS,
    LAMBDA_PROJECT,
    LAMBDA_RESIDUAL,
    LOG_STD_INIT,
    PILOT_UPDATES_N,
    PPO,
    ROOT,
    SERVICE_SAFETY_MARGIN_M,
    SMOKE_UPDATES_N,
    action_max,
    epsilon_base,
    high_sha,
    load_pools,
    projector_sha,
    refuse_phase2_validation,
    refuse_untouched,
)
from wt.lowlevel_v2.phase3.protocol import PHASE2_VAL_FROZEN, PHASE3, refuse_historical

PHASE4A = Path(__file__).resolve().parent
OUTPUT = PHASE4A / "outputs"
PREFREEZE_FILE = PHASE4A / "LLV2_PHASE4A_PREFREEZE.json"
PHASE3_GATE = PHASE3 / "LLV2_PHASE3_GATE.json"
PHASE3_CKPT = PHASE3 / "outputs"
PILOT_SEEDS = (0, 1)
LAMBDA_COLLISION_DISTILL = 0.10
PAYLOAD_DROP_TOLERANCE = 0.05
TEACHER_MIN_ACTIVE_FRACTION = 0.005
PHASE3_VAL_FROZEN = frozenset(range(223000, 223064))
DISTILL_EPS = 1.0e-8
INTERVENE_EPS = 1.0e-6


def load_phase3_gate() -> dict:
    payload = json.loads(PHASE3_GATE.read_text(encoding="utf-8"))
    if payload.get("LLV2_MS2") != "A":
        raise RuntimeError("Phase-3 gate is not LLV2-MS2-A")
    if payload.get("stable_residual_advantage") is not True:
        raise RuntimeError("Phase-3 stable_residual_advantage is not true")
    if str(payload.get("REL2")) != "0":
        raise RuntimeError("Phase-3 REL2 is not 0")
    return payload


def refuse_phase4a(seed: int) -> None:
    refuse_historical(int(seed), allow_phase3_val=False)
    s = int(seed)
    if s in PHASE3_VAL_FROZEN:
        raise RuntimeError(f"refused consumed Phase-3 validation seed {s}")
    if s in PHASE2_VAL_FROZEN:
        raise RuntimeError(f"refused consumed Phase-2 validation seed {s}")


__all__ = [
    "ALPHA_RESIDUAL",
    "BETA_RESIDUAL",
    "D_SAFE_M",
    "DISTILL_EPS",
    "EPSILON_BASE_FRACTION",
    "EVAL_EVERY",
    "EXPECTED_HIGH_SHA",
    "HOLD_STEPS",
    "INTERVENE_EPS",
    "LAMBDA_COLLISION_DISTILL",
    "LAMBDA_PROJECT",
    "LAMBDA_RESIDUAL",
    "LOG_STD_INIT",
    "OUTPUT",
    "PAYLOAD_DROP_TOLERANCE",
    "PHASE3",
    "PHASE3_CKPT",
    "PHASE3_GATE",
    "PHASE3_VAL_FROZEN",
    "PHASE4A",
    "PILOT_SEEDS",
    "PILOT_UPDATES_N",
    "PPO",
    "PREFREEZE_FILE",
    "ROOT",
    "SERVICE_SAFETY_MARGIN_M",
    "SMOKE_UPDATES_N",
    "TEACHER_MIN_ACTIVE_FRACTION",
    "action_max",
    "epsilon_base",
    "high_sha",
    "load_phase3_gate",
    "load_pools",
    "projector_sha",
    "refuse_historical",
    "refuse_phase2_validation",
    "refuse_phase4a",
    "refuse_untouched",
]
