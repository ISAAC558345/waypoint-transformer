"""LL-V2.2D protocol. Only Actor active-sample optimization is new."""
from __future__ import annotations

import json
from pathlib import Path

from wt.lowlevel_v2.phase2b.protocol import (
    ACTION_SCALE,
    ALPHA_RESIDUAL,
    BETA_RESIDUAL,
    DEV_SEEDS,
    D_SAFE_M,
    EPSILON_BASE_FRACTION,
    EVAL_EVERY,
    EXPECTED_HIGH_SHA,
    HOLD_STEPS,
    LAMBDA_PROJECT,
    LAMBDA_RESIDUAL,
    LOG_STD_INIT,
    PAYLOAD_SCALE,
    PHASE2B,
    PHASE2_VAL,
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
from wt.lowlevel_v2.phase2.protocol import PHASE1B_CKPT0, vector_seed_pilot, vector_seed_smoke
from wt.lowlevel_v2.phase2b.protocol import OLD_SEED1_CKPT

PHASE2D = Path(__file__).resolve().parent
OUTPUT = PHASE2D / "outputs"
PREFREEZE_FILE = PHASE2D / "LLV2_PHASE2D_PREFREEZE.json"
PHASE2C_GATE = PHASE2D.parent / "phase2c" / "LLV2_PHASE2C_MECHANISM_GATE.json"
ADV_EPS = 1.0e-6
MAX_ACTOR_MINIBATCHES = 4


def load_phase2c_gate() -> dict:
    payload = json.loads(PHASE2C_GATE.read_text(encoding="utf-8"))
    if payload.get("MASK") != "B":
        raise RuntimeError("Phase-2C MASK gate is not B")
    if payload.get("BASEZERO") != "A":
        raise RuntimeError("Phase-2C BASEZERO gate is not A")
    return payload


__all__ = [
    "ACTION_SCALE",
    "ADV_EPS",
    "ALPHA_RESIDUAL",
    "BETA_RESIDUAL",
    "DEV_SEEDS",
    "D_SAFE_M",
    "EPSILON_BASE_FRACTION",
    "EVAL_EVERY",
    "EXPECTED_HIGH_SHA",
    "HOLD_STEPS",
    "LAMBDA_PROJECT",
    "LAMBDA_RESIDUAL",
    "LOG_STD_INIT",
    "MAX_ACTOR_MINIBATCHES",
    "OLD_SEED1_CKPT",
    "OUTPUT",
    "PAYLOAD_SCALE",
    "PHASE1B_CKPT0",
    "PHASE2B",
    "PHASE2C_GATE",
    "PHASE2D",
    "PHASE2_VAL",
    "PILOT_UPDATES_N",
    "PPO",
    "PREFREEZE_FILE",
    "ROOT",
    "SERVICE_SAFETY_MARGIN_M",
    "SMOKE_UPDATES_N",
    "action_max",
    "epsilon_base",
    "high_sha",
    "load_phase2c_gate",
    "load_pools",
    "projector_sha",
    "refuse_phase2_validation",
    "refuse_untouched",
    "vector_seed_pilot",
    "vector_seed_smoke",
]
