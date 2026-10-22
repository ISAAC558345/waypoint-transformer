"""LL-V2.1B constants. Hyperparameters copied from Phase-1; only service lock is new."""
from __future__ import annotations

import json
from pathlib import Path

from wt.lowlevel_v2.phase1.protocol import (
    ACTION_SCALE,
    ALPHA_RESIDUAL,
    D_SAFE_M,
    EVAL_EVERY,
    EXPECTED_HIGH_SHA,
    FINAL_CKPT,
    HOLD_STEPS,
    LAMBDA_PROJECT,
    LAMBDA_RESIDUAL,
    LOG_STD_INIT,
    MODEL_SEED,
    PAYLOAD_SCALE,
    PILOT_UPDATES_N,
    PPO,
    SERVICE_SAFETY_MARGIN_M,
    SMOKE_UPDATES_N,
    VECTOR_SEED_PILOT,
    VECTOR_SEED_SMOKE,
    high_sha,
    load_pools,
    refuse_untouched,
)
from wt.lowlevel_v2.protocol import ROOT

PHASE1B = Path(__file__).resolve().parent
OUTPUT = PHASE1B / "outputs"
PREFREEZE_FILE = PHASE1B / "LLV2_PHASE1B_PREFREEZE.json"
OLD_P1_U100_PAYLOAD = 9.25e6
OLD_P1_U100_ENVELOPE_AMONG = 0.753968253968254
OLD_P1_U100_PROJECTOR = 0.328125

SERVICE_RESIDUAL_LOCK = True
POLICY_LOSS_MASK_LOCKED = True


def load_prefreeze() -> dict:
    payload = json.loads(PREFREEZE_FILE.read_text(encoding="utf-8"))
    if float(payload["d_safe_m"]) != D_SAFE_M:
        raise RuntimeError("prefreeze d_safe mismatch")
    if float(payload["alpha_residual"]) != ALPHA_RESIDUAL:
        raise RuntimeError("prefreeze alpha mismatch")
    if float(payload["service_safety_margin_m"]) != SERVICE_SAFETY_MARGIN_M:
        raise RuntimeError("prefreeze margin mismatch")
    if payload.get("tuned_from_payload") is True:
        raise RuntimeError("prefreeze illegally tuned from payload")
    if payload.get("service_residual_lock") is not True:
        raise RuntimeError("prefreeze missing service_residual_lock")
    if payload.get("policy_loss_mask_locked_states") is not True:
        raise RuntimeError("prefreeze missing policy mask flag")
    return payload


__all__ = [
    "ACTION_SCALE",
    "ALPHA_RESIDUAL",
    "D_SAFE_M",
    "EVAL_EVERY",
    "EXPECTED_HIGH_SHA",
    "FINAL_CKPT",
    "HOLD_STEPS",
    "LAMBDA_PROJECT",
    "LAMBDA_RESIDUAL",
    "LOG_STD_INIT",
    "MODEL_SEED",
    "OLD_P1_U100_ENVELOPE_AMONG",
    "OLD_P1_U100_PAYLOAD",
    "OLD_P1_U100_PROJECTOR",
    "OUTPUT",
    "PAYLOAD_SCALE",
    "PHASE1B",
    "PILOT_UPDATES_N",
    "POLICY_LOSS_MASK_LOCKED",
    "PPO",
    "PREFREEZE_FILE",
    "ROOT",
    "SERVICE_RESIDUAL_LOCK",
    "SERVICE_SAFETY_MARGIN_M",
    "SMOKE_UPDATES_N",
    "VECTOR_SEED_PILOT",
    "VECTOR_SEED_SMOKE",
    "high_sha",
    "load_pools",
    "load_prefreeze",
    "refuse_untouched",
]
