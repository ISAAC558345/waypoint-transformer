"""LL-V2.2B frozen protocol. Only residual trust-region is new."""
from __future__ import annotations

import json
from pathlib import Path

from wt.htw_phase941.protocol import sha256_bytes
from wt.lowlevel_v2.phase1.protocol import ACTION_SCALE
from wt.lowlevel_v2.phase1b.protocol import (
    ALPHA_RESIDUAL,
    D_SAFE_M,
    EVAL_EVERY,
    EXPECTED_HIGH_SHA,
    FINAL_CKPT,
    HOLD_STEPS,
    LAMBDA_PROJECT,
    LAMBDA_RESIDUAL,
    LOG_STD_INIT,
    PAYLOAD_SCALE,
    PILOT_UPDATES_N,
    PPO,
    SERVICE_SAFETY_MARGIN_M,
    SMOKE_UPDATES_N,
    high_sha,
    load_pools,
    refuse_untouched,
)
from wt.lowlevel_v2.phase2.protocol import PHASE1B_CKPT0, vector_seed_pilot, vector_seed_smoke
from wt.lowlevel_v2.phase2b.trust_region import BETA_RESIDUAL, EPSILON_BASE_FRACTION, action_max, epsilon_base
from wt.lowlevel_v2.protocol import ROOT

PHASE2B = Path(__file__).resolve().parent
OUTPUT = PHASE2B / "outputs"
PREFREEZE_FILE = PHASE2B / "LLV2_PHASE2B_PREFREEZE.json"
PROJECTOR_FILE = PHASE2B.parent / "safety_projector.py"
PHASE2_VAL = frozenset(range(222000, 222064))
DEV_SEEDS = (0, 1)
OLD_SEED1_CKPT = PHASE2B.parent / "phase2" / "outputs" / "ckpt_seed1_update100.pt"
OLD_SEED2_CKPT = PHASE2B.parent / "phase2" / "outputs" / "ckpt_seed2_update100.pt"


def refuse_phase2_validation(seed: int) -> None:
    refuse_untouched(int(seed))
    if int(seed) in PHASE2_VAL:
        raise RuntimeError(f"refused Phase-2 consumed validation seed {seed}")


def projector_sha() -> str:
    return sha256_bytes(PROJECTOR_FILE.read_bytes())


def load_prefreeze() -> dict:
    payload = json.loads(PREFREEZE_FILE.read_text(encoding="utf-8"))
    if float(payload["beta_residual"]) != BETA_RESIDUAL:
        raise RuntimeError("prefreeze beta mismatch")
    if float(payload["epsilon_base_fraction"]) != EPSILON_BASE_FRACTION:
        raise RuntimeError("prefreeze epsilon fraction mismatch")
    if float(payload["alpha_residual"]) != ALPHA_RESIDUAL:
        raise RuntimeError("prefreeze alpha mismatch")
    if float(payload["d_safe_m"]) != D_SAFE_M:
        raise RuntimeError("prefreeze d_safe mismatch")
    if payload.get("tuned_from_payload") is True:
        raise RuntimeError("prefreeze illegally tuned from payload")
    if payload.get("beta_grid_search") is True:
        raise RuntimeError("prefreeze illegally searched beta")
    return payload


__all__ = [
    "ACTION_SCALE",
    "ALPHA_RESIDUAL",
    "BETA_RESIDUAL",
    "DEV_SEEDS",
    "D_SAFE_M",
    "EPSILON_BASE_FRACTION",
    "EVAL_EVERY",
    "EXPECTED_HIGH_SHA",
    "FINAL_CKPT",
    "HOLD_STEPS",
    "LAMBDA_PROJECT",
    "LAMBDA_RESIDUAL",
    "LOG_STD_INIT",
    "OLD_SEED1_CKPT",
    "OLD_SEED2_CKPT",
    "OUTPUT",
    "PAYLOAD_SCALE",
    "PHASE1B_CKPT0",
    "PHASE2B",
    "PHASE2_VAL",
    "PILOT_UPDATES_N",
    "PPO",
    "PREFREEZE_FILE",
    "PROJECTOR_FILE",
    "ROOT",
    "SERVICE_SAFETY_MARGIN_M",
    "SMOKE_UPDATES_N",
    "action_max",
    "epsilon_base",
    "high_sha",
    "load_pools",
    "load_prefreeze",
    "projector_sha",
    "refuse_phase2_validation",
    "refuse_untouched",
    "vector_seed_pilot",
    "vector_seed_smoke",
]
