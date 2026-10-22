"""LL-V2.2 multi-seed independent validation. Method frozen from LL-V2.1B."""
from __future__ import annotations

import json
from pathlib import Path

from wt.htw_phase941.protocol import sha256_bytes
from wt.htw_phase103.protocol import ENV_FILE
from wt.lowlevel_v2.phase1b.protocol import (
    ALPHA_RESIDUAL,
    D_SAFE_M,
    EVAL_EVERY,
    EXPECTED_HIGH_SHA,
    FINAL_CKPT,
    HOLD_STEPS,
    LOG_STD_INIT,
    PILOT_UPDATES_N,
    PPO,
    SERVICE_SAFETY_MARGIN_M,
    SMOKE_UPDATES_N,
    high_sha,
    load_pools,
    refuse_untouched,
)
from wt.lowlevel_v2.protocol import ROOT

PHASE2 = Path(__file__).resolve().parent
OUTPUT = PHASE2 / "outputs"
PHASE1B_CKPT0 = PHASE2.parent / "phase1b" / "outputs" / "ckpt_update100.pt"
P1B_PREFREEZE = PHASE2.parent / "phase1b" / "LLV2_PHASE1B_PREFREEZE.json"
VALIDATION_N = 64
VALIDATION_BASE = 222000
N_BOOT = 2000
BOOT_SEED = 22201
LOW_SEEDS = (0, 1, 2, 3, 4)
NEW_SEEDS = (1, 2, 3, 4)
CURVE_TRAIN_N = 16


def vector_seed_smoke(model_seed: int) -> int:
    return 223100 + int(model_seed)


def vector_seed_pilot(model_seed: int) -> int:
    return 223200 + int(model_seed)


def env_sha() -> str:
    return sha256_bytes(ENV_FILE.read_bytes())


def load_p1b_prefreeze() -> dict:
    return json.loads(P1B_PREFREEZE.read_text(encoding="utf-8"))


__all__ = [
    "ALPHA_RESIDUAL",
    "BOOT_SEED",
    "CURVE_TRAIN_N",
    "D_SAFE_M",
    "ENV_FILE",
    "EVAL_EVERY",
    "EXPECTED_HIGH_SHA",
    "FINAL_CKPT",
    "HOLD_STEPS",
    "LOG_STD_INIT",
    "LOW_SEEDS",
    "N_BOOT",
    "NEW_SEEDS",
    "OUTPUT",
    "PHASE1B_CKPT0",
    "PHASE2",
    "PILOT_UPDATES_N",
    "PPO",
    "ROOT",
    "SERVICE_SAFETY_MARGIN_M",
    "SMOKE_UPDATES_N",
    "VALIDATION_BASE",
    "VALIDATION_N",
    "env_sha",
    "high_sha",
    "load_p1b_prefreeze",
    "load_pools",
    "refuse_untouched",
    "vector_seed_pilot",
    "vector_seed_smoke",
]
