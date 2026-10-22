"""LL-V2.3 protocol. Frozen Phase-2D method. No algorithm changes."""
from __future__ import annotations

import json
from pathlib import Path

from wt.lowlevel_v2.phase2.protocol import env_sha, vector_seed_pilot, vector_seed_smoke
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
    PHASE2D,
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
from wt.lowlevel_v2.protocol import PHASE104_INDEPENDENT, PHASE105_INDEPENDENT, PHASE106_UNTOUCHED

PHASE3 = Path(__file__).resolve().parent
OUTPUT = PHASE3 / "outputs"
PREFREEZE_FILE = PHASE3 / "LLV2_PHASE3_PREFREEZE.json"
MANIFEST_FILE = PHASE3 / "LLV2_PHASE3_VALIDATION_MANIFEST.json"
PHASE2D_GATE_ACTUAL = PHASE2D / "LLV2_PHASE2D_GATE.json"
PHASE2D_CKPT = PHASE2D / "outputs"
LOW_SEEDS = (0, 1, 2, 3, 4)
NEW_SEEDS = (2, 3, 4)
REUSED_SEEDS = (0, 1)
VALIDATION_N = 64
VALIDATION_BASE = 223000
N_BOOT = 2000
BOOT_SEED = 22301
PHASE2_VAL_FROZEN = frozenset(range(222000, 222064))


def load_phase2d_gate() -> dict:
    payload = json.loads(PHASE2D_GATE_ACTUAL.read_text(encoding="utf-8"))
    if payload.get("P2D") != "A":
        raise RuntimeError("Phase-2D gate is not P2D-A")
    if payload.get("active_sample_optimization_supported") is not True:
        raise RuntimeError("active_sample_optimization_supported is not true")
    return payload


def load_val_seeds() -> list[int]:
    payload = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    return [int(s) for s in payload["scenario_seeds"]]


def refuse_historical(seed: int, *, allow_phase3_val: bool = False) -> None:
    refuse_untouched(int(seed))
    refuse_phase2_validation(int(seed))
    s = int(seed)
    if s in PHASE106_UNTOUCHED or s in PHASE105_INDEPENDENT or s in PHASE104_INDEPENDENT:
        raise RuntimeError(f"refused historical independent/untouched seed {s}")
    if s in PHASE2_VAL_FROZEN:
        raise RuntimeError(f"refused Phase-2 consumed validation seed {s}")
    if MANIFEST_FILE.exists():
        val = set(load_val_seeds())
        if s in val and not allow_phase3_val:
            raise RuntimeError(f"refused Phase-3 validation seed {s} before/outside eval")


__all__ = [
    "ALPHA_RESIDUAL",
    "BETA_RESIDUAL",
    "BOOT_SEED",
    "D_SAFE_M",
    "EPSILON_BASE_FRACTION",
    "EVAL_EVERY",
    "EXPECTED_HIGH_SHA",
    "HOLD_STEPS",
    "LAMBDA_PROJECT",
    "LAMBDA_RESIDUAL",
    "LOG_STD_INIT",
    "LOW_SEEDS",
    "MANIFEST_FILE",
    "N_BOOT",
    "NEW_SEEDS",
    "OUTPUT",
    "PHASE2D",
    "PHASE2D_CKPT",
    "PHASE2D_GATE_ACTUAL",
    "PHASE2_VAL_FROZEN",
    "PHASE3",
    "PILOT_UPDATES_N",
    "PPO",
    "PREFREEZE_FILE",
    "REUSED_SEEDS",
    "ROOT",
    "SERVICE_SAFETY_MARGIN_M",
    "SMOKE_UPDATES_N",
    "VALIDATION_BASE",
    "VALIDATION_N",
    "action_max",
    "env_sha",
    "epsilon_base",
    "high_sha",
    "load_phase2d_gate",
    "load_pools",
    "load_val_seeds",
    "projector_sha",
    "refuse_historical",
    "refuse_phase2_validation",
    "refuse_untouched",
    "vector_seed_pilot",
    "vector_seed_smoke",
]
