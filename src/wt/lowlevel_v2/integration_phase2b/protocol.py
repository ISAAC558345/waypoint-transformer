"""HL-ABS-SUPPORT protocol. Absolute set-likelihood utility only. No other changes."""
from __future__ import annotations

import json
from pathlib import Path

from wt.htw_phase92.run_phase92 import BATCH, EPOCHS, LR
from wt.htw_phase941.protocol import MODEL_SEEDS, sha256_bytes
from wt.htw_phase945.protocol import HOLD, LAMBDA_RANK, LAMBDA_REF, POOL_SIZE, TAU, VAL_EVERY
from wt.lowlevel_v2.integration_phase2.protocol import (
    CHECKPOINT_RULE,
    EXPECTED_HIGH_SHA,
    LOW_SEEDS,
    OLD_HL_CKPT,
    OUTPUT as PHASE2_OUTPUT,
    UTILITY_EPS,
    ckpt_path,
    file_sha,
    high_sha,
    load_compat_gate,
    refuse_compat,
)
from wt.lowlevel_v2.integration_phase2a.protocol import BEAM_UTILITY_K, BEAM_WIDTH, load_phase2_gate, new_ckpt
from wt.lowlevel_v2.protocol import ROOT

INTEG2B = Path(__file__).resolve().parent
OUTPUT = INTEG2B / "outputs"
NEAR_OPT_FRAC = 0.95
EXPECTED_UTIL_SHA = "ff493cf0828e3fa8232b4d5d54cfcee7d34dddc1f42976bc6e9670beaefa3f57"
PHASE2A_GATE = ROOT / "results/root_exports/HL_RL_SUPPORT_MECHANISM_GATE.json"
PHASE2A_DIR = Path(__file__).resolve().parent.parent / "integration_phase2a" / "outputs"
COND_CKPT = tuple(PHASE2_OUTPUT / f"deepsets_rlalign_seed{int(s)}.pt" for s in MODEL_SEEDS)
IDENTITY_N = 100
SHIFT_CS = (-5.0, 5.0)
NORM_TOL = 1.0e-5
IDENTITY_TOL = 1.0e-5


def load_phase2a_gate() -> dict:
    path = PHASE2A_GATE if PHASE2A_GATE.is_file() else PHASE2A_DIR / "HL_RL_SUPPORT_MECHANISM_GATE.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("MECHANISM_GATE") != "A" and payload.get("gate_name") != "HLSUP-A":
        raise RuntimeError("Phase-2A mechanism gate is not HLSUP-A")
    return payload


def abs_ckpt(seed: int) -> Path:
    return OUTPUT / f"deepsets_abs_seed{int(seed)}.pt"


__all__ = [
    "BATCH",
    "BEAM_UTILITY_K",
    "BEAM_WIDTH",
    "CHECKPOINT_RULE",
    "COND_CKPT",
    "EPOCHS",
    "EXPECTED_HIGH_SHA",
    "EXPECTED_UTIL_SHA",
    "HOLD",
    "IDENTITY_N",
    "IDENTITY_TOL",
    "INTEG2B",
    "LAMBDA_RANK",
    "LAMBDA_REF",
    "LOW_SEEDS",
    "LR",
    "MODEL_SEEDS",
    "NEAR_OPT_FRAC",
    "NORM_TOL",
    "OLD_HL_CKPT",
    "OUTPUT",
    "PHASE2A_DIR",
    "PHASE2_OUTPUT",
    "POOL_SIZE",
    "ROOT",
    "SHIFT_CS",
    "TAU",
    "UTILITY_EPS",
    "VAL_EVERY",
    "abs_ckpt",
    "ckpt_path",
    "file_sha",
    "high_sha",
    "load_compat_gate",
    "load_phase2_gate",
    "load_phase2a_gate",
    "new_ckpt",
    "refuse_compat",
    "sha256_bytes",
]
