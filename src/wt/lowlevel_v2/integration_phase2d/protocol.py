"""HL-PREFIX-PILOT protocol. Prefix-consistent utility only. Two seeds from scratch."""
from __future__ import annotations

import json
from pathlib import Path

from wt.htw_phase92.run_phase92 import BATCH, EPOCHS, LR
from wt.htw_phase941.protocol import MODEL_SEEDS
from wt.htw_phase945.protocol import HOLD, LAMBDA_RANK, LAMBDA_REF, POOL_SIZE, TAU, VAL_EVERY
from wt.lowlevel_v2.integration_phase2.protocol import (
    CHECKPOINT_RULE,
    EXPECTED_HIGH_SHA,
    LOW_SEEDS,
    OLD_HL_CKPT,
    OUTPUT as PHASE2_OUTPUT,
    UTILITY_EPS,
    ckpt_path,
    high_sha,
    load_compat_gate,
)
from wt.lowlevel_v2.integration_phase2a.protocol import load_phase2_gate, new_ckpt
from wt.lowlevel_v2.integration_phase2b.protocol import EXPECTED_UTIL_SHA, abs_ckpt, load_phase2a_gate
from wt.lowlevel_v2.protocol import ROOT

INTEG2D = Path(__file__).resolve().parent
OUTPUT = INTEG2D / "outputs"
PHASE2B_OUTPUT = Path(__file__).resolve().parent.parent / "integration_phase2b" / "outputs"
PHASE2C_OUTPUT = Path(__file__).resolve().parent.parent / "integration_phase2c" / "outputs"
PHASE2C_GATE = ROOT / "results/root_exports/HL_SEARCH_MECHANISM_GATE.json"
PILOT_SEEDS = (0, 1)
ABS_AGG_DECODE_NR = 0.326
ENUM_TOL = 1.0e-5
PERM_TOL = 1.0e-5
SCALE_N = 8


def load_phase2c_gate() -> dict:
    path = PHASE2C_GATE if PHASE2C_GATE.is_file() else PHASE2C_OUTPUT / "HL_SEARCH_MECHANISM_GATE.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("SEARCH_GATE") != "B" and payload.get("gate_key") != "HLSETPATH-B":
        raise RuntimeError("Phase-2C gate is not HLSETPATH-B")
    if payload.get("training") is True or payload.get("training_low_level") is True:
        raise RuntimeError("Phase-2C unexpectedly trained")
    return payload


def prefix_ckpt(seed: int) -> Path:
    return OUTPUT / f"deepsets_prefix_seed{int(seed)}.pt"


__all__ = [
    "ABS_AGG_DECODE_NR",
    "BATCH",
    "CHECKPOINT_RULE",
    "ENUM_TOL",
    "EPOCHS",
    "EXPECTED_HIGH_SHA",
    "EXPECTED_UTIL_SHA",
    "HOLD",
    "INTEG2D",
    "LAMBDA_RANK",
    "LAMBDA_REF",
    "LOW_SEEDS",
    "LR",
    "MODEL_SEEDS",
    "OLD_HL_CKPT",
    "OUTPUT",
    "PERM_TOL",
    "PHASE2B_OUTPUT",
    "PHASE2C_OUTPUT",
    "PHASE2_OUTPUT",
    "PILOT_SEEDS",
    "POOL_SIZE",
    "ROOT",
    "SCALE_N",
    "TAU",
    "UTILITY_EPS",
    "VAL_EVERY",
    "abs_ckpt",
    "ckpt_path",
    "high_sha",
    "load_compat_gate",
    "load_phase2_gate",
    "load_phase2a_gate",
    "load_phase2c_gate",
    "new_ckpt",
    "prefix_ckpt",
]
