"""HL-RL-SUPPORT-AUDIT protocol. Frozen models only. No training."""
from __future__ import annotations

import json
from pathlib import Path

from wt.htw_phase941.protocol import MODEL_SEEDS, sha256_bytes
from wt.htw_phase945.protocol import HOLD, LAMBDA_RANK, LAMBDA_REF, POOL_SIZE, TAU
from wt.lowlevel_v2.integration_phase2.protocol import (
    CHECKPOINT_RULE,
    COMPAT_GATE,
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
from wt.lowlevel_v2.protocol import ROOT

INTEG2A = Path(__file__).resolve().parent
OUTPUT = INTEG2A / "outputs"
BEAM_WIDTH = 32
BEAM_UTILITY_K = 8
NEAR_OPT_FRAC = 0.95
PHASE2_GATE = PHASE2_OUTPUT / "HL_RL_ALIGN_GATE.json"
NEW_HL_CKPT = tuple(PHASE2_OUTPUT / f"deepsets_rlalign_seed{int(seed)}.pt" for seed in MODEL_SEEDS)


def load_phase2_gate() -> dict:
    payload = json.loads(PHASE2_GATE.read_text(encoding="utf-8"))
    if payload.get("HLALIGN") != "C":
        raise RuntimeError("Phase-2 HLALIGN is not C")
    if payload.get("high_level_rl_alignment_confirmed") is True:
        raise RuntimeError("Phase-2 unexpectedly confirmed alignment")
    if payload.get("training_low_level") is True:
        raise RuntimeError("Phase-2 unexpectedly trained low-level")
    return payload


def new_ckpt(seed: int) -> Path:
    path = PHASE2_OUTPUT / f"deepsets_rlalign_seed{int(seed)}.pt"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


__all__ = [
    "BEAM_UTILITY_K",
    "BEAM_WIDTH",
    "CHECKPOINT_RULE",
    "COMPAT_GATE",
    "EXPECTED_HIGH_SHA",
    "HOLD",
    "INTEG2A",
    "LAMBDA_RANK",
    "LAMBDA_REF",
    "LOW_SEEDS",
    "MODEL_SEEDS",
    "NEAR_OPT_FRAC",
    "NEW_HL_CKPT",
    "OLD_HL_CKPT",
    "OUTPUT",
    "PHASE2_GATE",
    "PHASE2_OUTPUT",
    "POOL_SIZE",
    "ROOT",
    "TAU",
    "UTILITY_EPS",
    "ckpt_path",
    "file_sha",
    "high_sha",
    "load_compat_gate",
    "load_phase2_gate",
    "new_ckpt",
    "refuse_compat",
    "sha256_bytes",
]
