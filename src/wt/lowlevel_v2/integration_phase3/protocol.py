"""HL-EXEC-UTIL protocol. Executor-normalized q only. Five seeds from scratch."""
from __future__ import annotations

import json
from pathlib import Path

from wt.htw_phase92.run_phase92 import BATCH, EPOCHS, LR
from wt.htw_phase941.protocol import MODEL_SEEDS
from wt.htw_phase945.protocol import HOLD, LAMBDA_RANK, LAMBDA_REF, POOL_SIZE, VAL_EVERY
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
from wt.lowlevel_v2.integration_phase2a.protocol import load_phase2_gate
from wt.lowlevel_v2.integration_phase2b.protocol import EXPECTED_UTIL_SHA, abs_ckpt, load_phase2a_gate
from wt.lowlevel_v2.protocol import ROOT

INTEG3 = Path(__file__).resolve().parent
OUTPUT = INTEG3 / "outputs"
PHASE2D_OUTPUT = Path(__file__).resolve().parent.parent / "integration_phase2d" / "outputs"
PHASE2B_OUTPUT = Path(__file__).resolve().parent.parent / "integration_phase2b" / "outputs"
PHASE2D_GATE = ROOT / "results/root_exports/HL_PREFIX_GATE.json"
Q_EPS = 1.0e-8
TRAIN_WORKERS = 1
TRAIN_NAME = "hl_exec_align_train"
VAL_NAME = "hl_exec_align_val"


def load_phase2d_gate() -> dict:
    path = PHASE2D_GATE if PHASE2D_GATE.is_file() else PHASE2D_OUTPUT / "HL_PREFIX_GATE.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("PREFIX") != "B":
        raise RuntimeError("Phase-2D gate is not PREFIX-B")
    if payload.get("training_low_level") is True:
        raise RuntimeError("Phase-2D unexpectedly trained the low-level")
    return payload


def exec_ckpt(seed: int) -> Path:
    return OUTPUT / f"deepsets_exec_seed{int(seed)}.pt"


def prefix_ckpt(seed: int) -> Path:
    return PHASE2D_OUTPUT / f"deepsets_prefix_seed{int(seed)}.pt"


__all__ = [
    "BATCH",
    "CHECKPOINT_RULE",
    "EPOCHS",
    "EXPECTED_HIGH_SHA",
    "EXPECTED_UTIL_SHA",
    "HOLD",
    "INTEG3",
    "LAMBDA_RANK",
    "LAMBDA_REF",
    "LOW_SEEDS",
    "LR",
    "MODEL_SEEDS",
    "OLD_HL_CKPT",
    "OUTPUT",
    "PHASE2B_OUTPUT",
    "PHASE2D_OUTPUT",
    "PHASE2_OUTPUT",
    "POOL_SIZE",
    "Q_EPS",
    "ROOT",
    "TRAIN_NAME",
    "TRAIN_WORKERS",
    "UTILITY_EPS",
    "VAL_EVERY",
    "VAL_NAME",
    "abs_ckpt",
    "ckpt_path",
    "exec_ckpt",
    "high_sha",
    "load_compat_gate",
    "load_phase2_gate",
    "load_phase2a_gate",
    "load_phase2d_gate",
    "prefix_ckpt",
]
