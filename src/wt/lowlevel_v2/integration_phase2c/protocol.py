"""HL-SEARCH-AUDIT protocol. Frozen models only. No training."""
from __future__ import annotations

import json
from pathlib import Path

from wt.htw_phase941.protocol import MODEL_SEEDS
from wt.lowlevel_v2.integration_phase2.protocol import (
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
from wt.lowlevel_v2.integration_phase2a.protocol import load_phase2_gate, new_ckpt
from wt.lowlevel_v2.integration_phase2b.protocol import EXPECTED_UTIL_SHA, abs_ckpt, load_phase2a_gate
from wt.lowlevel_v2.protocol import ROOT

INTEG2C = Path(__file__).resolve().parent
OUTPUT = INTEG2C / "outputs"
PHASE2B_OUTPUT = Path(__file__).resolve().parent.parent / "integration_phase2b" / "outputs"
PHASE2B_GATE = ROOT / "results/root_exports/HL_ABS_GATE.json"
BEAM_WIDTHS = (1, 2, 4, 8, 16, 32, 64)
RECOVER_NR = 0.10
TOY_N_UAV = 3
TOY_N_BUOY = 4
TOY_SEEDS = tuple(range(20))
NR_EPS = UTILITY_EPS


def load_phase2b_gate() -> dict:
    path = PHASE2B_GATE if PHASE2B_GATE.is_file() else PHASE2B_OUTPUT / "HL_ABS_GATE.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("ABSSUP") != "B":
        raise RuntimeError("Phase-2B ABSSUP is not B")
    if payload.get("absolute_support_supervision_confirmed") is True:
        raise RuntimeError("Phase-2B unexpectedly confirmed absolute support")
    if payload.get("training_low_level") is True:
        raise RuntimeError("Phase-2B unexpectedly trained low-level")
    return payload


__all__ = [
    "BEAM_WIDTHS",
    "EXPECTED_HIGH_SHA",
    "EXPECTED_UTIL_SHA",
    "INTEG2C",
    "LOW_SEEDS",
    "MODEL_SEEDS",
    "NR_EPS",
    "OLD_HL_CKPT",
    "OUTPUT",
    "PHASE2B_OUTPUT",
    "PHASE2_OUTPUT",
    "RECOVER_NR",
    "ROOT",
    "TOY_N_BUOY",
    "TOY_N_UAV",
    "TOY_SEEDS",
    "UTILITY_EPS",
    "abs_ckpt",
    "ckpt_path",
    "file_sha",
    "high_sha",
    "load_compat_gate",
    "load_phase2_gate",
    "load_phase2a_gate",
    "load_phase2b_gate",
    "new_ckpt",
    "refuse_compat",
]
