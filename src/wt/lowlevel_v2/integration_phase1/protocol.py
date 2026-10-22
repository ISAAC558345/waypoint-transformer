"""HL-LL COMPAT protocol. Frozen models only. No training."""
from __future__ import annotations

import json
from pathlib import Path

from wt.htw_phase945.protocol import HOLD, POOL_SEED, POOL_SIZE
from wt.lowlevel_v2.phase3.protocol import (
    EXPECTED_HIGH_SHA,
    HOLD_STEPS,
    LOW_SEEDS,
    OUTPUT as PHASE3_OUTPUT,
    PHASE2_VAL_FROZEN,
    PHASE3,
    high_sha,
    load_pools,
    refuse_historical,
)
from wt.lowlevel_v2.phase4a.protocol import PHASE3_VAL_FROZEN, load_phase3_gate
from wt.lowlevel_v2.phase4b.protocol import PHASE4B, load_phase4a_gate
from wt.lowlevel_v2.protocol import PHASE104_INDEPENDENT, PHASE105_INDEPENDENT, PHASE106_UNTOUCHED, ROOT

INTEG1 = Path(__file__).resolve().parent
OUTPUT = INTEG1 / "outputs"
PHASE4A_GATE = Path(__file__).resolve().parents[1] / "phase4a" / "LLV2_PHASE4A_GATE.json"
PHASE4B_GATE = PHASE4B / "LLV2_PHASE4B_GATE.json"
PHASE106_DIR = ROOT / "src" / "wt" / "htw_phase106"
NEAR_OPT_FRAC = 0.95
UTILITY_EPS = 1.0
N_STATES = 64
TRAINING = False
CKPT_UPDATE = 100


def load_phase4b_gate() -> dict:
    payload = json.loads(PHASE4B_GATE.read_text(encoding="utf-8"))
    if payload.get("COMM") != "B":
        raise RuntimeError("Phase-4B COMM is not B")
    if payload.get("communication_trajectory_headroom") is True:
        raise RuntimeError("Phase-4B unexpectedly set communication_trajectory_headroom")
    return payload


def refuse_compat(seed: int) -> None:
    refuse_historical(int(seed), allow_phase3_val=False)
    s = int(seed)
    if s in PHASE3_VAL_FROZEN or s in range(223000, 223064):
        raise RuntimeError(f"refused consumed Phase-3 validation seed {s}")
    if s in PHASE2_VAL_FROZEN or s in range(222000, 222064):
        raise RuntimeError(f"refused consumed Phase-2 validation seed {s}")
    if s in PHASE106_UNTOUCHED or s in range(106000, 106064):
        raise RuntimeError(f"refused Phase-10.6 untouched seed {s}")
    if s in PHASE105_INDEPENDENT or s in PHASE104_INDEPENDENT:
        raise RuntimeError(f"refused historical independent seed {s}")


def ckpt_path(seed: int) -> Path:
    path = PHASE3_OUTPUT / f"ckpt_seed{int(seed)}_update{CKPT_UPDATE}.pt"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


__all__ = [
    "CKPT_UPDATE",
    "EXPECTED_HIGH_SHA",
    "HOLD",
    "HOLD_STEPS",
    "INTEG1",
    "LOW_SEEDS",
    "NEAR_OPT_FRAC",
    "N_STATES",
    "OUTPUT",
    "PHASE106_DIR",
    "PHASE3",
    "PHASE3_OUTPUT",
    "PHASE4A_GATE",
    "PHASE4B_GATE",
    "POOL_SEED",
    "POOL_SIZE",
    "ROOT",
    "TRAINING",
    "UTILITY_EPS",
    "ckpt_path",
    "high_sha",
    "load_phase3_gate",
    "load_phase4a_gate",
    "load_phase4b_gate",
    "load_pools",
    "refuse_compat",
]
