"""Phase-10.3 protocol. Frozen executor gate only; no training."""
from __future__ import annotations

from pathlib import Path

from wt.htw_phase941.protocol import dump_frozen, sha256_bytes
from wt.htw_phase945.protocol import ROOT
from wt.htw_phase101.protocol import DESIGN_VAL_SEEDS, HIGH_CKPT, HOLD, PPO
from wt.htw_phase101.protocol import OUTPUT as PHASE101_OUTPUT
from wt.htw_phase102.protocol import OUTPUT as PHASE102_OUTPUT

OUTPUT = Path(__file__).resolve().parent / "outputs" / "gate"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase10_3_in_range_zero_motion_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase10_3_in_range_zero_motion_report.md"
LOW_FINAL = PHASE101_OUTPUT / "checkpoints" / "pilot_final.pt"
ENV_FILE = Path(__file__).resolve().parents[1] / "l3b" / "env.py"

__all__ = [
    "DESIGN_VAL_SEEDS",
    "ENV_FILE",
    "HIGH_CKPT",
    "HOLD",
    "LOW_FINAL",
    "OUTPUT",
    "PACK_REPORT",
    "PHASE102_OUTPUT",
    "PPO",
    "REPORT",
    "ROOT",
    "dump_frozen",
    "sha256_bytes",
]
