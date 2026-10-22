"""Phase-10.2 frozen diagnosis protocol. No training."""
from __future__ import annotations

from pathlib import Path

from wt.htw_phase941.protocol import dump_frozen, sha256_bytes
from wt.htw_phase945.protocol import ROOT
from wt.htw_phase101.protocol import (
    DESIGN_VAL_SEEDS,
    HIGH_CKPT,
    HIGH_MODEL_SEED,
    HOLD,
    OUTPUT as PHASE101_OUTPUT,
    PPO,
)

OUTPUT = Path(__file__).resolve().parent / "outputs" / "diagnosis"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase10_2_in_range_service_diagnosis_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase10_2_in_range_service_diagnosis_report.md"
LOW_FINAL = PHASE101_OUTPUT / "checkpoints" / "pilot_final.pt"
MAX_EPISODES = 16
MAX_SNAPSHOTS = 16
MAX_PER_SCENARIO = 2
MAX_BRANCH_STEPS = HOLD
ATOL_REPLAY = 1.0e-9

SNAPSHOT_RULE = (
    "Scan B1 episodes in scenario order 8-15. Inside each episode scan increasing time_step, "
    "then UAV index 0..M-1. Accept a snapshot at the start of a step iff: target not idle, "
    "target in 0..K-1, target queue > 0, UAV currently in service range, remaining_hold > 0, "
    "episode not terminated. Take at most MAX_PER_SCENARIO per scenario and stop at MAX_SNAPSHOTS. "
    "Do not filter by later C1/C2 payoff."
)

__all__ = [
    "ATOL_REPLAY",
    "DESIGN_VAL_SEEDS",
    "HIGH_CKPT",
    "HIGH_MODEL_SEED",
    "HOLD",
    "LOW_FINAL",
    "MAX_BRANCH_STEPS",
    "MAX_EPISODES",
    "MAX_PER_SCENARIO",
    "MAX_SNAPSHOTS",
    "OUTPUT",
    "PACK_REPORT",
    "PHASE101_OUTPUT",
    "PPO",
    "REPORT",
    "ROOT",
    "SNAPSHOT_RULE",
    "dump_frozen",
    "sha256_bytes",
]
