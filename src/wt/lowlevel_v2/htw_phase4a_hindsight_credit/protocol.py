"""Phase-4A export protocol. Frozen Phase-3 tables only."""
from __future__ import annotations

from pathlib import Path

from wt.lowlevel_v2.integration_phase2.protocol import UTILITY_EPS
from wt.lowlevel_v2.protocol import ROOT
from wt.lowlevel_v2.traj_audit.protocol import LL_RUN, LOW_SEEDS, OUTPUT as TRAJ_OUT

PHASE = Path(__file__).resolve().parent
OUTPUT = PHASE / "outputs"
FIG = OUTPUT / "figures"
P3C = ROOT / "src" / "wt" / "lowlevel_v2" / "phase3c_credit_counterfactual" / "outputs"
REPORT_NAME = "HTW_Phase4A_HindsightCredit_DatasetAudit_CN.md"

FAMS = ("OLD", "ABS", "PREFIX", "EXEC")
PRIMARY = "EXEC"
LOCKED_EXEC = {
    "NATURAL": 5.8125e6,
    "LOCAL-ORACLE": 9.000e6,
    "TRAJ-ORACLE": 11.875e6,
}
ALIGN_ATOL = 1.0
CKPT_ID = "ckpt_seed0_update100"
N_HOLD = 4
EPISODE_HORIZON = 48
HOLD_STEPS = 15
POOL_SIZE = 24
N_SCENARIO = 16

SOURCE = {
    "candidates": P3C / "first_hold_all_candidates.csv",
    "interventions": P3C / "first_hold_counterfactual.csv",
    "pool": TRAJ_OUT / "first_hold_pool.csv",
    "counterfactual": TRAJ_OUT / "counterfactual.json",
    "natural": TRAJ_OUT / "natural_trajectories.json",
}

__all__ = [
    "ALIGN_ATOL",
    "CKPT_ID",
    "EPISODE_HORIZON",
    "FAMS",
    "FIG",
    "HOLD_STEPS",
    "LL_RUN",
    "LOCKED_EXEC",
    "LOW_SEEDS",
    "N_HOLD",
    "N_SCENARIO",
    "OUTPUT",
    "P3C",
    "PHASE",
    "POOL_SIZE",
    "PRIMARY",
    "REPORT_NAME",
    "ROOT",
    "SOURCE",
    "TRAJ_OUT",
    "UTILITY_EPS",
]
