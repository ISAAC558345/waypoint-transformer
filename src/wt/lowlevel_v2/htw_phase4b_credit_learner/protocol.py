"""Phase-4B offline residual-credit protocol. Reads Phase-4A only."""
from __future__ import annotations

from pathlib import Path

from wt.lowlevel_v2.protocol import ROOT

PHASE = Path(__file__).resolve().parent
OUTPUT = PHASE / "outputs"
FIG = OUTPUT / "figures"
P4A = ROOT / "src" / "wt" / "lowlevel_v2" / "htw_phase4a_hindsight_credit" / "outputs"
P4A_CSV = P4A / "phase4a_hindsight_credit.csv"
REPORT_NAME = "HTW_Phase4B_TrajectoryAwareCreditLearner_Audit_CN.md"

FAMS = ("OLD", "ABS", "PREFIX", "EXEC")
PRIMARY = "EXEC"
N_SPLITS = 5
N_PERM = 50
ALIGN_ATOL = 1.0
N_SCENARIO = 16
POOL_SIZE = 24
ET = dict(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=0, n_jobs=1)
PERM_SEED = 0

LOCKED_LOCAL_R = 0.758
LOCKED_LOCAL_U = 9.000e6
LOCKED_TRAJ_U = 11.875e6
LOCKED_FV_SPEARMAN = 0.553
LOCKED_FV_R_GAIN = 0.130
LOCKED_FV_U = 9.375e6

__all__ = [
    "ALIGN_ATOL",
    "ET",
    "FAMS",
    "FIG",
    "LOCKED_FV_R_GAIN",
    "LOCKED_FV_SPEARMAN",
    "LOCKED_FV_U",
    "LOCKED_LOCAL_R",
    "LOCKED_LOCAL_U",
    "LOCKED_TRAJ_U",
    "N_PERM",
    "N_SCENARIO",
    "N_SPLITS",
    "OUTPUT",
    "P4A_CSV",
    "PERM_SEED",
    "PHASE",
    "POOL_SIZE",
    "PRIMARY",
    "REPORT_NAME",
    "ROOT",
]
