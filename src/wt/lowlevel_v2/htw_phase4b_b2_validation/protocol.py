"""B2 verification protocol. Frozen Phase-4A labels only."""
from __future__ import annotations

from pathlib import Path

from wt.lowlevel_v2.protocol import ROOT

PHASE = Path(__file__).resolve().parent
OUTPUT = PHASE / "outputs"
FIG = OUTPUT / "figures"
REPORT_NAME = "HTW_Phase4B_B2_IndependentValidation_Audit_CN.md"
PRIMARY = "EXEC"
N_SPLITS = 5
N_PERM = 50
PERM_SEED = 0
ET = dict(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=0, n_jobs=1)

__all__ = ["ET", "FIG", "N_PERM", "N_SPLITS", "OUTPUT", "PERM_SEED", "PHASE", "PRIMARY", "REPORT_NAME", "ROOT"]
