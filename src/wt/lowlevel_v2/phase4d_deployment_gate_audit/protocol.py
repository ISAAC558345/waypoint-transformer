"""Phase-4D gate protocol. Threshold locked from Phase-4B shadow, not from Phase-4C U_traj."""
from __future__ import annotations

from pathlib import Path

from wt.lowlevel_v2.protocol import ROOT

PHASE = Path(__file__).resolve().parent
OUTPUT = PHASE / "outputs"
FIG = OUTPUT / "figures"
REPORT_NAME = "HTW_Phase4D_DeploymentGate_Audit_CN.md"

P4C = ROOT / "src" / "wt" / "lowlevel_v2" / "phase4c_independent_validation" / "outputs"
P4C_DECISIONS = P4C / "independent_decisions.csv"
P4C_LABELS = P4C / "independent_eval_labels.csv"
P4C_SUMMARY = P4C / "independent_summary.json"

# Pre-fixed from Phase-4B B2 shadow audit: mean dU_hat on the 13 changed design states.
# Not chosen from the 64 independent U_traj values.
TAU_UHAT = 1534933.5944128654
TAU_SOURCE = "HTW_Phase4B_B2_ShadowIntegration_Audit_CN.md changed-state mean dU_hat"
N_INDEPENDENT = 64
INDEPENDENT_IDS = tuple(range(220000, 220064))

__all__ = [
    "FIG",
    "INDEPENDENT_IDS",
    "N_INDEPENDENT",
    "OUTPUT",
    "P4C_DECISIONS",
    "P4C_LABELS",
    "P4C_SUMMARY",
    "PHASE",
    "REPORT_NAME",
    "ROOT",
    "TAU_SOURCE",
    "TAU_UHAT",
]
