"""Phase-4D-2 protocol. Tau remains the Phase-4B shadow lock; do not retune."""
from __future__ import annotations

from pathlib import Path

from wt.lowlevel_v2.phase4d_deployment_gate_audit.protocol import (
    INDEPENDENT_IDS,
    N_INDEPENDENT,
    P4C_DECISIONS,
    P4C_LABELS,
    P4C_SUMMARY,
    TAU_SOURCE,
    TAU_UHAT,
)
from wt.lowlevel_v2.protocol import ROOT

PHASE = Path(__file__).resolve().parent
OUTPUT = PHASE / "outputs"
REPORT_NAME = "HTW_Phase4D2_DeploymentSafety_Audit_CN.md"

# Same frozen tau as Phase-4D. Not selected from independent U_traj.
TAU_DISPLAY = 1.535e6

KNOWN_CRITERION_FILES = (
    ROOT / "docs/archive/phase_reports/HTW_Phase4B_B2_ShadowIntegration_Audit_CN.md",
    ROOT / "docs/archive/phase_reports/HTW_Phase4C_IndependentValidation_Audit_CN.md",
    ROOT / "docs/archive/phase_reports/HTW_Phase4D_DeploymentGate_Audit_CN.md",
    ROOT / "src" / "wt" / "lowlevel_v2" / "phase4d_deployment_gate_audit" / "outputs" / "phase4d_gate_summary.json",
    ROOT / "src" / "wt" / "lowlevel_v2" / "phase4c_independent_validation" / "outputs" / "independent_summary.json",
    ROOT / "src" / "wt" / "lowlevel_v2" / "htw_phase4b_b2_shadow" / "outputs" / "b2_shadow_summary.json",
)

__all__ = [
    "INDEPENDENT_IDS",
    "KNOWN_CRITERION_FILES",
    "N_INDEPENDENT",
    "OUTPUT",
    "P4C_DECISIONS",
    "P4C_LABELS",
    "P4C_SUMMARY",
    "PHASE",
    "REPORT_NAME",
    "ROOT",
    "TAU_DISPLAY",
    "TAU_SOURCE",
    "TAU_UHAT",
]
