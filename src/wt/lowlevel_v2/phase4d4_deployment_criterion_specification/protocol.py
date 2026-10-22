"""Phase-4D-4 specification constants. Numeric acceptance fields stay null."""
from __future__ import annotations

from pathlib import Path

from wt.lowlevel_v2.phase4d_deployment_gate_audit.protocol import TAU_SOURCE, TAU_UHAT
from wt.lowlevel_v2.protocol import ROOT

PHASE = Path(__file__).resolve().parent
OUTPUT = PHASE / "outputs"
REPORT_NAME = "HTW_Phase4D4_DeploymentCriterionSpecification_CN.md"

# Frozen Gate-1 observations. Reference only. Not acceptance criteria.
OBSERVED_REFERENCE = {
    "gate": "Gate-1",
    "definition": "if margin >= tau: B2 else LOCAL",
    "margin": "U_hat(argmax U_hat) - U_hat(argmax U_single)",
    "tau": TAU_UHAT,
    "tau_display": 1.535e6,
    "tau_source": TAU_SOURCE,
    "n_B2": "26/64",
    "mean_R": 0.874,
    "mean_delta_R": 0.111,
    "degrade": "3/64",
    "degrade_rate": 0.047,
    "worst_case_delta_R": -0.214,
    "disclaimer": (
        "These are observed evaluation results and are NOT deployment acceptance criteria."
    ),
}

# Human-decision fields must remain unset. Do not auto-select.
HUMAN_PENDING = "PENDING_HUMAN_DECISION"

__all__ = [
    "HUMAN_PENDING",
    "OBSERVED_REFERENCE",
    "OUTPUT",
    "PHASE",
    "REPORT_NAME",
    "ROOT",
    "TAU_SOURCE",
    "TAU_UHAT",
]
