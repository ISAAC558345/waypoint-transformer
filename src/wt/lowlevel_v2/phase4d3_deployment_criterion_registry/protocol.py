"""Phase-4D-3 search corpus and frozen Gate-1 observations. Tau is not retuned."""
from __future__ import annotations

from pathlib import Path

from wt.lowlevel_v2.phase4d_deployment_gate_audit.protocol import TAU_SOURCE, TAU_UHAT
from wt.lowlevel_v2.protocol import ROOT

PHASE = Path(__file__).resolve().parent
OUTPUT = PHASE / "outputs"
REPORT_NAME = "HTW_Phase4D3_DeploymentCriterionRegistry_Audit_CN.md"

# Frozen observed Gate-1 numbers from Phase-4D / Phase-4D-2. Do not retune.
FROZEN_GATE1 = {
    "n_B2": "26/64",
    "n_B2_int": 26,
    "n_total": 64,
    "mean_R": 0.874,
    "mean_dR": 0.111,
    "n_degrade": "3/64",
    "n_degrade_int": 3,
    "degrade_rate": 0.047,
    "worst_case_delta_R": -0.214,
    "tau": TAU_UHAT,
    "tau_display": 1.535e6,
    "tau_source": TAU_SOURCE,
}

SEARCH_KEYWORDS = (
    "deployment-safe",
    "deployment safety",
    "deployment_criterion",
    "safety threshold",
    "acceptance criterion",
    "acceptance rule",
    "degradation",
    "worst-case",
    "degrade rate",
    "degrade_rate",
    "regression tolerance",
    "planner_ready",
    "deployment_ready",
    "safety gate",
    "conservative deployment",
    "allowed_degrade",
    "max_degrade",
)

SEARCH_GLOBS = (
    "HTW_Phase3*.md",
    "HTW_Phase4*.md",
    "FINAL_*.md",
    "FINAL_*.json",
    "AGENTS.md",
    "PLANS.md",
    "src/wt/lowlevel_v2/htw_phase4a_hindsight_credit/**/*.md",
    "src/wt/lowlevel_v2/htw_phase4b_*/**/*.{md,json}",
    "src/wt/lowlevel_v2/phase4c_independent_validation/**/*.{md,json}",
    "src/wt/lowlevel_v2/phase4d_deployment_gate_audit/**/*.{md,json}",
    "src/wt/lowlevel_v2/phase4d2_deployment_safety_audit/**/*.{md,json}",
)

NEAR_MISS_DOCS = (
    {
        "path": "HTW_Phase4B_B2_ShadowIntegration_Audit_CN.md",
        "section": "C. Conservative deployment gate",
        "why_not_criterion": (
            "Gate-1/2 are scientific sanity checks (mean R > LOCAL; permutation drop>0.03). "
            "The same section states Gate-3 deployment threshold is undefined and planner_ready=False."
        ),
    },
    {
        "path": "HTW_Phase4D_DeploymentGate_Audit_CN.md",
        "section": "E/F",
        "why_not_criterion": "Explicitly states the project has no predefined deployment-safety threshold.",
    },
    {
        "path": "HTW_Phase4D2_DeploymentSafety_Audit_CN.md",
        "section": "既有工程阈值检索",
        "why_not_criterion": "Marks deployment criterion = NOT PRE-REGISTERED and leaves A-E thresholds null.",
    },
)

__all__ = [
    "FROZEN_GATE1",
    "NEAR_MISS_DOCS",
    "OUTPUT",
    "PHASE",
    "REPORT_NAME",
    "ROOT",
    "SEARCH_GLOBS",
    "SEARCH_KEYWORDS",
    "TAU_SOURCE",
    "TAU_UHAT",
]
