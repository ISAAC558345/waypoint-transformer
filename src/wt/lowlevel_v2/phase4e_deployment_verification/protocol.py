"""Phase-4E reads frozen artifacts and writes only its own outputs."""
from __future__ import annotations

from pathlib import Path

from wt.lowlevel_v2.htw_phase4b_b2_validation.protocol import ET
from wt.lowlevel_v2.protocol import ROOT

PHASE = Path(__file__).resolve().parent
OUTPUT = PHASE / "outputs"
REPORT_NAME = "HTW_Phase4E_DeploymentVerification_CN.md"
HOLDOUT = (
    ROOT
    / "src"
    / "wt"
    / "lowlevel_v2"
    / "phase4d7b_independent_holdout_generation"
    / "outputs"
)
CRITERION = (
    ROOT
    / "src"
    / "wt"
    / "lowlevel_v2"
    / "phase4d_deployment_criterion_registry"
    / "deployment_criterion_registry.json"
)
TAU = 1534933.5944128654
DESIGN_IDS = tuple(range(221000, 221016))
EXPECTED_ET = dict(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=0, n_jobs=1)
DEGRADE_EPS = 1e-12

__all__ = [
    "CRITERION",
    "DEGRADE_EPS",
    "DESIGN_IDS",
    "ET",
    "EXPECTED_ET",
    "HOLDOUT",
    "OUTPUT",
    "REPORT_NAME",
    "ROOT",
    "TAU",
]
