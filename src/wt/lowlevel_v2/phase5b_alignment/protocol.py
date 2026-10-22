"""Phase-5B constants. Diagnosis only."""
from __future__ import annotations

from pathlib import Path

from wt.lowlevel_v2.phase5_hgnn_urgency.protocol import assert_phase5_seed

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/phase5b_alignment"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5B_HierarchicalObjectiveAlignmentDiagnosis_CN.md"
PHASE5_CKPT = ROOT / "results/phase5_hgnn_urgency" / "checkpoints"

EVAL_SEEDS = (310100, 310101)
RANK_SEEDS = (310100, 310101, 310102, 310103)
HOLD_STEPS = 15
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False


def assert_diagnosis_seed(seed: int) -> None:
    assert_phase5_seed(int(seed))
    if 230000 <= int(seed) <= 230127:
        raise RuntimeError("phase-5b refused deployment holdout")


__all__ = [
    "DEPLOYMENT_VERIFIED",
    "EVAL_SEEDS",
    "HOLD_STEPS",
    "OUTPUT",
    "PHASE5_CKPT",
    "PLANNER_READY",
    "RANK_SEEDS",
    "REPORT",
    "ROOT",
    "assert_diagnosis_seed",
]
