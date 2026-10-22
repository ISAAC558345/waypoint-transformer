"""Phase-5E dataset constants. No training and no deployment."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/phase5e_utility_dataset"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5E_TrajectoryAwareUtilityDatasetAudit_CN.md"
DATASET_SEEDS = (310100, 310101)
DECISION_TIMES = (0, 15, 30)
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False

FEATURE_COLUMNS = (
    "queue",
    "channel",
    "distance",
    "ETA",
    "interference",
    "urgency",
)
LABEL_COLUMNS = (
    "payload_gain",
    "utility_score",
    "utility_rank",
)
DATASET_COLUMNS = (
    "scenario_id",
    "time_step",
    "uav_id",
    "candidate_id",
    *FEATURE_COLUMNS,
    *LABEL_COLUMNS,
)


def assert_dataset_seed(seed: int) -> None:
    value = int(seed)
    if 230000 <= value <= 230127:
        raise RuntimeError("phase-5e refused deployment holdout")
    protected = (
        (220000, 220064),
        (221000, 221016),
        (222000, 222064),
        (223000, 223064),
        (104000, 104064),
        (105000, 105064),
        (106000, 106064),
    )
    for low, high in protected:
        if low <= value < high:
            raise RuntimeError(f"phase-5e refused protected seed {value}")


__all__ = [
    "DATASET_COLUMNS",
    "DATASET_SEEDS",
    "DECISION_TIMES",
    "DEPLOYMENT_VERIFIED",
    "FEATURE_COLUMNS",
    "LABEL_COLUMNS",
    "OUTPUT",
    "PLANNER_READY",
    "REPORT",
    "assert_dataset_seed",
]
