"""Phase-5 probe constants. Isolated seeds. Deployment flags stay false."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PACKAGE = Path(__file__).resolve().parent
OUTPUT = ROOT / "results/phase5_hgnn_urgency"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase5_HGNN_SharedUrgency_Analysis_CN.md"

# New band. Does not overlap consumed development or deployment holdout ids.
TRAIN_SEEDS = tuple(range(310000, 310008))
EVAL_SEEDS = (310100, 310101)

FORBIDDEN_RANGES = (
    (221000, 221016),
    (220000, 220064),
    (222000, 222064),
    (223000, 223064),
    (104000, 104064),
    (105000, 105064),
    (106000, 106064),
    (230000, 230128),
)

HL_EPOCHS = 6
HL_LR = 1.0e-3
MAPPO_UPDATES = 8
URGENCY_EPS = 1.0e-8
HOLD_STEPS = 15

# Bounded probe. Not a deployment claim.
PLANNER_READY = False
DEPLOYMENT_VERIFIED = False


def assert_phase5_seed(seed: int) -> None:
    value = int(seed)
    for low, high in FORBIDDEN_RANGES:
        if low <= value < high:
            raise RuntimeError(f"phase-5 refused protected seed {value}")


def assert_pool(seeds: tuple[int, ...]) -> None:
    if not seeds:
        raise RuntimeError("empty phase-5 seed pool")
    for seed in seeds:
        assert_phase5_seed(int(seed))
    if set(seeds) & set(EVAL_SEEDS) and seeds != EVAL_SEEDS:
        raise RuntimeError("train pool overlaps the phase-5 eval seeds")


__all__ = [
    "DEPLOYMENT_VERIFIED",
    "EVAL_SEEDS",
    "FORBIDDEN_RANGES",
    "HL_EPOCHS",
    "HL_LR",
    "HOLD_STEPS",
    "MAPPO_UPDATES",
    "OUTPUT",
    "PACKAGE",
    "PLANNER_READY",
    "REPORT",
    "ROOT",
    "TRAIN_SEEDS",
    "URGENCY_EPS",
    "assert_phase5_seed",
    "assert_pool",
]
