"""Stage C settings. Seeds, budget, and lambda are fixed before training returns are read."""
from __future__ import annotations

from pathlib import Path

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    DEV_POOL,
    HOLD,
    MODEL_SEEDS,
    POSITIVE_BIT,
    PPO,
    TRAIN_POOL,
    UPDATES,
    assert_allowed,
)
from wt.benchmark_v2.phase6i_urgency_stage_a.protocol import AGE_SCALE_STEPS, SCHEMA_VERSION
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST
from wt.lowlevel_v2.observation import TASK_DIM

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_urgency_lowlevel"
CHECKPOINTS = OUTPUT / "checkpoints"
INSTRUCTION_COMMIT = "3a4db69783679fff8796852d43dae74de814e8b2"
MODEL_SEEDS_STAGE_C = (0, 1, 2)
LAMBDA_URGENCY = 1.0
URGENCY_TASK_EXTRA = 2
ACTOR_IN_DIM = 36
CRITIC_EXTRA = 12
CRITIC_IN_DIM = 200
HIGH_URGENCY_QUANTILE = 0.75
FIXED_URGENCY_THRESHOLD = 0.5
VARIANTS = ("L0", "L1", "L2")


def require_train_scenario(scenario_id: int) -> int:
    value = int(scenario_id)
    if value in RESERVED_TEST or value in DEV_POOL:
        raise RuntimeError(f"stage C training refuses scenario {value}")
    assert_allowed(value, split="train")
    return value


def require_dev_scenario(scenario_id: int) -> int:
    value = int(scenario_id)
    if value in RESERVED_TEST or value in TRAIN_POOL:
        raise RuntimeError(f"stage C evaluation refuses scenario {value}")
    assert_allowed(value, split="dev")
    return value


def train_scenario_ids() -> tuple[int, ...]:
    chosen = tuple(int(item) for item in TRAIN_POOL)
    if chosen != tuple(range(321000, 321096)):
        raise RuntimeError("phase6g train pool is not 321000-321095")
    return chosen


def dev_scenario_ids() -> tuple[int, ...]:
    chosen = tuple(int(item) for item in DEV_POOL)
    if chosen != tuple(range(321096, 321128)):
        raise RuntimeError("phase6g dev pool is not 321096-321127")
    return chosen


__all__ = [
    "ACTOR_IN_DIM",
    "AGE_SCALE_STEPS",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CHECKPOINTS",
    "CRITIC_EXTRA",
    "CRITIC_IN_DIM",
    "FIXED_URGENCY_THRESHOLD",
    "HIGH_URGENCY_QUANTILE",
    "HOLD",
    "INSTRUCTION_COMMIT",
    "LAMBDA_URGENCY",
    "MODEL_SEEDS",
    "MODEL_SEEDS_STAGE_C",
    "OUTPUT",
    "POSITIVE_BIT",
    "PPO",
    "SCHEMA_VERSION",
    "TASK_DIM",
    "UPDATES",
    "URGENCY_TASK_EXTRA",
    "VARIANTS",
    "dev_scenario_ids",
    "require_dev_scenario",
    "require_train_scenario",
    "train_scenario_ids",
]
