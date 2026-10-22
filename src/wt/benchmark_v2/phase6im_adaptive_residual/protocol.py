"""Round-1 adaptive residual map. These limits are not the official low-level config."""
from __future__ import annotations

from pathlib import Path

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import DEV_POOL, HOLD, PPO, ROOT, TRAIN_POOL
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST

OUTPUT = ROOT / "results" / "phase6im_adaptive_residual"
CHECKPOINT_DIR = OUTPUT / "checkpoints"
SOURCE_ACTOR = primary_checkpoint()
MODEL_SEEDS = (0, 1, 2)
UPDATES = 80
EVAL_POOL = DEV_POOL
CALIBRATION_SCENARIOS = TRAIN_POOL[:8]
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
MAX_TURN_DEG = 45.0
MAX_SPEED_M_S = 22.0
INIT_TURN_TARGET_DEG = 5.0
EXCESSIVE_HEADING_DEG = 15.0
EXCESSIVE_SPEED_FRACTION = 0.20
OFFICIAL_CONFIG = False


def assert_phase_output(path: Path) -> Path:
    resolved = Path(path).resolve()
    blocked = ("phase6g_lowlevel_mappo", "phase6ik_lowlevel_opt", "phase6il_direct_action")
    if resolved == SOURCE_ACTOR.resolve() or any(part in resolved.parts for part in blocked):
        raise RuntimeError(f"phase-6im refused to write {resolved}")
    if any(str(scenario) in resolved.name for scenario in RESERVED_TEST):
        raise RuntimeError(f"phase-6im refused a reserved-test path: {resolved}")
    return resolved


def assert_eval_scenario(scenario_id: int) -> None:
    value = int(scenario_id)
    if value in RESERVED_TEST or value not in EVAL_POOL:
        raise RuntimeError(f"phase-6im eval refuses scenario {value}")


__all__ = [
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CALIBRATION_SCENARIOS",
    "CHECKPOINT_DIR",
    "EVAL_POOL",
    "EXCESSIVE_HEADING_DEG",
    "EXCESSIVE_SPEED_FRACTION",
    "HOLD",
    "INIT_TURN_TARGET_DEG",
    "MAX_SPEED_M_S",
    "MAX_TURN_DEG",
    "MODEL_SEEDS",
    "OFFICIAL_CONFIG",
    "OUTPUT",
    "PPO",
    "SOURCE_ACTOR",
    "TRAIN_POOL",
    "UPDATES",
    "assert_eval_scenario",
    "assert_phase_output",
]
