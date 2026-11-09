"""Observation and reward audit paths. These constants are not the official low-level config."""
from __future__ import annotations

from pathlib import Path

from wt.control.mappo.protocol import DEV_POOL, HOLD, PPO, ROOT, TRAIN_POOL
from wt.control.executor.protocol import primary_checkpoint
from wt.models.hgnn_assets.protocol import RESERVED_TEST
from wt.legacy.historical1.protocol import ALPHA_RESIDUAL

OUTPUT = ROOT / "results" / "phase6io_obs_reward"
CHECKPOINT_DIR = OUTPUT / "checkpoints"
SOURCE_ACTOR = primary_checkpoint()
OFFICIAL_ALPHA = float(ALPHA_RESIDUAL)
AUDIT_SCENARIOS = (321096, 321100, 321104, 321108)
EVAL_POOL = DEV_POOL
TRAIN_CHECK_POOL = tuple(range(321000, 321032, 4))
MODEL_SEEDS = (0, 1, 2)
UPDATES = 80
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
SAFETY_MARGIN = 0.02
OFFICIAL_CONFIG = False


def assert_phase_output(path: Path) -> Path:
    resolved = Path(path).resolve()
    blocked = (
        "phase6g_lowlevel_mappo",
        "phase6ik_lowlevel_opt",
        "phase6il_direct_action",
        "phase6im_adaptive_residual",
        "phase6in_residual_scale",
    )
    if resolved == SOURCE_ACTOR.resolve() or any(part in resolved.parts for part in blocked):
        raise RuntimeError(f"phase-6io refused to write {resolved}")
    if any(str(scenario) in resolved.name for scenario in RESERVED_TEST):
        raise RuntimeError(f"phase-6io refused a reserved-test path: {resolved}")
    return resolved


def assert_report_scenario(scenario_id: int) -> None:
    value = int(scenario_id)
    if value in RESERVED_TEST or value not in set(EVAL_POOL) | set(TRAIN_POOL):
        raise RuntimeError(f"phase-6io refused scenario {value}")


__all__ = [
    "AUDIT_SCENARIOS",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CHECKPOINT_DIR",
    "EVAL_POOL",
    "HOLD",
    "MODEL_SEEDS",
    "OFFICIAL_ALPHA",
    "OFFICIAL_CONFIG",
    "OUTPUT",
    "PPO",
    "SAFETY_MARGIN",
    "SOURCE_ACTOR",
    "TRAIN_CHECK_POOL",
    "TRAIN_POOL",
    "UPDATES",
    "assert_phase_output",
    "assert_report_scenario",
]