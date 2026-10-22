"""Potential-shaping paths. These constants are not the official low-level config."""
from __future__ import annotations

from pathlib import Path

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import DEV_POOL, HOLD, PPO, ROOT, TRAIN_POOL
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST
from wt.lowlevel_v2.phase1.protocol import ALPHA_RESIDUAL

OUTPUT = ROOT / "results" / "phase6ip_potential_shaping"
CHECKPOINT_DIR = OUTPUT / "checkpoints"
SOURCE_ACTOR = primary_checkpoint()
EXPECTED_SOURCE_SHA256 = "92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e"
OFFICIAL_ALPHA = float(ALPHA_RESIDUAL)
EVAL_POOL = DEV_POOL
MODEL_SEEDS = (0, 1, 2)
UPDATES = 80
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
SAFETY_MARGIN = 0.02
MIN_SUPPORTED_SEEDS = 2
OFFICIAL_CONFIG = False


def assert_phase_output(path: Path) -> Path:
    resolved = Path(path).resolve()
    blocked = (
        "phase6g_lowlevel_mappo",
        "phase6ik_lowlevel_opt",
        "phase6il_direct_action",
        "phase6im_adaptive_residual",
        "phase6in_residual_scale",
        "phase6io_obs_reward",
        "phase6ip_dynamic_sinr",
    )
    if resolved == SOURCE_ACTOR.resolve() or any(part in resolved.parts for part in blocked):
        raise RuntimeError(f"phase-6ip potential shaping refused to write {resolved}")
    if any(str(scenario) in resolved.name for scenario in RESERVED_TEST):
        raise RuntimeError(f"phase-6ip potential shaping refused a reserved-test path: {resolved}")
    return resolved


def assert_report_scenario(scenario_id: int) -> None:
    value = int(scenario_id)
    if value in RESERVED_TEST or value not in EVAL_POOL:
        raise RuntimeError(f"phase-6ip potential shaping refused scenario {value}")


__all__ = [
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CHECKPOINT_DIR",
    "EVAL_POOL",
    "EXPECTED_SOURCE_SHA256",
    "HOLD",
    "MIN_SUPPORTED_SEEDS",
    "MODEL_SEEDS",
    "OFFICIAL_ALPHA",
    "OFFICIAL_CONFIG",
    "OUTPUT",
    "PPO",
    "SAFETY_MARGIN",
    "SOURCE_ACTOR",
    "TRAIN_POOL",
    "UPDATES",
    "assert_phase_output",
    "assert_report_scenario",
]
