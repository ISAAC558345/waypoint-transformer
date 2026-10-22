"""Read-only task-distribution audit. It does not train or write checkpoints."""
from __future__ import annotations

from pathlib import Path

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import DEV_POOL, HOLD, ROOT, TRAIN_POOL
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST
from wt.lowlevel_v2.phase1.protocol import ALPHA_RESIDUAL

OUTPUT = ROOT / "results" / "phase6iq_task_distribution"
SOURCE_ACTOR = primary_checkpoint()
OFFICIAL_ALPHA = float(ALPHA_RESIDUAL)
EVAL_POOL = DEV_POOL
TRAIN_PROBE = (321000, 321016, 321032, 321048, 321064, 321080, 321088, 321095)
DECISION_TIMES = (0, 15, 30, 45)
BIT_WINDOWS = (0, 15, 30)
STATIC_DRAWS = 32
ROLLOUT_DRAWS = 2
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
SUPPORT_OUTSIDE_LIMIT = 0.20
BIT_GAP = 1.0e6
DISTANCE_GAP_M = 2.0
RADIUS_GAP = 0.05
QUEUE_REL_GAP = 0.10
SINR_GAP = 0.05
INTERFERENCE_REL_GAP = 0.10


def assert_phase_output(path: Path) -> Path:
    resolved = Path(path).resolve()
    blocked = (
        "phase6g_lowlevel_mappo",
        "phase6ik_lowlevel_opt",
        "phase6il_direct_action",
        "phase6im_adaptive_residual",
        "phase6in_residual_scale",
        "phase6io_obs_reward",
        "phase6ip_potential_shaping",
        "phase6ip_dynamic_sinr",
        "checkpoints",
    )
    if resolved == SOURCE_ACTOR.resolve() or any(part in resolved.parts for part in blocked):
        raise RuntimeError(f"phase-6iq refused to write {resolved}")
    if resolved.suffix == ".pt":
        raise RuntimeError(f"phase-6iq refused a checkpoint write: {resolved}")
    if any(str(scenario) in resolved.name for scenario in RESERVED_TEST):
        raise RuntimeError(f"phase-6iq refused a reserved-test path: {resolved}")
    return resolved


def assert_audit_scenario(scenario_id: int) -> None:
    value = int(scenario_id)
    if value in RESERVED_TEST or value not in set(EVAL_POOL) | set(TRAIN_POOL):
        raise RuntimeError(f"phase-6iq refused scenario {value}")


__all__ = [
    "BIT_GAP",
    "BIT_WINDOWS",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "DECISION_TIMES",
    "DISTANCE_GAP_M",
    "EVAL_POOL",
    "HOLD",
    "INTERFERENCE_REL_GAP",
    "OFFICIAL_ALPHA",
    "OUTPUT",
    "QUEUE_REL_GAP",
    "RADIUS_GAP",
    "ROLLOUT_DRAWS",
    "SINR_GAP",
    "SOURCE_ACTOR",
    "STATIC_DRAWS",
    "SUPPORT_OUTSIDE_LIMIT",
    "TRAIN_PROBE",
    "assert_audit_scenario",
    "assert_phase_output",
]
