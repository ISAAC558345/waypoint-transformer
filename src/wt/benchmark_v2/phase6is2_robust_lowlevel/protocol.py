"""Cross-policy low-level training. The official checkpoint is not replaced."""
from __future__ import annotations

from pathlib import Path

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import DEV_POOL, HOLD, PPO, ROOT, TRAIN_POOL
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST
from wt.lowlevel_v2.phase1.protocol import ALPHA_RESIDUAL

OUTPUT = ROOT / "results" / "phase6is2_robust_lowlevel"
CHECKPOINT_DIR = OUTPUT / "checkpoints"
SOURCE_ACTOR = primary_checkpoint()
EXPECTED_SOURCE_SHA256 = "92aa68785c898415439ad8c6480427593e10c851e3d392ee3a8110f3a58dbd0e"
TRAIN_SCENARIOS = TRAIN_POOL
EVAL_POOL = DEV_POOL
TRAIN_SOURCES = ("scheme_c", "g1_rate", "uniform_g1_k6")
EVAL_POLICIES = ("scheme_c", "g1_rate", "g2_distance")
HELD_OUT_SOURCE = "g2_distance"
DECISION_TIMES = (0, 15, 30, 45)
MODEL_SEEDS = (0, 1, 2)
UPDATES = 80
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
SAFETY_MARGIN = 0.02
MIN_SUPPORTED_SEEDS = 2
OFFICIAL_ALPHA = float(ALPHA_RESIDUAL)
GAIN_RULE = (
    "A policy is supported when at least 2 of 3 seeds have closed-loop development "
    "payload mean above 0, bootstrap 95 percent lower bound above 0, safety within "
    "0.02 of that policy's frozen MAPPO baseline, no incomplete episode, and the gain "
    "is not concentrated in one scenario. CROSS_POLICY_GENERALIZATION_SUPPORTED requires "
    "scheme_c, g1_rate, and the held-out g2_distance policy all to be supported. "
    "Scheme C is degraded when at least 2 seeds have mean below 0 and upper bound below 0. "
    "Fixed-assignment deltas do not change the gate."
)


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
        "phase6iq_task_distribution",
        "phase6ir_scheme_c_align",
        "phase6is_assignment_head",
    )
    if resolved == SOURCE_ACTOR.resolve() or any(part in resolved.parts for part in blocked):
        raise RuntimeError(f"phase-6is2 refused to write {resolved}")
    if any(str(scenario) in resolved.name for scenario in RESERVED_TEST):
        raise RuntimeError(f"phase-6is2 refused a reserved-test path: {resolved}")
    return resolved


def assert_split_scenario(scenario_id: int, split: str) -> None:
    value = int(scenario_id)
    pool = TRAIN_SCENARIOS if split == "train" else EVAL_POOL
    if value in RESERVED_TEST or value not in pool:
        raise RuntimeError(f"phase-6is2 refused {split} scenario {value}")


__all__ = [
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CHECKPOINT_DIR",
    "DECISION_TIMES",
    "EVAL_POLICIES",
    "EVAL_POOL",
    "EXPECTED_SOURCE_SHA256",
    "GAIN_RULE",
    "HELD_OUT_SOURCE",
    "HOLD",
    "MIN_SUPPORTED_SEEDS",
    "MODEL_SEEDS",
    "OFFICIAL_ALPHA",
    "OUTPUT",
    "PPO",
    "SAFETY_MARGIN",
    "SOURCE_ACTOR",
    "TRAIN_SCENARIOS",
    "TRAIN_SOURCES",
    "UPDATES",
    "assert_phase_output",
    "assert_split_scenario",
]
