"""Phase-6I-L direct velocity actions. Residual MAPPO checkpoints stay untouched."""
from __future__ import annotations

from pathlib import Path

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import DEV_POOL, HOLD, PPO, ROOT, TRAIN_POOL
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST

OUTPUT = ROOT / "results" / "phase6il_direct_action"
AUDIT_OUTPUT = ROOT / "results" / "phase6il_residual_audit"
CHECKPOINT_DIR = OUTPUT / "checkpoints"
SOURCE_ACTOR = primary_checkpoint()
PHASE6IK_SUMMARY = ROOT / "results" / "phase6ik_lowlevel_opt" / "summary.json"
MODEL_SEEDS = (0, 1, 2)
UPDATES = 80
BC_SCENARIOS = TRAIN_POOL[:4]
BC_EPOCHS = 80
EVAL_POOL = DEV_POOL
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
BC_COSINE_MIN = 0.85


def assert_phase_output(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved == SOURCE_ACTOR.resolve() or "phase6g_lowlevel_mappo" in resolved.parts:
        raise RuntimeError(f"phase-6il refused to write {resolved}")
    if "phase6ik_lowlevel_opt" in resolved.parts and resolved.suffix == ".pt":
        raise RuntimeError(f"phase-6il refused to overwrite a Phase-6I-K checkpoint: {resolved}")
    if any(str(scenario) in resolved.name for scenario in RESERVED_TEST):
        raise RuntimeError(f"phase-6il refused a reserved-test path: {resolved}")
    return resolved


def assert_eval_scenario(scenario_id: int) -> None:
    value = int(scenario_id)
    if value in RESERVED_TEST or value not in EVAL_POOL:
        raise RuntimeError(f"phase-6il eval refuses scenario {value}")


__all__ = [
    "AUDIT_OUTPUT",
    "BC_COSINE_MIN",
    "BC_EPOCHS",
    "BC_SCENARIOS",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CHECKPOINT_DIR",
    "EVAL_POOL",
    "HOLD",
    "MODEL_SEEDS",
    "OUTPUT",
    "PHASE6IK_SUMMARY",
    "PPO",
    "SOURCE_ACTOR",
    "TRAIN_POOL",
    "UPDATES",
    "assert_eval_scenario",
    "assert_phase_output",
]
