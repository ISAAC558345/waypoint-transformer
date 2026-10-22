"""Candidate residual scales. They are not written into the official low-level config."""
from __future__ import annotations

from pathlib import Path

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import DEV_POOL, HOLD, ROOT
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST
from wt.lowlevel_v2.phase1.protocol import ALPHA_RESIDUAL

OUTPUT = ROOT / "results" / "phase6in_residual_scale"
SOURCE_ACTOR = primary_checkpoint()
OFFICIAL_ALPHA = float(ALPHA_RESIDUAL)
CANDIDATE_ALPHAS = (0.25, 0.35, 0.50, 0.75)
SCREEN_POOL = tuple(range(321096, 321128, 4))
EVAL_POOL = DEV_POOL
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
CLEAR_LOSS_BITS = 1.0e6
SAFETY_MARGIN = 0.02
OFFICIAL_CONFIG = False


def assert_phase_output(path: Path) -> Path:
    resolved = Path(path).resolve()
    blocked = (
        "phase6g_lowlevel_mappo",
        "phase6ik_lowlevel_opt",
        "phase6il_direct_action",
        "phase6im_adaptive_residual",
    )
    if resolved == SOURCE_ACTOR.resolve() or any(part in resolved.parts for part in blocked):
        raise RuntimeError(f"phase-6in refused to write {resolved}")
    if any(str(scenario) in resolved.name for scenario in RESERVED_TEST):
        raise RuntimeError(f"phase-6in refused a reserved-test path: {resolved}")
    return resolved


def assert_eval_scenario(scenario_id: int) -> None:
    value = int(scenario_id)
    if value in RESERVED_TEST or value not in EVAL_POOL:
        raise RuntimeError(f"phase-6in eval refuses scenario {value}")


__all__ = [
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CANDIDATE_ALPHAS",
    "CLEAR_LOSS_BITS",
    "EVAL_POOL",
    "HOLD",
    "OFFICIAL_ALPHA",
    "OFFICIAL_CONFIG",
    "OUTPUT",
    "SAFETY_MARGIN",
    "SCREEN_POOL",
    "SOURCE_ACTOR",
    "assert_eval_scenario",
    "assert_phase_output",
]
