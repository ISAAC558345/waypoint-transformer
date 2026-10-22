"""Phase-6I-AB stability rules. Thresholds are fixed before the episodes."""
from __future__ import annotations

from pathlib import Path

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import ROOT
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6iaa_formal_planner.protocol import POLICY_FILE
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST, VALIDATION_POOL
from wt.benchmark_v2.phase6is_assignment_head.protocol import EPOCH6_CKPT, SCHEME_C_CKPT

OUTPUT = ROOT / "results/phase6iab_planner_stability"
SMOKE = (322048, 322049)
SCENARIOS = tuple(VALIDATION_POOL)
BIT_TIE = 1.0
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
TORCH_SEED_OFFSET = 1_000_000
DECISION_TIMES = (0, 15, 30, 45)
HORIZON = 48


def assert_output_path(path: Path) -> Path:
    resolved = Path(path).resolve()
    forbidden = (
        EPOCH6_CKPT.resolve(),
        SCHEME_C_CKPT.resolve(),
        primary_checkpoint().resolve(),
        POLICY_FILE.resolve(),
    )
    if resolved in forbidden:
        raise RuntimeError(f"phase-6iab refused to write {resolved}")
    if any(str(scenario) in resolved.name for scenario in RESERVED_TEST):
        raise RuntimeError(f"phase-6iab refused a reserved-test path: {resolved}")
    return resolved


def assert_scenario(scenario_id: int) -> None:
    value = int(scenario_id)
    if value in RESERVED_TEST:
        raise RuntimeError(f"phase-6iab refused reserved test scenario {value}")
    if value not in SCENARIOS:
        raise RuntimeError(f"phase-6iab refuses scenario {value}")


def assign_gate(*, failures: int, complete_pairs: int, ci_low: float, ci_high: float) -> str:
    """A positive lower bound is the only gain result. A crossing interval stays inconclusive."""
    if int(failures) > 0 or int(complete_pairs) != len(SCENARIOS):
        return "PLANNER_STABILITY_AUDIT_BLOCKED"
    if float(ci_low) > 0.0:
        return "PLANNER_STABILITY_GAIN_SUPPORTED"
    if float(ci_high) < 0.0:
        return "PLANNER_STABILITY_GAIN_NEGATIVE"
    return "PLANNER_STABILITY_GAIN_INCONCLUSIVE"


__all__ = [
    "BIT_TIE",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "DECISION_TIMES",
    "HORIZON",
    "OUTPUT",
    "POLICY_FILE",
    "SCENARIOS",
    "SMOKE",
    "TORCH_SEED_OFFSET",
    "assert_output_path",
    "assert_scenario",
    "assign_gate",
]
