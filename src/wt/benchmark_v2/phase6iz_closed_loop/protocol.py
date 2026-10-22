"""Phase-6I-Z paired-execution rules. Thresholds are fixed before the rollouts."""
from __future__ import annotations

from pathlib import Path

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import ROOT
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import HOLD, VALIDATION_POOL
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST
from wt.benchmark_v2.phase6is_assignment_head.protocol import EPOCH6_CKPT, SCHEME_C_CKPT

OUTPUT = ROOT / "results/phase6iz_closed_loop"
HORIZON = int(HOLD)
DECISION_TIME = 15
BIT_TIE = 1.0
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
SMOKE = (322048, 322049)
RATE_ATOL = 1.0e-6


def assert_output_path(path: Path) -> Path:
    resolved = Path(path).resolve()
    forbidden = (EPOCH6_CKPT.resolve(), SCHEME_C_CKPT.resolve(), primary_checkpoint().resolve())
    if resolved in forbidden:
        raise RuntimeError(f"phase-6iz refused to write {resolved}")
    if any(str(scenario) in resolved.name for scenario in RESERVED_TEST):
        raise RuntimeError(f"phase-6iz refused a reserved-test path: {resolved}")
    return resolved


def assign_gate(*, failures: int, ci_low: float, ci_high: float) -> str:
    """Positive evidence requires a strictly positive lower confidence bound."""
    if int(failures) > 0:
        return "CLOSED_LOOP_AUDIT_BLOCKED"
    if float(ci_low) > 0.0:
        return "CLOSED_LOOP_GAIN_SUPPORTED"
    if float(ci_high) < 0.0:
        return "CLOSED_LOOP_GAIN_NEGATIVE"
    return "CLOSED_LOOP_GAIN_INCONCLUSIVE"


__all__ = [
    "BIT_TIE",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "DECISION_TIME",
    "EPOCH6_CKPT",
    "HORIZON",
    "OUTPUT",
    "RATE_ATOL",
    "RESERVED_TEST",
    "SCHEME_C_CKPT",
    "SMOKE",
    "VALIDATION_POOL",
    "assert_output_path",
    "assign_gate",
]
