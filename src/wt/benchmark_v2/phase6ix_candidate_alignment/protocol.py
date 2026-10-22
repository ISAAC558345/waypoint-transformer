"""Phase-6I-X pool-alignment rules. Neither generator constant is edited."""
from __future__ import annotations

from pathlib import Path

from wt.benchmark_v2.phase6f_system_bringup.protocol import CANDIDATE_K, GENERATOR
from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import ROOT
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import HOLD, SELECTED_K
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST, TRAIN_POOL, VALIDATION_POOL
from wt.benchmark_v2.phase6is_assignment_head.protocol import EPOCH6_CKPT, SCHEME_C_CKPT

OUTPUT = ROOT / "results/phase6ix_candidate_alignment"
PROBE = ((322000, 0), (322000, 15), (322000, 30))
FORMAL_K = int(CANDIDATE_K)
FROZEN_K = int(SELECTED_K)
RATE_ATOL = 1.0e-6


def assert_output_path(path: Path) -> Path:
    resolved = Path(path).resolve()
    forbidden = (EPOCH6_CKPT.resolve(), SCHEME_C_CKPT.resolve(), primary_checkpoint().resolve())
    if resolved in forbidden:
        raise RuntimeError(f"phase-6ix refused to write {resolved}")
    if any(str(scenario) in resolved.name for scenario in RESERVED_TEST):
        raise RuntimeError(f"phase-6ix refused a reserved-test path: {resolved}")
    return resolved


def assign_gate(*, comparable: bool, identical: bool) -> str:
    """Supported only when every checked state has the same rows and the same indices."""
    if not comparable:
        return "CANDIDATE_POOL_ALIGNMENT_INCONCLUSIVE"
    if identical:
        return "CANDIDATE_POOL_ALIGNMENT_SUPPORTED"
    return "CANDIDATE_POOL_ALIGNMENT_BLOCKED"


__all__ = [
    "EPOCH6_CKPT",
    "FORMAL_K",
    "FROZEN_K",
    "GENERATOR",
    "HOLD",
    "OUTPUT",
    "PROBE",
    "RATE_ATOL",
    "RESERVED_TEST",
    "SCHEME_C_CKPT",
    "TRAIN_POOL",
    "VALIDATION_POOL",
    "assert_output_path",
    "assign_gate",
]
