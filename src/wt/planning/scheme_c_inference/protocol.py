"""Phase-6I-W interface rules. The official planner is not an import target."""
from __future__ import annotations

from pathlib import Path

from wt.control.mappo.protocol import ROOT
from wt.control.executor.protocol import primary_checkpoint
from wt.models.hgnn_assets.protocol import RESERVED_TEST, TRAIN_POOL, VALIDATION_POOL
from wt.models.assignment_head.protocol import EPOCH6_CKPT, SCHEME_C_CKPT

OUTPUT = ROOT / "results/phase6iw_inference_interface"
PROBE = ((322000, 0), (322000, 15), (322048, 15))
CANDIDATE_COUNT = 13327
PUBLISHED_TOP1_COUNTS = {"epoch6": 27, "scheme_c": 32}
REPEAT_ATOL = 1.0e-5
MAPPO_CKPT = primary_checkpoint()


def assert_output_path(path: Path) -> Path:
    resolved = Path(path).resolve()
    forbidden = (EPOCH6_CKPT.resolve(), SCHEME_C_CKPT.resolve(), MAPPO_CKPT.resolve())
    if resolved in forbidden or any(resolved.is_relative_to(item) for item in forbidden if item.exists() and item.is_dir()):
        raise RuntimeError(f"phase-6iw refused to write {resolved}")
    if any(str(scenario) in resolved.name for scenario in RESERVED_TEST):
        raise RuntimeError(f"phase-6iw refused a reserved-test path: {resolved}")
    return resolved


def assign_gate(blocked: bool) -> str:
    """Ready means the isolated scorer matches its saved pool.

    It does not mean the official planner can call it, and it does not
    mean a closed-loop gain.
    """
    if blocked:
        return "HIGH_LEVEL_INFERENCE_INTERFACE_BLOCKED"
    return "HIGH_LEVEL_INFERENCE_INTERFACE_READY"


__all__ = [
    "CANDIDATE_COUNT",
    "EPOCH6_CKPT",
    "MAPPO_CKPT",
    "OUTPUT",
    "PROBE",
    "PUBLISHED_TOP1_COUNTS",
    "REPEAT_ATOL",
    "RESERVED_TEST",
    "SCHEME_C_CKPT",
    "TRAIN_POOL",
    "VALIDATION_POOL",
    "assert_output_path",
    "assign_gate",
]