"""Phase-6I-S constants. Existing checkpoints are read-only."""
from __future__ import annotations

from pathlib import Path

from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import (
    DECODER_CKPT,
    ENCODER_CKPT,
    RESERVED_TEST,
    REVISION_DATASET,
    TRAIN_POOL,
    VALIDATION_POOL,
)
from wt.lowlevel_v2.phase5l_joint_encoder.protocol import LEARNING_RATE, WEIGHT_DECAY

ROOT = Path(__file__).resolve().parents[4]
EPOCH6_CKPT = ROOT / "results/phase6id_joint_utility_training" / "epoch_006.pt"
SCHEME_C_CKPT = ROOT / "results/phase6if_ranking_loss" / "runs" / "c" / "epoch_001.pt"
OUTPUT = ROOT / "results/phase6is_assignment_conditioned"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase6I_AssignmentConditionedRepresentation_CN.md"

SEEDS = (0, 1)
MAX_EPOCHS = 3
GRAD_CLIP = 1.0
PAIR_CHUNK = 1024
HIDDEN = 64
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
MATERIAL_BITS = 1.0
MIN_POSITIVE_SCENARIOS = 12
PUBLISHED_TOP1_COUNTS = {"epoch6": 27, "scheme_c": 32}
SWAP_PROBE = (322051, 15, (1, 2))

RAW_NAMES = (
    "rel_x",
    "rel_y",
    "distance_norm",
    "steps_norm",
    "in_range",
    "hypothetical_rate_norm",
    "switch",
    "abandoned_progress",
    "queue_fraction",
    "log_interference",
    "same_cluster_fraction",
    "idle",
)


def assert_output_path(path: Path) -> Path:
    resolved = Path(path).resolve()
    forbidden = {
        ENCODER_CKPT.resolve(),
        DECODER_CKPT.resolve(),
        EPOCH6_CKPT.resolve(),
        SCHEME_C_CKPT.resolve(),
        REVISION_DATASET.resolve(),
    }
    if resolved in forbidden or any(resolved.is_relative_to(item) for item in forbidden if item.exists()):
        raise RuntimeError(f"phase-6is refused to write {resolved}")
    return resolved


def assign_gate(blocked: bool, checks_passed: bool, seed_reports: list[dict]) -> str:
    """Support needs every seed to clear the same decision bar.

    A mean J15 increase alone is not support. Top-1 must rise against both
    published checkpoints, the scenario-cluster interval must sit above zero,
    and the scenario signs must not split.
    """
    if blocked or not checks_passed or len(seed_reports) != len(SEEDS):
        return "ASSIGNMENT_CONDITIONED_REPRESENTATION_BLOCKED"
    if all(bool(report["supported"]) for report in seed_reports):
        return "ASSIGNMENT_CONDITIONED_REPRESENTATION_SUPPORTED"
    return "ASSIGNMENT_CONDITIONED_REPRESENTATION_INCONCLUSIVE"


def seed_supported(report: dict) -> bool:
    required = (
        "top1_above_epoch6",
        "top1_above_scheme_c",
        "t15_top1_above_epoch6",
        "t15_top1_above_scheme_c",
        "mean_ci_above_epoch6",
        "mean_ci_above_scheme_c",
        "median_above_epoch6",
        "median_above_scheme_c",
        "t15_mean_ci_above_epoch6",
        "t15_mean_ci_above_scheme_c",
        "scenarios_consistent_epoch6",
        "scenarios_consistent_scheme_c",
    )
    return all(bool(report[name]) for name in required)


__all__ = [
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "DECODER_CKPT",
    "ENCODER_CKPT",
    "EPOCH6_CKPT",
    "GRAD_CLIP",
    "HIDDEN",
    "LEARNING_RATE",
    "MATERIAL_BITS",
    "MAX_EPOCHS",
    "MIN_POSITIVE_SCENARIOS",
    "OUTPUT",
    "PAIR_CHUNK",
    "PUBLISHED_TOP1_COUNTS",
    "RAW_NAMES",
    "REPORT",
    "RESERVED_TEST",
    "SCHEME_C_CKPT",
    "SEEDS",
    "SWAP_PROBE",
    "TRAIN_POOL",
    "VALIDATION_POOL",
    "WEIGHT_DECAY",
    "assert_output_path",
    "assign_gate",
    "seed_supported",
]
