"""Phase-6I-U constants. Thresholds are fixed before training."""
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
from wt.benchmark_v2.phase6is_assignment_head.protocol import EPOCH6_CKPT, SCHEME_C_CKPT
from wt.lowlevel_v2.phase5l_joint_encoder.protocol import LEARNING_RATE, WEIGHT_DECAY, WIDTH

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/phase6iu_pairwise_interaction"
HEAD_OUTPUT = ROOT / "results/phase6is_assignment_conditioned"

SEEDS = (0, 1, 2, 3, 4)
MAX_EPOCHS = 5
HIDDEN = 32
LAMBDA_PAIR = 1.0
CANDIDATE_CHUNK = 1024
GRAD_CLIP = 1.0
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0
BIT_TIE = 1.0
MATERIAL_BITS = 1.0e6
SCORE_DUST = 1.0e-3
MIN_POSITIVE_SCENARIOS = 12
ACTIVE_FRACTION = 0.50
SIGN_AGREEMENT = 0.60
CONCENTRATION = 0.50
PUBLISHED_TOP1_COUNTS = {"epoch6": 27, "scheme_c": 32}


def assert_output_path(path: Path) -> Path:
    resolved = Path(path).resolve()
    forbidden = (
        ENCODER_CKPT.resolve(),
        DECODER_CKPT.resolve(),
        EPOCH6_CKPT.resolve(),
        SCHEME_C_CKPT.resolve(),
        REVISION_DATASET.resolve(),
        HEAD_OUTPUT.resolve(),
    )
    if resolved in forbidden or any(resolved.is_relative_to(item) for item in forbidden if item.exists()):
        raise RuntimeError(f"phase-6iu refused to write {resolved}")
    if any(str(scenario) in resolved.name for scenario in RESERVED_TEST):
        raise RuntimeError(f"phase-6iu refused a reserved-test path: {resolved}")
    return resolved


def assign_gate(blocked: bool, checks_passed: bool, seed_reports: list[dict]) -> str:
    """Support needs every seed to clear the same pre-registered bar.

    A lower training loss or a higher mean J15 is not enough. Top-1, regret,
    the scenario-cluster interval, gain concentration, and the disjoint-swap
    audit all have to pass on all five seeds.
    """
    if blocked or not checks_passed or len(seed_reports) != len(SEEDS):
        return "PAIRWISE_INTERACTION_IMPLEMENTATION_BLOCKED"
    if all(bool(report["supported"]) for report in seed_reports):
        return "PAIRWISE_INTERACTION_GAIN_SUPPORTED"
    return "PAIRWISE_INTERACTION_GAIN_INCONCLUSIVE"


def seed_supported(report: dict) -> bool:
    required = (
        "top1_above_scheme_c",
        "regret_not_worse",
        "mean_ci_above_scheme_c",
        "scenarios_consistent",
        "gain_not_concentrated",
        "interaction_active",
        "interaction_sign",
        "quadruple_improves",
    )
    return all(bool(report[name]) for name in required)


__all__ = [
    "ACTIVE_FRACTION",
    "BIT_TIE",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CANDIDATE_CHUNK",
    "CONCENTRATION",
    "DECODER_CKPT",
    "ENCODER_CKPT",
    "EPOCH6_CKPT",
    "GRAD_CLIP",
    "HEAD_OUTPUT",
    "HIDDEN",
    "LAMBDA_PAIR",
    "LEARNING_RATE",
    "MATERIAL_BITS",
    "MAX_EPOCHS",
    "MIN_POSITIVE_SCENARIOS",
    "OUTPUT",
    "PUBLISHED_TOP1_COUNTS",
    "RESERVED_TEST",
    "REVISION_DATASET",
    "SCHEME_C_CKPT",
    "SCORE_DUST",
    "SEEDS",
    "SIGN_AGREEMENT",
    "TRAIN_POOL",
    "VALIDATION_POOL",
    "WEIGHT_DECAY",
    "WIDTH",
    "assert_output_path",
    "assign_gate",
    "seed_supported",
]
