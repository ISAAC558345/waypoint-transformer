"""Phase-6I-D training constants. Checkpoints never share a directory with Phase-5L."""
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
OUTPUT = ROOT / "results/phase6id_joint_utility_training"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase6I_JointUtilityHGNN_Training_CN.md"
MAX_EPOCHS = 10
PATIENCE = 4
GRAD_CLIP = 1.0
COLLAPSE_EPS = 1.0e-6
COLLAPSE_STOP = 0.5


def assert_output_path(path: Path) -> Path:
    resolved = Path(path).resolve()
    forbidden = {ENCODER_CKPT.resolve(), DECODER_CKPT.resolve(), REVISION_DATASET.resolve()}
    if resolved in forbidden or any(resolved.is_relative_to(item) for item in forbidden):
        raise RuntimeError(f"phase-6id refused to write {resolved}")
    if "phase5l_joint_encoder" in resolved.parts or "phase5k_joint_decoder" in resolved.parts:
        raise RuntimeError(f"phase-6id refused to write inside a Phase-5 checkpoint tree: {resolved}")
    if any(int(scenario) in RESERVED_TEST for scenario in _numbers(resolved.name)):
        raise RuntimeError(f"phase-6id refused a reserved-test path: {resolved}")
    return resolved


def _numbers(name: str) -> list[int]:
    digits = ""
    found = []
    for char in name:
        if char.isdigit():
            digits += char
        elif digits:
            found.append(int(digits))
            digits = ""
    if digits:
        found.append(int(digits))
    return found


def selection_key(metrics: dict[str, float]) -> tuple[float, float, float, float]:
    """Validation ranking first. Train loss is only the last tie break."""
    spearman = float(metrics["spearman"])
    if spearman != spearman:
        spearman = -1.0e18
    return (
        spearman,
        float(metrics["top1"]),
        -float(metrics["normalized_regret"]),
        -float(metrics["loss"]),
    )


def clear_improvement(trained: dict[str, float], frozen: dict[str, float]) -> bool:
    """A clear gain needs a better rank correlation, no worse Top-1 or regret, and a higher J15 payload."""
    required = ("spearman", "top1", "normalized_regret", "payload")
    for name in required:
        if float(trained[name]) != float(trained[name]) or float(frozen[name]) != float(frozen[name]):
            return False
    return bool(
        float(trained["spearman"]) > float(frozen["spearman"])
        and float(trained["top1"]) >= float(frozen["top1"])
        and float(trained["normalized_regret"]) <= float(frozen["normalized_regret"])
        and float(trained["payload"]) > float(frozen["payload"])
    )


__all__ = [
    "COLLAPSE_EPS",
    "COLLAPSE_STOP",
    "DECODER_CKPT",
    "ENCODER_CKPT",
    "GRAD_CLIP",
    "LEARNING_RATE",
    "MAX_EPOCHS",
    "OUTPUT",
    "PATIENCE",
    "REPORT",
    "RESERVED_TEST",
    "TRAIN_POOL",
    "VALIDATION_POOL",
    "WEIGHT_DECAY",
    "assert_output_path",
    "clear_improvement",
    "selection_key",
]
