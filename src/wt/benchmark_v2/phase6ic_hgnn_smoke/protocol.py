"""Phase-6I-C split rules. Reserved scenarios are never opened."""
from __future__ import annotations

from pathlib import Path

from wt.benchmark_v2.phase6ib_highlevel_dataset.hgnn_revision import REVISION_DATASET
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import PRIMARY_TIMES

ROOT = Path(__file__).resolve().parents[4]
ENCODER_CKPT = ROOT / "results/phase5l_joint_encoder" / "joint_utility_encoder.pt"
DECODER_CKPT = ROOT / "results/phase5k_joint_decoder" / "matching_decoder.pt"
TRAIN_POOL = tuple(range(322000, 322048))
VALIDATION_POOL = tuple(range(322048, 322064))
RESERVED_TEST = tuple(range(322064, 322080))
GRAPH_KEYS = ("uav", "buoy", "uav_mask", "buoy_mask", "global", "edge", "edge_uu")


def assert_open(scenario: int, split: str) -> None:
    value = int(scenario)
    if value in RESERVED_TEST:
        raise RuntimeError(f"phase-6ic refused reserved test scenario {value}")
    pool = {"train": TRAIN_POOL, "validation": VALIDATION_POOL}[split]
    if value not in pool:
        raise RuntimeError(f"phase-6ic {split} split does not include {value}")


__all__ = [
    "DECODER_CKPT",
    "ENCODER_CKPT",
    "GRAPH_KEYS",
    "PRIMARY_TIMES",
    "RESERVED_TEST",
    "REVISION_DATASET",
    "TRAIN_POOL",
    "VALIDATION_POOL",
    "assert_open",
]
