"""Frozen FINAL-HIER constants copied from Phase-10.5/10.6 manifests."""
from __future__ import annotations

from pathlib import Path

from . import _bootstrap  # noqa: F401

from wt.models.edge_bound.const import HOLD, IDLE_INDEX, MAX_BUOYS, MAX_UAVS
from wt.legacy.htw_historical101.protocol import HIGH_CKPT
from wt.legacy.htw_historical941.protocol import sha256_bytes
from wt.common.paths import repo_root

ROOT = repo_root()
EXPECTED_HIGH_SHA = "dbd6196b23c111a88a4d86a9054ca1709e459e5b035c1d9210639bedd2dda109"
EXPECTED_ENV_SHA = "de22347ea625a330f61e8f62b2a973066f4669c7498fd58bac2fd8bf706632ba"
DEFAULT_DEMO_SEED = 8
UNTOUCHED_SEEDS = frozenset(range(106000, 106064))
ARTIFACTS = ROOT / "final_artifacts"
FINAL_CKPT = ARTIFACTS / "checkpoints" / "deepsets_util_seed0.pt"
SOURCE_CKPT = HIGH_CKPT
DEMO_SEEDS_ALLOWED_NOTE = "design/demo seeds such as 8-15; never Phase-10.6 untouched 106000-106063"

__all__ = [
    "ARTIFACTS",
    "DEFAULT_DEMO_SEED",
    "EXPECTED_ENV_SHA",
    "EXPECTED_HIGH_SHA",
    "FINAL_CKPT",
    "HOLD",
    "IDLE_INDEX",
    "MAX_BUOYS",
    "MAX_UAVS",
    "ROOT",
    "SOURCE_CKPT",
    "UNTOUCHED_SEEDS",
    "sha256_bytes",
]