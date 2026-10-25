"""Stage B evaluation settings. Seeds are fixed before any development-pool result is read."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import (
    BENCHMARK_DEV,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    DEV_POOL,
    HISTORICAL,
    POSITIVE_BIT,
    RESERVED_POOL,
    assert_allowed,
)
from wt.benchmark_v2.phase6i_urgency_stage_a.protocol import AGE_SCALE_STEPS, SCHEMA_VERSION
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_urgency_stage_b"
FIGURES = OUTPUT / "figures"
LOG_SCHEMA_VERSION = "crosstask_baseline_v1"
INSTRUCTION_COMMIT = "97d35ad5d99a1f49e3124828e6ecc60e2ede84ff"
AGE_SCHEMA_VERSION = SCHEMA_VERSION
HIGH_URGENCY_QUANTILE = 0.75
FIXED_URGENCY_THRESHOLD = 0.5
REPLICATION_SEEDS = (0, 1, 2)
CONDITION_SPECS = (
    {
        "condition_id": "A",
        "condition_name": "scheme_c_dynamic",
        "random": False,
        "assignment_rule": "FormalPlanner scheme_c on the frozen G1_RATE k=6 legal pool",
    },
    {
        "condition_id": "B",
        "condition_name": "g1_rate_k6_dynamic",
        "random": False,
        "assignment_rule": "FormalPlanner g1_rate additive_assignment and the existing tie-break",
    },
    {
        "condition_id": "C",
        "condition_name": "g1_rate_k5_uniform",
        "random": True,
        "assignment_rule": "uniform draw from build_pools G1_RATE k=5, the phase6g training distribution",
    },
    {
        "condition_id": "D",
        "condition_name": "g1_rate_k6_uniform",
        "random": True,
        "assignment_rule": "uniform draw from frozen_candidate_keys; pool size is the returned length",
    },
)


def instruction_commit() -> str:
    """The Stage B instruction object. Kept beside the constant so a short hash cannot drift silently."""
    return INSTRUCTION_COMMIT


def require_dev_scenario(scenario_id: int) -> int:
    """Accept only the phase6g development pool. High-level guards are not disabled."""
    value = int(scenario_id)
    if value in RESERVED_TEST or value in RESERVED_POOL or value in BENCHMARK_DEV or value in HISTORICAL:
        raise RuntimeError(f"stage B refuses scenario {value}")
    assert_allowed(value, split="dev")
    return value


def dev_scenario_ids() -> tuple[int, ...]:
    chosen = tuple(int(item) for item in DEV_POOL)
    if len(chosen) != 32 or len(set(chosen)) != 32:
        raise RuntimeError("phase6g dev pool is not 32 distinct scenarios")
    for item in chosen:
        require_dev_scenario(item)
    return chosen


def replication_rng(scenario_id: int, condition_code: int, replication_seed: int) -> np.random.Generator:
    """Derive the episode RNG from the scenario, condition, and pre-registered replication seed."""
    sequence = np.random.SeedSequence([int(scenario_id), int(condition_code), int(replication_seed)])
    return np.random.default_rng(sequence)


def condition_code(condition_id: str) -> int:
    codes = {"A": 1, "B": 2, "C": 3, "D": 4}
    if condition_id not in codes:
        raise RuntimeError(f"unknown condition {condition_id}")
    return int(codes[condition_id])


def pool_hash(keys: list[tuple[int, ...]]) -> str:
    """Same byte hash the formal planner uses for a key matrix."""
    digest = hashlib.sha256()
    digest.update(np.asarray(keys, dtype=np.int64).tobytes())
    return digest.hexdigest()


__all__ = [
    "AGE_SCALE_STEPS",
    "AGE_SCHEMA_VERSION",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CONDITION_SPECS",
    "FIGURES",
    "FIXED_URGENCY_THRESHOLD",
    "HIGH_URGENCY_QUANTILE",
    "INSTRUCTION_COMMIT",
    "LOG_SCHEMA_VERSION",
    "OUTPUT",
    "POSITIVE_BIT",
    "REPLICATION_SEEDS",
    "condition_code",
    "dev_scenario_ids",
    "pool_hash",
    "replication_rng",
    "require_dev_scenario",
]
