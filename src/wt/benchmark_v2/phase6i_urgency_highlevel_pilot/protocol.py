"""Fixed Stage D0 rules. Sampling and signal thresholds are set before any urgent rollout."""
from __future__ import annotations

import json
from pathlib import Path

from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6iaa_formal_planner.protocol import (
    ADAPTER_CONFIG_SHA256,
    EPOCH6_SHA256,
    MAPPO_SHA256,
    SCHEME_C_SHA256,
)
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import (
    DEV_TEST_RESERVED,
    EPS_BITS,
    GENERATOR,
    HOLD,
    LABEL_NAME,
    PHASE,
    PILOT_POOL,
    PRIMARY_TIMES,
    SELECTED_K,
    TRAIN_POOL,
    VALIDATION_POOL,
    assert_allowed,
)
from wt.benchmark_v2.phase6is_assignment_head.protocol import EPOCH6_CKPT, SCHEME_C_CKPT
from wt.benchmark_v2.phase6iy_candidate_adapter.protocol import config_sha256

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results" / "phase6i_urgency_highlevel_pilot"
REGISTRY = PHASE / "phase6ib_data_registry.json"
DATASET = PHASE / "dataset"
INSTRUCTION_COMMIT = "a83869d5155c485c8cc752a43d41d246130e784c"
# Parent of the commit that deleted the npz dumps. Blobs are read from here and never written back.
FROZEN_SHARD_COMMIT = "aadbb3c5c5954669dd335dce9afb3462d7de114c"
PILOT_SEED = 601
TOP_K = 32
SAMPLE_K = 96
STRATA = 8
PER_STRATUM = 12
MAX_CANDIDATES_PER_STATE = TOP_K + SAMPLE_K
PRIMARY_TIME_COUNT = len(PRIMARY_TIMES)
MAX_ROLLOUTS = 4 * PRIMARY_TIME_COUNT * MAX_CANDIDATES_PER_STATE
# One timed state above this stops the pilot. Twelve such states would pass three hours.
MAX_STATE_SECONDS = 900.0
# A full-state repeat is extra confirmation, not a larger candidate set.
REPEAT_STATE_IF_UNDER_SECONDS = 120.0
REPEAT_CANDIDATES_IF_SLOW = 4
PAYLOAD_SCALE = 1.0e6
# Rank signal is assessed after the rollouts. These cutoffs are not retuned from the outcome.
USEFUL_SPEARMAN_BELOW = 0.95
USEFUL_REVERSAL_ABOVE = 0.05
GATE_PASS = "HIGHLEVEL_URGENCY_LABEL_PILOT_PASS"
GATE_INCONCLUSIVE = "HIGHLEVEL_URGENCY_LABEL_PILOT_INCONCLUSIVE"
GATE_BLOCKED = "HIGHLEVEL_URGENCY_LABEL_PILOT_BLOCKED"
CANDIDATE_KEYS = ("candidate_id", "assignment", "j15_bits", "utility_rank", "tie_group_id")
LAMBDA_H = None


def pilot_scenarios() -> tuple[int, ...]:
    """First four train ids in the Phase-6I-B registry. Validation and reserved ids are refused."""
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    train = tuple(int(item) for item in payload["highlevel_v2_train"])
    validation = tuple(int(item) for item in payload["highlevel_v2_validation"])
    reserved = tuple(int(item) for item in payload["highlevel_v2_dev_test_reserved"])
    if train != tuple(int(item) for item in TRAIN_POOL):
        raise RuntimeError("registry train pool does not match TRAIN_POOL")
    if validation != tuple(int(item) for item in VALIDATION_POOL):
        raise RuntimeError("registry validation pool does not match VALIDATION_POOL")
    if reserved != tuple(int(item) for item in DEV_TEST_RESERVED):
        raise RuntimeError("registry reserved pool does not match DEV_TEST_RESERVED")
    chosen = train[:4]
    if chosen != tuple(int(item) for item in PILOT_POOL):
        raise RuntimeError("registry prefix does not match the pre-registered four train scenarios")
    if any(item in validation or item in reserved for item in chosen):
        raise RuntimeError("pilot scenarios overlap validation or reserved test")
    for item in chosen:
        assert_allowed(int(item), split="train")
    return chosen


def assert_pilot_scenario(scenario_id: int) -> int:
    value = int(scenario_id)
    if value not in pilot_scenarios():
        raise RuntimeError(f"stage D0 refuses scenario {value}")
    return value


def assert_train_scenario(scenario_id: int) -> int:
    """Any Phase-6I-B train id. Validation and reserved ids stay outside this loader."""
    value = int(scenario_id)
    assert_allowed(value, split="train")
    if value in VALIDATION_POOL or value in DEV_TEST_RESERVED:
        raise RuntimeError(f"train shard loader refuses scenario {value}")
    return value


def frozen_relpath(scenario_id: int, time_step: int, kind: str) -> str:
    scenario = assert_train_scenario(scenario_id)
    if int(time_step) not in PRIMARY_TIMES:
        raise RuntimeError(f"stage D0 refuses time {time_step}")
    if kind not in {"state", "candidates", "manifest"}:
        raise RuntimeError(f"stage D0 refuses shard kind {kind}")
    suffix = "json" if kind == "manifest" else "npz"
    return (
        "src/wt/benchmark_v2/phase6ib_highlevel_dataset/dataset/train/"
        f"scenario_{scenario}/t{int(time_step):03d}_{kind}.{suffix}"
    )


def choose_gate(*, mechanical_ok: bool, budget_ok: bool, mean_spearman: float | None, mean_reversal: float | None) -> str:
    """Mechanical or budget failure blocks. A weak rank change stays inconclusive."""
    if not mechanical_ok or not budget_ok:
        return GATE_BLOCKED
    spearman_useful = mean_spearman is not None and float(mean_spearman) < float(USEFUL_SPEARMAN_BELOW)
    reversal_useful = mean_reversal is not None and float(mean_reversal) > float(USEFUL_REVERSAL_ABOVE)
    if spearman_useful or reversal_useful:
        return GATE_PASS
    return GATE_INCONCLUSIVE


__all__ = [
    "ADAPTER_CONFIG_SHA256",
    "CANDIDATE_KEYS",
    "DATASET",
    "EPOCH6_CKPT",
    "EPOCH6_SHA256",
    "EPS_BITS",
    "FROZEN_SHARD_COMMIT",
    "GATE_BLOCKED",
    "GATE_INCONCLUSIVE",
    "GATE_PASS",
    "GENERATOR",
    "HOLD",
    "INSTRUCTION_COMMIT",
    "LABEL_NAME",
    "LAMBDA_H",
    "MAPPO_SHA256",
    "MAX_CANDIDATES_PER_STATE",
    "MAX_ROLLOUTS",
    "MAX_STATE_SECONDS",
    "OUTPUT",
    "PAYLOAD_SCALE",
    "PER_STRATUM",
    "PILOT_SEED",
    "PRIMARY_TIMES",
    "REPEAT_CANDIDATES_IF_SLOW",
    "REPEAT_STATE_IF_UNDER_SECONDS",
    "ROOT",
    "SAMPLE_K",
    "SCHEME_C_CKPT",
    "SCHEME_C_SHA256",
    "SELECTED_K",
    "STRATA",
    "TOP_K",
    "USEFUL_REVERSAL_ABOVE",
    "USEFUL_SPEARMAN_BELOW",
    "assert_pilot_scenario",
    "assert_train_scenario",
    "choose_gate",
    "config_sha256",
    "frozen_relpath",
    "pilot_scenarios",
    "primary_checkpoint",
]
