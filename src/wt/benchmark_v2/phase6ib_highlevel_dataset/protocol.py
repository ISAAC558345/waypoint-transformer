"""Phase-6I-B dataset rules. Splits are fixed before any new J15 label."""
from __future__ import annotations

import hashlib
from pathlib import Path

PHASE = Path(__file__).resolve().parent
ROOT = PHASE.parents[3]
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase6IB_HighLevelMacroUtilityDataset_CN.md"
CHECKPOINT_DIR = ROOT / "src" / "wt" / "benchmark_v2" / "phase6g_lowlevel_mappo" / "checkpoints"
PRIMARY_SEED = 0
TRAIN_POOL = tuple(range(322000, 322048))
VALIDATION_POOL = tuple(range(322048, 322064))
DEV_TEST_RESERVED = tuple(range(322064, 322080))
PILOT_POOL = tuple(range(322000, 322004))
OPEN_POOL = TRAIN_POOL + VALIDATION_POOL
PRIMARY_TIMES = (0, 15, 30)
HOLD = 15
SELECTED_K = 6
GENERATOR = "G1_RATE"
LABEL_NAME = "J15_MAPPO_CURRENT"
EPS_BITS = 1.0
DEGENERACY_MAX = 0.10
INTEGRITY_PER_10000 = 500
INTEGRITY_STAMP = "6IB-INTEGRITY-1"
PROTOCOL_STAMP = "6IB-PRE-J15-1"
STATE_SOURCE = "MAX_RATE_INDEPENDENT held for 15 steps and executed by frozen MAPPO-V2 seed0"
RESERVED_FRESH = tuple(range(320100, 320132))
HISTORICAL_DEV = tuple(range(320000, 320064))
LOWLEVEL_POOL = tuple(range(321000, 321128))


def assert_allowed(scenario_id: int, *, split: str) -> None:
    value = int(scenario_id)
    if value in DEV_TEST_RESERVED or value in RESERVED_FRESH or value in HISTORICAL_DEV or value in LOWLEVEL_POOL or 310100 <= value <= 310125:
        raise RuntimeError(f"phase-6ib refuses scenario {value}")
    pools = {"train": TRAIN_POOL, "validation": VALIDATION_POOL, "pilot": PILOT_POOL}
    if value not in pools[split]:
        raise RuntimeError(f"phase-6ib {split} split does not include {value}")


def integrity_selected(scenario_id: int, time_step: int, candidate_id: int) -> bool:
    """Fixed 5 percent of state-assignment pairs. The stamp is not retuned after labels exist."""
    raw = hashlib.sha256(f"{INTEGRITY_STAMP}|{int(scenario_id)}|{int(time_step)}|{int(candidate_id)}".encode("utf-8")).digest()
    return int.from_bytes(raw[:4], "little") % 10000 < INTEGRITY_PER_10000


def choose_gate(flags: dict) -> dict:
    if bool(flags["lineage_failure"]):
        gate, name = "DATA-E", "DATA_LINEAGE_FAILURE"
    elif bool(flags["leakage"]):
        gate, name = "DATA-C", "STATE_OR_FUTURE_LEAKAGE"
    elif bool(flags["reproducibility_failure"]):
        gate, name = "DATA-B", "LABEL_REPRODUCIBILITY_FAILURE"
    elif bool(flags["degenerate"]):
        gate, name = "DATA-D", "DATASET_DEGENERATE"
    elif bool(flags["data_a"]):
        gate, name = "DATA-A", "HIGHLEVEL_MACRO_DATASET_VALID"
    else:
        gate, name = "DATA-B", "LABEL_REPRODUCIBILITY_FAILURE"
    frozen = gate == "DATA-A"
    return {
        "gate": gate,
        "gate_name": name,
        "highlevel_dataset_frozen": frozen,
        "candidate_generator": GENERATOR,
        "selected_k": SELECTED_K,
        "label": LABEL_NAME,
        "primary_checkpoint": "phase6g_seed0.pt",
        "primary_seed_rule": "MINIMUM_SEED_ID",
        "mappo_v2_frozen": True,
        "high_level_dataset_only": True,
        "high_level_not_trained": True,
        "trained_hgnn": False,
        "trained_deepsets": False,
        "trained_mlp": False,
        "retrained_mappo": False,
        "planner_ready": False,
        "deployment_verified": False,
        "deployment_pass": False,
        "fresh_confirmation_untouched": True,
        "dev_test_untouched": True,
        "next_phase_may_train_encoders": frozen,
    }
