"""HL-RL-ALIGN protocol. Retrain DeepSets labels only. Low-level frozen."""
from __future__ import annotations

import json
from pathlib import Path

from wt.htw_phase92.run_phase92 import BATCH, COLLECT_SEEDS, EPOCHS, EVAL_SEEDS, LR
from wt.htw_phase941.protocol import MODEL_SEEDS, sha256_bytes
from wt.htw_phase945.protocol import (
    HOLD,
    LAMBDA_RANK,
    LAMBDA_REF,
    NORM_EPS,
    OUTPUT as PHASE945_OUTPUT,
    POOL_SEED,
    POOL_SIZE,
    TAU,
    VAL_EVERY,
)
from wt.lowlevel_v2.integration_phase1.protocol import (
    CKPT_UPDATE,
    EXPECTED_HIGH_SHA,
    LOW_SEEDS,
    ckpt_path,
    high_sha,
    load_phase3_gate,
    load_phase4a_gate,
    load_phase4b_gate,
    load_pools,
    refuse_compat,
)
from wt.lowlevel_v2.protocol import FINAL_CKPT, ROOT

INTEG2 = Path(__file__).resolve().parent
OUTPUT = INTEG2 / "outputs"
SHARD_DIR = OUTPUT / "utility_shards"
COMPAT_GATE = ROOT / "results/root_exports/HL_LL_COMPAT_GATE.json"
OLD_HL_CKPT = FINAL_CKPT
UTILITY_EPS = 1.0
NEAR_OPT_FRAC = 0.95
CHECKPOINT_RULE = "best_validation_mean_candidate_regret"


def load_compat_gate() -> dict:
    payload = json.loads(COMPAT_GATE.read_text(encoding="utf-8"))
    if payload.get("COMPAT") != "B":
        raise RuntimeError("HL-LL COMPAT gate is not B")
    if payload.get("high_level_retraining_needed") is not True:
        raise RuntimeError("high_level_retraining_needed is not true")
    return payload


def file_sha(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


__all__ = [
    "BATCH",
    "CHECKPOINT_RULE",
    "CKPT_UPDATE",
    "COLLECT_SEEDS",
    "COMPAT_GATE",
    "EPOCHS",
    "EVAL_SEEDS",
    "EXPECTED_HIGH_SHA",
    "FINAL_CKPT",
    "HOLD",
    "INTEG2",
    "LAMBDA_RANK",
    "LAMBDA_REF",
    "LOW_SEEDS",
    "LR",
    "MODEL_SEEDS",
    "NEAR_OPT_FRAC",
    "NORM_EPS",
    "OLD_HL_CKPT",
    "OUTPUT",
    "PHASE945_OUTPUT",
    "POOL_SEED",
    "POOL_SIZE",
    "ROOT",
    "SHARD_DIR",
    "TAU",
    "UTILITY_EPS",
    "VAL_EVERY",
    "ckpt_path",
    "file_sha",
    "high_sha",
    "load_compat_gate",
    "load_phase3_gate",
    "load_phase4a_gate",
    "load_phase4b_gate",
    "load_pools",
    "refuse_compat",
    "sha256_bytes",
]
