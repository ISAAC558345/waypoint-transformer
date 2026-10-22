"""Phase-9.4.5 frozen protocol. Objective change only; 9.4.3 architecture stays locked."""
from __future__ import annotations

from pathlib import Path

from wt.htw_phase9.const import EVAL_SEEDS, HOLD, TRAIN_SEEDS
from wt.htw_phase92.run_phase92 import BATCH, COLLECT_SEEDS, EPOCHS, LR
from wt.htw_phase941.protocol import MODEL_SEEDS, dump_frozen, sha256_bytes

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "outputs" / "utility_aligned"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase9_4_5_utility_aligned_objective_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase9_4_5_utility_aligned_objective_report.md"
PHASE943_CKPT = ROOT / "src" / "wt" / "htw_phase943" / "outputs" / "order_free_decoder"
PHASE942_INTER = ROOT / "src" / "wt" / "htw_phase942" / "outputs" / "conditional_graph" / "phase942_interference_thresholds.json"
PHASE942_COMP = ROOT / "src" / "wt" / "htw_phase942" / "outputs" / "conditional_graph" / "phase942_competition_thresholds.json"

POOL_SIZE = 24
POOL_SEED = 94501
TAU = 1.0
LAMBDA_RANK = 0.25
LAMBDA_REF = 0.10
NORM_EPS = 1.0
DELTA_RANGE_FRAC = 0.05
VAL_EVERY = 10
N_PER_REGIME = 16
N_BOOT = 2000
BOOT_SEED = 94502
# 0-1320 used or scanned through Phase-9.4.3; 9000-9031 permutation audit in 9.4.3/9.4.4.
BLOCKED_SCENARIO_SEEDS = tuple(sorted(set(range(0, 1321)) | set(range(9000, 9032))))
SCAN_START = 1321

__all__ = [
    "BATCH",
    "BLOCKED_SCENARIO_SEEDS",
    "BOOT_SEED",
    "COLLECT_SEEDS",
    "DELTA_RANGE_FRAC",
    "EPOCHS",
    "EVAL_SEEDS",
    "HOLD",
    "LAMBDA_RANK",
    "LAMBDA_REF",
    "LR",
    "MODEL_SEEDS",
    "NORM_EPS",
    "N_BOOT",
    "N_PER_REGIME",
    "OUTPUT",
    "PACK_REPORT",
    "PHASE942_COMP",
    "PHASE942_INTER",
    "PHASE943_CKPT",
    "POOL_SEED",
    "POOL_SIZE",
    "REPORT",
    "ROOT",
    "SCAN_START",
    "TAU",
    "TRAIN_SEEDS",
    "VAL_EVERY",
    "dump_frozen",
    "sha256_bytes",
]
