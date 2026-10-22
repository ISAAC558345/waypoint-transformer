"""Phase-10.0 freeze protocol. No training."""
from __future__ import annotations

from pathlib import Path

from wt.htw_phase9.const import EVAL_SEEDS, HOLD, TRAIN_SEEDS
from wt.htw_phase92.run_phase92 import COLLECT_SEEDS
from wt.htw_phase941.protocol import MODEL_SEEDS, dump_frozen, sha256_bytes
from wt.htw_phase945.protocol import OUTPUT as PHASE945_OUTPUT
from wt.htw_phase945.protocol import PHASE943_CKPT, POOL_SEED, POOL_SIZE, ROOT

OUTPUT = Path(__file__).resolve().parent / "outputs" / "handoff"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase10_0_high_level_freeze_and_low_level_handoff.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase10_0_high_level_freeze_and_low_level_handoff.md"
ARCHIVE = OUTPUT / "frozen_deepsets_util"
DESIGN_SEEDS = tuple(int(s) for s in TRAIN_SEEDS[:4])
ATOL_MOVE = 1.0e-12
ATOL_POS = 1.0e-12
ATOL_QUEUE = 1.0e-12
ATOL_PROGRESS = 1.0e-12
ATOL_PAYLOAD = 0.0

__all__ = [
    "ARCHIVE",
    "ATOL_MOVE",
    "ATOL_PAYLOAD",
    "ATOL_POS",
    "ATOL_PROGRESS",
    "ATOL_QUEUE",
    "COLLECT_SEEDS",
    "DESIGN_SEEDS",
    "EVAL_SEEDS",
    "HOLD",
    "MODEL_SEEDS",
    "OUTPUT",
    "PACK_REPORT",
    "PHASE943_CKPT",
    "PHASE945_OUTPUT",
    "POOL_SEED",
    "POOL_SIZE",
    "REPORT",
    "ROOT",
    "dump_frozen",
    "sha256_bytes",
]
