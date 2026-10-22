"""Phase-9.4.3 protocol. Order-free decoder only; encoders and L3-B stay frozen."""
from __future__ import annotations

from pathlib import Path

from wt.htw_phase92.run_phase92 import BATCH, COLLECT_SEEDS, EPOCHS, LR
from wt.htw_phase94.training import AUX_RANK_WEIGHT
from wt.htw_phase941.protocol import dump_frozen, sha256_bytes
from wt.htw_phase941.protocol import MODEL_SEEDS

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "outputs" / "order_free_decoder"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase9_4_3_order_free_decoder_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase9_4_3_order_free_decoder_report.md"
PHASE942_INTER = ROOT / "src" / "wt" / "htw_phase942" / "outputs" / "conditional_graph" / "phase942_interference_thresholds.json"
PHASE942_COMP = ROOT / "src" / "wt" / "htw_phase942" / "outputs" / "conditional_graph" / "phase942_competition_thresholds.json"
PHASE942_CONF = ROOT / "src" / "wt" / "htw_phase942" / "outputs" / "conditional_graph" / "phase942_relational_confirmation_manifest.json"

# 0-39 HTW L3-B history; 40-598 scanned or selected in Phase-9.4.2.
BLOCKED_SCENARIO_SEEDS = tuple(range(0, 600))
SCAN_START = 600
N_PER_REGIME = 16
N_CONFIRM = 64
N_BOOT = 2000
BOOT_SEED = 94301
AUDIT_SEEDS = tuple(range(9000, 9032))
TIE_ATOL = 1.0e-6

__all__ = [
    "AUDIT_SEEDS",
    "AUX_RANK_WEIGHT",
    "BATCH",
    "BLOCKED_SCENARIO_SEEDS",
    "BOOT_SEED",
    "COLLECT_SEEDS",
    "EPOCHS",
    "LR",
    "MODEL_SEEDS",
    "N_BOOT",
    "N_CONFIRM",
    "N_PER_REGIME",
    "OUTPUT",
    "PACK_REPORT",
    "PHASE942_COMP",
    "PHASE942_CONF",
    "PHASE942_INTER",
    "REPORT",
    "ROOT",
    "SCAN_START",
    "TIE_ATOL",
    "dump_frozen",
    "sha256_bytes",
]
