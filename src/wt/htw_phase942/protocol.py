"""Phase-9.4.2 frozen inference-only protocol. Does not train or edit encoders/decoders."""
from __future__ import annotations

from pathlib import Path

from wt.htw_phase92.run_phase92 import COLLECT_SEEDS
from wt.htw_phase941.protocol import CONFIRMATION_SEEDS, MODEL_SEEDS, USED_L3B_SCENARIO_SEEDS, dump_frozen, sha256_bytes

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "outputs" / "conditional_graph"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase9_4_2_conditional_graph_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase9_4_2_conditional_graph_report.md"
PHASE941_LOCK = ROOT / "src" / "wt" / "htw_phase941" / "outputs" / "graph_stability" / "phase941_strata_lock.json"
PHASE941_CKPT = ROOT / "src" / "wt" / "htw_phase941" / "outputs" / "graph_stability" / "phase941_checkpoint_manifest.json"

BLOCKED_SCENARIO_SEEDS = tuple(sorted(set(list(USED_L3B_SCENARIO_SEEDS) + list(CONFIRMATION_SEEDS))))
N_BOOT = 2000
BOOT_SEED = 94201
ORDER_SEED = 94202
SCAN_START = 40
SCAN_LIMIT = 20000
N_PER_REGIME = 8


__all__ = [
    "BLOCKED_SCENARIO_SEEDS",
    "BOOT_SEED",
    "COLLECT_SEEDS",
    "MODEL_SEEDS",
    "N_BOOT",
    "N_PER_REGIME",
    "ORDER_SEED",
    "OUTPUT",
    "PACK_REPORT",
    "PHASE941_CKPT",
    "PHASE941_LOCK",
    "REPORT",
    "ROOT",
    "SCAN_LIMIT",
    "SCAN_START",
    "dump_frozen",
    "sha256_bytes",
]
