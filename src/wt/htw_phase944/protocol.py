"""Phase-9.4.4 frozen diagnostic protocol. Does not edit Phase-9.4.3 models."""
from __future__ import annotations

from pathlib import Path

from wt.htw_phase9.const import EVAL_SEEDS, TRAIN_SEEDS
from wt.htw_phase92.run_phase92 import BATCH, COLLECT_SEEDS, EPOCHS, LR
from wt.htw_phase941.protocol import dump_frozen, sha256_bytes
from wt.htw_phase941.protocol import MODEL_SEEDS
from wt.htw_phase94.training import AUX_RANK_WEIGHT

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "outputs" / "hgnn_instability"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase9_4_4_HGNN_instability_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase9_4_4_HGNN_instability_report.md"
CURVE_AUDIT = OUTPUT / "phase944_training_curve_audit.md"
ATTR_MD = OUTPUT / "phase944_encoder_decoder_attribution.md"
PHASE943 = ROOT / "src" / "wt" / "htw_phase943" / "outputs" / "order_free_decoder"
PHASE943_CKPT = PHASE943
LOG_CANDIDATES = (
    ROOT / "src" / "wt" / "htw_phase944" / "inputs" / "phase943_training_log.txt",
    Path(r"C:\Users\郑天成\.cursor\projects\e-srdp-NeurIPS-2023-waypoint-transformer-reinforcement-learning-via-supervised-learning-with-intermediate-targets-Supplemental-Conference-waypoint-transformer-official\terminals\500335.txt"),
)

N_NLL = 48
N_REP = 64
N_GRAD = 8
N_DECODE = 48
N_SENS = 8
BOOT_SEED = 94401
DESIGN_VAL_SEEDS = EVAL_SEEDS
DIAG_SPLIT = "historical_eval_8_15_plus_collect_states"

__all__ = [
    "ATTR_MD",
    "AUX_RANK_WEIGHT",
    "BATCH",
    "BOOT_SEED",
    "COLLECT_SEEDS",
    "CURVE_AUDIT",
    "DESIGN_VAL_SEEDS",
    "DIAG_SPLIT",
    "EPOCHS",
    "EVAL_SEEDS",
    "LOG_CANDIDATES",
    "LR",
    "MODEL_SEEDS",
    "N_DECODE",
    "N_GRAD",
    "N_NLL",
    "N_REP",
    "N_SENS",
    "OUTPUT",
    "PACK_REPORT",
    "PHASE943",
    "PHASE943_CKPT",
    "REPORT",
    "ROOT",
    "TRAIN_SEEDS",
    "dump_frozen",
    "sha256_bytes",
]
