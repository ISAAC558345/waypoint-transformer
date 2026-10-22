"""Phase-9.4.1 frozen protocol. Does not edit Phase-9.4 models or L3-B."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from wt.htw_phase9.const import EVAL_SEEDS, HOLD, SEED, TRAIN_SEEDS
from wt.htw_phase92.run_phase92 import BATCH, COLLECT_SEEDS, EPOCHS, LR
from wt.htw_phase94.training import AUX_RANK_WEIGHT

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "outputs" / "graph_stability"
PHASE94_CKPT = ROOT / "src" / "wt" / "htw_phase94" / "outputs" / "autoregressive_assignment"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase9_4_1_graph_stability_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase9_4_1_graph_stability_report.md"

USED_L3B_SCENARIO_SEEDS = tuple(sorted(set(list(TRAIN_SEEDS) + list(EVAL_SEEDS) + list(COLLECT_SEEDS))))
CONFIRMATION_SEEDS = tuple(range(24, 40))
MODEL_SEEDS = (0, 1, 2, 3, 4)
PHASE94_OFFICIAL_MODEL_SEED = 0
N_BOOT = 2000
BOOT_SEED = 94101
CHECKPOINT_RULE = "last_epoch_fixed_phase94"
HUNGARIAN_DEEPSETS = 4.00e6
HUNGARIAN_HGNN = 2.625e6


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def dump_frozen(path: Path, payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw, encoding="utf-8")
    return sha256_bytes(raw.encode("utf-8"))


def confirmation_ok() -> None:
    overlap = set(CONFIRMATION_SEEDS) & set(USED_L3B_SCENARIO_SEEDS)
    if overlap:
        raise ValueError(f"confirmation seeds collide with used L3-B seeds: {sorted(overlap)}")
    if len(CONFIRMATION_SEEDS) < 16:
        raise ValueError("need at least 16 confirmation seeds")


def protocol_payload() -> dict[str, Any]:
    confirmation_ok()
    return {
        "phase": "9.4.1",
        "architecture_frozen": True,
        "ppo": False,
        "hold": HOLD,
        "epochs": EPOCHS,
        "batch": BATCH,
        "lr": LR,
        "aux_rank_weight": AUX_RANK_WEIGHT,
        "train_scenario_seeds": list(TRAIN_SEEDS),
        "eval_history_seeds": list(EVAL_SEEDS),
        "collect_seeds": list(COLLECT_SEEDS),
        "used_l3b_scenario_seeds": list(USED_L3B_SCENARIO_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "model_seeds": list(MODEL_SEEDS),
        "phase94_official_model_seed": PHASE94_OFFICIAL_MODEL_SEED,
        "checkpoint_rule": CHECKPOINT_RULE,
        "checkpoint_rule_frozen_before_training": True,
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "hungarian_deepsets_payload": HUNGARIAN_DEEPSETS,
        "hungarian_hgnn_payload": HUNGARIAN_HGNN,
        "heuristic_name": "shadow-search (n_near, n_far, idle)",
        "modified_decoder": False,
        "modified_hgnn": False,
        "modified_l3b": False,
        "phase94_init_seed": SEED,
    }


__all__ = [
    "BOOT_SEED",
    "CHECKPOINT_RULE",
    "COLLECT_SEEDS",
    "CONFIRMATION_SEEDS",
    "HUNGARIAN_DEEPSETS",
    "HUNGARIAN_HGNN",
    "MODEL_SEEDS",
    "N_BOOT",
    "OUTPUT",
    "PACK_REPORT",
    "PHASE94_CKPT",
    "PHASE94_OFFICIAL_MODEL_SEED",
    "REPORT",
    "ROOT",
    "USED_L3B_SCENARIO_SEEDS",
    "confirmation_ok",
    "dump_frozen",
    "protocol_payload",
    "sha256_bytes",
]
