"""Phase-10.4 frozen validation protocol. Methods copied from 10.1/10.3; not retuned."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wt.htw_phase9.const import HOLD
from wt.htw_phase92.run_phase92 import COLLECT_SEEDS
from wt.htw_phase941.protocol import CONFIRMATION_SEEDS, USED_L3B_SCENARIO_SEEDS, dump_frozen, sha256_bytes
from wt.htw_phase943.protocol import AUDIT_SEEDS
from wt.htw_phase943.protocol import BLOCKED_SCENARIO_SEEDS as BLOCKED_943
from wt.htw_phase945.protocol import BLOCKED_SCENARIO_SEEDS as BLOCKED_945
from wt.htw_phase945.protocol import ROOT
from wt.htw_phase101.protocol import DESIGN_TRAIN_SEEDS, DESIGN_VAL_SEEDS, HIGH_CKPT, PILOT_UPDATES, PPO, VECTOR_SEED_PILOT
from wt.htw_phase101.protocol import OUTPUT as PHASE101_OUTPUT
from wt.htw_phase103.protocol import ENV_FILE, LOW_FINAL

OUTPUT = Path(__file__).resolve().parent / "outputs" / "validation"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase10_4_multiseed_low_level_validation_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase10_4_multiseed_low_level_validation_report.md"

LOW_SEEDS = (0, 1, 2, 3, 4)
NEW_LOW_SEEDS = (1, 2, 3, 4)
SEED0_FINAL = LOW_FINAL
N_SCENARIOS = 64
SCENARIO_BASE = 104000
N_BOOT = 2000
BOOT_SEED = 10401
UPDATES = int(PILOT_UPDATES)
TRANSITIONS = UPDATES * int(PPO.rollout_length) * int(PPO.num_envs)

PHASE103_B0 = 8500000.0
PHASE103_B1 = 7250000.0
PHASE103_B2 = 12875000.0
PHASE103_B3 = 10750000.0
PHASE103_B2_MINUS_B3 = 2125000.0

MANIFEST_SEED_KEYS = (
    "blocked_scenario_seeds",
    "collect_seeds",
    "confirmation_seeds",
    "design_train_seeds",
    "design_val_seeds",
    "eval_history_seeds",
    "eval_seeds",
    "independent_scenario_seeds",
    "scenario_seeds",
    "train_scenario_seeds",
    "used_l3b_scenario_seeds",
    "val_seeds",
)


def protocol_blocked_seeds() -> set[int]:
    blocked = set(int(s) for s in BLOCKED_945)
    blocked.update(int(s) for s in BLOCKED_943)
    blocked.update(int(s) for s in USED_L3B_SCENARIO_SEEDS)
    blocked.update(int(s) for s in CONFIRMATION_SEEDS)
    blocked.update(int(s) for s in COLLECT_SEEDS)
    blocked.update(int(s) for s in DESIGN_TRAIN_SEEDS)
    blocked.update(int(s) for s in DESIGN_VAL_SEEDS)
    blocked.update(int(s) for s in AUDIT_SEEDS)
    return blocked


def scan_manifest_scenario_seeds(root: Path | None = None) -> set[int]:
    found: set[int] = set()
    base = ROOT if root is None else Path(root)
    patterns = (
        "*confirmation*manifest*.json",
        "*scenario*manifest*.json",
        "*seed*manifest*.json",
        "*design*manifest*.json",
        "phase10*_protocol.json",
        "phase9*_protocol.json",
    )
    files: list[Path] = []
    for pattern in patterns:
        files.extend(base.rglob(pattern))
    for path in files:
        if "backup" in path.parts or "_backup_" in path.as_posix():
            continue
        if "htw_phase104" in path.parts and "outputs" in path.parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        found.update(_extract_seed_lists(payload))
    return found


def _extract_seed_lists(payload: Any) -> set[int]:
    found: set[int] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key) in MANIFEST_SEED_KEYS and isinstance(value, list):
                for item in value:
                    if isinstance(item, int) or (isinstance(item, str) and str(item).lstrip("-").isdigit()):
                        found.add(int(item))
            else:
                found.update(_extract_seed_lists(value))
    elif isinstance(payload, list):
        for item in payload:
            found.update(_extract_seed_lists(item))
    return found


def all_blocked_scenario_seeds() -> set[int]:
    return protocol_blocked_seeds() | scan_manifest_scenario_seeds()


__all__ = [
    "BOOT_SEED",
    "DESIGN_TRAIN_SEEDS",
    "DESIGN_VAL_SEEDS",
    "ENV_FILE",
    "HIGH_CKPT",
    "HOLD",
    "LOW_SEEDS",
    "N_BOOT",
    "N_SCENARIOS",
    "NEW_LOW_SEEDS",
    "OUTPUT",
    "PACK_REPORT",
    "PHASE101_OUTPUT",
    "PHASE103_B0",
    "PHASE103_B1",
    "PHASE103_B2",
    "PHASE103_B2_MINUS_B3",
    "PHASE103_B3",
    "PPO",
    "REPORT",
    "ROOT",
    "SCENARIO_BASE",
    "SEED0_FINAL",
    "TRANSITIONS",
    "UPDATES",
    "VECTOR_SEED_PILOT",
    "all_blocked_scenario_seeds",
    "dump_frozen",
    "protocol_blocked_seeds",
    "scan_manifest_scenario_seeds",
    "sha256_bytes",
]
