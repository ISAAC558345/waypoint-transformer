"""Phase-10.5 frozen FINAL-HIER benchmark protocol. No training."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wt.htw_phase9.const import HOLD
from wt.htw_phase92.run_phase92 import COLLECT_SEEDS
from wt.htw_phase941.protocol import CONFIRMATION_SEEDS, dump_frozen, sha256_bytes
from wt.htw_phase943.protocol import AUDIT_SEEDS, PHASE942_COMP, PHASE942_INTER
from wt.htw_phase943.protocol import BLOCKED_SCENARIO_SEEDS as BLOCKED_943
from wt.htw_phase945.protocol import BLOCKED_SCENARIO_SEEDS as BLOCKED_945
from wt.htw_phase945.protocol import PHASE943_CKPT, ROOT
from wt.htw_phase945.protocol import TAU, LAMBDA_RANK, LAMBDA_REF, POOL_SIZE, POOL_SEED, NORM_EPS
from wt.htw_phase101.protocol import DESIGN_TRAIN_SEEDS, DESIGN_VAL_SEEDS, HIGH_CKPT
from wt.htw_phase103.protocol import ENV_FILE
from wt.htw_phase104.protocol import MANIFEST_SEED_KEYS, SCENARIO_BASE as PHASE104_BASE, N_SCENARIOS as PHASE104_N

OUTPUT = Path(__file__).resolve().parent / "outputs" / "benchmark"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase10_5_final_architecture_benchmark_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase10_5_final_architecture_benchmark_report.md"

N_SCENARIOS = 64
SCENARIO_BASE = 105000
N_BOOT = 2000
BOOT_SEED = 10501
OLD_DEEPSETS = PHASE943_CKPT / "deepsets_seed0.pt"
HIGH_SEED = 0


def protocol_blocked_seeds() -> set[int]:
    blocked = set(int(s) for s in BLOCKED_945)
    blocked.update(int(s) for s in BLOCKED_943)
    blocked.update(int(s) for s in CONFIRMATION_SEEDS)
    blocked.update(int(s) for s in COLLECT_SEEDS)
    blocked.update(int(s) for s in DESIGN_TRAIN_SEEDS)
    blocked.update(int(s) for s in DESIGN_VAL_SEEDS)
    blocked.update(int(s) for s in AUDIT_SEEDS)
    blocked.update(range(int(PHASE104_BASE), int(PHASE104_BASE) + int(PHASE104_N)))
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
        if "htw_phase105" in path.parts and "outputs" in path.parts:
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
    "AUDIT_SEEDS",
    "BOOT_SEED",
    "DESIGN_TRAIN_SEEDS",
    "DESIGN_VAL_SEEDS",
    "ENV_FILE",
    "HIGH_CKPT",
    "HIGH_SEED",
    "HOLD",
    "LAMBDA_RANK",
    "LAMBDA_REF",
    "N_BOOT",
    "N_SCENARIOS",
    "NORM_EPS",
    "OLD_DEEPSETS",
    "OUTPUT",
    "PACK_REPORT",
    "PHASE942_COMP",
    "PHASE942_INTER",
    "PHASE943_CKPT",
    "POOL_SEED",
    "POOL_SIZE",
    "REPORT",
    "ROOT",
    "SCENARIO_BASE",
    "TAU",
    "all_blocked_scenario_seeds",
    "dump_frozen",
    "protocol_blocked_seeds",
    "sha256_bytes",
]
