"""Phase-10.6 one-shot untouched protocol. Methods frozen from Phase-10.5."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wt.htw_phase9.const import HOLD
from wt.htw_phase941.protocol import dump_frozen, sha256_bytes
from wt.htw_phase945.protocol import ROOT
from wt.htw_phase101.protocol import HIGH_CKPT
from wt.htw_phase103.protocol import ENV_FILE
from wt.htw_phase104.protocol import MANIFEST_SEED_KEYS
from wt.htw_phase105.protocol import (
    BOOT_SEED,
    N_BOOT,
    N_SCENARIOS,
    OLD_DEEPSETS,
    protocol_blocked_seeds,
)
from wt.htw_phase105.protocol import OUTPUT as PHASE105_OUTPUT
from wt.htw_phase105.protocol import SCENARIO_BASE as PHASE105_BASE

OUTPUT = Path(__file__).resolve().parent / "outputs" / "untouched"
REPORT = ROOT / "docs/archive/phase_reports/HTW_Phase10_6_final_untouched_evaluation_report.md"
PACK_REPORT = Path(__file__).resolve().parent / "HTW_Phase10_6_final_untouched_evaluation_report.md"

EXPECTED_HIGH_SHA = "dbd6196b23c111a88a4d86a9054ca1709e459e5b035c1d9210639bedd2dda109"
EXPECTED_ENV_SHA = "de22347ea625a330f61e8f62b2a973066f4669c7498fd58bac2fd8bf706632ba"
SCENARIO_BASE = 106000
PHASE105_N = 64
ARCH_MANIFEST = PHASE105_OUTPUT / "phase105_final_architecture_manifest.json"
FREEZE_FILE = PHASE105_OUTPUT / "phase105_final_test_protocol_freeze.json"


def historical_blocked_seeds() -> set[int]:
    blocked = protocol_blocked_seeds()
    blocked.update(range(int(PHASE105_BASE), int(PHASE105_BASE) + int(PHASE105_N)))
    blocked.update(_scan_including_phase105())
    return blocked


def _scan_including_phase105() -> set[int]:
    found: set[int] = set()
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
        files.extend(ROOT.rglob(pattern))
    for path in files:
        if "backup" in path.parts or "_backup_" in path.as_posix():
            continue
        if "htw_phase106" in path.parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        found.update(_extract(payload))
    return found


def _extract(payload: Any) -> set[int]:
    found: set[int] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key) in MANIFEST_SEED_KEYS and isinstance(value, list):
                for item in value:
                    if isinstance(item, int) or (isinstance(item, str) and str(item).lstrip("-").isdigit()):
                        found.add(int(item))
            else:
                found.update(_extract(value))
    elif isinstance(payload, list):
        for item in payload:
            found.update(_extract(item))
    return found


def candidate_untouched_seeds() -> list[int]:
    return list(range(int(SCENARIO_BASE), int(SCENARIO_BASE) + int(N_SCENARIOS)))


__all__ = [
    "ARCH_MANIFEST",
    "BOOT_SEED",
    "ENV_FILE",
    "EXPECTED_ENV_SHA",
    "EXPECTED_HIGH_SHA",
    "FREEZE_FILE",
    "HIGH_CKPT",
    "HOLD",
    "N_BOOT",
    "N_SCENARIOS",
    "OLD_DEEPSETS",
    "OUTPUT",
    "PACK_REPORT",
    "REPORT",
    "ROOT",
    "SCENARIO_BASE",
    "candidate_untouched_seeds",
    "dump_frozen",
    "historical_blocked_seeds",
    "sha256_bytes",
]
