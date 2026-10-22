"""New-branch seed governance. Does not open Phase-10.6 per-scenario CSVs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wt.htw_phase104.protocol import MANIFEST_SEED_KEYS, protocol_blocked_seeds
from wt.htw_phase945.protocol import ROOT

from .protocol import (
    LLV2_DESIGN_BASE,
    LLV2_DESIGN_N,
    LLV2_TRAIN_BASE,
    LLV2_TRAIN_N,
    PHASE104_INDEPENDENT,
    PHASE105_INDEPENDENT,
    PHASE106_UNTOUCHED,
)


FORBIDDEN_CSV_NAMES = {
    "phase106_per_scenario_metrics.csv",
    "phase106_final_results.csv",
}


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


def scan_historical_seeds(root: Path | None = None) -> set[int]:
    found = set(protocol_blocked_seeds())
    found.update(PHASE106_UNTOUCHED)
    found.update(PHASE105_INDEPENDENT)
    found.update(PHASE104_INDEPENDENT)
    found.update(range(0, 24))
    base = ROOT if root is None else Path(root)
    patterns = (
        "*confirmation*manifest*.json",
        "*scenario*manifest*.json",
        "*seed*manifest*.json",
        "*design*manifest*.json",
        "phase10*_protocol.json",
        "phase9*_protocol.json",
    )
    for pattern in patterns:
        for path in base.rglob(pattern):
            if "backup" in path.parts or "_backup_" in path.as_posix():
                continue
            if path.name in FORBIDDEN_CSV_NAMES:
                raise RuntimeError("refused to scan Phase-10.6 result CSV")
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            found.update(_extract(payload))
    return found


def allocate_new_pool(blocked: set[int], base: int, count: int) -> list[int]:
    seeds: list[int] = []
    cursor = int(base)
    while len(seeds) < int(count):
        if cursor in blocked or cursor in PHASE106_UNTOUCHED:
            cursor += 1
            continue
        seeds.append(int(cursor))
        cursor += 1
    if any(seed in PHASE106_UNTOUCHED for seed in seeds):
        raise RuntimeError("new pool overlapped Phase-10.6 untouched")
    return seeds


def build_registry() -> dict[str, Any]:
    blocked = scan_historical_seeds()
    train = allocate_new_pool(blocked, LLV2_TRAIN_BASE, LLV2_TRAIN_N)
    design = allocate_new_pool(blocked | set(train), LLV2_DESIGN_BASE, LLV2_DESIGN_N)
    return {
        "historical_blocked_count": len(blocked),
        "phase106_untouched_blocked": True,
        "phase106_untouched": sorted(PHASE106_UNTOUCHED),
        "opened_phase106_result_csv": False,
        "LLV2_train": train,
        "LLV2_design": design,
        "LLV2_confirmation": None,
        "LLV2_final_untouched": None,
        "note": "confirmation and final untouched pools are not created in LL-V2.0",
        "overlap_train_design": bool(set(train) & set(design)),
        "overlap_train_history": bool(set(train) & blocked),
        "overlap_design_history": bool(set(design) & blocked),
    }


def refuse_untouched(seed: int) -> None:
    if int(seed) in PHASE106_UNTOUCHED:
        raise RuntimeError(f"refused consumed Phase-10.6 untouched seed {seed}")


__all__ = ["build_registry", "refuse_untouched", "scan_historical_seeds"]
