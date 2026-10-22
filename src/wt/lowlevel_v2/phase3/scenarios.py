"""Allocate LLV2_PHASE3_INDEPENDENT_VALIDATION_V2 before any validation model run."""
from __future__ import annotations

from pathlib import Path

from wt.htw_phase104.protocol import all_blocked_scenario_seeds
from wt.htw_phase941.protocol import dump_frozen
from wt.lowlevel_v2.data_registry import scan_historical_seeds
from wt.lowlevel_v2.phase1b.protocol import HOLD_STEPS, load_pools
from wt.lowlevel_v2.phase3.protocol import (
    EXPECTED_HIGH_SHA,
    MANIFEST_FILE,
    PHASE2_VAL_FROZEN,
    VALIDATION_BASE,
    VALIDATION_N,
    env_sha,
    high_sha,
)
from wt.lowlevel_v2.protocol import PHASE104_INDEPENDENT, PHASE105_INDEPENDENT, PHASE106_UNTOUCHED


def blocked_universe() -> set[int]:
    blocked = set(all_blocked_scenario_seeds())
    blocked.update(scan_historical_seeds())
    blocked.update(PHASE106_UNTOUCHED)
    blocked.update(PHASE105_INDEPENDENT)
    blocked.update(PHASE104_INDEPENDENT)
    blocked.update(PHASE2_VAL_FROZEN)
    train, design = load_pools()
    blocked.update(int(s) for s in train)
    blocked.update(int(s) for s in design)
    blocked.update(range(0, 40))
    blocked.update(range(220000, 221200))
    blocked.update(range(222000, 222064))
    return blocked


def allocate_validation(blocked: set[int]) -> list[int]:
    seeds: list[int] = []
    cursor = int(VALIDATION_BASE)
    while len(seeds) < int(VALIDATION_N):
        if cursor not in blocked:
            seeds.append(int(cursor))
        cursor += 1
        if cursor > VALIDATION_BASE + 100000:
            raise RuntimeError("could not allocate Phase-3 validation seeds")
    if set(seeds) & blocked:
        raise RuntimeError("Phase-3 validation overlapped blocked seeds")
    if set(seeds) & PHASE106_UNTOUCHED:
        raise RuntimeError("Phase-3 validation overlapped untouched")
    if set(seeds) & PHASE2_VAL_FROZEN:
        raise RuntimeError("Phase-3 validation overlapped Phase-2 validation")
    return seeds


def freeze_validation_manifest(path: Path | None = None) -> dict:
    dest = Path(path) if path is not None else MANIFEST_FILE
    blocked = blocked_universe()
    seeds = allocate_validation(blocked)
    train, design = load_pools()
    payload = {
        "name": "LLV2_PHASE3_INDEPENDENT_VALIDATION_V2",
        "not_confirmation": True,
        "not_final_untouched": True,
        "consumed": False,
        "n": len(seeds),
        "scenario_seeds": seeds,
        "M": 6,
        "K": 12,
        "T": 48,
        "hold": int(HOLD_STEPS),
        "environment_sha": env_sha(),
        "high_sha": high_sha(),
        "expected_high_sha": EXPECTED_HIGH_SHA,
        "arrival_config": "L3BEnv.reset(seed) default arrivals",
        "sinr_config": "L3BConfig tx_power/noise/path_loss/kappas frozen in src/wt/l3b/config.py",
        "generator_config": f"first {VALIDATION_N} unused integers from {VALIDATION_BASE} after full manifest scan",
        "blocked_count": len(blocked),
        "overlap_train": bool(set(seeds) & set(int(s) for s in train)),
        "overlap_design": bool(set(seeds) & set(int(s) for s in design)),
        "overlap_phase2_val": False,
        "overlap_untouched": False,
    }
    dump_frozen(dest, payload)
    return payload


__all__ = ["allocate_validation", "blocked_universe", "freeze_validation_manifest"]
