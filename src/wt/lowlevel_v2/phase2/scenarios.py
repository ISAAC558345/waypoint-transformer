"""Allocate and freeze LLV2_PHASE2_VALIDATION_V1 before any model run."""
from __future__ import annotations

from pathlib import Path

from wt.htw_phase104.protocol import all_blocked_scenario_seeds
from wt.htw_phase941.protocol import dump_frozen
from wt.lowlevel_v2.data_registry import scan_historical_seeds
from wt.lowlevel_v2.phase1b.protocol import HOLD_STEPS, load_pools
from wt.lowlevel_v2.protocol import PHASE104_INDEPENDENT, PHASE105_INDEPENDENT, PHASE106_UNTOUCHED

from .protocol import EXPECTED_HIGH_SHA, VALIDATION_BASE, VALIDATION_N, env_sha, high_sha


def blocked_universe() -> set[int]:
    blocked = set(all_blocked_scenario_seeds())
    blocked.update(scan_historical_seeds())
    blocked.update(PHASE106_UNTOUCHED)
    blocked.update(PHASE105_INDEPENDENT)
    blocked.update(PHASE104_INDEPENDENT)
    train, design = load_pools()
    blocked.update(int(s) for s in train)
    blocked.update(int(s) for s in design)
    blocked.update(range(0, 40))
    blocked.update(range(220000, 221200))
    return blocked


def allocate_validation(blocked: set[int]) -> list[int]:
    seeds: list[int] = []
    cursor = int(VALIDATION_BASE)
    while len(seeds) < int(VALIDATION_N):
        if cursor not in blocked:
            seeds.append(int(cursor))
        cursor += 1
        if cursor > VALIDATION_BASE + 100000:
            raise RuntimeError("could not allocate validation seeds")
    overlap = set(seeds) & blocked
    if overlap:
        raise RuntimeError(f"validation overlapped blocked seeds {sorted(overlap)[:8]}")
    if set(seeds) & PHASE106_UNTOUCHED:
        raise RuntimeError("validation overlapped Phase-10.6 untouched")
    return seeds


def freeze_validation_manifest(path: Path) -> dict:
    blocked = blocked_universe()
    seeds = allocate_validation(blocked)
    payload = {
        "name": "LLV2_PHASE2_VALIDATION_V1",
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
        "overlap_train": False,
        "overlap_design": False,
        "overlap_untouched": False,
    }
    dump_frozen(path, payload)
    return payload
