"""Holdout ID allocation. Does not call allocate_new_pool."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent / "outputs"
ALLOCATED = OUTPUT / "allocated_holdout_ids.json"

EXPLICIT_RANGES = (
    (220000, 220063),
    (220000, 221199),
    (221000, 221015),
    (222000, 222063),
    (223000, 223063),
    (104000, 104063),
    (105000, 105063),
    (106000, 106063),
)


def historical_ids() -> set[int]:
    used: set[int] = set()
    for start, end in EXPLICIT_RANGES:
        used.update(range(int(start), int(end) + 1))
    registry = ROOT / "src" / "wt" / "lowlevel_v2" / "outputs" / "phase0" / "LLV2_DATA_REGISTRY.json"
    if registry.is_file():
        payload = json.loads(registry.read_text(encoding="utf-8"))
        for key in ("LLV2_train", "LLV2_design", "LLV2_confirmation", "LLV2_final_untouched"):
            value = payload.get(key)
            if isinstance(value, list):
                used.update(int(item) for item in value)
    return used


def load_saved_allocation() -> tuple[list[int], list[int], int] | None:
    if not ALLOCATED.is_file():
        return None
    payload = json.loads(ALLOCATED.read_text(encoding="utf-8"))
    return list(payload["batch_1"]), list(payload["batch_2"]), int(payload["namespace_start"])


def save_allocation(batch1: list[int], batch2: list[int], start: int) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ALLOCATED.write_text(
        json.dumps(
            {"batch_1": batch1, "batch_2": batch2, "namespace_start": int(start), "allocate_new_pool_called": False},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def allocate_holdout_ids(used: set[int]) -> tuple[list[int], list[int], int]:
    start = 230000
    while True:
        window = list(range(start, start + 128))
        if set(window).isdisjoint(used):
            return window[:64], window[64:], start
        start += 128
        if start > 2000000:
            raise RuntimeError("no free 128-id holdout window")
