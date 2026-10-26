"""Read one formal step and update the tag ledger from returned fields."""
from __future__ import annotations

import numpy as np

from .ledger import PriorityTagLedger
from .protocol import DUST_BITS
from .tags import draw_tag

STEP_FIELDS = (
    "served_bits",
    "generated_arrival_bits",
    "admitted_arrival_bits",
    "dropped_bits_step",
)
SNAP_INFO_FIELDS = ("float_snap_bits", "snap_bits")


def snap_from_step(env, snap_before: np.ndarray, info: dict) -> tuple[np.ndarray, str]:
    """Use a snap field when the step exposes one. Otherwise use the cumulative counter."""
    count = int(env.config.num_buoys)
    for key in SNAP_INFO_FIELDS:
        if key in info:
            return np.asarray(info[key], dtype=np.float64).reshape(count), f"info.{key}"
    if not hasattr(env, "cumulative_snap_bits"):
        raise RuntimeError("formal step exposes neither float snap nor cumulative_snap_bits")
    delta = np.asarray(env.cumulative_snap_bits, dtype=np.float64).reshape(count) - np.asarray(snap_before, dtype=np.float64).reshape(count)
    return delta, "cumulative_snap_bits_delta"


def arrival_tags(scenario_id: int, admission_step: int, generated: np.ndarray) -> list[int | None]:
    tags: list[int | None] = []
    for buoy, bits in enumerate(np.asarray(generated, dtype=np.float64).reshape(-1)):
        if float(bits) > DUST_BITS:
            tags.append(draw_tag(int(scenario_id), int(admission_step), int(buoy)))
        else:
            tags.append(None)
    return tags


def follow_step(env, ledger: PriorityTagLedger, movement: np.ndarray, association: np.ndarray, scenario_id: int) -> dict:
    """Step the formal env once. The ledger consumes the returned vectors and does not call arrivals."""
    time_step = int(env.t)
    snap_before = np.asarray(env.cumulative_snap_bits, dtype=np.float64).copy()
    info = env.step(movement, association)
    missing = [key for key in STEP_FIELDS if key not in info]
    if missing:
        raise RuntimeError(f"formal step is missing {missing}")
    snap, snap_source = snap_from_step(env, snap_before, info)
    generated = np.asarray(info["generated_arrival_bits"], dtype=np.float64)
    tags = arrival_tags(int(scenario_id), time_step, generated)
    split = ledger.apply_transition(
        info["served_bits"],
        generated,
        info["admitted_arrival_bits"],
        info["dropped_bits_step"],
        snap,
        time_step,
        tags,
        float(env.tol_bits),
    )
    gap = ledger.assert_matches(env.q_bits, float(env.tol_bits))
    return {
        "time_step": time_step,
        "info_keys": sorted(info.keys()),
        "snap_source": snap_source,
        "served_bits": float(np.sum(info["served_bits"])),
        "payload_increment_bits": float(info["payload_increment_bits"]),
        "generated_bits": float(np.sum(generated)),
        "admitted_bits": float(np.sum(info["admitted_arrival_bits"])),
        "dropped_bits": float(np.sum(info["dropped_bits_step"])),
        "snap_bits": float(np.sum(snap)),
        "served_ordinary_bits": float(np.sum(split["served_ordinary_bits"])),
        "served_tagged_bits": float(np.sum(split["served_tagged_bits"])),
        "dropped_ordinary_bits": float(np.sum(split["dropped_ordinary_bits"])),
        "dropped_tagged_bits": float(np.sum(split["dropped_tagged_bits"])),
        "admitted_ordinary_bits": float(np.sum(split["admitted_ordinary_bits"])),
        "admitted_tagged_bits": float(np.sum(split["admitted_tagged_bits"])),
        "urgency_served_bits": float(np.sum(split["urgency_served_bits"])),
        "ledger_gap_bits": float(np.max(np.abs(gap))),
        "q_bits": float(np.sum(env.q_bits)),
        "payload_bits": float(env.payload_bits),
        "tags": tags,
    }
