"""Short scripted smoke. It does not call the formal planner or a learned policy."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2

from .follow import follow_step
from .ledger import PriorityTagLedger
from .protocol import SMOKE_SCENARIO, SMOKE_STEPS, TOL_BITS


def scripted_action(env) -> tuple[np.ndarray, np.ndarray]:
    """Hover inside the service radius. Fly toward a fixed buoy when outside it.

    The formal step moves first and serves second. A full east step leaves the
    16 m radius before service, so this script holds position while the chosen
    buoy is already in range. UAVs 0-2 prefer their near buoy. UAVs 3-5 fly to
    the matching far buoy so later arrivals can be served inside 16 steps.
    """
    num_uavs = int(env.config.num_uavs)
    movement = np.zeros((num_uavs, 2), dtype=np.float64)
    assignment = np.full(num_uavs, -1, dtype=np.int64)
    radius = float(env.config.service_radius_m)
    for uav in range(num_uavs):
        primary = int(uav) if uav < 3 else int(uav) + 6
        secondary = int(uav) + 6 if uav < 3 else int(uav)
        target = _script_target(env, uav, primary, secondary, radius)
        assignment[uav] = int(target)
        delta = np.asarray(env.buoy_positions[target] - env.uav_positions[uav], dtype=np.float64)
        distance = float(np.linalg.norm(delta))
        if distance > radius:
            movement[uav] = delta / max(distance, 1.0e-9)
    return movement, assignment


def _script_target(env, uav: int, primary: int, secondary: int, radius: float) -> int:
    if int(uav) >= 3:
        return int(primary)
    if _within(env, uav, primary, radius) and float(env.q_bits[primary]) > float(TOL_BITS):
        return int(primary)
    if float(env.q_bits[primary]) > float(TOL_BITS):
        return int(primary)
    return int(secondary)


def _within(env, uav: int, buoy: int, radius: float) -> bool:
    delta = np.asarray(env.buoy_positions[int(buoy)] - env.uav_positions[int(uav)], dtype=np.float64)
    return float(np.linalg.norm(delta)) <= float(radius)


def _fingerprint(env) -> dict[str, float]:
    return {
        "q_bits": float(np.sum(env.q_bits)),
        "payload_bits": float(env.payload_bits),
        "served_sum": float(np.sum(env.cumulative_served_bits)),
        "dropped_sum": float(np.sum(env.cumulative_dropped_bits)),
        "position_sum": float(np.sum(env.uav_positions)),
        "t": float(env.t),
    }


def run_smoke(scenario_id: int = SMOKE_SCENARIO, steps: int = SMOKE_STEPS) -> dict:
    if not 321000 <= int(scenario_id) <= 321095:
        raise RuntimeError("smoke scenario is outside the low-level train pool")
    followed = _roll(int(scenario_id), int(steps), use_ledger=True)
    plain = _roll(int(scenario_id), int(steps), use_ledger=False)
    repeated = _roll(int(scenario_id), int(steps), use_ledger=True)
    transition_gap = max(abs(float(followed["trace"][key]) - float(plain["trace"][key])) for key in followed["trace"])
    repeat_gap = max(abs(float(followed["trace"][key]) - float(repeated["trace"][key])) for key in followed["trace"])
    tag_match = followed["tag_rows"] == repeated["tag_rows"]
    return {
        "scenario_id": int(scenario_id),
        "steps": int(steps),
        "synthetic": True,
        "rows": followed["rows"],
        "info_keys": followed["info_keys"],
        "snap_source": followed["snap_source"],
        "max_ledger_gap_bits": float(max(float(row["ledger_gap_bits"]) for row in followed["rows"])),
        "transition_gap": float(transition_gap),
        "repeat_gap": float(repeat_gap),
        "tag_repeat_match": bool(tag_match),
        "served_ordinary_bits": float(sum(float(row["served_ordinary_bits"]) for row in followed["rows"])),
        "served_tagged_bits": float(sum(float(row["served_tagged_bits"]) for row in followed["rows"])),
        "dropped_ordinary_bits": float(sum(float(row["dropped_ordinary_bits"]) for row in followed["rows"])),
        "dropped_tagged_bits": float(sum(float(row["dropped_tagged_bits"]) for row in followed["rows"])),
        "admitted_ordinary_bits": float(sum(float(row["admitted_ordinary_bits"]) for row in followed["rows"])),
        "admitted_tagged_bits": float(sum(float(row["admitted_tagged_bits"]) for row in followed["rows"])),
        "urgency_served_bits": float(sum(float(row["urgency_served_bits"]) for row in followed["rows"])),
        "generated_bits": float(sum(float(row["generated_bits"]) for row in followed["rows"])),
        "admitted_bits": float(sum(float(row["admitted_bits"]) for row in followed["rows"])),
        "dropped_bits": float(sum(float(row["dropped_bits"]) for row in followed["rows"])),
        "snap_bits": float(sum(float(row["snap_bits"]) for row in followed["rows"])),
        "remaining_ordinary_bits": float(followed["remaining_ordinary_bits"]),
        "remaining_tagged_bits": float(followed["remaining_tagged_bits"]),
        "final_q_bits": float(followed["trace"]["q_bits"]),
        "tag_draws": int(followed["tag_draws"]),
        "tagged_draws": int(followed["tagged_draws"]),
    }


def _roll(scenario_id: int, steps: int, *, use_ledger: bool) -> dict:
    env = L3BBitQueueEnvV2()
    env.reset(seed=int(scenario_id))
    ledger = PriorityTagLedger(int(env.config.num_buoys))
    ledger.reset_from_queue(env.q_bits)
    ledger.assert_matches(env.q_bits, float(env.tol_bits))
    rows = []
    tag_rows = []
    info_keys: list[str] = []
    snap_source = ""
    for _ in range(int(steps)):
        movement, association = scripted_action(env)
        if use_ledger:
            row = follow_step(env, ledger, movement, association, int(scenario_id))
            rows.append(row)
            tag_rows.append(row["tags"])
            info_keys = list(row["info_keys"])
            snap_source = str(row["snap_source"])
        else:
            env.step(movement, association)
    ordinary, tagged = ledger.bits_by_tag()
    draws = [tag for row in tag_rows for tag in row if tag is not None]
    return {
        "rows": rows,
        "tag_rows": tag_rows,
        "trace": _fingerprint(env),
        "info_keys": info_keys,
        "snap_source": snap_source,
        "remaining_ordinary_bits": float(np.sum(ordinary)),
        "remaining_tagged_bits": float(np.sum(tagged)),
        "tag_draws": int(len(draws)),
        "tagged_draws": int(sum(int(tag) for tag in draws)),
    }
