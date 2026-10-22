"""Aggregate metrics, windows, and reward audit from event logs."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def payload_caliber(b0: list[float], b1: list[float]) -> dict[str, Any]:
    b0a = np.asarray(b0, dtype=np.float64)
    b1a = np.asarray(b1, dtype=np.float64)
    mean_b0 = float(np.mean(b0a))
    mean_b1 = float(np.mean(b1a))
    per_ratio = b1a / np.maximum(b0a, 1.0e-9)
    return {
        "mean_b0": mean_b0,
        "mean_b1": mean_b1,
        "mean_b1_over_mean_b0": mean_b1 / max(mean_b0, 1.0e-9),
        "mean_delta_over_mean_b0": (mean_b1 - mean_b0) / max(mean_b0, 1.0e-9),
        "mean_of_per_scene_ratio": float(np.mean(per_ratio)),
        "per_scene_ratio_definition": "mean_i (payload_B1(i) / payload_B0(i)); this is what Phase-10.1 mean_ratio stored",
        "gap_frac": 1.0 - mean_b1 / max(mean_b0, 1.0e-9),
        "n": int(b0a.size),
    }


def slot_metrics(events: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    rows = [e for e in events if e["mode"] == mode]
    n = max(len(rows), 1)
    raw_in = sum(1 for e in rows if e["in_range_after"])
    active = [e for e in rows if not e["idle"]]
    queued = [e for e in active if not e["empty_queue_before"]]
    n_uav_scene = {}
    completed_uavs = set()
    for e in rows:
        key = (e["scenario"], e["uav"])
        n_uav_scene[key] = True
        if e["completed_block"]:
            completed_uavs.add(key)
    never = set(n_uav_scene) - completed_uavs
    return {
        "mode": mode,
        "n_uav_slots": len(rows),
        "raw_in_range_frac": raw_in / n,
        "active_in_range_frac": (sum(1 for e in active if e["in_range_after"]) / max(len(active), 1)),
        "active_queued_in_range_frac": (sum(1 for e in queued if e["in_range_after"]) / max(len(queued), 1)),
        "n_active": len(active),
        "n_active_queued": len(queued),
        "n_uav_episodes": len(n_uav_scene),
        "n_uav_completed_service": len(completed_uavs),
        "n_uav_never_completed": len(never),
        "n_completed_block_events": sum(1 for e in rows if e["completed_block"]),
    }


def reset_breakdown(events: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    rows = [e for e in events if e["mode"] == mode]
    counts = defaultdict(int)
    lost_progress = 0.0
    for e in rows:
        labels = str(e["labels"]).split("|")
        dropped = float(e["rho_before"]) > 1.0e-12 and float(e["rho_after"]) <= 1.0e-12
        if dropped:
            if "NORMAL_COMPLETION" in labels:
                counts["NORMAL_COMPLETION"] += 1
            elif "TARGET_CHANGE" in labels:
                counts["TARGET_CHANGE"] += 1
                lost_progress += float(e["rho_before"])
            else:
                counts["OTHER_DROP"] += 1
                lost_progress += float(e["rho_before"])
        for lab in labels:
            counts[f"label_{lab}"] += 1
        if "OUT_OF_RANGE" in labels and float(e["rho_before"]) > 1.0e-12 and "NORMAL_COMPLETION" not in labels:
            counts["out_of_range_with_progress"] += 1
    return {"mode": mode, "lost_progress_mass_non_completion": float(lost_progress), **{k: int(v) if isinstance(v, (int, np.integer)) else v for k, v in counts.items()}}


def windows(events: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    rows = [e for e in events if e["mode"] == mode]
    groups: dict[tuple, list] = defaultdict(list)
    for e in rows:
        hid = int(e["t_before"]) // 15
        groups[(e["scenario"], hid, e["uav"], e["target"])].append(e)
    out = []
    for key, seq in groups.items():
        scenario, hid, uav, tgt = key
        if tgt < 0:
            continue
        queued_any = any(not e["empty_queue_before"] for e in seq)
        if not queued_any:
            continue
        in_idx = [i for i, e in enumerate(seq) if e["in_range_after"] and not e["empty_queue_before"]]
        n_complete = sum(1 for e in seq if e["completed_block"])
        leftover = float(seq[-1]["rho_after"])
        first_in_rem_hold = int(seq[in_idx[0]]["remaining_hold_after"]) if in_idx else None
        first_in_rem_hzn = int(seq[in_idx[0]]["remaining_horizon_before"]) - 1 if in_idx else None
        mean_rate = float(np.mean([e["rate_bps"] for e in seq if e["in_range_after"] and not e["empty_queue_before"]])) if in_idx else 0.0
        lost_switch = any("TARGET_CHANGE" in str(e["labels"]) and float(e["rho_before"]) > 1.0e-12 and not e["completed_block"] for e in seq)
        left_range = any("OUT_OF_RANGE" in str(e["labels"]) and e["in_range_before"] for e in seq)
        emptied = any(e["empty_queue_before"] and e["in_range_after"] for e in seq)
        tags = []
        if n_complete == 0 and in_idx and first_in_rem_hold is not None and first_in_rem_hold < 3:
            tags.append("A_late_or_short_hold")
        if n_complete == 0 and in_idx and mean_rate < 1.0e5:
            tags.append("B_in_range_low_rate")
        if lost_switch or left_range:
            tags.append("C_switch_or_leave_range")
        if emptied:
            tags.append("D_queue_empty")
        if n_complete == 0 and in_idx and leftover > 1.0e-12 and not tags:
            tags.append("F_unexplained")
        if n_complete == 0 and not in_idx:
            tags.append("F_never_in_range_with_queue")
        if n_complete > 0:
            tags.append("completed")
        out.append(
            {
                "mode": mode,
                "scenario": scenario,
                "hold_id": hid,
                "uav": uav,
                "target": tgt,
                "n_slots": len(seq),
                "n_in_range_queued": len(in_idx),
                "n_completed_blocks": n_complete,
                "leftover_rho": leftover,
                "mean_in_range_rate_bps": mean_rate,
                "first_in_remaining_hold": first_in_rem_hold,
                "first_in_remaining_horizon": first_in_rem_hzn,
                "tags": "|".join(tags) if tags else "none",
            }
        )
    return out


def reward_audit(events: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    team = [e for e in events if e["mode"] == mode and e["uav"] == 0]
    uav_rows = [e for e in events if e["mode"] == mode]
    in_range_aux_steps = 0
    in_range_no_block = 0
    empty_queue_in_range = 0
    idle_in_range = 0
    for e in uav_rows:
        if e["idle"] and e["in_range_after"]:
            idle_in_range += 1
        if (not e["idle"]) and e["in_range_after"]:
            in_range_aux_steps += 1
            if not e["completed_block"]:
                in_range_no_block += 1
            if e["empty_queue_before"]:
                empty_queue_in_range += 1
    return {
        "mode": mode,
        "sum_r_env": float(sum(e["r_env"] for e in team)),
        "sum_r_aux": float(sum(e["r_aux"] for e in team)),
        "n_team_steps": len(team),
        "uav_in_range_steps": in_range_aux_steps,
        "uav_in_range_without_block": in_range_no_block,
        "frac_in_range_without_block": in_range_no_block / max(in_range_aux_steps, 1),
        "uav_empty_queue_in_range": empty_queue_in_range,
        "idle_flagged_in_range": idle_in_range,
        "note": "r_aux uses after-step positions and does not require nonempty queue; idle UAVs get no in-range term",
    }


__all__ = ["payload_caliber", "reset_breakdown", "reward_audit", "slot_metrics", "windows"]
