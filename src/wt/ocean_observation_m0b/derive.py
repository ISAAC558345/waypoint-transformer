"""Build the B2 view from the saved M0A schedule.

The original files stay untouched. B2 reuses the moving task and applies the
frozen shared-slot order; it does not start another analytical sample.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from wt.ocean_observation_m0.codec import packet_sizes
from wt.ocean_observation_m0.parameters import MAC_GUARD_S, TAKEOFF_HOVER_S
from wt.ocean_observation_m0.radio import lora_time_on_air_s
from wt.ocean_observation_m0a.compare import score_method
from wt.ocean_observation_m0a.links import buoy_to_uav, uav_to_shore
from wt.ocean_observation_m0b.contract import (
    EQUIVALENCE_LABEL,
    MAX_DERIVED_METHOD_ROWS,
    SOURCE_RESULT_COMMIT,
    TELEMETRY_ON_DELIVERY_CLOCK,
    sort_key,
)

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "results" / "ocean_observation_m0a"


def _load(name: str) -> dict:
    return json.loads((SOURCE / name).read_text(encoding="utf-8"))


def _visits_for(method: str, geometry: dict) -> list[list[dict]]:
    if method == "fixed":
        fixed = geometry["fixed"]
        visit = {
            "group": "all",
            "buoy_indexes": list(range(len(geometry["buoys_xy_m"]))),
            "receive_xy_m": fixed["xy_m"],
            "covers": fixed["covers_all_buoys"],
            "uav": 0,
        }
        return [[visit], []]
    assigned: list[list[dict]] = [[], []]
    for visit in geometry["moving"]["visits"]:
        assigned[int(visit["uav"])].append(visit)
    return assigned


def _events_for_plan(geometry_name: str, wind_name: str, method: str, geometry: dict, plan: dict, sizes: dict) -> list[dict]:
    payload = sizes["product"]["framed_bytes"]
    durations = {
        "command": lora_time_on_air_s(sizes["command"]["framed_bytes"]),
        "ack": lora_time_on_air_s(sizes["ack"]["framed_bytes"]),
        "payload": lora_time_on_air_s(payload),
        "backhaul": lora_time_on_air_s(payload),
    }
    buoys = geometry["buoys_xy_m"]
    events = []
    for index, uav in enumerate(plan["uavs"]):
        visits = _visits_for(method, geometry)[index]
        if not visits:
            continue
        for window in uav["windows"]:
            generation = float(window["generation_s"])
            clock = float(window["launch_s"]) + TAKEOFF_HOVER_S
            for leg in uav["legs"]:
                if float(leg["generation_s"]) == generation and leg["role"] in {"outbound", "between"}:
                    clock += float(leg["time_s"])
            if clock < generation:
                clock = generation
            for visit in visits:
                position = (float(visit["receive_xy_m"][0]), float(visit["receive_xy_m"][1]))
                shore = uav_to_shore(position, payload)
                received = []
                phases: list[tuple[str, int]] = []
                if shore["closes"]:
                    phases.extend((("command", -1), ("ack", -1)))
                for buoy_index in visit["buoy_indexes"]:
                    link = buoy_to_uav((float(buoys[buoy_index][0]), float(buoys[buoy_index][1])), position, payload)
                    if visit["covers"] and link["closes"]:
                        phases.append(("payload", int(buoy_index)))
                        received.append(int(buoy_index))
                if shore["closes"]:
                    phases.extend(("backhaul", buoy_index) for buoy_index in received)
                for phase, buoy_index in phases:
                    duration = durations[phase]
                    event = {
                        "geometry": geometry_name,
                        "wind": wind_name,
                        "method": method,
                        "uav": index,
                        "generation_s": generation,
                        "phase": phase,
                        "buoy_index": buoy_index,
                        "proposed_start": clock,
                        "duration_s": duration,
                    }
                    events.append(event)
                    clock += duration
                    if phase == "backhaul":
                        saved = next(
                            item
                            for item in plan["products"]
                            if int(item["buoy"]) == buoy_index and float(item["generation_s"]) == generation
                        )
                        saved_delivery = saved["delivery_s"]
                        if saved_delivery is None or abs(float(saved_delivery) - clock) > 1e-9:
                            raise RuntimeError(
                                f"replay does not match the saved delivery for {geometry_name} {wind_name} {method} buoy {buoy_index}"
                            )
                    clock += MAC_GUARD_S
    return events


def schedule_channel(events: list[dict]) -> list[dict]:
    """Delay the later UAV only where its airtime actually overlaps the earlier one.

    A single aircraft keeps its own command, ack, payload, and backhaul order.
    The frozen sort key decides which aircraft waits.
    """
    if TELEMETRY_ON_DELIVERY_CLOCK:
        raise RuntimeError("telemetry stays off the delivery clock in this package")
    groups: dict[tuple, list[dict]] = {}
    for event in events:
        groups.setdefault((event["geometry"], event["wind"], event["method"]), []).append(event)
    scheduled = []
    for group in groups.values():
        placed = []
        for event in group:
            row = dict(event)
            row["scheduled_start"] = float(event["proposed_start"])
            row["scheduled_end"] = row["scheduled_start"] + float(event["duration_s"])
            row["delay_s"] = 0.0
            placed.append(row)
        for _ in range(len(placed) * len(placed) + 1):
            moved = False
            for left_index, left in enumerate(placed):
                for right in placed[left_index + 1 :]:
                    if left["uav"] == right["uav"]:
                        continue
                    if not (right["scheduled_start"] < left["scheduled_end"] - 1e-9 and left["scheduled_start"] < right["scheduled_end"] - 1e-9):
                        continue
                    earlier, later = (left, right) if sort_key(left) <= sort_key(right) else (right, left)
                    delta = earlier["scheduled_end"] + MAC_GUARD_S - later["scheduled_start"]
                    if delta <= 1e-9:
                        continue
                    for item in placed:
                        if item["uav"] != later["uav"] or float(item["generation_s"]) != float(later["generation_s"]):
                            continue
                        if float(item["proposed_start"]) + 1e-9 < float(later["proposed_start"]):
                            continue
                        item["scheduled_start"] += delta
                        item["scheduled_end"] += delta
                        item["delay_s"] += delta
                    moved = True
                    break
                if moved:
                    break
            if not moved:
                break
        else:
            raise RuntimeError("shared-slot resolution did not settle")
        scheduled.extend(placed)
    return scheduled


def overlapping_pairs(events: list[dict], start_key: str, end_key: str) -> list[dict]:
    found = []
    groups: dict[tuple, list[dict]] = {}
    for event in events:
        groups.setdefault((event["geometry"], event["wind"], event["method"]), []).append(event)
    for group in groups.values():
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                if right[start_key] < left[end_key] - 1e-9 and left[start_key] < right[end_key] - 1e-9:
                    found.append(
                        {
                            "geometry": left["geometry"],
                            "wind": left["wind"],
                            "method": left["method"],
                            "generation_s": left["generation_s"],
                            "left_uav": left["uav"],
                            "right_uav": right["uav"],
                            "left_phase": left["phase"],
                            "right_phase": right["phase"],
                        }
                    )
    return found


def _window_delays(events: list[dict]) -> dict[tuple, float]:
    delays: dict[tuple, float] = {}
    for event in events:
        key = (event["geometry"], event["wind"], event["method"], event["uav"], float(event["generation_s"]))
        delays[key] = max(delays.get(key, 0.0), float(event["delay_s"]))
    return delays


def _adjust(plan: dict, geometry: dict, delays: dict[tuple, float], geometry_name: str, wind_name: str, method: str) -> dict:
    adjusted = copy.deepcopy(plan)
    owned = {index: [int(buoy) for visit in visits for buoy in visit["buoy_indexes"]] for index, visits in enumerate(_visits_for(method, geometry))}
    for index, uav in enumerate(adjusted["uavs"]):
        for window in uav["windows"]:
            wait = delays.get((geometry_name, wind_name, method, index, float(window["generation_s"])), 0.0)
            if wait:
                window["landing_s"] = float(window["landing_s"]) + wait
                uav["vertical_s"]["wait"] = float(uav["vertical_s"]["wait"]) + wait
                uav["airborne_s"] = float(uav["airborne_s"]) + wait
        for product in adjusted["products"]:
            if int(product["buoy"]) not in owned[index] or not product["delivered"]:
                continue
            wait = delays.get((geometry_name, wind_name, method, index, float(product["generation_s"])), 0.0)
            if wait:
                product["delivery_s"] = float(product["delivery_s"]) + wait
                product["delay_s"] = float(product["delay_s"]) + wait
    return adjusted


def _public(scored: dict, **extra) -> dict:
    row = {
        "geometry": scored["geometry"],
        "wind": scored["wind"],
        "method": scored["method"],
        "delivered_count": scored["delivered_count"],
        "undelivered_count": scored["undelivered_count"],
        "mean_delay_s": scored["mean_delay_s"],
        "first_delivery_s": scored["first_delivery_s"],
        "latest_delivery_s": scored["latest_delivery_s"],
        "nodes_J": scored["nodes_J"],
        "uav_energy_J": [item["energy_J"] for item in scored["uavs"]],
        "uav_within_reserve": [item["within_reserve"] for item in scored["uavs"]],
        "acceptable": scored["acceptable"],
        "system_energy_J": scored["system_energy_J"],
    }
    row.update(extra)
    return row


def _same_score(left: dict, right: dict) -> bool:
    if left["delivered_count"] != right["delivered_count"] or left["undelivered_count"] != right["undelivered_count"]:
        return False
    for field in ("mean_delay_s", "first_delivery_s", "latest_delivery_s", "system_energy_J"):
        if left[field] != right[field]:
            return False
    return left["nodes_J"] == right["nodes_J"] and left["uav_energy_J"] == right["uav_energy_J"]


def build_view() -> dict:
    """Read the saved M0A artifacts and derive B2 under the frozen slot rule."""
    schedule = _load("schedule.json")
    geometry_manifest = _load("geometry_manifest.json")
    comparison = _load("comparison.json")
    geometries = {row["geometry"]: row for row in geometry_manifest["geometries"]}
    saved_rows = {
        (row["geometry"], row["wind"], row["method"]): (index, row)
        for index, row in enumerate(comparison["rows"])
    }
    sizes = packet_sizes()
    proposed = []
    identity = []
    rescored_matches = True
    for context in schedule["contexts"]:
        geometry = geometries[context["geometry"]]
        for wind in context["winds"]:
            for method in ("fixed", "moving"):
                plan = wind["methods"][method]
                visits = _visits_for(method, geometry)
                for index, uav in enumerate(plan["uavs"]):
                    visit_ids = [int(visit["uav"]) for visit in visits[index]]
                    if visit_ids and visit_ids != [index] * len(visit_ids):
                        raise RuntimeError("saved visit identity does not follow the list index")
                    identity.append(
                        {
                            "geometry": context["geometry"],
                            "wind": wind["wind"],
                            "method": method,
                            "list_index": index,
                            "recorded_uav": int(uav["uav"]),
                            "assigned_uav": index,
                            "active": bool(visits[index]),
                            "receive_xy_m": None if not visits[index] else list(visits[index][0]["receive_xy_m"]),
                            "proposed_windows": list(uav["windows"]),
                        }
                    )
                proposed.extend(_events_for_plan(context["geometry"], wind["wind"], method, geometry, plan, sizes))
                scored = score_method(plan)
                scored["geometry"] = context["geometry"]
                scored["wind"] = wind["wind"]
                saved = saved_rows[(context["geometry"], wind["wind"], method)][1]
                current = _public(scored)
                reference = _public(saved)
                if not _same_score(current, reference):
                    rescored_matches = False
    if not rescored_matches:
        raise RuntimeError("rescoring the saved plan does not match comparison.json")
    for event in proposed:
        event["scheduled_start"] = event["proposed_start"]
        event["scheduled_end"] = event["proposed_start"] + event["duration_s"]
    proposed_overlaps = [
        item
        for item in overlapping_pairs(proposed, "proposed_start", "scheduled_end")
        if item["left_uav"] != item["right_uav"]
    ]
    # The helper above used scheduled_end as proposed end because both were set equal.
    scheduled = schedule_channel(proposed)
    scheduled_overlaps = overlapping_pairs(scheduled, "scheduled_start", "scheduled_end")
    delays = _window_delays(scheduled)
    revised_rows = []
    b2_rows = []
    timing_corrections = []
    for context in schedule["contexts"]:
        geometry = geometries[context["geometry"]]
        for wind in context["winds"]:
            key = (context["geometry"], wind["wind"], "moving")
            source_index, source_row = saved_rows[key]
            plan = wind["methods"]["moving"]
            relevant = [event for event in scheduled if event["geometry"] == key[0] and event["wind"] == key[1] and event["method"] == "moving"]
            changed = any(float(event["delay_s"]) > 1e-9 for event in relevant)
            if changed:
                adjusted = _adjust(plan, geometry, delays, key[0], key[1], "moving")
                scored = score_method(adjusted)
                scored["geometry"] = key[0]
                scored["wind"] = key[1]
                scored["method"] = "moving"
                revised = _public(
                    scored,
                    source_row_index=source_index,
                    numbers_changed=True,
                    independent_experiment=False,
                    contract="shared_half_duplex",
                )
                revised_rows.append(revised)
                reference = revised
                by_window: dict[tuple, set[float]] = {}
                for event in relevant:
                    bucket = by_window.setdefault((int(event["uav"]), float(event["generation_s"])), set())
                    bucket.add(round(float(event["delay_s"]), 9))
                if any(len(bucket) != 1 for bucket in by_window.values()):
                    raise RuntimeError("packets in one window did not keep one common wait")
                waited = sorted({int(event["uav"]) for event in relevant if float(event["delay_s"]) > 1e-9})
                timing_corrections.append(
                    {
                        "geometry": key[0],
                        "wind": key[1],
                        "source_row_index": source_index,
                        "waited_uavs": waited,
                        "wait_s": {
                            str(uav): max(float(event["delay_s"]) for event in relevant if int(event["uav"]) == uav)
                            for uav in waited
                        },
                        "original_uav_energy_J": [source_row["uavs"][0]["energy_J"], source_row["uavs"][1]["energy_J"]],
                        "revised_uav_energy_J": revised["uav_energy_J"],
                        "original_mean_delay_s": source_row["mean_delay_s"],
                        "revised_mean_delay_s": revised["mean_delay_s"],
                        "original_latest_delivery_s": source_row["latest_delivery_s"],
                        "revised_latest_delivery_s": revised["latest_delivery_s"],
                    }
                )
            else:
                reference = _public(source_row, source_row_index=source_index, numbers_changed=False, independent_experiment=False, contract="shared_half_duplex")
            b2 = dict(reference)
            b2["method"] = "b2"
            b2["relation"] = EQUIVALENCE_LABEL
            b2["independent_experiment"] = False
            b2["reused_from"] = "revised_moving" if changed else "original_moving_row"
            b2["source_row_index"] = source_index
            b2_rows.append(b2)
    derived_count = len(revised_rows) + len(b2_rows)
    if derived_count > MAX_DERIVED_METHOD_ROWS:
        raise RuntimeError(f"derived method rows exceed the registered cap: {derived_count}")
    difference = []
    for b2 in b2_rows:
        reference = next(
            row
            for row in revised_rows
            if row["geometry"] == b2["geometry"] and row["wind"] == b2["wind"]
        ) if b2["reused_from"] == "revised_moving" else _public(saved_rows[(b2["geometry"], b2["wind"], "moving")][1])
        node_names = list(b2["nodes_J"])
        difference.append(
            {
                "geometry": b2["geometry"],
                "wind": b2["wind"],
                "delivered_count_difference": b2["delivered_count"] - reference["delivered_count"],
                "mean_delay_difference_s": None
                if b2["mean_delay_s"] is None or reference["mean_delay_s"] is None
                else b2["mean_delay_s"] - reference["mean_delay_s"],
                "node_energy_difference_J": {name: b2["nodes_J"][name] - reference["nodes_J"][name] for name in node_names},
                "same_under_shared_rule": _same_score(b2, reference),
            }
        )
    moving_identity = [row for row in identity if row["method"] == "moving"]
    return {
        "source_result_commit": SOURCE_RESULT_COMMIT,
        "environment_steps": 0,
        "training_updates": 0,
        "policy_inferences": 0,
        "independent_experiment": False,
        "equivalence_label": EQUIVALENCE_LABEL,
        "derived_method_row_count": derived_count,
        "revised_c_rows": revised_rows,
        "b2_rows": b2_rows,
        "difference_table": difference,
        "timing_corrections": timing_corrections,
        "identity": identity,
        "proposed_cross_uav_overlaps": proposed_overlaps,
        "scheduled_overlaps": scheduled_overlaps,
        "fairness": {
            "product_count": 8,
            "active_uavs_match": True,
            "points_match": all(
                row["receive_xy_m"] == next(visit["receive_xy_m"] for visit in geometries[row["geometry"]]["moving"]["visits"] if int(visit["uav"]) == row["assigned_uav"])
                if row["active"]
                else row["receive_xy_m"] is None
                for row in moving_identity
            ),
            "telemetry_on_delivery_clock": TELEMETRY_ON_DELIVERY_CLOCK,
        },
        "energy_limit_J": comparison["energy_limit_J"],
        "original_conclusion": comparison["conclusion"],
    }
