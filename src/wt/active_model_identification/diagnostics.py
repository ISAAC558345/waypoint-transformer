"""Deployment-matched fixed receiver and same-contract direct accounting.

Neither path is an environment mission. Direct delivery is credited only when
the registered short-range bulk contract closes.
"""

from __future__ import annotations

import math

from wt.ocean_observation_m0.parameters import BUOY_ANTENNA_M, SHORE_ANTENNA_M
from wt.ocean_observation_m0a.links import buoy_to_shore


def _workload_bits(spec: dict) -> float:
    return math.fsum(float(buoy["packets"]) * float(buoy["bits_each"]) for buoy in spec["geometry"]["buoys"])


def fixed_receiver_row(spec: dict, scenario: dict) -> dict:
    energy = spec["energy"]
    delivered = 0.0
    links = []
    for buoy in spec["geometry"]["buoys"]:
        best = None
        for uav in spec["geometry"]["uavs"]:
            distance = math.hypot(float(buoy["xy"][0]) - float(uav["home_xy"][0]), float(buoy["xy"][1]) - float(uav["home_xy"][1]))
            closes = distance <= float(energy["bulk_range_m"])
            row = {"buoy": int(buoy["id"]), "receiver_xy": list(uav["home_xy"]), "distance_m": distance, "closes": closes}
            if best is None or distance < best["distance_m"]:
                best = row
        links.append(best)
        if best["closes"]:
            delivered += float(buoy["packets"]) * float(buoy["bits_each"])
    return {
        "kind": "fixed_receiver",
        "scenario_id": scenario["id"],
        "environment_mission": False,
        "delivered_bits": delivered,
        "workload_bits": _workload_bits(spec),
        "propulsion_j": 0.0,
        "links": links,
    }


def direct_row(spec: dict, scenario: dict) -> dict:
    energy = spec["energy"]
    delivered = 0.0
    links = []
    lora_aside = []
    for buoy in spec["geometry"]["buoys"]:
        distance = math.hypot(float(buoy["xy"][0]) - float(spec["geometry"]["shore_xy"][0]), float(buoy["xy"][1]) - float(spec["geometry"]["shore_xy"][1]))
        closes = distance <= float(energy["bulk_range_m"])
        bits = float(buoy["packets"]) * float(buoy["bits_each"])
        if closes:
            delivered += bits
        links.append({"buoy": int(buoy["id"]), "distance_m": distance, "bulk_closes": closes, "bits": bits})
        aside = buoy_to_shore((float(buoy["xy"][0]), float(buoy["xy"][1])), 200)
        lora_aside.append(
            {
                "buoy": int(buoy["id"]),
                "closes": bool(aside["closes"]),
                "distance_m": aside["distance_m"],
                "horizon_m": aside["horizon_m"],
                "credited": False,
                "reason": "LoRa closure is not the registered bulk contract",
            }
        )
    return {
        "kind": "direct_bulk",
        "scenario_id": scenario["id"],
        "environment_mission": False,
        "delivered_bits": delivered,
        "workload_bits": _workload_bits(spec),
        "counted_in_executed_tasks": False,
        "links": links,
        "lora_aside_not_credited": lora_aside,
        "antenna_heights_m": {"buoy": BUOY_ANTENNA_M, "shore": SHORE_ANTENNA_M},
    }
