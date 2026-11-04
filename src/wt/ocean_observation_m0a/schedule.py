"""Kinematic schedule for the frozen geometries. It does not rank methods by energy."""
from __future__ import annotations

from wt.ocean_observation_m0.codec import packet_sizes
from wt.ocean_observation_m0.flight import leg_budget, select_cruise_airspeed_m_s
from wt.ocean_observation_m0.parameters import (
    DEPOT_XY,
    EVALUATION_HORIZON_S,
    LANDING_HOVER_S,
    PRODUCT_TIMES_S,
    TAKEOFF_HOVER_S,
    WIND_CALM,
    WIND_EAST_8,
)
from wt.ocean_observation_m0.radio import lora_time_on_air_s
from wt.ocean_observation_m0a.links import buoy_to_shore, buoy_to_uav, product_airtime_s, uav_to_shore

Point = tuple[float, float]
WINDS = (("calm", WIND_CALM), ("wind_8mps_east", WIND_EAST_8))
UAV_COUNT = 2


def _point(raw) -> Point:
    return (float(raw[0]), float(raw[1]))


def _blank_uav(index: int) -> dict:
    return {
        "uav": index,
        "legs": [],
        "vertical_s": {"takeoff": 0.0, "landing": 0.0, "wait": 0.0, "radio": 0.0},
        "radio_s": {"rx": 0.0, "tx": 0.0},
        "airborne_s": 0.0,
        "track_feasible": True,
        "windows": [],
        "telemetry_attempts": 0,
        "telemetry_air_s": 0.0,
    }


def _add_vertical(uav: dict, kind: str, duration: float) -> float:
    uav["vertical_s"][kind] += float(duration)
    uav["airborne_s"] += float(duration)
    return float(duration)


def _products(buoy_count: int) -> list[dict]:
    rows = []
    for generation in PRODUCT_TIMES_S:
        for buoy in range(buoy_count):
            rows.append(
                {
                    "product_id": f"b{buoy}_t{int(generation)}",
                    "buoy": buoy,
                    "generation_s": float(generation),
                    "delivered": False,
                    "delivery_s": None,
                    "delay_s": None,
                }
            )
    return rows


def _mark(products: list[dict], buoy: int, generation: float, delivery_s: float | None) -> None:
    for row in products:
        if row["buoy"] == buoy and row["generation_s"] == float(generation):
            if delivery_s is not None and delivery_s <= EVALUATION_HORIZON_S:
                row["delivered"] = True
                row["delivery_s"] = float(delivery_s)
                row["delay_s"] = float(delivery_s - generation)
            return


def _direct(buoys: list, payload_bytes: int) -> dict:
    products = _products(len(buoys))
    buoy_tx = [0.0 for _ in buoys]
    shore_rx = 0.0
    airtime = product_airtime_s(payload_bytes)
    from wt.ocean_observation_m0.parameters import MAC_GUARD_S

    guard = MAC_GUARD_S
    for generation in PRODUCT_TIMES_S:
        clock = float(generation)
        for index, buoy in enumerate(buoys):
            link = buoy_to_shore(_point(buoy), payload_bytes)
            if not link["closes"]:
                continue
            clock += airtime
            buoy_tx[index] += airtime
            shore_rx += airtime
            _mark(products, index, generation, clock)
            clock += guard
    return {
        "method": "direct",
        "products": products,
        "uavs": [_blank_uav(index) for index in range(UAV_COUNT)],
        "buoy_tx_s": buoy_tx,
        "shore_tx_s": 0.0,
        "shore_rx_s": shore_rx,
        "track_feasible": True,
    }


def _time_to_first(visits: list[dict], wind: Point, airspeed: float) -> float | None:
    if not visits:
        return 0.0
    leg = leg_budget(DEPOT_XY, _point(visits[0]["receive_xy_m"]), wind, airspeed)
    if not leg["feasible"]:
        return None
    return TAKEOFF_HOVER_S + float(leg["time_s"])


def _fly(visits: list[dict], buoys: list, payload_bytes: int, wind: Point, airspeed: float, command_bytes: int, ack_bytes: int, telemetry_bytes: int) -> dict:
    products = _products(len(buoys))
    owned = {index for visit in visits for index in visit["buoy_indexes"]}
    uav = _blank_uav(0)
    buoy_tx = [0.0 for _ in buoys]
    shore_tx = 0.0
    shore_rx = 0.0
    airtime = product_airtime_s(payload_bytes)
    command_air = lora_time_on_air_s(command_bytes)
    ack_air = lora_time_on_air_s(ack_bytes)
    telemetry_air = lora_time_on_air_s(telemetry_bytes)
    from wt.ocean_observation_m0.parameters import MAC_GUARD_S

    ready = 0.0
    windows = []
    uav["windows"] = windows
    if not visits:
        return {
            "products": products,
            "uav": uav,
            "buoy_tx_s": buoy_tx,
            "shore_tx_s": shore_tx,
            "shore_rx_s": shore_rx,
            "owned": sorted(owned),
        }
    for generation in PRODUCT_TIMES_S:
        preview = _time_to_first(visits, wind, airspeed)
        if preview is None:
            uav["track_feasible"] = False
            return {
                "products": products,
                "uav": uav,
                "buoy_tx_s": buoy_tx,
                "shore_tx_s": shore_tx,
                "shore_rx_s": shore_rx,
                "owned": sorted(owned),
            }
        clock = max(ready, float(generation) - preview, 0.0)
        launch = clock
        position = DEPOT_XY
        _add_vertical(uav, "takeoff", TAKEOFF_HOVER_S)
        clock += TAKEOFF_HOVER_S
        for visit_index, visit in enumerate(visits):
            target = _point(visit["receive_xy_m"])
            leg = leg_budget(position, target, wind, airspeed)
            if not leg["feasible"]:
                uav["track_feasible"] = False
                return {
                    "products": products,
                    "uav": uav,
                    "buoy_tx_s": buoy_tx,
                    "shore_tx_s": shore_tx,
                    "shore_rx_s": shore_rx,
                    "owned": sorted(owned),
                }
            role = "outbound" if visit_index == 0 else "between"
            uav["legs"].append({"role": role, "generation_s": float(generation), **leg})
            uav["airborne_s"] += float(leg["time_s"])
            clock += float(leg["time_s"])
            position = target
            if clock < float(generation):
                waited = float(generation) - clock
                _add_vertical(uav, "wait", waited)
                clock += waited
            shore = uav_to_shore(position, payload_bytes)
            if shore["closes"]:
                clock += command_air
                _add_vertical(uav, "radio", command_air)
                uav["radio_s"]["rx"] += command_air
                shore_tx += command_air
                clock += MAC_GUARD_S
                _add_vertical(uav, "radio", MAC_GUARD_S)
                clock += ack_air
                _add_vertical(uav, "radio", ack_air)
                uav["radio_s"]["tx"] += ack_air
                shore_rx += ack_air
                clock += MAC_GUARD_S
                _add_vertical(uav, "radio", MAC_GUARD_S)
            received = []
            for buoy_index in visit["buoy_indexes"]:
                link = buoy_to_uav(_point(buoys[buoy_index]), position, payload_bytes)
                if not visit["covers"] or not link["closes"]:
                    continue
                clock += airtime
                _add_vertical(uav, "radio", airtime)
                uav["radio_s"]["rx"] += airtime
                buoy_tx[buoy_index] += airtime
                received.append(buoy_index)
                clock += MAC_GUARD_S
                _add_vertical(uav, "radio", MAC_GUARD_S)
            if shore["closes"]:
                for buoy_index in received:
                    clock += airtime
                    _add_vertical(uav, "radio", airtime)
                    uav["radio_s"]["tx"] += airtime
                    shore_rx += airtime
                    _mark(products, buoy_index, generation, clock)
                    clock += MAC_GUARD_S
                    _add_vertical(uav, "radio", MAC_GUARD_S)
        return_leg = leg_budget(position, DEPOT_XY, wind, airspeed)
        if not return_leg["feasible"]:
            uav["track_feasible"] = False
            return {
                "products": products,
                "uav": uav,
                "buoy_tx_s": buoy_tx,
                "shore_tx_s": shore_tx,
                "shore_rx_s": shore_rx,
                "owned": sorted(owned),
            }
        uav["legs"].append({"role": "return", "generation_s": float(generation), **return_leg})
        uav["airborne_s"] += float(return_leg["time_s"])
        clock += float(return_leg["time_s"])
        _add_vertical(uav, "landing", LANDING_HOVER_S)
        clock += LANDING_HOVER_S
        ready = clock
        windows.append({"launch_s": launch, "landing_s": ready, "generation_s": float(generation)})
    attempts = int(uav["airborne_s"] // 30.0)
    uav["telemetry_attempts"] = attempts
    uav["telemetry_air_s"] = attempts * telemetry_air
    uav["radio_s"]["tx"] += attempts * telemetry_air
    if any(uav_to_shore(_point(visit["receive_xy_m"]), payload_bytes)["closes"] for visit in visits):
        shore_rx += attempts * telemetry_air
    uav["windows"] = windows
    uav["ready_s"] = ready
    return {
        "products": products,
        "uav": uav,
        "buoy_tx_s": buoy_tx,
        "shore_tx_s": shore_tx,
        "shore_rx_s": shore_rx,
        "owned": sorted(owned),
    }


def _merge_flight(plans: list[dict], buoy_count: int) -> dict:
    products = _products(buoy_count)
    by_id = {row["product_id"]: row for row in products}
    uavs = []
    buoy_tx = [0.0 for _ in range(buoy_count)]
    shore_tx = 0.0
    shore_rx = 0.0
    feasible = True
    for plan in plans:
        feasible = feasible and bool(plan["uav"]["track_feasible"])
        uavs.append(plan["uav"])
        shore_tx += plan["shore_tx_s"]
        shore_rx += plan["shore_rx_s"]
        for index, value in enumerate(plan["buoy_tx_s"]):
            buoy_tx[index] += value
        for row in plan["products"]:
            if row["delivered"]:
                by_id[row["product_id"]] = row
    while len(uavs) < UAV_COUNT:
        uavs.append(_blank_uav(len(uavs)))
    return {
        "products": [by_id[row["product_id"]] for row in products],
        "uavs": uavs,
        "buoy_tx_s": buoy_tx,
        "shore_tx_s": shore_tx,
        "shore_rx_s": shore_rx,
        "track_feasible": feasible,
    }


def _visits_for(method: str, geometry: dict) -> list[list[dict]]:
    if method == "fixed":
        fixed = geometry["fixed"]
        visit = {
            "group": "all",
            "buoy_indexes": list(range(len(geometry["buoys_xy_m"]))),
            "receive_xy_m": fixed["xy_m"],
            "covers": fixed["covers_all_buoys"],
        }
        return [[visit], []]
    assigned = [[], []]
    for visit in geometry["moving"]["visits"]:
        assigned[int(visit["uav"])].append(visit)
    return assigned


def schedule_geometry(geometry: dict, wind_name: str, wind: Point, airspeed: float, sizes: dict) -> dict:
    payload = sizes["product"]["framed_bytes"]
    command = sizes["command"]["framed_bytes"]
    ack = sizes["ack"]["framed_bytes"]
    telemetry = sizes["telemetry"]["framed_bytes"]
    if wind_name == "direct_unused":
        raise RuntimeError(wind_name)
    direct = _direct(geometry["buoys_xy_m"], payload)
    rows = {"direct": direct}
    for method in ("fixed", "moving"):
        plans = [
            _fly(visits, geometry["buoys_xy_m"], payload, wind, airspeed, command, ack, telemetry)
            for visits in _visits_for(method, geometry)
        ]
        merged = _merge_flight(plans, len(geometry["buoys_xy_m"]))
        merged["method"] = method
        rows[method] = merged
    return rows


def build_schedule(geometries: list[dict]) -> dict:
    sizes = packet_sizes()
    airspeed = select_cruise_airspeed_m_s()
    contexts = []
    for geometry in geometries:
        winds = []
        for name, wind in WINDS:
            winds.append({"wind": name, "wind_m_s": [wind[0], wind[1]], "methods": schedule_geometry(geometry, name, wind, airspeed, sizes)})
        contexts.append({"geometry": geometry["geometry"], "winds": winds})
    return {
        "airspeed_m_s": airspeed,
        "generation_times_s": list(PRODUCT_TIMES_S),
        "deadline_s": EVALUATION_HORIZON_S,
        "timing_rule": "launch at max(previous landing, generation time minus takeoff and the first leg); receive after one command and one ack; backhaul immediately; return and land",
        "contexts": contexts,
    }
