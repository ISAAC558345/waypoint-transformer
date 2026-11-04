"""Twenty-four analytical cases. The catalog is fixed before any feasibility number is used."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from wt.ocean_observation_m0.codec import packet_sizes
from wt.ocean_observation_m0.flight import (
    energy_limit_j,
    hover_power_w,
    horizontal_distance_m,
    leg_budget,
    propulsion_power_w,
    select_cruise_airspeed_m_s,
    terminal_hover_j,
    usable_energy_j,
    vertical_service_j,
)
from wt.ocean_observation_m0.parameters import (
    ABUNDANT_MIN_FRAME_S,
    AREA_HEIGHT_M,
    AREA_WIDTH_M,
    BACKHAUL_CLOSED_S,
    BACKHAUL_OPEN_S,
    BUOY_ANTENNA_M,
    BUOY_GAIN_DBI,
    BUOY_IDLE_W,
    COMPUTE_W,
    CORRELATION_LENGTH_M,
    COVERAGE_GRID_M,
    DEPOT_XY,
    EVALUATION_HORIZON_S,
    FLIGHT_ALTITUDE_M,
    GROUND_IDLE_W,
    JOINT_CYCLE_S,
    LANDED_ANTENNA_M,
    LAYOUT_NAMES,
    PRODUCT_TIMES_S,
    REGIMES,
    SCALES,
    SHORE_ANTENNA_M,
    SHORE_GAIN_DBI,
    SHORE_XY,
    UAV_GAIN_DBI,
    WIND_NAMES,
    layout_buoys,
    wind_vector,
)
from wt.ocean_observation_m0.radio import (
    evaluate_link,
    lora_time_on_air_s,
    radio_horizon_m,
    range_for_zero_margin_m,
    rx_power_w,
    slot_s,
    tx_power_w,
)

Point = Tuple[float, float]


def case_catalog() -> List[dict]:
    rows = []
    for n_uav, n_buoy in SCALES:
        for regime in REGIMES:
            for wind_name in WIND_NAMES:
                for layout in LAYOUT_NAMES:
                    buoys = layout_buoys(layout, n_buoy)
                    rows.append(
                        {
                            "case_id": f"u{n_uav}_b{n_buoy}_{regime}_{wind_name}_{layout}",
                            "n_uav": n_uav,
                            "n_buoy": n_buoy,
                            "regime": regime,
                            "wind_name": wind_name,
                            "wind_m_s": list(wind_vector(wind_name)),
                            "layout": layout,
                            "shore_xy_m": list(SHORE_XY),
                            "depot_xy_m": list(DEPOT_XY),
                            "buoys_xy_m": [list(point) for point in buoys],
                            "analytical_case": True,
                            "environment_episode": False,
                            "policy_score": False,
                        }
                    )
    if len(rows) != 24:
        raise RuntimeError(len(rows))
    return rows


def _session(regime: str, traffic_s: float) -> dict:
    if traffic_s < 0.0:
        raise ValueError("traffic")
    if regime == "communication_abundant":
        return {"occupation_s": max(ABUNDANT_MIN_FRAME_S, traffic_s), "fits": True, "reason": "traffic_frame"}
    if regime == "joint_access_and_communication_constraint":
        cycles = max(1, math.ceil(traffic_s / JOINT_CYCLE_S - 1e-12))
        return {"occupation_s": cycles * JOINT_CYCLE_S, "fits": True, "reason": "planning_cycle"}
    if regime == "intermittent_backhaul":
        if traffic_s > BACKHAUL_OPEN_S:
            return {"occupation_s": None, "fits": False, "reason": "traffic_exceeds_open_window"}
        return {
            "occupation_s": BACKHAUL_CLOSED_S + traffic_s,
            "fits": True,
            "reason": "worst_case_closed_interval_then_open_window",
        }
    raise KeyError(regime)


def _collection_regime(regime: str) -> str:
    if regime == "intermittent_backhaul":
        return "communication_abundant"
    return regime


def _slant_m(a: Point, za: float, b: Point, zb: float) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (za - zb) ** 2)


def _nearest_path(depot: Point, buoys: Sequence[Point]) -> List[Point]:
    remaining = list(enumerate(buoys))
    path = [depot]
    current = depot
    while remaining:
        def sort_key(item, origin=current):
            index, point = item
            return (horizontal_distance_m(origin, point), point[0], point[1], index)

        _index, chosen = min(remaining, key=sort_key)
        remaining = [item for item in remaining if item[1] != chosen]
        path.append(chosen)
        current = chosen
    path.append(depot)
    return path


def _groups(buoys: Sequence[Point], n_uav: int) -> List[List[Point]]:
    groups: List[List[Point]] = [[] for _ in range(n_uav)]
    for index, point in enumerate(buoys):
        groups[index % n_uav].append(point)
    return groups


def _farthest(depot: Point, buoys: Sequence[Point]) -> Point:
    def sort_key(item):
        index, point = item
        return (horizontal_distance_m(depot, point), point[0], point[1], index)

    return max(enumerate(buoys), key=sort_key)[1]


def _path_motion(path: Sequence[Point], wind: Point, airspeed: float) -> dict:
    legs = [leg_budget(path[i], path[i + 1], wind, airspeed) for i in range(len(path) - 1)]
    if any(not leg["feasible"] for leg in legs):
        return {"track_feasible": False, "legs": legs}
    outbound = legs[:-1]
    ret = legs[-1]
    return {
        "track_feasible": True,
        "legs": legs,
        "outbound_distance_m": sum(leg["distance_m"] for leg in outbound),
        "return_distance_m": ret["distance_m"],
        "outbound_propulsion_J": sum(leg["propulsion_J"] for leg in outbound),
        "outbound_compute_J": sum(leg["compute_J"] for leg in outbound),
        "outbound_s": sum(leg["time_s"] for leg in outbound),
        "return_propulsion_J": ret["propulsion_J"],
        "return_compute_J": ret["compute_J"],
        "return_s": ret["time_s"],
        "min_ground_speed_m_s": min(leg["ground_speed_m_s"] for leg in legs),
    }


def _radio_pair_j(airtime_s: float) -> float:
    return (tx_power_w() + rx_power_w()) * airtime_s


def evaluate_route(
    buoys: Sequence[Point],
    n_uav: int,
    regime: str,
    wind: Point,
    airspeed: float,
    sizes: dict,
    assign: str,
) -> dict:
    if assign == "farthest":
        groups = [[_farthest(DEPOT_XY, buoys)]]
    elif assign == "split_tour":
        groups = [group for group in _groups(buoys, n_uav) if group]
    else:
        raise KeyError(assign)
    product_air = lora_time_on_air_s(sizes["product"]["framed_bytes"])
    command_air = lora_time_on_air_s(sizes["command"]["framed_bytes"])
    ack_air = lora_time_on_air_s(sizes["ack"]["framed_bytes"])
    telemetry_air = lora_time_on_air_s(sizes["telemetry"]["framed_bytes"])
    product_slot = slot_s(sizes["product"]["framed_bytes"])
    command_slot = slot_s(sizes["command"]["framed_bytes"])
    ack_slot = slot_s(sizes["ack"]["framed_bytes"])

    vehicles = []
    for group in groups:
        path = _nearest_path(DEPOT_XY, group)
        motion = _path_motion(path, wind, airspeed)
        visits = len(group)
        collection = _session(_collection_regime(regime), product_slot)
        visit_hover = vertical_service_j(collection["occupation_s"])
        terminal = terminal_hover_j()
        link = evaluate_link(
            abs(FLIGHT_ALTITUDE_M - BUOY_ANTENNA_M),
            BUOY_ANTENNA_M,
            FLIGHT_ALTITUDE_M,
            BUOY_GAIN_DBI,
            UAV_GAIN_DBI,
            sizes["product"]["framed_bytes"],
        )
        vehicles.append(
            {
                "path_xy_m": [list(point) for point in path],
                "visits": visits,
                "motion": {key: value for key, value in motion.items() if key != "legs"},
                "legs": motion["legs"],
                "collection_occupation_s": collection["occupation_s"],
                "collection_hover_J": visit_hover["propulsion_J"] * visits,
                "collection_compute_J": visit_hover["compute_J"] * visits,
                "collection_s": visit_hover["time_s"] * visits,
                "terminal_propulsion_J": terminal["propulsion_J"],
                "terminal_compute_J": terminal["compute_J"],
                "terminal_s": terminal["time_s"],
                "collection_link": link,
            }
        )

    n_launched = len(vehicles)
    n_packets = sum(vehicle["visits"] for vehicle in vehicles)
    command_traffic = n_launched * (command_slot + ack_slot)
    backhaul_traffic = n_packets * product_slot
    command_session = _session(regime, command_traffic)
    backhaul_session = _session(regime, backhaul_traffic)
    command_link = evaluate_link(
        _slant_m(DEPOT_XY, LANDED_ANTENNA_M, SHORE_XY, SHORE_ANTENNA_M),
        LANDED_ANTENNA_M,
        SHORE_ANTENNA_M,
        UAV_GAIN_DBI,
        SHORE_GAIN_DBI,
        sizes["command"]["framed_bytes"],
    )
    sessions_fit = command_session["fits"] and backhaul_session["fits"] and command_link["closes"]

    per_vehicle = []
    for vehicle in vehicles:
        motion = vehicle["motion"]
        flyable = motion["track_feasible"] and vehicle["collection_link"]["closes"] and vehicle["collection_occupation_s"] is not None
        airborne_s = None
        if motion["track_feasible"]:
            airborne_s = (
                motion["outbound_s"]
                + motion["return_s"]
                + vehicle["collection_s"]
                + vehicle["terminal_s"]
            )
        per_vehicle.append({**vehicle, "flyable_track_and_collection": flyable, "airborne_s": airborne_s})

    flyable_times = [item["airborne_s"] for item in per_vehicle if item["airborne_s"] is not None and item["motion"]["track_feasible"]]
    max_airborne = max(flyable_times) if flyable_times and all(item["motion"]["track_feasible"] for item in per_vehicle) else None

    fleet = _blank_costs()
    vehicle_reports = []
    all_tracks = all(item["motion"]["track_feasible"] for item in per_vehicle)
    for item in per_vehicle:
        report = _vehicle_cost(
            item,
            max_airborne if all_tracks else None,
            command_session,
            backhaul_session,
            product_air,
            command_air,
            ack_air,
            telemetry_air,
            regime,
        )
        vehicle_reports.append(report)
        if report["defined"]:
            for key in fleet:
                fleet[key] += report[key]

    one_sortie_defined = all_tracks and sessions_fit and all(item["collection_link"]["closes"] for item in per_vehicle)
    sortie_s = None
    if one_sortie_defined and max_airborne is not None:
        sortie_s = command_session["occupation_s"] + max_airborne + backhaul_session["occupation_s"]
    episode_energy = None
    second_end = None
    timescale_ok = False
    energy_ok = False
    if one_sortie_defined and sortie_s is not None:
        episode_energy = {key: 2.0 * value for key, value in fleet.items()}
        first_end = sortie_s
        second_start = max(PRODUCT_TIMES_S[1], first_end)
        second_end = second_start + sortie_s
        timescale_ok = second_end <= EVALUATION_HORIZON_S
        limit = energy_limit_j()
        energy_ok = all(2.0 * report["uav_J"] <= limit for report in vehicle_reports)
    delivery = bool(one_sortie_defined and timescale_ok and energy_ok)
    reasons = []
    if not all_tracks:
        reasons.append("track_infeasible")
    if not sessions_fit:
        reasons.append("shore_session_or_command_link")
    if any(not item["collection_link"]["closes"] for item in per_vehicle):
        reasons.append("collection_link")
    if one_sortie_defined and not timescale_ok:
        reasons.append("timescale")
    if one_sortie_defined and not energy_ok:
        reasons.append("energy_reserve")
    if delivery:
        reasons.append("feasible")
    return {
        "assignment": assign,
        "delivery_feasible": delivery,
        "reasons": reasons,
        "defined": one_sortie_defined,
        "sortie_s": sortie_s,
        "second_product_end_s": second_end,
        "timescale_fits_horizon": timescale_ok,
        "energy_within_reserve": energy_ok,
        "energy_limit_J": energy_limit_j(),
        "usable_energy_per_uav_J": usable_energy_j(),
        "command_session": command_session,
        "backhaul_session": backhaul_session,
        "command_link_margin_db": command_link["margin_db"],
        "vehicles": vehicle_reports,
        "one_sortie": fleet if one_sortie_defined else None,
        "episode": episode_energy,
        "min_collection_margin_db": min(item["collection_link"]["margin_db"] for item in per_vehicle),
        "in_flight_telemetry_delivered": regime != "intermittent_backhaul",
        "radio": {
            "tx_dbm": 13.0,
            "bandwidth_hz": 125000.0,
            "sf": 7,
            "packet_error_model": None,
        },
    }


def _blank_costs() -> Dict[str, float]:
    return {
        "flight_J": 0.0,
        "return_J": 0.0,
        "wait_J": 0.0,
        "handshake_J": 0.0,
        "payload_radio_J": 0.0,
        "telemetry_J": 0.0,
        "total_J": 0.0,
        "uav_J": 0.0,
    }


def _vehicle_cost(item, max_airborne, command_session, backhaul_session, product_air, command_air, ack_air, telemetry_air, regime) -> dict:
    motion = item["motion"]
    if not motion["track_feasible"] or command_session["occupation_s"] is None or backhaul_session["occupation_s"] is None:
        blank = _blank_costs()
        blank["defined"] = False
        blank["total_J"] = None
        return blank
    flight = (
        item["terminal_propulsion_J"]
        + item["terminal_compute_J"]
        + motion["outbound_propulsion_J"]
        + motion["outbound_compute_J"]
    )
    ret = motion["return_propulsion_J"] + motion["return_compute_J"]
    early = 0.0
    if max_airborne is not None:
        early = GROUND_IDLE_W * max(0.0, max_airborne - item["airborne_s"])
    ground = GROUND_IDLE_W * (command_session["occupation_s"] + backhaul_session["occupation_s"]) + early
    collection_wait = item["collection_hover_J"] + item["collection_compute_J"]
    buoy_idle = BUOY_IDLE_W * item["collection_s"]
    wait = collection_wait + ground + buoy_idle
    handshake = _radio_pair_j(command_air + ack_air)
    payload = _radio_pair_j(product_air) * item["visits"] * 2.0
    # Two radio hops: buoy to UAV, then UAV to shore. Each hop is one TX plus one RX.
    attempts = int(item["airborne_s"] // 30.0)
    telemetry_tx = tx_power_w() * telemetry_air * attempts
    telemetry_rx = 0.0 if regime == "intermittent_backhaul" else rx_power_w() * telemetry_air * attempts
    telemetry = telemetry_tx + telemetry_rx
    total = flight + ret + wait + handshake + payload + telemetry
    uav_radio = (
        rx_power_w() * command_air
        + tx_power_w() * ack_air
        + rx_power_w() * product_air * item["visits"]
        + tx_power_w() * product_air * item["visits"]
        + telemetry_tx
    )
    uav = flight + ret + collection_wait + ground + uav_radio
    return {
        "defined": True,
        "flight_J": flight,
        "return_J": ret,
        "wait_J": wait,
        "handshake_J": handshake,
        "payload_radio_J": payload,
        "telemetry_J": telemetry,
        "total_J": total,
        "uav_J": uav,
        "airborne_s": item["airborne_s"],
        "visits": item["visits"],
        "path_xy_m": item["path_xy_m"],
        "min_ground_speed_m_s": motion["min_ground_speed_m_s"],
        "collection_margin_db": item["collection_link"]["margin_db"],
        "telemetry_attempts": attempts,
    }


def evaluate_direct(buoys: Sequence[Point], regime: str, sizes: dict) -> dict:
    airtime = lora_time_on_air_s(sizes["product"]["framed_bytes"])
    traffic = len(buoys) * slot_s(sizes["product"]["framed_bytes"])
    session = _session(regime, traffic)
    links = [
        evaluate_link(
            _slant_m(point, BUOY_ANTENNA_M, SHORE_XY, SHORE_ANTENNA_M),
            BUOY_ANTENNA_M,
            SHORE_ANTENNA_M,
            BUOY_GAIN_DBI,
            SHORE_GAIN_DBI,
            sizes["product"]["framed_bytes"],
        )
        for point in buoys
    ]
    links_ok = all(link["closes"] for link in links)
    if not session["fits"] or not links_ok:
        return {
            "delivery_feasible": False,
            "reasons": ["shore_window" if not session["fits"] else "direct_link"],
            "defined": False,
            "session": session,
            "min_margin_db": min(link["margin_db"] for link in links),
            "episode": None,
            "second_product_end_s": None,
        }
    occupation = session["occupation_s"]
    idle = BUOY_IDLE_W * occupation * len(buoys)
    radio = _radio_pair_j(airtime) * len(buoys)
    one = {
        "flight_J": 0.0,
        "return_J": 0.0,
        "wait_J": idle,
        "handshake_J": 0.0,
        "payload_radio_J": radio,
        "telemetry_J": 0.0,
        "total_J": idle + radio,
    }
    episode = {key: 2.0 * value for key, value in one.items()}
    second_end = PRODUCT_TIMES_S[1] + occupation
    timescale_ok = second_end <= EVALUATION_HORIZON_S
    delivery = links_ok and timescale_ok
    return {
        "delivery_feasible": delivery,
        "reasons": ["feasible"] if delivery else ["timescale"],
        "defined": True,
        "session": session,
        "min_margin_db": min(link["margin_db"] for link in links),
        "max_distance_m": max(link["distance_m"] for link in links),
        "one_window": one,
        "episode": episode,
        "second_product_end_s": second_end,
        "timescale_fits_horizon": timescale_ok,
        "radio": {"tx_dbm": 13.0, "bandwidth_hz": 125000.0, "sf": 7, "packet_error_model": None},
    }


def coverage_report(buoys: Sequence[Point]) -> dict:
    step = COVERAGE_GRID_M
    xs = [index * step for index in range(int(AREA_WIDTH_M / step) + 1)]
    ys = [index * step for index in range(int(AREA_HEIGHT_M / step) + 1)]
    total = 0
    covered = 0
    for x in xs:
        for y in ys:
            total += 1
            nearest = min(math.hypot(x - buoy[0], y - buoy[1]) for buoy in buoys)
            if nearest <= CORRELATION_LENGTH_M:
                covered += 1
    redundant = 0
    pairs = 0
    for i, left in enumerate(buoys):
        for right in buoys[i + 1 :]:
            pairs += 1
            if math.hypot(left[0] - right[0], left[1] - right[1]) <= CORRELATION_LENGTH_M:
                redundant += 1
    return {
        "grid_points": total,
        "covered_points": covered,
        "covered_fraction": covered / total,
        "prior_only_points": total - covered,
        "pairs": pairs,
        "pairs_within_correlation_length": redundant,
        "correlation_length_m": CORRELATION_LENGTH_M,
        "correlation_length_is_measurement": False,
    }


def drift_report(buoys: Sequence[Point], wind: Point) -> dict:
    from wt.ocean_observation_m0.information import advance

    finals = []
    exits = 0
    for point in buoys:
        end = advance(point, wind, EVALUATION_HORIZON_S)
        outside = not (0.0 <= end[0] <= AREA_WIDTH_M and 0.0 <= end[1] <= AREA_HEIGHT_M)
        exits += int(outside)
        finals.append({"start_m": list(point), "end_m": [end[0], end[1]], "exited": outside})
    return {
        "horizon_s": EVALUATION_HORIZON_S,
        "exited": exits,
        "reflected": False,
        "positions": finals,
    }


def dominant_bucket(costs: Optional[dict]) -> Optional[str]:
    if not costs:
        return None
    keys = ("flight_J", "return_J", "wait_J", "handshake_J", "payload_radio_J")
    return max(keys, key=lambda key: costs[key])


def link_properties(sizes: dict) -> dict:
    product_bytes = sizes["product"]["framed_bytes"]
    overhead = evaluate_link(
        abs(FLIGHT_ALTITUDE_M - BUOY_ANTENNA_M),
        FLIGHT_ALTITUDE_M,
        BUOY_ANTENNA_M,
        UAV_GAIN_DBI,
        BUOY_GAIN_DBI,
        product_bytes,
    )
    corner = (AREA_WIDTH_M, AREA_HEIGHT_M)
    direct_corner = evaluate_link(
        _slant_m(corner, BUOY_ANTENNA_M, SHORE_XY, SHORE_ANTENNA_M),
        BUOY_ANTENNA_M,
        SHORE_ANTENNA_M,
        BUOY_GAIN_DBI,
        SHORE_GAIN_DBI,
        product_bytes,
    )
    uav_corner = evaluate_link(
        _slant_m(corner, FLIGHT_ALTITUDE_M, SHORE_XY, SHORE_ANTENNA_M),
        FLIGHT_ALTITUDE_M,
        SHORE_ANTENNA_M,
        UAV_GAIN_DBI,
        SHORE_GAIN_DBI,
        product_bytes,
    )
    landed = evaluate_link(
        _slant_m(DEPOT_XY, LANDED_ANTENNA_M, SHORE_XY, SHORE_ANTENNA_M),
        LANDED_ANTENNA_M,
        SHORE_ANTENNA_M,
        UAV_GAIN_DBI,
        SHORE_GAIN_DBI,
        product_bytes,
    )
    return {
        "product_airtime_s": lora_time_on_air_s(product_bytes),
        "product_rate_bps": overhead["rate_bps"],
        "packet_error_model": None,
        "friis_zero_margin_buoy_shore_m": range_for_zero_margin_m(BUOY_GAIN_DBI, SHORE_GAIN_DBI),
        "horizon_buoy_shore_m": radio_horizon_m(BUOY_ANTENNA_M, SHORE_ANTENNA_M),
        "horizon_uav_buoy_m": radio_horizon_m(FLIGHT_ALTITUDE_M, BUOY_ANTENNA_M),
        "horizon_uav_shore_m": radio_horizon_m(FLIGHT_ALTITUDE_M, SHORE_ANTENNA_M),
        "overhead_collection": overhead,
        "domain_corner_direct": direct_corner,
        "domain_corner_uav": uav_corner,
        "landed_backhaul": landed,
        "same_tx_bw_sf_for_all_links": True,
    }


def build_cases() -> List[dict]:
    catalog = case_catalog()
    sizes = packet_sizes()
    airspeed = select_cruise_airspeed_m_s()
    results = []
    for spec in catalog:
        buoys = [tuple(point) for point in spec["buoys_xy_m"]]
        wind = (float(spec["wind_m_s"][0]), float(spec["wind_m_s"][1]))
        direct = evaluate_direct(buoys, spec["regime"], sizes)
        farthest = evaluate_route(buoys, spec["n_uav"], spec["regime"], wind, airspeed, sizes, "farthest")
        tour = evaluate_route(buoys, spec["n_uav"], spec["regime"], wind, airspeed, sizes, "split_tour")
        results.append(
            {
                **spec,
                "cruise_airspeed_m_s": airspeed,
                "direct": direct,
                "farthest": farthest,
                "tour": tour,
                "direct_dominant": dominant_bucket(direct.get("episode")),
                "farthest_dominant": dominant_bucket(farthest.get("episode")),
                "tour_dominant": dominant_bucket(tour.get("episode")),
                "coverage": coverage_report(buoys),
                "drift": drift_report(buoys, wind),
            }
        )
    if len(results) != 24:
        raise RuntimeError("case count changed")
    return results


def conclude(cases: Sequence[dict], missing_parameters: Sequence[dict]) -> dict:
    if any(row.get("missing") for row in missing_parameters):
        return {
            "conclusion": "NEEDS_PARAMETER_EVIDENCE",
            "reason": "at least one required parameter has no source and no synthetic contract",
        }
    abundant = [case for case in cases if case["regime"] == "communication_abundant"]
    if abundant and all(case["direct"]["delivery_feasible"] for case in abundant):
        return {
            "conclusion": "REFRAME_REQUIRED",
            "reason": "direct buoy-to-shore delivery is feasible in every registered abundant case, so a UAV ferry is not required for this product and geometry",
        }
    direct_any = any(case["direct"]["delivery_feasible"] for case in cases)
    ferry_any = any(case["farthest"]["delivery_feasible"] for case in cases)
    if ferry_any and not direct_any:
        return {
            "conclusion": "DESIGN_READY",
            "reason": "a registered collect-and-return route is feasible and direct delivery is not",
        }
    if ferry_any and direct_any:
        return {
            "conclusion": "REFRAME_REQUIRED",
            "reason": "direct delivery succeeds in some registered conditions, so the UAV problem needs a narrower domain before a closed loop",
        }
    return {
        "conclusion": "REFRAME_REQUIRED",
        "reason": "no registered case has a feasible delivery path",
    }


def wind_illustration(airspeed_m_s: float) -> dict:
    """One kilometre east-west legs. Not one of the 24 mission cases."""
    calm = leg_budget((0.0, 0.0), (1000.0, 0.0), (0.0, 0.0), airspeed_m_s)
    tail = leg_budget((0.0, 0.0), (1000.0, 0.0), (8.0, 0.0), airspeed_m_s)
    head = leg_budget((1000.0, 0.0), (0.0, 0.0), (8.0, 0.0), airspeed_m_s)
    return {
        "leg_m": 1000.0,
        "airspeed_m_s": airspeed_m_s,
        "wind_m_s": [8.0, 0.0],
        "calm": calm,
        "tailwind": tail,
        "headwind": head,
        "round_trip_propulsion_J": tail["propulsion_J"] + head["propulsion_J"],
        "two_headwind_propulsion_J": 2.0 * head["propulsion_J"],
    }


def propulsion_facts() -> dict:
    cruise = select_cruise_airspeed_m_s()
    return {
        "P0_W": (PROFILE_DRAG_POWER()),
        "Pi_W": induced_from_flight(),
        "hover_W": hover_power_w(),
        "cruise_airspeed_m_s": cruise,
        "cruise_power_W": propulsion_power_w(cruise),
        "usable_energy_J": usable_energy_j(),
        "energy_limit_J": energy_limit_j(),
        "compute_W": COMPUTE_W,
    }


def PROFILE_DRAG_POWER() -> float:
    from wt.ocean_observation_m0.flight import profile_power_w

    return profile_power_w()


def induced_from_flight() -> float:
    from wt.ocean_observation_m0.flight import induced_power_w

    return induced_power_w()
