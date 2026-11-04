"""Registered geometries and the one fixed-position rule.

Positions come from the horizon and the link test. Energy and delivery are not inputs.
"""
from __future__ import annotations

import math

from wt.ocean_observation_m0.codec import packet_sizes
from wt.ocean_observation_m0.parameters import (
    BUOY_ANTENNA_M,
    DEPOT_XY,
    FLIGHT_ALTITUDE_M,
    SHORE_ANTENNA_M,
    SHORE_XY,
    dispersed_buoys,
)
from wt.ocean_observation_m0a.horizon import PRIMARY_K, geometric_horizon_m
from wt.ocean_observation_m0a.links import buoy_to_uav, uav_to_shore

Point = tuple[float, float]
MARCH_STEP_M = 100.0
OUTSIDE_DIRECT_MARGIN_M = 5000.0
SPLIT_MARGIN_M = 1000.0
CLUSTER_OFFSETS_M = ((-200.0, -200.0), (-200.0, 200.0), (200.0, -200.0), (200.0, 200.0))


def _horizontal_radius_m(horizon_m: float, height_a_m: float, height_b_m: float) -> float:
    gap = abs(float(height_a_m) - float(height_b_m))
    if horizon_m <= gap:
        return 0.0
    return math.sqrt(horizon_m * horizon_m - gap * gap)


def _cluster(center: Point) -> tuple[Point, ...]:
    return tuple((center[0] + dx, center[1] + dy) for dx, dy in CLUSTER_OFFSETS_M)


def _pair(center: Point) -> tuple[Point, ...]:
    """Two buoys on one side. A separated context then has four buoys in total."""
    return ((center[0] - 200.0, center[1]), (center[0] + 200.0, center[1]))


def _centroid(points: tuple[Point, ...]) -> Point:
    return (sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points))


def contexts() -> tuple[dict, ...]:
    """Three geometries. The split is one declared margin past a single coverage disk."""
    product_bytes = packet_sizes()["product"]["framed_bytes"]
    direct = geometric_horizon_m(BUOY_ANTENNA_M, SHORE_ANTENNA_M, PRIMARY_K)
    direct_radius = _horizontal_radius_m(direct, BUOY_ANTENNA_M, SHORE_ANTENNA_M)
    relay = geometric_horizon_m(FLIGHT_ALTITUDE_M, BUOY_ANTENNA_M, PRIMARY_K)
    relay_radius = _horizontal_radius_m(relay, FLIGHT_ALTITUDE_M, BUOY_ANTENNA_M)
    far_center = (direct_radius + OUTSIDE_DIRECT_MARGIN_M, 0.0)
    half = relay_radius + SPLIT_MARGIN_M
    return (
        {
            "geometry": "original_domain",
            "reason": "Four buoys inside the original 2 km square, where a direct buoy-shore link already closes.",
            "groups": ({"name": "domain", "buoys": dispersed_buoys(4)},),
        },
        {
            "geometry": "single_region_beyond_direct",
            "reason": "One cluster 5 km beyond the k=4/3 buoy-shore horizon, still inside one UAV coverage disk from the depot.",
            "groups": ({"name": "far", "buoys": _cluster(far_center)},),
        },
        {
            "geometry": "separated_regions",
            "reason": "Two clusters placed one declared kilometre beyond opposite UAV-buoy coverage radii, so one hover cannot see both.",
            "groups": (
                {"name": "east", "buoys": _pair((half, 0.0))},
                {"name": "west", "buoys": _pair((-half, 0.0))},
            ),
        },
    )


def cover_point(origin: Point, buoys: tuple[Point, ...], payload_bytes: int) -> dict:
    """First point on the depot-centroid segment, in 100 m steps, that closes every buoy and the shore."""
    center = _centroid(buoys)
    length = math.hypot(center[0] - origin[0], center[1] - origin[1])
    count = 0 if length == 0.0 else int(math.ceil(length / MARCH_STEP_M))
    checked = []
    for index in range(count + 1):
        distance = min(length, index * MARCH_STEP_M)
        fraction = 0.0 if length == 0.0 else distance / length
        point = (origin[0] + fraction * (center[0] - origin[0]), origin[1] + fraction * (center[1] - origin[1]))
        buoy_links = [buoy_to_uav(buoy, point, payload_bytes) for buoy in buoys]
        shore = uav_to_shore(point, payload_bytes)
        closed = all(link["closes"] for link in buoy_links) and bool(shore["closes"])
        checked.append({"xy_m": [point[0], point[1]], "closed": closed})
        if closed:
            return {"xy_m": [point[0], point[1]], "covers": True, "checked_points": len(checked)}
    return {"xy_m": [center[0], center[1]], "covers": False, "checked_points": len(checked)}


def fixed_position(groups: tuple[dict, ...], payload_bytes: int) -> dict:
    """One hover for every buoy. A miss stays at the centroid and does not add another candidate."""
    buoys = tuple(buoy for group in groups for buoy in group["buoys"])
    chosen = cover_point(DEPOT_XY, buoys, payload_bytes)
    return {
        "rule": "first 100 m step from the depot toward the centroid that closes every buoy link and the shore link",
        "xy_m": chosen["xy_m"],
        "covers_all_buoys": chosen["covers"],
        "checked_points": chosen["checked_points"],
    }


def moving_route(groups: tuple[dict, ...], payload_bytes: int) -> dict:
    """Each cluster gets the same segment rule. East goes to UAV 0 and west goes to UAV 1."""
    visits = []
    for group in groups:
        chosen = cover_point(DEPOT_XY, group["buoys"], payload_bytes)
        center = _centroid(group["buoys"])
        visits.append(
            {
                "group": group["name"],
                "buoy_indexes": [],
                "centroid_xy_m": [center[0], center[1]],
                "receive_xy_m": chosen["xy_m"],
                "covers": chosen["covers"],
                "uav": 0 if center[0] >= 0.0 else 1,
            }
        )
    # Indexes follow the flattened buoy order used by the context.
    cursor = 0
    for visit, group in zip(visits, groups):
        visit["buoy_indexes"] = list(range(cursor, cursor + len(group["buoys"])))
        cursor += len(group["buoys"])
    return {
        "rule": "each cluster uses the fixed-position segment rule; UAV 0 takes centroids with x>=0 and UAV 1 takes the rest",
        "visits": visits,
    }


def geometry_manifest() -> dict:
    product = packet_sizes()["product"]
    rows = []
    for context in contexts():
        groups = context["groups"]
        buoys = tuple(buoy for group in groups for buoy in group["buoys"])
        rows.append(
            {
                "geometry": context["geometry"],
                "reason": context["reason"],
                "buoys_xy_m": [[point[0], point[1]] for point in buoys],
                "groups": [{"name": group["name"], "buoys_xy_m": [[point[0], point[1]] for point in group["buoys"]]} for group in groups],
                "fixed": fixed_position(groups, product["framed_bytes"]),
                "moving": moving_route(groups, product["framed_bytes"]),
            }
        )
    return {
        "k": PRIMARY_K,
        "earth_radius_m": 6371000.0,
        "march_step_m": MARCH_STEP_M,
        "outside_direct_margin_m": OUTSIDE_DIRECT_MARGIN_M,
        "split_margin_m": SPLIT_MARGIN_M,
        "depot_xy_m": [DEPOT_XY[0], DEPOT_XY[1]],
        "shore_xy_m": [SHORE_XY[0], SHORE_XY[1]],
        "altitude_m": FLIGHT_ALTITUDE_M,
        "drift": "not applied; every method uses these coordinates at both generation times",
        "product_framed_bytes": product["framed_bytes"],
        "geometries": rows,
    }
