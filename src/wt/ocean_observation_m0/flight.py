"""Steady rotary-wing energy. Power uses airspeed. Progress uses ground speed."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

from wt.ocean_observation_m0.parameters import (
    ACCEL_M_S2,
    AIR_DENSITY_KG_M3,
    AIRSPEED_GRID_M_S,
    COMPUTE_W,
    HOVER_ENDURANCE_CONTRACT_S,
    INDUCED_K,
    INDUCED_VELOCITY_M_S,
    LANDING_HOVER_S,
    PARASITE_D0,
    PROFILE_DRAG,
    RESERVE_FRACTION,
    ROTOR_AREA_M2,
    ROTOR_OMEGA_RAD_S,
    ROTOR_RADIUS_M,
    SOLIDITY,
    TAKEOFF_HOVER_S,
    TIP_SPEED_M_S,
    WEIGHT_N,
)

Point = Tuple[float, float]


def profile_power_w() -> float:
    return (
        (PROFILE_DRAG / 8.0)
        * AIR_DENSITY_KG_M3
        * SOLIDITY
        * ROTOR_AREA_M2
        * (ROTOR_OMEGA_RAD_S ** 3)
        * (ROTOR_RADIUS_M ** 3)
    )


def induced_power_w() -> float:
    return (1.0 + INDUCED_K) * (WEIGHT_N ** 1.5) / math.sqrt(2.0 * AIR_DENSITY_KG_M3 * ROTOR_AREA_M2)


def propulsion_power_w(airspeed_m_s: float) -> float:
    """Zeng, Xu and Zhang equation (6). The argument is airspeed."""
    if airspeed_m_s < 0.0:
        raise ValueError("airspeed")
    p0 = profile_power_w()
    pi = induced_power_w()
    v2 = airspeed_m_s * airspeed_m_s
    blade = p0 * (1.0 + 3.0 * v2 / (TIP_SPEED_M_S ** 2))
    induced_term = math.sqrt(1.0 + (v2 * v2) / (4.0 * INDUCED_VELOCITY_M_S ** 4)) - v2 / (
        2.0 * INDUCED_VELOCITY_M_S ** 2
    )
    induced = pi * math.sqrt(max(0.0, induced_term))
    parasite = 0.5 * PARASITE_D0 * AIR_DENSITY_KG_M3 * SOLIDITY * ROTOR_AREA_M2 * v2 * airspeed_m_s
    return blade + induced + parasite


def select_cruise_airspeed_m_s() -> float:
    """Zero-wind joule per ground metre. Ties keep the slower airspeed."""
    best_v = AIRSPEED_GRID_M_S[0]
    best_cost = propulsion_power_w(best_v) / best_v
    for airspeed in AIRSPEED_GRID_M_S[1:]:
        cost = propulsion_power_w(airspeed) / airspeed
        if cost < best_cost:
            best_v = airspeed
            best_cost = cost
    return best_v


def hover_power_w() -> float:
    return propulsion_power_w(0.0)


def usable_energy_j() -> float:
    return HOVER_ENDURANCE_CONTRACT_S * hover_power_w()


def energy_limit_j() -> float:
    return usable_energy_j() * (1.0 - RESERVE_FRACTION)


def horizontal_distance_m(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def unit(a: Point, b: Point) -> Optional[Point]:
    length = horizontal_distance_m(a, b)
    if length <= 0.0:
        return None
    return ((b[0] - a[0]) / length, (b[1] - a[1]) / length)


def ground_speed_m_s(airspeed_m_s: float, wind_m_s: Point, heading: Point) -> Optional[float]:
    """Track-aligned ground speed. None when the airspeed cannot hold the track."""
    w_par = wind_m_s[0] * heading[0] + wind_m_s[1] * heading[1]
    w_perp = wind_m_s[0] * (-heading[1]) + wind_m_s[1] * heading[0]
    if airspeed_m_s * airspeed_m_s < w_perp * w_perp:
        return None
    speed = math.sqrt(airspeed_m_s * airspeed_m_s - w_perp * w_perp) + w_par
    if speed <= 1e-9:
        return None
    return speed


def leg_budget(a: Point, b: Point, wind_m_s: Point, airspeed_m_s: float) -> dict:
    """One straight leg plus a hover-power accel and decel. No negative energy."""
    length = horizontal_distance_m(a, b)
    heading = unit(a, b)
    if heading is None:
        return {
            "feasible": True,
            "reason": "zero_length",
            "distance_m": 0.0,
            "ground_speed_m_s": None,
            "cruise_s": 0.0,
            "transient_s": 0.0,
            "propulsion_J": 0.0,
            "compute_J": 0.0,
            "time_s": 0.0,
        }
    vg = ground_speed_m_s(airspeed_m_s, wind_m_s, heading)
    if vg is None:
        return {
            "feasible": False,
            "reason": "track_infeasible",
            "distance_m": length,
            "ground_speed_m_s": None,
            "cruise_s": None,
            "transient_s": None,
            "propulsion_J": None,
            "compute_J": None,
            "time_s": None,
        }
    cruise_s = length / vg
    transient_s = 2.0 * (airspeed_m_s / ACCEL_M_S2)
    cruise_power = propulsion_power_w(airspeed_m_s)
    hover = hover_power_w()
    propulsion = cruise_power * cruise_s + hover * transient_s
    compute = COMPUTE_W * (cruise_s + transient_s)
    return {
        "feasible": True,
        "reason": "flown",
        "distance_m": length,
        "ground_speed_m_s": vg,
        "cruise_s": cruise_s,
        "transient_s": transient_s,
        "propulsion_J": propulsion,
        "compute_J": compute,
        "time_s": cruise_s + transient_s,
        "along_track_wind_m_s": wind_m_s[0] * heading[0] + wind_m_s[1] * heading[1],
    }


def vertical_service_j(duration_s: float) -> dict:
    if duration_s < 0.0:
        raise ValueError("duration")
    hover = hover_power_w()
    return {
        "time_s": duration_s,
        "propulsion_J": hover * duration_s,
        "compute_J": COMPUTE_W * duration_s,
    }


def terminal_hover_j() -> dict:
    return vertical_service_j(TAKEOFF_HOVER_S + LANDING_HOVER_S)


def add_energy(parts: Sequence[dict]) -> dict:
    propulsion = 0.0
    compute = 0.0
    time_s = 0.0
    for part in parts:
        propulsion += part["propulsion_J"]
        compute += part["compute_J"]
        time_s += part["time_s"]
    if propulsion < -1e-9 or compute < -1e-9:
        raise RuntimeError("negative energy")
    return {"propulsion_J": propulsion, "compute_J": compute, "time_s": time_s}
