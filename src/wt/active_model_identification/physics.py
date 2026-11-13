"""Nominal propulsion and trip-energy estimates.

Propulsion power is the existing rotary-wing nominal model. Theta scales that
power only. Integration-factor measurement is a property of this controller,
taken before any trial reward is known.
"""

from __future__ import annotations

import math

from wt.ocean_observation_m0.flight import hover_power_w, propulsion_power_w

CRUISE_M_S = 18.0
ACCEL_M_S2 = 2.0


def clamp(value: float, limit: float) -> float:
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def seek_velocity(x: float, y: float, target_x: float, target_y: float) -> tuple[float, float]:
    dx = target_x - x
    dy = target_y - y
    distance = math.hypot(dx, dy)
    if distance < 2.0:
        return 0.0, 0.0
    stopping = math.sqrt(max(0.0, 2.0 * ACCEL_M_S2 * max(distance - 3.0, 0.0)))
    speed = min(CRUISE_M_S, stopping)
    return dx / distance * speed, dy / distance * speed


def step_kinematics(
    x: float,
    y: float,
    vx: float,
    vy: float,
    desired_vx: float,
    desired_vy: float,
    propulsion: bool,
    dt: float,
) -> tuple[float, float, float, float, float]:
    """Return x, y, vx, vy and the nominal propulsion energy over dt."""
    if not propulsion:
        return x, y, 0.0, 0.0, 0.0
    dv_limit = ACCEL_M_S2 * dt
    vx += clamp(desired_vx - vx, dv_limit)
    vy += clamp(desired_vy - vy, dv_limit)
    x += vx * dt
    y += vy * dt
    airspeed = math.hypot(vx, vy)
    return x, y, vx, vy, propulsion_power_w(airspeed) * dt


def measure_integration_factor() -> float:
    """Simulated straight-flight joules divided by P(cruise) * distance / cruise."""
    x = y = vx = vy = 0.0
    spent = 0.0
    for _ in range(200):
        x, y, vx, vy, z_j = step_kinematics(x, y, vx, vy, CRUISE_M_S, 0.0, True, 1.0)
        spent += z_j
    distance = math.hypot(x, y)
    analytic = propulsion_power_w(CRUISE_M_S) * distance / CRUISE_M_S
    if analytic <= 0.0:
        raise ValueError("integration reference")
    return spent / analytic


def straight_energy_j(distance_m: float, theta: float, aux_w: float, factor: float) -> float:
    if distance_m < 1.0:
        return 0.0
    duration = factor * distance_m / CRUISE_M_S
    nominal = propulsion_power_w(CRUISE_M_S) * duration
    return theta * nominal + aux_w * duration


def hover_energy_j(seconds: float, theta: float, aux_w: float) -> float:
    if seconds <= 0.0:
        return 0.0
    return theta * hover_power_w() * seconds + aux_w * seconds


def trip_energy_j(
    x: float,
    y: float,
    target_x: float,
    target_y: float,
    home_x: float,
    home_y: float,
    theta: float,
    aux_w: float,
    factor: float,
    landing_s: float,
) -> float:
    outbound = math.hypot(target_x - x, target_y - y)
    inbound = math.hypot(home_x - target_x, home_y - target_y)
    return (
        straight_energy_j(outbound, theta, aux_w, factor)
        + hover_energy_j(2.0, theta, aux_w)
        + straight_energy_j(inbound, theta, aux_w, factor)
        + hover_energy_j(landing_s, theta, aux_w)
    )
