"""Cross-UAV wind observation for one pre-registered decision.

The simplified field is known structure plus an unknown mean. A samples it
during ordinary flight. B may use the message only when it has arrived.
Power uses airspeed once: commanded velocity stays the ground velocity.
"""

from __future__ import annotations

import math
import random
from typing import Callable

from wt.active_model_identification.belief import Belief
from wt.active_model_identification.physics import CRUISE_M_S, propulsion_power_w
from wt.active_model_identification.world import (
    Action,
    World,
    advance,
    energy_used_j,
    mission_terminal,
    on_time_bits,
    pad_distance_m,
    pad_radius_m,
    utility,
)
from wt.ocean_observation_m0.radio import lora_time_on_air_s, rx_power_w, tx_power_w

WindFn = Callable[[float, float, float], tuple[float, float]]


def _ripple(x: float, y: float, t: float, field: dict) -> tuple[float, float]:
    lx = float(field["Lx_m"])
    ly = float(field["Ly_m"])
    period = float(field["period_s"])
    bx = float(field["bx"])
    by = float(field["by"])
    wx = bx * math.cos(2.0 * math.pi * x / lx) * math.cos(2.0 * math.pi * t / period)
    wy = by * math.sin(2.0 * math.pi * y / ly) * math.sin(2.0 * math.pi * t / period)
    return wx, wy


def true_wind(condition: dict, x: float, y: float, t: float, field: dict) -> tuple[float, float]:
    rx, ry = _ripple(x, y, t, field)
    return float(condition["ax"]) + rx, float(condition["ay"]) + ry


def observe(
    condition: dict,
    x: float,
    y: float,
    t: float,
    field: dict,
    sigma: float,
    rng: random.Random,
) -> dict:
    wx, wy = true_wind(condition, x, y, t, field)
    return {
        "x_m": x,
        "y_m": y,
        "sample_time_s": t,
        "wind_x_m_s": wx + rng.gauss(0.0, sigma),
        "wind_y_m_s": wy + rng.gauss(0.0, sigma),
        "sigma_m_s": sigma,
    }


def predict(message: dict | None, x: float, y: float, t: float, field: dict, prior_std: float) -> dict:
    """Mean wind and a symmetric interval. No message keeps the zero-mean prior."""
    rx, ry = _ripple(x, y, t, field)
    if message is None:
        return {"wind_xy": (rx, ry), "std_m_s": prior_std, "used_message": False}
    sample_rx, sample_ry = _ripple(float(message["x_m"]), float(message["y_m"]), float(message["sample_time_s"]), field)
    ax = float(message["wind_x_m_s"]) - sample_rx
    ay = float(message["wind_y_m_s"]) - sample_ry
    return {
        "wind_xy": (ax + rx, ay + ry),
        "std_m_s": float(message["sigma_m_s"]),
        "used_message": True,
        "ax_m_s": ax,
        "ay_m_s": ay,
    }


def mean_wind_fn(message: dict | None, field: dict, prior_std: float) -> WindFn:
    def wind_at(x: float, y: float, t: float) -> tuple[float, float]:
        found = predict(message, x, y, t, field, prior_std)
        return found["wind_xy"]

    return wind_at


def adverse_wind_fn(message: dict | None, field: dict, prior_std: float, origin: tuple[float, float], home: tuple[float, float]) -> tuple[WindFn, float]:
    """Add one headwind offset of the predictive std. Power still sees the sum once."""
    sample = predict(message, origin[0], origin[1], 0.0, field, prior_std)
    std = float(sample["std_m_s"])
    dx = home[0] - origin[0]
    dy = home[1] - origin[1]
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        extra = (std, 0.0)
    else:
        extra = (-std * dx / length, -std * dy / length)
    base = mean_wind_fn(message, field, prior_std)

    def wind_at(x: float, y: float, t: float) -> tuple[float, float]:
        mx, my = base(x, y, t)
        return mx + extra[0], my + extra[1]

    return wind_at, std


def message_cost_j(payload_bytes: int) -> dict:
    airtime = lora_time_on_air_s(payload_bytes)
    return {
        "payload_bytes": payload_bytes,
        "airtime_s": airtime,
        "sender_j": tx_power_w() * airtime,
        "receiver_j": rx_power_w() * airtime,
    }


def charge_message(world: World, sender: int, receiver: int, payload_bytes: int) -> dict:
    cost = message_cost_j(payload_bytes)
    drawn_sender = _draw_aux(world.uavs[sender], cost["sender_j"])
    drawn_receiver = _draw_aux(world.uavs[receiver], cost["receiver_j"])
    cost["sender_j_drawn"] = drawn_sender
    cost["receiver_j_drawn"] = drawn_receiver
    return cost


def _draw_aux(uav, joules: float) -> float:
    taken = min(joules, max(0.0, uav.energy_j))
    uav.energy_j -= taken
    uav.consumed_j += taken
    uav.aux_j += taken
    uav.true_drop_j += taken
    return taken


def script_action(world: World, uav_index: int, action: Action, spec: dict, interval: int) -> Action | None:
    """Continue one chosen action.

    ``return`` flies home even when the cache still holds packets. ``backhaul``
    delivers that cache first. ``service`` finishes the named buoy, then the cache.
    """
    kind, target = action
    uav = world.uavs[uav_index]
    holding = any(packet.holder == f"uav:{uav_index}" for packet in world.packets)
    landed = (not uav.airborne) and pad_distance_m(uav) <= pad_radius_m(spec)
    if kind == "wait" and interval == 0:
        return ("wait", None)
    if kind == "service" and target is not None:
        pending = any(packet.buoy == int(target) and packet.holder == "buoy" for packet in world.packets)
        if pending:
            return ("service", int(target))
    delivers = kind in ("service", "backhaul")
    if delivers and holding:
        return ("backhaul", None)
    if landed:
        return None
    return ("return", None)


def _keep(_uav, action: Action, _belief: Belief, _world: World) -> tuple[Action, bool]:
    return action, False


def fly_script(
    world: World,
    beliefs: list[Belief],
    spec: dict,
    factor: float,
    uav_index: int,
    action: Action,
    thetas: list[float],
    wind_at: WindFn | None,
    noise,
    *,
    other_actions: dict[int, dict[int, Action]] | None = None,
    admit=None,
    return_guard=None,
    sentinel_battery: bool = False,
    on_interval=None,
) -> dict:
    """Replay one script on a copy. The caller's world and beliefs stay put."""
    sim = world.snapshot()
    if sentinel_battery:
        for uav in sim.uavs:
            uav.energy_j = 1.0e8
            uav.initial_energy_j = 1.0e8
            uav.failed = False
            uav.failure_recorded = False
            uav.pending_failure = False
    local = [item.copy() for item in beliefs]
    consumed_before = [uav.consumed_j for uav in sim.uavs]
    bits_before = on_time_bits(sim)
    decision = float(spec["time"]["decision_s"])
    mission = float(spec["time"]["mission_s"])
    interval0 = int(round(sim.t / decision))
    limit = int(mission // decision) + 4
    requested = []
    executed_last = action
    first_executed = None
    for step in range(limit):
        if sim.t >= mission - 1e-9:
            break
        if on_interval is not None:
            on_interval(sim, local, interval0 + step)
        chosen = script_action(sim, uav_index, action, spec, step)
        epoch = interval0 + step
        if chosen is None and (other_actions is None or epoch not in other_actions):
            break
        actions: list[Action] = [("wait", None) for _ in sim.uavs]
        if chosen is not None:
            actions[uav_index] = chosen
        if other_actions is not None:
            for index, other in other_actions.get(epoch, {}).items():
                if index != uav_index:
                    actions[index] = other
        requested.append([f"{item[0]}:{item[1]}" for item in actions])
        observations = advance(
            sim,
            actions,
            list(thetas),
            local,
            spec,
            factor,
            None,
            planning=False,
            interval_index=interval0 + step,
            noise_j=noise,
            wind_at=wind_at,
            admit=admit if admit is not None else _keep,
            return_guard=return_guard if return_guard is not None else _keep,
        )
        executed_last = tuple(observations[uav_index]["executed"])
        if first_executed is None:
            first_executed = executed_last
    terminals = mission_terminal(sim, spec)
    return {
        "energy_j": math.fsum(uav.consumed_j - consumed_before[uav.index] for uav in sim.uavs),
        "focus_energy_j": sim.uavs[uav_index].consumed_j - consumed_before[uav_index],
        "on_time_bits": on_time_bits(sim) - bits_before,
        "total_on_time_bits": on_time_bits(sim),
        "landed": (not sim.uavs[uav_index].airborne) and pad_distance_m(sim.uavs[uav_index]) <= pad_radius_m(spec),
        "failed": bool(sim.uavs[uav_index].failed),
        "t_end_s": sim.t,
        "on_time": sim.t <= mission + 1e-9,
        "executed": f"{executed_last[0]}:{executed_last[1]}",
        "first_executed": None if first_executed is None else f"{first_executed[0]}:{first_executed[1]}",
        "interventions": sim.interventions,
        "requested_tail": requested[-1] if requested else [],
        "terminals": terminals,
        "utility": utility(sim, spec),
        "world_energy_j": math.fsum(energy_used_j(sim)),
        "sim": sim,
    }


def return_energy_j(uav, theta: float, spec: dict, factor: float, wind_xy: tuple[float, float]) -> float:
    """Straight return plus landing, with airspeed applied once."""
    distance = pad_distance_m(uav)
    aux = float(spec["energy"]["compute_w"])
    landing = float(spec["time"]["landing_s"])
    if distance < 1.0:
        cruise_j = 0.0
    else:
        duration = factor * distance / CRUISE_M_S
        heading_x = (uav.home_x - uav.x) / distance
        heading_y = (uav.home_y - uav.y) / distance
        airspeed = math.hypot(CRUISE_M_S * heading_x - wind_xy[0], CRUISE_M_S * heading_y - wind_xy[1])
        cruise_j = theta * propulsion_power_w(airspeed) * duration + aux * duration
    hover_airspeed = math.hypot(wind_xy[0], wind_xy[1])
    hover_j = theta * propulsion_power_w(hover_airspeed) * landing + aux * landing
    return cruise_j + hover_j


def make_return_guard(message: dict | None, field: dict, prior_std: float, spec: dict, factor: float, focus: int):
    """Force the decider to return when the predicted return no longer fits.

    The other UAV is left on its frozen action. The wind argument is the
    information-set mean, not the simulator truth.
    """

    def guard(uav, action: Action, belief: Belief, world: World) -> tuple[Action, bool]:
        if uav.index != focus or uav.failed or action[0] == "return":
            return action, False
        theta = belief.quantile(float(spec["belief"]["guard_probability"]))
        wind_xy = predict(message, uav.x, uav.y, world.t, field, prior_std)["wind_xy"]
        need = return_energy_j(uav, theta, spec, factor, wind_xy) + float(spec["energy"]["reserve_j"])
        if uav.energy_j < need:
            return ("return", None), True
        return action, False

    return guard
