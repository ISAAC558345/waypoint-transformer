"""Non-learning schedulers for the four M1 controls.

Finite lookahead is coordinate-wise: each UAV scores its own action while the
other waits, over a 7-node marginal quadrature. The explicit horizon is one
decision interval plus an analytic terminal value. It is not an exact
belief-space optimum.
"""

from __future__ import annotations

import math

from wt.active_model_identification.belief import Belief, gaussian_quadrature, quadrature
from wt.active_model_identification.physics import CRUISE_M_S, hover_energy_j, straight_energy_j
from wt.active_model_identification.world import (
    Action,
    World,
    action_allowed,
    action_cost_j,
    advance,
    energy_used_j,
    on_time_bits,
    pad_distance_m,
    tx_seconds,
)

WAIT: Action = ("wait", None)


def _bits_at_buoy(world: World, buoy_id: int) -> float:
    return math.fsum(packet.bits for packet in world.packets if packet.holder == "buoy" and packet.buoy == buoy_id)


def _cached_bits(world: World, uav_index: int) -> float:
    holder = f"uav:{uav_index}"
    return math.fsum(packet.bits for packet in world.packets if packet.holder == holder)


def candidates(spec: dict, world: World, uav_index: int, *, include_calibrate: bool) -> list[Action]:
    actions: list[Action] = [WAIT, ("backhaul", None), ("return", None)]
    if include_calibrate:
        actions.append(("calibrate", None))
    for buoy in spec["geometry"]["buoys"]:
        if _bits_at_buoy(world, int(buoy["id"])) > 0.0:
            actions.append(("service", int(buoy["id"])))
    return actions


def _trip_duration_s(world: World, uav_index: int, buoy_id: int, spec: dict, factor: float) -> float:
    uav = world.uavs[uav_index]
    buoy = spec["geometry"]["buoys"][buoy_id]
    outbound = math.hypot(float(buoy["xy"][0]) - uav.x, float(buoy["xy"][1]) - uav.y)
    inbound = math.hypot(uav.home_x - float(buoy["xy"][0]), uav.home_y - float(buoy["xy"][1]))
    return factor * (outbound + inbound) / CRUISE_M_S + float(spec["time"]["landing_s"]) + 2.0


def _deliverable_cached_bits(world: World, beliefs: list[Belief], spec: dict, factor: float, focus: int) -> float:
    """Credit cached bits only when backhaul and the return both fit.

    Airtime is the focus UAV's own LoRa time. Contention with the other UAV
    is omitted; that is part of the one-interval terminal approximation.
    """
    bits = _cached_bits(world, focus)
    uav = world.uavs[focus]
    if bits <= 0.0 or uav.failed:
        return 0.0
    airtime = tx_seconds(bits, spec)
    decision = float(spec["time"]["decision_s"])
    occupied = math.ceil(max(airtime, 1e-9) / decision) * decision
    home = pad_distance_m(uav)
    landing = float(spec["time"]["landing_s"])
    return_s = factor * home / CRUISE_M_S + landing
    deadline = float(spec["time"]["mission_s"])
    if world.t + airtime > deadline + 1e-9 or world.t + occupied + return_s > deadline + 1e-9:
        return 0.0
    theta = beliefs[focus].quantile(float(spec["belief"]["guard_probability"]))
    aux = float(spec["energy"]["compute_w"])
    need = hover_energy_j(occupied, theta, aux)
    need += straight_energy_j(home, theta, aux, factor)
    need += hover_energy_j(landing, theta, aux)
    need += float(spec["energy"]["reserve_j"])
    if uav.energy_j + 1e-6 < need:
        return 0.0
    return bits


def terminal_value(world: World, beliefs: list[Belief], spec: dict, factor: float, focus: int) -> float:
    block = spec["utility"]
    scale_b = float(block["b_ref_bits"])
    scale_e = float(block["e_ref_j"])
    weight = float(block["lambda_e"])
    value = _deliverable_cached_bits(world, beliefs, spec, factor, focus) / scale_b
    uav = world.uavs[focus]
    if uav.failed:
        return value
    theta = beliefs[focus].quantile(float(spec["belief"]["guard_probability"]))
    remaining = float(spec["time"]["mission_s"]) - world.t
    best = 0.0
    for buoy in spec["geometry"]["buoys"]:
        buoy_id = int(buoy["id"])
        bits = _bits_at_buoy(world, buoy_id)
        if bits <= 0.0:
            continue
        action: Action = ("service", buoy_id)
        if not action_allowed(uav, action, theta, spec, factor):
            continue
        if _trip_duration_s(world, focus, buoy_id, spec, factor) > remaining:
            continue
        cost = action_cost_j(uav, action, theta, spec, factor)
        gain = bits / scale_b - weight * cost / scale_e
        if gain > best:
            best = gain
    return value + best


def _utility_delta(before: World, after: World, spec: dict) -> float:
    block = spec["utility"]
    bits = on_time_bits(after) - on_time_bits(before)
    energy = math.fsum(energy_used_j(after)) - math.fsum(energy_used_j(before))
    return bits / float(block["b_ref_bits"]) - float(block["lambda_e"]) * energy / float(block["e_ref_j"])


def _roll_value(
    world: World,
    beliefs: list[Belief],
    action: Action,
    uav_index: int,
    theta: float,
    epsilon_j: float,
    spec: dict,
    factor: float,
    stats: dict | None,
) -> float:
    snap = world.snapshot()
    if stats is not None:
        stats["planning_copies"] = int(stats.get("planning_copies", 0)) + 1
    copied = [item.copy() for item in beliefs]
    dynamics = [item.mean() for item in beliefs]
    dynamics[uav_index] = theta
    actions: list[Action] = [WAIT for _ in world.uavs]
    actions[uav_index] = action
    advance(
        snap,
        actions,
        dynamics,
        copied,
        spec,
        factor,
        stats,
        planning=True,
        interval_index=0,
        noise_j=lambda uav, _interval, _epsilon=epsilon_j, _focus=uav_index: _epsilon if uav == _focus else 0.0,
    )
    immediate = _utility_delta(world, snap, spec)
    return immediate + terminal_value(snap, copied, spec, factor, uav_index)


def score_actions(
    world: World,
    beliefs: list[Belief],
    spec: dict,
    factor: float,
    uav_index: int,
    *,
    include_calibrate: bool,
    stats: dict | None,
) -> list[tuple[float, Action]]:
    nodes = int(spec["belief"]["quadrature_nodes"])
    noise_nodes = int(spec.get("integral", {}).get("noise_nodes", 1))
    grid = quadrature(beliefs[uav_index], nodes)
    noises = gaussian_quadrature(float(spec["belief"]["sigma_j"]), noise_nodes)
    scored: list[tuple[float, Action]] = []
    for action in candidates(spec, world, uav_index, include_calibrate=include_calibrate):
        total = 0.0
        for theta, theta_weight in grid:
            for epsilon, noise_weight in noises:
                total += theta_weight * noise_weight * _roll_value(
                    world, beliefs, action, uav_index, theta, epsilon, spec, factor, stats
                )
        scored.append((total, action))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def _greedy_scores(
    world: World,
    belief: Belief,
    spec: dict,
    factor: float,
    uav_index: int,
    *,
    include_calibrate: bool,
) -> list[tuple[float, Action]]:
    uav = world.uavs[uav_index]
    theta = belief.quantile(float(spec["belief"]["guard_probability"]))
    block = spec["utility"]
    scored: list[tuple[float, Action]] = [(0.0, WAIT)]
    cached = _cached_bits(world, uav_index)
    if cached > 0.0 and action_allowed(uav, ("backhaul", None), theta, spec, factor):
        scored.append((cached / float(block["b_ref_bits"]), ("backhaul", None)))
    if include_calibrate and action_allowed(uav, ("calibrate", None), theta, spec, factor):
        cost = action_cost_j(uav, ("calibrate", None), theta, spec, factor)
        scored.append((-float(block["lambda_e"]) * cost / float(block["e_ref_j"]), ("calibrate", None)))
    remaining = float(spec["time"]["mission_s"]) - world.t
    for buoy in spec["geometry"]["buoys"]:
        buoy_id = int(buoy["id"])
        bits = _bits_at_buoy(world, buoy_id)
        action: Action = ("service", buoy_id)
        if bits <= 0.0 or not action_allowed(uav, action, theta, spec, factor):
            continue
        if _trip_duration_s(world, uav_index, buoy_id, spec, factor) > remaining:
            continue
        cost = action_cost_j(uav, action, theta, spec, factor)
        gain = bits / float(block["b_ref_bits"]) - float(block["lambda_e"]) * cost / float(block["e_ref_j"])
        scored.append((gain, action))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


_ACTION_ORDER = {"service": 0, "backhaul": 1, "return": 2, "wait": 3, "calibrate": 4}


def action_rank(action: Action) -> tuple[int, int]:
    kind, target = action
    return (_ACTION_ORDER[kind], -1 if target is None else int(target))


def tie_abs(spec: dict) -> float:
    return float(spec.get("integral", {}).get("tie_abs", 0.0))


def pick_action(scored: list[tuple[float, Action]], tau: float) -> tuple[float, Action]:
    """Highest score. Scores inside the numerical tolerance share one action order."""
    best = max(score for score, _action in scored)
    pool = [item for item in scored if best - item[0] <= tau + 1e-15]
    pool.sort(key=lambda item: (action_rank(item[1]), -item[0]))
    return pool[0]


def _repair(chosen: list[tuple[float, Action]], tables: list[list[tuple[float, Action]]], tau: float) -> list[Action]:
    actions = [item[1] for item in chosen]
    if len(actions) == 2 and actions[0][0] == "service" and actions[0] == actions[1]:
        if abs(chosen[0][0] - chosen[1][0]) <= tau + 1e-15:
            weaker = 1
        else:
            weaker = 0 if chosen[0][0] < chosen[1][0] else 1
        for _score, action in tables[weaker]:
            if action != actions[stronger_index(weaker)]:
                actions[weaker] = action
                break
    return actions


def stronger_index(weaker: int) -> int:
    return 1 - weaker


LOOKAHEAD_METHODS = {
    "finite_lookahead": True,
    "lookahead_calibration_on": True,
    "lookahead_calibration_off": False,
}


def select_lookahead(
    method: str,
    world: World,
    beliefs: list[Belief],
    spec: dict,
    factor: float,
    stats: dict | None = None,
) -> tuple[list[Action], list[list[tuple[float, Action]]]]:
    if method not in LOOKAHEAD_METHODS:
        raise KeyError(method)
    include = LOOKAHEAD_METHODS[method]
    tables = [
        score_actions(world, beliefs, spec, factor, index, include_calibrate=include, stats=stats)
        for index in range(len(world.uavs))
    ]
    tau = tie_abs(spec)
    chosen = [pick_action(table, tau) for table in tables]
    return _repair(chosen, tables, tau), tables


def select_actions(
    method: str,
    world: World,
    beliefs: list[Belief],
    spec: dict,
    factor: float,
    epoch: int,
    stats: dict | None = None,
) -> list[Action]:
    if method in LOOKAHEAD_METHODS:
        actions, _tables = select_lookahead(method, world, beliefs, spec, factor, stats)
        return actions
    if method not in ("fixed_model", "passive", "fixed_calibration"):
        raise KeyError(method)
    include_calibrate = method == "fixed_calibration" and epoch == 0
    tables = [
        _greedy_scores(world, beliefs[index], spec, factor, index, include_calibrate=include_calibrate)
        for index in range(len(world.uavs))
    ]
    if include_calibrate:
        forced: list[tuple[float, Action]] = []
        for index, table in enumerate(tables):
            calibrate = next((item for item in table if item[1][0] == "calibrate"), None)
            if calibrate is None:
                forced.append((0.0, ("calibrate", None)))
            else:
                forced.append((calibrate[0] + 1.0e6, ("calibrate", None)))
            _ = index
        chosen = forced
    else:
        chosen = [table[0] for table in tables]
    return _repair(chosen, tables, tie_abs(spec))
