"""Registered M1 scenarios and the battery rule.

Probe flights use a sentinel battery and are not trial samples. Their step
count is returned separately so it is not added to the reported mission steps.
"""

from __future__ import annotations

import json
import math

from wt.active_model_identification.belief import Belief, prior_belief
from wt.active_model_identification.physics import measure_integration_factor
from wt.active_model_identification.world import Action, World, action_allowed, advance, make_world

SENTINEL_BATTERY_J = 1.0e7
_BUILD_CACHE: dict[str, tuple[list[dict], dict, float]] = {}


def observation_noise(seed: int, sigma_j: float):
    import random

    def _noise(uav_index: int, interval_index: int) -> float:
        generator = random.Random(f"m1-obs-{seed}-{uav_index}-{interval_index}")
        return generator.gauss(0.0, sigma_j)

    return _noise


def _point_belief(spec: dict, theta: float) -> Belief:
    lo, hi = (float(spec["belief"]["support"][0]), float(spec["belief"]["support"][1]))
    std = 1.0e-4
    return Belief(float(theta), 1.0 / (std * std), lo, hi)


def _far_id(spec: dict, uav_index: int) -> int:
    return int(spec["geometry"]["uavs"][uav_index]["far_buoy"])


def _near_id(spec: dict, uav_index: int) -> int:
    return int(spec["geometry"]["uavs"][uav_index]["near_buoy"])


def _actions(spec: dict, active: int, action: Action) -> list[Action]:
    actions: list[Action] = [("wait", None) for _ in spec["geometry"]["uavs"]]
    actions[active] = action
    return actions


def _finished(world: World, uav_index: int, buoy_id: int) -> bool:
    uav = world.uavs[uav_index]
    pending = any(packet.buoy == buoy_id and packet.holder != "shore" for packet in world.packets)
    away = math.hypot(uav.x - uav.home_x, uav.y - uav.home_y) > 15.0
    return (not pending) and (not away) and (not uav.airborne) and (not uav.cache_ids)


def _run(
    spec: dict,
    factor: float,
    battery: list[float],
    beliefs: list[Belief],
    dynamics: list[float],
    action_at,
    intervals: int,
    tag: str,
    counter: dict | None = None,
) -> World:
    world = make_world(spec, battery, tag)
    for interval in range(intervals):
        actions = action_at(world, interval)
        if actions is None:
            break
        if counter is not None:
            counter["intervals"] = int(counter.get("intervals", 0)) + 1
        advance(
            world,
            actions,
            dynamics,
            beliefs,
            spec,
            factor,
            None,
            planning=True,
            interval_index=interval,
            noise_j=lambda _uav, _interval: 0.0,
        )
    return world


def _service_script(spec: dict, uav_index: int, buoy_id: int):
    def action_at(world: World, _interval: int):
        if _finished(world, uav_index, buoy_id):
            return None
        uav = world.uavs[uav_index]
        holding = any(packet.holder == f"uav:{uav_index}" for packet in world.packets)
        at_buoy = any(packet.buoy == buoy_id and packet.holder == "buoy" for packet in world.packets)
        if at_buoy:
            return _actions(spec, uav_index, ("service", buoy_id))
        if holding:
            return _actions(spec, uav_index, ("backhaul", None))
        if math.hypot(uav.x - uav.home_x, uav.y - uav.home_y) > 12.0 or uav.airborne:
            return _actions(spec, uav_index, ("return", None))
        return _actions(spec, uav_index, ("wait", None))

    return action_at


def _cal_then_service(spec: dict, uav_index: int, buoy_id: int):
    follow = _service_script(spec, uav_index, buoy_id)

    def action_at(world: World, interval: int):
        if interval == 0:
            return _actions(spec, uav_index, ("calibrate", None))
        return follow(world, interval)

    return action_at


def _delivered(world: World, buoy_id: int) -> bool:
    packets = [packet for packet in world.packets if packet.buoy == buoy_id]
    return bool(packets) and all(packet.holder == "shore" for packet in packets)


def _measure_pair(spec: dict, factor: float, theta: float, uav_index: int, counter: dict | None = None) -> dict:
    buoy_id = _far_id(spec, uav_index)
    beliefs = [_point_belief(spec, theta), _point_belief(spec, theta)]
    battery = [SENTINEL_BATTERY_J, SENTINEL_BATTERY_J]
    chained = _run(
        spec,
        factor,
        battery,
        [item.copy() for item in beliefs],
        [theta, theta],
        _cal_then_service(spec, uav_index, buoy_id),
        24,
        f"probe-chain-{uav_index}",
        counter,
    )
    direct = _run(
        spec,
        factor,
        battery,
        [item.copy() for item in beliefs],
        [theta, theta],
        _service_script(spec, uav_index, buoy_id),
        24,
        f"probe-far-{uav_index}",
        counter,
    )
    if not _delivered(chained, buoy_id) or chained.uavs[uav_index].failed:
        raise RuntimeError(f"probe calibration-plus-far did not finish for uav {uav_index} at theta {theta}")
    if not _delivered(direct, buoy_id) or direct.uavs[uav_index].failed:
        raise RuntimeError(f"probe far trip did not finish for uav {uav_index} at theta {theta}")
    return {
        "chain_j": chained.uavs[uav_index].consumed_j,
        "far_j": direct.uavs[uav_index].consumed_j,
        "chain_return_failures": chained.return_failures,
        "far_return_failures": direct.return_failures,
    }


def _chain_status(spec: dict, factor: float, battery: list[float], uav_index: int, counter: dict | None) -> dict:
    try:
        _assert_chain_delivers(spec, factor, battery, uav_index, 1.0, counter)
    except RuntimeError as exc:
        return {"uav": uav_index, "delivers": False, "detail": str(exc)}
    return {"uav": uav_index, "delivers": True, "detail": ""}


def assess_budgets(spec: dict) -> dict:
    """Apply the registered chain-cost formula and report whether the two classes still exist."""
    key = json.dumps(spec, sort_keys=True)
    cached = _BUILD_CACHE.get(key)
    if cached is not None:
        return cached
    factor = measure_integration_factor()
    reserve = float(spec["energy"]["reserve_j"])
    margin = float(spec["energy"]["budget_margin_j"])
    sensitive = []
    insensitive = []
    probes = []
    counter = {"intervals": 0}
    for uav_index in range(2):
        low = _measure_pair(spec, factor, 1.0, uav_index, counter)
        high = _measure_pair(spec, factor, 1.4, uav_index, counter)
        probes.append({"uav": uav_index, "theta_1": low, "theta_1_4": high})
        sensitive.append(low["chain_j"] + reserve + margin)
        insensitive.append(high["chain_j"] + reserve + margin)
    sensitive_predicates = predicate_rows(spec, factor, sensitive)
    insensitive_predicates = predicate_rows(spec, factor, insensitive)
    chain = [_chain_status(spec, factor, sensitive, uav_index, counter) for uav_index in range(2)]
    far_blocked = all(not row["far_allowed_under_prior"] for row in sensitive_predicates)
    far_open = all(row["far_allowed_under_prior"] for row in insensitive_predicates)
    chain_ok = all(row["delivers"] for row in chain)
    separates = far_blocked and far_open and chain_ok
    if not far_blocked:
        reason = (
            "Recomputed sensitive battery allows the high-value service under the prior guard, "
            "so the registered chain-cost formula does not create the decision-sensitive class."
        )
    elif not chain_ok:
        reason = "Calibration then the high-value trip does not finish on the recomputed sensitive battery."
    elif not far_open:
        reason = "Recomputed insensitive battery still blocks the high-value service under the prior guard."
    else:
        reason = ""
    scenarios = []
    for base in spec["scenarios"]:
        for condition, battery in (("sensitive", sensitive), ("insensitive", insensitive)):
            scenarios.append(
                {
                    "id": f"{base['base']}_{condition}",
                    "base": base["base"],
                    "condition": condition,
                    "theta": [float(value) for value in base["theta"]],
                    "seed": int(base["seed"]),
                    "battery_j": [float(value) for value in battery],
                    "wind_xy_m_s": [float(value) for value in spec["motion"]["wind_xy_m_s"]],
                }
            )
    budget = {
        "integration_factor": factor,
        "reserve_j": reserve,
        "margin_j": margin,
        "sensitive_battery_j": sensitive,
        "insensitive_battery_j": insensitive,
        "probes": probes,
        "probe_battery_j": SENTINEL_BATTERY_J,
        "probe_intervals": int(counter["intervals"]),
        "probe_physical_steps": int(counter["intervals"]) * int(spec["time"]["decision_s"]),
        "probe_note": "Sentinel-battery probes and the predicate checks measure route cost before the trial. They are not mission samples.",
        "sensitive_predicates": sensitive_predicates,
        "insensitive_predicates": insensitive_predicates,
        "sensitive_chain": chain,
        "separates": separates,
    }
    result = {
        "separates": separates,
        "reason": reason,
        "integration_factor": factor,
        "scenarios": scenarios,
        "budget": budget,
    }
    _BUILD_CACHE[key] = result
    return result


def build_scenarios(spec: dict) -> tuple[list[dict], dict, float]:
    assessed = assess_budgets(spec)
    if not assessed["separates"]:
        raise RuntimeError(assessed["reason"])
    _validate(spec, assessed["integration_factor"], assessed["budget"]["sensitive_battery_j"], assessed["budget"]["insensitive_battery_j"])
    return assessed["scenarios"], assessed["budget"], assessed["integration_factor"]


def _validate(spec: dict, factor: float, sensitive: list[float], insensitive: list[float], counter: dict | None = None) -> None:
    prior = prior_belief(spec)
    guard = float(spec["belief"]["guard_probability"])
    for uav_index, battery in enumerate(sensitive):
        world = make_world(spec, sensitive, "validate-sensitive")
        far = ("service", _far_id(spec, uav_index))
        if action_allowed(world.uavs[uav_index], far, prior.quantile(guard), spec, factor):
            raise RuntimeError(f"sensitive battery still allows the high-value trip for uav {uav_index} under the prior")
        _assert_chain_delivers(spec, factor, sensitive, uav_index, 1.0, counter)
        blocked = _run(
            spec,
            factor,
            sensitive,
            [prior.copy(), prior.copy()],
            [1.25, 1.25],
            _service_script(spec, uav_index, _far_id(spec, uav_index)),
            8,
            "validate-block",
            counter,
        )
        if _delivered(blocked, _far_id(spec, uav_index)):
            raise RuntimeError("prior guard delivered the high-value buoy on the sensitive battery")
    for uav_index, _battery in enumerate(insensitive):
        world = make_world(spec, insensitive, "validate-insensitive")
        far = ("service", _far_id(spec, uav_index))
        if not action_allowed(world.uavs[uav_index], far, prior.quantile(guard), spec, factor):
            raise RuntimeError(f"insensitive battery blocks the high-value trip for uav {uav_index} under the prior")


def _assert_chain_delivers(
    spec: dict,
    factor: float,
    battery: list[float],
    uav_index: int,
    theta: float,
    counter: dict | None = None,
) -> None:
    beliefs = [prior_belief(spec), prior_belief(spec)]
    buoy_id = _far_id(spec, uav_index)
    world = _run(
        spec,
        factor,
        battery,
        beliefs,
        [theta, theta],
        _cal_then_service(spec, uav_index, buoy_id),
        24,
        "validate-chain",
        counter,
    )
    if world.uavs[uav_index].failed or not _delivered(world, buoy_id):
        raise RuntimeError(
            f"calibration then high-value service failed for uav {uav_index}: "
            f"failed={world.uavs[uav_index].failed} delivered={_delivered(world, buoy_id)} "
            f"energy={world.uavs[uav_index].energy_j:.1f}"
        )


def predicate_rows(spec: dict, factor: float, battery: list[float]) -> list[dict]:
    """Whether the prior guard allows the far service. Does not change the battery."""
    prior = prior_belief(spec)
    guard = float(spec["belief"]["guard_probability"])
    rows = []
    for uav_index in range(2):
        world = make_world(spec, battery, f"predicate-{uav_index}")
        far = ("service", _far_id(spec, uav_index))
        near = ("service", _near_id(spec, uav_index))
        theta = prior.quantile(guard)
        rows.append(
            {
                "uav": uav_index,
                "guard_theta": theta,
                "far_allowed_under_prior": action_allowed(world.uavs[uav_index], far, theta, spec, factor),
                "near_allowed_under_prior": action_allowed(world.uavs[uav_index], near, theta, spec, factor),
            }
        )
    return rows


def ablate_beliefs(before: list[Belief], after: list[Belief], executed: list[Action]) -> tuple[list[Belief], list[int]]:
    """Drop only the belief update of UAVs that actually executed calibrate."""
    kept: list[Belief] = []
    removed: list[int] = []
    for index, action in enumerate(executed):
        if action[0] == "calibrate":
            kept.append(before[index].copy())
            removed.append(index)
        else:
            kept.append(after[index].copy())
    return kept, removed
