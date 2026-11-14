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
        if action_allowed(world.uavs[uav_index], far, prior.quantile(guard), spec, factor, world):
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
        if not action_allowed(world.uavs[uav_index], far, prior.quantile(guard), spec, factor, world):
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
                "far_allowed_under_prior": action_allowed(world.uavs[uav_index], far, theta, spec, factor, world),
                "near_allowed_under_prior": action_allowed(world.uavs[uav_index], near, theta, spec, factor, world),
            }
        )
    return rows


def _posterior_case(spec: dict, factor: float, battery: list[float], uav_index: int, epsilon_j: float) -> dict:
    from wt.active_model_identification.completion import admits, completion_quote

    world = make_world(spec, battery, f"posterior-{uav_index}")
    beliefs = [prior_belief(spec), prior_belief(spec)]
    advance(
        world,
        _actions(spec, uav_index, ("calibrate", None)),
        [1.0, 1.0],
        beliefs,
        spec,
        factor,
        None,
        planning=True,
        interval_index=0,
        noise_j=lambda index, _interval, _epsilon=epsilon_j, _focus=uav_index: _epsilon if index == _focus else 0.0,
    )
    guard = float(spec["belief"]["guard_probability"])
    theta = beliefs[uav_index].quantile(guard)
    quote = completion_quote(world, uav_index, _far_id(spec, uav_index), theta, spec)
    uav = world.uavs[uav_index]
    reserve = float(spec["energy"]["reserve_j"])
    return {
        "uav": uav_index,
        "epsilon_j": epsilon_j,
        "posterior_mean": beliefs[uav_index].mean(),
        "posterior_q95": theta,
        "posterior_std": math.sqrt(beliefs[uav_index].variance()),
        "calibration_z_j": uav.z_j,
        "x": uav.x,
        "y": uav.y,
        "airborne": uav.airborne,
        "energy_left_j": uav.energy_j,
        "quote_j": quote.energy_j,
        "quote_plus_reserve_j": quote.energy_j + reserve,
        "quote_id": quote.quote_id,
        "seconds": quote.seconds,
        "on_time": quote.on_time,
        "admitted": admits(uav, quote, spec),
        "slack_j": uav.energy_j - (quote.energy_j + reserve),
    }


def assess_m12(spec: dict) -> dict:
    """Recheck the registered battery with the shared completion quote.

    The sensitive candidate is not replaced. The insensitive figure is only the
    registered chain at theta 1.4 plus reserve and margin. Neither is edited
    after a mission reward, and this function does not read one.
    """
    from wt.active_model_identification.completion import admits, chain_quote, completion_quote, standard_quantile

    factor = measure_integration_factor()
    candidate = float(spec["energy"]["candidate_sensitive_battery_j"])
    reserve = float(spec["energy"]["reserve_j"])
    margin = float(spec["energy"]["budget_margin_j"])
    prior = prior_belief(spec)
    guard_p = float(spec["belief"]["guard_probability"])
    guard_theta = prior.quantile(guard_p)
    sigma = float(spec["belief"]["sigma_j"])
    adverse = standard_quantile(guard_p) * sigma
    battery = [candidate, candidate]
    uavs = []
    insensitive = []
    for uav_index in range(2):
        world = make_world(spec, battery, f"m12-budget-{uav_index}")
        far_id = _far_id(spec, uav_index)
        near_id = _near_id(spec, uav_index)
        far_prior = completion_quote(world, uav_index, far_id, guard_theta, spec)
        far_true = completion_quote(world, uav_index, far_id, 1.0, spec)
        near_prior = completion_quote(world, uav_index, near_id, guard_theta, spec)
        chain = chain_quote(world, uav_index, far_id, 1.0, spec)
        chain_high = chain_quote(world, uav_index, far_id, 1.4, spec)
        zero = _posterior_case(spec, factor, battery, uav_index, 0.0)
        adverse_row = _posterior_case(spec, factor, battery, uav_index, adverse)
        ideal_low = chain.energy_j + reserve
        ideal_high = far_prior.energy_j + reserve
        insensitive.append(chain_high.energy_j + reserve + margin)
        uavs.append(
            {
                "uav": uav_index,
                "physical_at_true_theta_1": admits(world.uavs[uav_index], far_true, spec),
                "physical_quote_j": far_true.energy_j,
                "physical_quote_plus_reserve_j": far_true.energy_j + reserve,
                "admitted_under_prior": admits(world.uavs[uav_index], far_prior, spec),
                "prior_quote_j": far_prior.energy_j,
                "prior_quote_plus_reserve_j": ideal_high,
                "near_admitted_under_prior": admits(world.uavs[uav_index], near_prior, spec),
                "near_quote_plus_reserve_j": near_prior.energy_j + reserve,
                "chain_theta_1_j": chain.energy_j,
                "chain_theta_1_plus_reserve_j": ideal_low,
                "ideal_script_window": ideal_low <= candidate + 1e-6 < ideal_high,
                "zero_noise_posterior": zero,
                "adverse_posterior": adverse_row,
                "information_changes_admission": (not admits(world.uavs[uav_index], far_prior, spec)) and zero["admitted"],
            }
        )
    ideal = all(row["ideal_script_window"] for row in uavs)
    posterior_ok = all(row["zero_noise_posterior"]["admitted"] for row in uavs)
    near_ok = all(row["near_admitted_under_prior"] for row in uavs)
    window = ideal and posterior_ok and near_ok
    if window:
        limiting = ""
        scenario_status = "FROZEN"
    elif not ideal:
        limiting = "理想脚本窗口本身不成立：校准后 theta=1 的完整成本加备用，或先验保守完整成本加备用，没有把原电量夹在中间。"
        scenario_status = "NOT_A_WINDOW"
    elif not posterior_ok:
        limiting = (
            "原电量落在理想脚本窗口内，但一次校准后的后验 0.95 分位完整成本加备用超过剩余能量。"
            "限制项是后验分位而不是 theta=1 的理想真值；登记裕量盖不住这个抬升。不利噪声更紧。"
            "普通近程服务在先验下仍然放行。"
        )
        scenario_status = "NOT_A_WINDOW"
    else:
        limiting = "完整成本准入把普通近程服务也拦住了。这不是信息窗口，本轮不把电量调低去制造对照。"
        scenario_status = "NOT_A_WINDOW"
    scenarios = []
    for base in spec["scenarios"]:
        for condition, values in (("sensitive", battery), ("insensitive", insensitive)):
            scenarios.append(
                {
                    "id": f"{base['base']}_{condition}",
                    "base": base["base"],
                    "condition": condition,
                    "theta": [float(value) for value in base["theta"]],
                    "seed": int(base["seed"]),
                    "battery_j": [float(value) for value in values],
                    "wind_xy_m_s": [float(value) for value in spec["motion"]["wind_xy_m_s"]],
                    "scheduled": window,
                }
            )
    return {
        "scenario_status": scenario_status,
        "window_valid": window,
        "ideal_script_window": ideal,
        "posterior_zero_admits": posterior_ok,
        "near_admitted_under_prior": near_ok,
        "battery_replaced": False,
        "limiting_condition": limiting,
        "candidate_battery_j": candidate,
        "sensitive_battery_j": battery,
        "insensitive_battery_j": insensitive,
        "reserve_j": reserve,
        "margin_j": margin,
        "guard_theta": guard_theta,
        "adverse_epsilon_j": adverse,
        "integration_factor": factor,
        "channel_assumption": "The other UAV waits and still occupies LoRa with its command and telemetry. Budget, admission and the terminal quote use that same assumption.",
        "uavs": uavs,
        "scenarios": scenarios,
        "reserve_note": "reserve_j is added once, outside the completion quote. It is not inside the replayed joules.",
    }


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


def assess_m13(spec: dict) -> dict:
    """Freeze tight, the posterior-cost window, and the registered loose battery.

    The window uses theta_ref and the adverse design noise on an untruncated
    copy. It is not recomputed from a mission's true theta. posterior_zero_admits
    stays an explanation and does not decide whether tight or loose exist.
    """
    from wt.active_model_identification.completion import admits, completion_quote, standard_quantile

    factor = measure_integration_factor()
    tight_value = float(spec["energy"]["tight_battery_j"])
    loose = [float(value) for value in spec["energy"]["loose_battery_j"]]
    reserve = float(spec["energy"]["reserve_j"])
    theta_ref = float(spec["energy"]["theta_ref"])
    design_p = float(spec["energy"]["epsilon_design_probability"])
    sigma = float(spec["belief"]["sigma_j"])
    epsilon_design = standard_quantile(design_p) * sigma
    prior = prior_belief(spec)
    guard_p = float(spec["belief"]["guard_probability"])
    guard_theta = prior.quantile(guard_p)
    sentinel = [1.0e8, 1.0e8]
    tight = [tight_value, tight_value]
    bounds = []
    probes = 0
    for uav_index in range(2):
        world = make_world(spec, sentinel, f"m13-window-{uav_index}")
        beliefs = [prior.copy(), prior.copy()]
        far = _far_id(spec, uav_index)
        start_quote = completion_quote(world, uav_index, far, guard_theta, spec)
        upper = start_quote.energy_j + reserve
        advance(
            world,
            _actions(spec, uav_index, ("calibrate", None)),
            [theta_ref, theta_ref],
            beliefs,
            spec,
            factor,
            None,
            planning=True,
            interval_index=0,
            noise_j=lambda index, _interval, _epsilon=epsilon_design, _focus=uav_index: _epsilon if index == _focus else 0.0,
        )
        probes += 1
        posterior = beliefs[uav_index]
        q95 = posterior.quantile(guard_p)
        post_quote = completion_quote(world, uav_index, far, q95, spec)
        lower = world.uavs[uav_index].consumed_j + post_quote.energy_j + reserve
        bounds.append(
            {
                "uav": uav_index,
                "theta_ref": theta_ref,
                "epsilon_design_j": epsilon_design,
                "calibration_cost_j": world.uavs[uav_index].consumed_j,
                "calibration_z_j": world.uavs[uav_index].z_j,
                "post_x": world.uavs[uav_index].x,
                "post_y": world.uavs[uav_index].y,
                "post_airborne": world.uavs[uav_index].airborne,
                "posterior_mean": posterior.mean(),
                "posterior_q95": q95,
                "posterior_std": math.sqrt(posterior.variance()),
                "post_quote_j": post_quote.energy_j,
                "post_quote_id": post_quote.quote_id,
                "start_quote_j": start_quote.energy_j,
                "start_quote_id": start_quote.quote_id,
                "L_j": lower,
                "U_j": upper,
                "interval": lower < upper,
            }
        )
    window_valid = all(row["interval"] for row in bounds)
    window_battery = [0.5 * (row["L_j"] + row["U_j"]) for row in bounds] if window_valid else None
    for row, battery in zip(bounds, window_battery or [None, None]):
        row["window_battery_j"] = battery
    zero_rows = []
    for uav_index in range(2):
        zero_rows.append(_posterior_case(spec, factor, tight, uav_index, 0.0))
        probes += 1
    posterior_zero = all(bool(row["admitted"]) for row in zero_rows)
    near_rows = []
    for uav_index in range(2):
        probe = make_world(spec, tight, f"m13-near-{uav_index}")
        quote = completion_quote(probe, uav_index, _near_id(spec, uav_index), guard_theta, spec)
        near_rows.append({"uav": uav_index, "admitted": admits(probe.uavs[uav_index], quote, spec), "quote_plus_reserve_j": quote.energy_j + reserve})
    tiers: list[tuple[str, list[float]]] = [("tight", tight), ("loose", loose)]
    if window_valid and window_battery is not None:
        tiers.insert(1, ("window", window_battery))
    scenarios = []
    for base in spec["scenarios"]:
        for condition, battery in tiers:
            scenarios.append(
                {
                    "id": f"{base['base']}_{condition}",
                    "base": base["base"],
                    "condition": condition,
                    "theta": [float(value) for value in base["theta"]],
                    "seed": int(base["seed"]),
                    "battery_j": [float(value) for value in battery],
                    "wind_xy_m_s": [float(value) for value in spec["motion"]["wind_xy_m_s"]],
                    "scheduled": True,
                }
            )
    if window_valid:
        limiting = ""
    else:
        limiting = "至少一架 UAV 的 L 不小于 U，window 档无效。上下界已保存。tight 和 loose 仍运行，不改设计噪声或电池。"
    return {
        "scenario_status": "FROZEN",
        "window_valid": window_valid,
        "window_status": "VALID" if window_valid else "INVALID",
        "posterior_zero_admits": posterior_zero,
        "near_admitted_under_prior": all(row["admitted"] for row in near_rows),
        "limiting_condition": limiting,
        "tight_battery_j": tight,
        "window_battery_j": window_battery,
        "loose_battery_j": loose,
        "reserve_j": reserve,
        "theta_ref": theta_ref,
        "epsilon_design_probability": design_p,
        "epsilon_design_j": epsilon_design,
        "guard_theta": guard_theta,
        "integration_factor": factor,
        "design_probe_intervals": probes,
        "design_probe_steps": probes * int(spec["time"]["decision_s"]),
        "formula": "L = calibration cost(theta_ref) + far completion at the design posterior q95 + reserve; U = far completion at the prior q95 + reserve; B = (L+U)/2 when L < U.",
        "channel_assumption": "The other UAV waits and still occupies LoRa with its command and telemetry. The design probe and the completion quote use that same assumption. Reserve is added once, outside the quote.",
        "uavs": bounds,
        "posterior_zero": zero_rows,
        "near": near_rows,
        "scenarios": scenarios,
        "scheduled_conditions": [name for name, _battery in tiers],
    }
