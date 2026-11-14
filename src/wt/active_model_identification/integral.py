"""Choose the quadrature before any mission utility is known.

Moment checks use the registered beliefs and a fine quantile reference.
Noise-node checks compare Gauss-Hermite rules on the registered development
states. The selection rule is the tolerances in the config, not the trial reward.
"""

from __future__ import annotations

import math

from wt.active_model_identification.belief import (
    Belief,
    equal_width_quadrature,
    expectation,
    gaussian_quadrature,
    prior_belief,
)
from wt.active_model_identification.policies import score_actions
from wt.active_model_identification.world import make_world


def _belief_from_case(spec: dict, case: dict) -> Belief:
    lo, hi = (float(value) for value in spec["belief"]["support"])
    std = float(case["std"])
    return Belief(float(case["mu"]), 1.0 / (std * std), lo, hi)


def _moment_errors(belief: Belief, nodes: int, reference: int) -> dict:
    mean, variance = belief.moments()
    approx_mean = expectation(belief, nodes, lambda theta: theta)
    approx_second = expectation(belief, nodes, lambda theta: (theta - mean) ** 2)
    ref_mean = expectation(belief, reference, lambda theta: theta)
    return {
        "analytic_mean": mean,
        "analytic_variance": variance,
        "node_mean": approx_mean,
        "node_variance": approx_second,
        "reference_mean": ref_mean,
        "mean_abs_error": abs(approx_mean - mean),
        "variance_abs_error": abs(approx_second - variance),
        "reference_mean_abs_error": abs(ref_mean - mean),
    }


def _development_belief(spec: dict, state: dict) -> Belief:
    named = state["belief"]
    if named == "prior":
        return prior_belief(spec)
    lo, hi = (float(value) for value in spec["belief"]["support"])
    std = float(named["std"])
    return Belief(float(named["mu"]), 1.0 / (std * std), lo, hi)


def _scores(spec: dict, state: dict, theta_nodes: int, noise_nodes: int) -> dict[str, float]:
    trial = dict(spec)
    trial["belief"] = dict(spec["belief"])
    trial["belief"]["quadrature_nodes"] = theta_nodes
    trial["integral"] = dict(spec.get("integral", {}))
    trial["integral"]["noise_nodes"] = noise_nodes
    battery = [float(state["battery_j"]), float(state["battery_j"])]
    world = make_world(trial, battery, state["name"])
    if "x" in state:
        probe = world.uavs[0]
        probe.x = float(state["x"])
        probe.y = float(state["y"])
        probe.vx = float(state.get("vx", 0.0))
        probe.vy = float(state.get("vy", 0.0))
        probe.airborne = bool(state.get("airborne", False))
    if "cache_buoy" in state:
        moved = False
        for packet in world.packets:
            if packet.buoy == int(state["cache_buoy"]) and packet.holder == "buoy" and not moved:
                packet.holder = "uav:0"
                world.uavs[0].cache_ids.append(packet.id)
                moved = True
    belief = _development_belief(spec, state)
    beliefs = [belief.copy(), belief.copy()]
    factor = 1.0
    stats = {"planning_steps": 0, "planning_copies": 0}
    scored: dict[str, float] = {}
    for index in range(len(world.uavs)):
        table = score_actions(
            world,
            beliefs,
            trial,
            factor,
            index,
            include_calibrate=True,
            stats=stats,
        )
        for value, action in table:
            scored[f"u{index}:{action[0]}:{action[1]}"] = value
    scored["_planning_copies"] = float(stats["planning_copies"])
    scored["_planning_steps"] = float(stats["planning_steps"])
    return scored


def _max_gap(left: dict[str, float], right: dict[str, float]) -> float:
    keys = [key for key in left if not key.startswith("_")]
    if not keys:
        raise RuntimeError("development state produced no actions")
    return max(abs(left[key] - right[key]) for key in keys)


def select_integral(spec: dict) -> dict:
    rule = spec["integral"]
    reference = int(rule["theta_reference_nodes"])
    mean_tol = float(rule["theta_mean_abs_tol"])
    variance_tol = float(rule["theta_variance_abs_tol"])
    cases = []
    chosen_theta = None
    for nodes in rule["theta_candidates"]:
        rows = []
        passed = True
        for case in rule["moment_cases"]:
            belief = _belief_from_case(spec, case)
            error = _moment_errors(belief, int(nodes), reference)
            width = equal_width_quadrature(belief, 7)
            error["equal_width_7_mean"] = math.fsum(weight * theta for theta, weight in width)
            error["case"] = case["name"]
            rows.append(error)
            if error["mean_abs_error"] > mean_tol or error["variance_abs_error"] > variance_tol:
                passed = False
        cases.append({"theta_nodes": int(nodes), "passed": passed, "cases": rows})
    q_tol = float(rule["q_abs_tol"])
    noise_rows = []
    chosen_theta = None
    chosen_noise = None
    max_gap = None
    copies = 0
    steps = 0
    for moment in cases:
        if not moment["passed"]:
            continue
        theta_nodes = int(moment["theta_nodes"])
        for nodes in rule["noise_candidates"]:
            gaps = []
            for state in rule["development_states"]:
                print(f"integral theta {theta_nodes} noise {nodes} vs {rule['noise_reference_nodes']} {state['name']}", flush=True)
                coarse = _scores(spec, state, theta_nodes, int(nodes))
                fine = _scores(spec, state, theta_nodes, int(rule["noise_reference_nodes"]))
                gap = _max_gap(coarse, fine)
                gaps.append({"state": state["name"], "max_abs_q": gap})
                copies += int(coarse["_planning_copies"] + fine["_planning_copies"])
                steps += int(coarse["_planning_steps"] + fine["_planning_steps"])
            gap_here = max(item["max_abs_q"] for item in gaps)
            passed = gap_here <= q_tol
            noise_rows.append(
                {"theta_nodes": theta_nodes, "noise_nodes": int(nodes), "passed": passed, "max_abs_q": gap_here, "states": gaps}
            )
            if passed:
                chosen_theta = theta_nodes
                chosen_noise = int(nodes)
                max_gap = gap_here
                break
        if chosen_noise is not None:
            break
    if chosen_noise is None and noise_rows:
        max_gap = max(float(item["max_abs_q"]) for item in noise_rows)
    numerical_status = "PASS" if chosen_theta is not None and chosen_noise is not None else "NOT_MET"
    if numerical_status == "PASS":
        selection = "smallest registered candidate inside the pre-set tolerances; mission utility was not read"
    else:
        selection = "NOT_MET: no registered candidate met the tolerances; the last candidate was not kept"
    return {
        "numerical_status": numerical_status,
        "theta_nodes": chosen_theta,
        "noise_nodes": chosen_noise,
        "tie_abs": float(rule["tie_abs"]),
        "measured_q_gap": max_gap,
        "moment_checks": cases,
        "noise_checks": noise_rows,
        "development_planning_copies": copies,
        "development_planning_steps": steps,
        "selection": selection,
    }
