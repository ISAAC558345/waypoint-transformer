"""Closed-loop fly-toward rollouts and same-state assignment diagnostics."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.l3b.policies import fly_toward

from .channel import geometry, hypothetical_rate
from .controllers import (
    evaluate_assignment,
    legalize,
    max_queue_scores,
    nearest_scores,
    one_step_optima,
    queue_rate_utility,
    random_legal,
)
from .protocol import HOLD_STEPS, assert_development

CONTROLLERS = (
    "random_legal",
    "nearest",
    "max_queue",
    "max_rate_independent",
    "queue_rate",
    "joint_rate",
    "joint_true_bit",
)


def _jain(values: np.ndarray) -> float:
    total = float(np.sum(values))
    if total <= 0.0:
        return 0.0
    return float((total ** 2) / (values.size * np.sum(values ** 2)))


def choose_assignment(controller: str, env: L3BBitQueueEnvV2, rng: np.random.Generator, geo: dict) -> tuple[np.ndarray, dict]:
    if controller == "random_legal":
        return random_legal(env, rng), {"duplicate_targets": 0, "optima": None, "marginal_rate": None}
    if controller == "nearest":
        legal, _raw, duplicates = legalize(nearest_scores(env))
        return legal, {"duplicate_targets": duplicates, "optima": None, "marginal_rate": None}
    if controller == "max_queue":
        legal, _raw, duplicates = legalize(max_queue_scores(env))
        return legal, {"duplicate_targets": duplicates, "optima": None, "marginal_rate": None}
    if controller == "max_rate_independent":
        marginal = hypothetical_rate(env, geo)
        legal, _raw, duplicates = legalize(marginal)
        return legal, {"duplicate_targets": duplicates, "optima": None, "marginal_rate": marginal}
    if controller == "queue_rate":
        marginal = hypothetical_rate(env, geo)
        legal, _raw, duplicates = legalize(queue_rate_utility(env.q_bits, marginal))
        return legal, {"duplicate_targets": duplicates, "optima": None, "marginal_rate": marginal}
    optima = one_step_optima(env, geo)
    if controller == "joint_rate":
        return np.asarray(optima["capacity_assignment"], dtype=np.int64), {
            "duplicate_targets": 0,
            "optima": optima,
            "marginal_rate": None,
        }
    if controller == "joint_true_bit":
        return np.asarray(optima["true_bit_assignment"], dtype=np.int64), {
            "duplicate_targets": 0,
            "optima": optima,
            "marginal_rate": None,
        }
    raise RuntimeError(f"unknown controller {controller}")


def _pressure(q_bits: np.ndarray) -> np.ndarray:
    return q_bits / (float(np.max(q_bits)) + 1.0e-9)


def assignment_panel(env: L3BBitQueueEnvV2, scenario: int) -> list[dict]:
    """One-step scores on a frozen reset state. Nothing is stepped."""
    assert_development(int(scenario))
    geo = geometry(env)
    optima = one_step_optima(env, geo)
    marginal = hypothetical_rate(env, geo)
    rows = []
    specs = {
        "random_legal": random_legal(env, np.random.default_rng(17_000_003 + int(scenario))),
        "nearest": legalize(nearest_scores(env))[0],
        "max_queue": legalize(max_queue_scores(env))[0],
        "max_rate_independent": legalize(marginal)[0],
        "queue_rate": legalize(queue_rate_utility(env.q_bits, marginal))[0],
        "joint_rate": np.asarray(optima["capacity_assignment"], dtype=np.int64),
        "joint_true_bit": np.asarray(optima["true_bit_assignment"], dtype=np.int64),
    }
    for name, assignment in specs.items():
        scored = evaluate_assignment(env, assignment, geo)
        rows.append(
            {
                "scenario_id": int(scenario),
                "time_step": int(env.t),
                "controller": name,
                "one_step_served_bits": float(scored["served_bits"]),
                "one_step_capacity_bits": float(scored["capacity_bits"]),
                "one_step_mean_sinr": float(scored["mean_sinr"]),
                "panel": "reset_state",
            }
        )
    return rows


def _ablation_rows(env: L3BBitQueueEnvV2, scenario: int, geo: dict, reference: np.ndarray) -> list[dict]:
    bits = env.q_bits.copy()
    true_score = float(evaluate_assignment(env, reference, geo)["served_bits"])
    rows = []

    def pack(name: str, assignment: np.ndarray, served_under_true_channel: float) -> dict:
        return {
            "scenario_id": int(scenario),
            "time_step": int(env.t),
            "feature": name,
            "assignment_disagreement": int(not np.array_equal(reference, assignment)),
            "true_served_bits": true_score,
            "ablated_assignment_served_bits": float(served_under_true_channel),
            "served_bits_drop": float(true_score - served_under_true_channel),
        }

    fixed = np.full_like(geo["distance"], float(env.config.service_radius_m))
    fixed_gain = np.maximum(fixed, float(env.config.min_link_m)) ** (-float(env.config.path_loss_exp))
    distance_opt = one_step_optima(env, geo, gain=fixed_gain)
    distance_assignment = np.asarray(distance_opt["true_bit_assignment"], dtype=np.int64)
    rows.append(pack("distance", distance_assignment, float(evaluate_assignment(env, distance_assignment, geo)["served_bits"])))

    flat_queue = bits.copy()
    positive = flat_queue > 1.0e-9
    if np.any(positive):
        flat_queue[positive] = float(np.max(flat_queue))
    queue_opt = one_step_optima(env, geo, q_bits=flat_queue)
    queue_assignment = np.asarray(queue_opt["true_bit_assignment"], dtype=np.int64)
    rows.append(pack("queue", queue_assignment, float(evaluate_assignment(env, queue_assignment, geo)["served_bits"])))

    marginal = hypothetical_rate(env, geo)
    rate_assignment = legalize(marginal)[0]
    rows.append(pack("rate_if_assigned", rate_assignment, float(evaluate_assignment(env, rate_assignment, geo)["served_bits"])))

    interference_opt = one_step_optima(env, geo, kappa_same=0.0, kappa_cross=0.0)
    interference_assignment = np.asarray(interference_opt["true_bit_assignment"], dtype=np.int64)
    interference_score = evaluate_assignment(env, interference_assignment, geo)
    true_interference = evaluate_assignment(env, reference, geo)
    row = pack("uav_uav_interference", interference_assignment, float(interference_score["served_bits"]))
    row["interference_mean"] = float(true_interference["mean_interference"])
    row["sinr_difference"] = float(true_interference["mean_sinr"] - interference_score["mean_sinr"])
    rows.append(row)

    rows.append(
        {
            "scenario_id": int(scenario),
            "time_step": int(env.t),
            "feature": "current_association",
            "assignment_disagreement": 0,
            "true_served_bits": true_score,
            "ablated_assignment_served_bits": true_score,
            "served_bits_drop": 0.0,
            "objective_reads_previous_assignment": False,
        }
    )

    saved = env.uav_positions.copy()
    env.uav_positions[:] = np.mean(saved, axis=0)
    moved = geometry(env)
    position_opt = one_step_optima(env, moved)
    env.uav_positions[:] = saved
    position_assignment = np.asarray(position_opt["true_bit_assignment"], dtype=np.int64)
    rows.append(pack("uav_positions", position_assignment, float(evaluate_assignment(env, position_assignment, geo)["served_bits"])))
    return rows


def _coverage_rows(env: L3BBitQueueEnvV2, scenario: int, geo: dict, optimal: np.ndarray) -> list[dict]:
    marginal = hypothetical_rate(env, geo)
    individual = np.minimum(env.q_bits[None, :], marginal * float(env.config.delta_t))
    utility = queue_rate_utility(env.q_bits, marginal)
    rankings = {
        "instantaneous_rate": np.argsort(-marginal, axis=1, kind="mergesort"),
        "queue_rate": np.argsort(-utility, axis=1, kind="mergesort"),
        "individual_served_bits": np.argsort(-individual, axis=1, kind="mergesort"),
    }
    rows = []
    for name, order in rankings.items():
        edges = 0
        covered = {k: 0 for k in (1, 2, 3, 4, 5)}
        for uav, buoy in enumerate(np.asarray(optimal, dtype=np.int64)):
            if int(buoy) < 0:
                continue
            edges += 1
            position = int(np.where(order[uav] == int(buoy))[0][0])
            for k in covered:
                if position < k:
                    covered[k] += 1
        for k, hits in covered.items():
            rows.append(
                {
                    "scenario_id": int(scenario),
                    "time_step": int(env.t),
                    "ranking": name,
                    "k": int(k),
                    "edges": int(edges),
                    "covered_edges": int(hits),
                }
            )
    return rows


def run_episode(scenario: int, controller: str, *, collect_diagnostics: bool = False) -> dict:
    assert_development(int(scenario))
    env = L3BBitQueueEnvV2()
    env.reset(seed=int(scenario))
    rng = np.random.default_rng(17_000_003 + int(scenario))
    previous = np.full(env.config.num_uavs, -2, dtype=np.int64)
    pending = 0
    assignment = np.full(env.config.num_uavs, -1, dtype=np.int64)
    sinr_sum = 0.0
    rate_sum = 0.0
    path = 0.0
    switches = 0
    clips = 0
    service_events = 0
    pressure_sum = 0.0
    pressure_count = 0
    holds: list[dict] = []
    decisions: list[dict] = []
    ablations: list[dict] = []
    coverage: list[dict] = []
    hold_payload_start = float(env.payload_bits)
    hold_queue_start = float(np.sum(env.q_bits))
    hold_sinr = 0.0
    hold_rate = 0.0
    hold_steps = 0
    hold_zero = 0
    hold_switch = 0
    first_rate = None
    last_rate = 0.0
    first_distance = None
    while int(env.t) < int(env.config.episode_horizon):
        if pending <= 0:
            if hold_steps > 0:
                holds.append(
                    {
                        "scenario_id": int(scenario),
                        "controller": controller,
                        "time_step": int(env.t) - hold_steps,
                        "payload_bits": float(env.payload_bits - hold_payload_start),
                        "queue_change_bits": float(hold_queue_start - np.sum(env.q_bits)),
                        "mean_sinr": float(hold_sinr / hold_steps),
                        "mean_rate": float(hold_rate / hold_steps),
                        "zero_service_steps": int(hold_zero),
                        "steps": int(hold_steps),
                        "switch_count": int(hold_switch),
                        "first_step_rate": float(0.0 if first_rate is None else first_rate),
                        "last_step_rate": float(last_rate),
                        "first_distance": float(0.0 if first_distance is None else first_distance),
                    }
                )
            geo = geometry(env)
            assignment, info = choose_assignment(controller, env, rng, geo)
            if int(previous[0]) != -2:
                hold_switch = int(np.sum(previous != assignment))
                switches += hold_switch
            else:
                hold_switch = 0
            scored = evaluate_assignment(env, assignment, geo)
            marginal = info["marginal_rate"]
            if marginal is None:
                marginal = hypothetical_rate(env, geo)
            independent, _raw, duplicates = legalize(marginal)
            independent_score = evaluate_assignment(env, independent, geo)
            decisions.append(
                {
                    "scenario_id": int(scenario),
                    "controller": controller,
                    "time_step": int(env.t),
                    "duplicate_targets": int(info["duplicate_targets"]),
                    "served_bits": float(scored["served_bits"]),
                    "capacity_bits": float(scored["capacity_bits"]),
                    "mean_sinr": float(scored["mean_sinr"]),
                    "mean_interference": float(scored["mean_interference"]),
                    "independent_served_bits": float(independent_score["served_bits"]),
                    "independent_mean_sinr": float(independent_score["mean_sinr"]),
                    "independent_capacity_bits": float(independent_score["capacity_bits"]),
                    "independent_interference": float(independent_score["mean_interference"]),
                    "disagreement_vs_independent_max_rate": int(not np.array_equal(assignment, independent)),
                }
            )
            if collect_diagnostics:
                ablations.extend(_ablation_rows(env, int(scenario), geo, assignment))
                coverage.extend(_coverage_rows(env, int(scenario), geo, assignment))
            chosen = _pressure(env.q_bits)
            for buoy in assignment:
                if int(buoy) >= 0:
                    pressure_sum += float(chosen[int(buoy)])
                    pressure_count += 1
            pending = int(HOLD_STEPS)
            hold_payload_start = float(env.payload_bits)
            hold_queue_start = float(np.sum(env.q_bits))
            hold_sinr = 0.0
            hold_rate = 0.0
            hold_steps = 0
            hold_zero = 0
            first_rate = None
            distances = []
            for uav, buoy in enumerate(assignment):
                if int(buoy) < 0:
                    continue
                distances.append(float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[int(buoy)])))
            first_distance = float(np.mean(distances)) if distances else 0.0
            previous = np.asarray(assignment, dtype=np.int64).copy()
        q_before = env.q_bits.copy()
        positions = env.uav_positions.copy()
        stepped = env.step(fly_toward(env, assignment), assignment)
        path += float(np.sum(np.linalg.norm(env.uav_positions - positions, axis=1)))
        sinr_sum += float(np.mean(stepped["sinr"]))
        rate_sum += float(np.mean(stepped["rate_bps"]))
        hold_sinr += float(np.mean(stepped["sinr"]))
        hold_rate += float(np.mean(stepped["rate_bps"]))
        if first_rate is None:
            first_rate = float(np.mean(stepped["rate_bps"]))
        last_rate = float(np.mean(stepped["rate_bps"]))
        if float(stepped["payload_increment_bits"]) <= env.tol_bits:
            hold_zero += 1
        for uav, buoy in enumerate(assignment):
            if int(buoy) < 0 or float(stepped["rate_bps"][uav]) <= 0.0:
                continue
            service_events += 1
            if float(stepped["rate_bps"][uav]) * float(env.config.delta_t) > float(q_before[int(buoy)]) + float(env.tol_bits):
                clips += 1
        pending -= 1
        hold_steps += 1
    if hold_steps > 0:
        holds.append(
            {
                "scenario_id": int(scenario),
                "controller": controller,
                "time_step": int(env.config.episode_horizon) - hold_steps,
                "payload_bits": float(env.payload_bits - hold_payload_start),
                "queue_change_bits": float(hold_queue_start - np.sum(env.q_bits)),
                "mean_sinr": float(hold_sinr / max(hold_steps, 1)),
                "mean_rate": float(hold_rate / max(hold_steps, 1)),
                "zero_service_steps": int(hold_zero),
                "steps": int(hold_steps),
                "switch_count": int(hold_switch),
                "first_step_rate": float(0.0 if first_rate is None else first_rate),
                "last_step_rate": float(last_rate),
                "first_distance": float(0.0 if first_distance is None else first_distance),
            }
        )
    error = np.abs(env.conservation_error_bits())
    horizon = max(int(env.config.episode_horizon), 1)
    return {
        "summary": {
            "scenario_id": int(scenario),
            "controller": controller,
            "cumulative_payload_bits": float(env.payload_bits),
            "residual_queue_bits": float(np.sum(env.q_bits)),
            "dropped_bits": float(np.sum(env.cumulative_dropped_bits)),
            "generated_arrival_bits": float(np.sum(env.cumulative_generated_arrival_bits)),
            "initial_bits": float(np.sum(env.initial_q_bits)),
            "mean_sinr": float(sinr_sum / horizon),
            "mean_rate": float(rate_sum / horizon),
            "jain_fairness": _jain(env.cumulative_served_bits),
            "switch_count": int(switches),
            "path_length_m": float(path),
            "completed_block_equivalent": float(np.sum(env.completed_block_equivalent)),
            "conservation_abs_error_bits": float(np.max(error)),
            "conservation_rel_error": float(np.max(error) / env.block_bits),
            "conservation_ok": bool(float(np.max(error)) <= env.tol_bits),
            "queue_clip_events": int(clips),
            "service_events": int(service_events),
            "mean_chosen_queue_pressure": float(pressure_sum / max(pressure_count, 1)),
            "movement": "fly_toward",
        },
        "holds": holds,
        "decisions": decisions,
        "ablations": ablations,
        "coverage": coverage,
    }


__all__ = ["CONTROLLERS", "assignment_panel", "run_episode"]
