"""One Scheme C episode with at most one urgency tie-break."""
from __future__ import annotations

import csv
import time

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6d.shadow import capture_state
from wt.benchmark_v2.phase6e.generators import is_legal
from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import D_SAFE_M, DECISION_TIMES
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info, make_option
from wt.benchmark_v2.phase6iaa_formal_planner.planner import FormalPlanner
from wt.benchmark_v2.phase6ib_highlevel_dataset.schema import state_arrays, state_hash
from wt.benchmark_v2.phase6i_urgency_stage_a.ledger import AdmissionAgeLedger, snap_vector
from wt.benchmark_v2.phase6i_urgency_stage_a.runner import communication_at_service, frozen_mappo_action
from wt.benchmark_v2.phase6i_urgency_tiebreak_closedloop.labels import score_assignment
from wt.benchmark_v2.phase6i_urgency_tiebreak_closedloop.pool import sample_subset
from wt.benchmark_v2.phase6i_urgency_tiebreak_closedloop.protocol import label_horizon
from wt.benchmark_v2.phase6i_urgency_tiebreak_closedloop.rules import choose_executed
from wt.lowlevel_v2.contracts import minimum_pairwise_distance

from .d3_reuse import load_t0
from .diagnostic import PAIRING, real_horizon
from .protocol import EPS_BITS, HOLD, OUTPUT


def _join(values) -> str:
    return "|".join(f"{float(item):.8g}" for item in np.asarray(values, dtype=np.float64).reshape(-1))


def _text(assignment) -> str:
    return "|".join(str(int(item)) for item in np.asarray(assignment, dtype=np.int64).reshape(-1))


class ScoreTap:
    """Remember the chosen Scheme C scalar without scoring the pool twice."""

    def __init__(self, planner: FormalPlanner) -> None:
        if planner.scorer is None:
            raise RuntimeError("scheme C scorer is not loaded")
        self.chosen_score = None
        original = planner.scorer.infer

        def infer(graph, assignments):
            result = original(graph, assignments)
            self.chosen_score = float(result.scores[int(result.chosen_index)])
            return result

        planner.scorer.infer = infer


def run_episode(scenario_id: int, intervention_time: int | None, planner: FormalPlanner, tap: ScoreTap) -> dict:
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=int(scenario_id))
    if int(env.config.episode_horizon) != 48:
        raise RuntimeError("episode horizon is not 48")
    ledger = AdmissionAgeLedger(int(env.config.num_buoys))
    ledger.reset_from_queue(env.q_bits)
    ledger.assert_matches(env.q_bits, float(env.tol_bits))
    position_prev = np.asarray(env.uav_positions, dtype=np.float64).copy()
    predecessor = None
    last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    option = None
    pending = 0
    decisions = []
    steps = []
    served_sum = 0.0
    urgent_sum = 0.0
    dropped_sum = 0.0
    generated_sum = 0.0
    path = 0.0
    sinr_sum = 0.0
    sinr_count = 0
    interference_sum = 0.0
    projector_steps = 0
    projection_norm_sum = 0.0
    safety_steps = 0
    in_radius = 0
    first_radius = [-1] * int(env.config.num_uavs)
    positive_steps = 0
    max_gap = 0.0
    max_account = 0.0
    oldest_max = 0.0
    oldest_sum = 0.0
    oldest_count = 0
    cumulative = 0.0
    started = time.perf_counter()
    arm = "scheme_c" if intervention_time is None else f"intervention_{int(intervention_time)}"
    while int(env.t) < int(env.config.episode_horizon):
        if pending <= 0:
            if int(env.t) not in DECISION_TIMES:
                raise RuntimeError(f"clock left the decision grid at t={int(env.t)}")
            held_position = position_prev
            held_predecessor = predecessor
            decision = _choose_high_level(
                env,
                ledger,
                planner,
                tap,
                int(scenario_id),
                arm,
                None if intervention_time is None else int(intervention_time),
                held_position,
                held_predecessor,
            )
            decisions.append(decision)
            vector = np.asarray(decision["executed_vector"], dtype=np.int64)
            if not is_legal(vector, int(env.config.num_uavs), int(env.config.num_buoys)):
                raise RuntimeError("selected assignment is illegal")
            option = make_option(env, vector)
            pending = int(HOLD)
            option.remaining_hold = int(HOLD)
        if option is None:
            raise RuntimeError("episode has no assignment")
        t_before = int(env.t)
        step_start = np.asarray(env.uav_positions, dtype=np.float64).copy()
        queue_before = np.asarray(env.q_bits, dtype=np.float64).copy()
        snap_before = np.asarray(env.cumulative_snap_bits, dtype=np.float64).copy()
        view = ledger.view(t_before, queue_before, float(env.tol_bits))
        position_prev = step_start
        predecessor = int(t_before)
        action = frozen_mappo_action(env, planner.actor, option, last)
        info = env.step(action.executed, option.association)
        snap = snap_vector(env, snap_before, info)
        ledger.apply_transition(
            info["served_bits"],
            info["generated_arrival_bits"],
            info["admitted_arrival_bits"],
            info["dropped_bits_step"],
            snap,
            t_before,
            float(env.tol_bits),
        )
        gap = ledger.assert_matches(env.q_bits, float(env.tol_bits))
        account = np.asarray(env.conservation_error_bits(), dtype=np.float64)
        positions_after = np.asarray(env.uav_positions, dtype=np.float64).copy()
        comm = communication_at_service(env, queue_before)
        served = np.asarray(info["served_bits"], dtype=np.float64)
        urgency = np.asarray(view["urgency"], dtype=np.float64)
        association = np.asarray(option.association, dtype=np.int64)
        distances = _target_distance(env, association, positions_after)
        served_sum += float(np.sum(served))
        urgent_sum += float(np.sum(urgency * served))
        dropped_sum += float(np.sum(info["dropped_bits_step"]))
        generated_sum += float(np.sum(info["generated_arrival_bits"]))
        cumulative += float(info["payload_increment_bits"])
        path += float(np.sum(np.linalg.norm(positions_after - step_start, axis=1)))
        oldest = np.asarray(view["oldest_wait_age"], dtype=np.float64)
        valid = np.asarray(view["urgency_valid"], dtype=np.int64) == 1
        oldest_max = max(oldest_max, float(np.max(oldest)))
        if np.any(valid):
            oldest_sum += float(np.sum(oldest[valid]))
            oldest_count += int(np.sum(valid))
        if float(np.sum(served)) > float(env.tol_bits):
            positive_steps += 1
        radius = float(env.config.service_radius_m)
        for uav, distance in enumerate(distances):
            if distance <= radius:
                in_radius += 1
                if first_radius[uav] < 0:
                    first_radius[uav] = int(t_before)
        projection_norm_sum += float(action.projection_norm)
        if action.constraint_status != "ok" or float(action.projection_norm) > 1.0e-3:
            projector_steps += 1
        if minimum_pairwise_distance(positions_after) < float(D_SAFE_M):
            safety_steps += 1
        for uav, buoy in enumerate(association):
            if int(buoy) < 0:
                continue
            sinr_sum += float(info["sinr"][uav])
            interference_sum += float(comm["total_interference"][uav])
            sinr_count += 1
        max_gap = max(max_gap, float(np.max(np.abs(gap))))
        max_account = max(max_account, float(np.max(np.abs(account))))
        steps.append(
            {
                "scenario_id": int(scenario_id),
                "arm": arm,
                "intervention_time": "" if intervention_time is None else int(intervention_time),
                "time_step": int(t_before),
                "assignment": _text(association),
                "payload_increment_bits": float(info["payload_increment_bits"]),
                "cumulative_payload_bits": float(cumulative),
                "served_bits": _join(served),
                "generated_arrival_bits": _join(info["generated_arrival_bits"]),
                "admitted_arrival_bits": _join(info["admitted_arrival_bits"]),
                "dropped_bits": _join(info["dropped_bits_step"]),
                "q_bits_after": _join(env.q_bits),
                "urgency": _join(urgency),
                "oldest_wait_age": _join(oldest),
                "uav_x_m": _join(positions_after[:, 0]),
                "uav_y_m": _join(positions_after[:, 1]),
                "target_distance_m": _join(distances),
                "sinr": _join(info["sinr"]),
                "total_interference": _join(comm["total_interference"]),
                "projection_norm": float(action.projection_norm),
                "constraint_status": str(action.constraint_status),
                "ledger_gap_bits": float(np.max(np.abs(gap))),
                "accounting_error_bits": float(np.max(np.abs(account))),
                "in_radius_links": int(sum(distance <= radius for distance in distances)),
            }
        )
        pending -= 1
        option.remaining_hold = int(pending)
        last = np.asarray(action.executed, dtype=np.float64).copy()
    if int(env.t) != 48 or len(steps) != 48 or len(decisions) != 4:
        raise RuntimeError(f"{arm} scenario {scenario_id} did not finish 48 steps")
    payload_gap = abs(float(env.payload_bits) - float(served_sum))
    status = "ok"
    if payload_gap > float(env.tol_bits) or max_account > float(env.tol_bits):
        status = "accounting"
    if max_gap > float(env.tol_bits):
        status = "ledger"
    return {
        "scenario_id": int(scenario_id),
        "arm": arm,
        "intervention_time": "" if intervention_time is None else int(intervention_time),
        "status": status,
        "final_t": int(env.t),
        "payload_bits": float(env.payload_bits),
        "served_bits_sum": float(served_sum),
        "payload_accounting_gap_bits": float(payload_gap),
        "urgent_service_bits": float(urgent_sum),
        "generated_arrival_bits": float(generated_sum),
        "dropped_bits": float(dropped_sum),
        "final_queue_bits": float(np.sum(env.q_bits)),
        "oldest_wait_max": float(oldest_max),
        "oldest_wait_mean": float(oldest_sum / oldest_count) if oldest_count else 0.0,
        "path_length_m": float(path),
        "mean_sinr": float(sinr_sum / sinr_count) if sinr_count else 0.0,
        "mean_total_interference": float(interference_sum / sinr_count) if sinr_count else 0.0,
        "projector_trigger_steps": int(projector_steps),
        "mean_projection_norm": float(projection_norm_sum / len(steps)),
        "safety_violation_steps": int(safety_steps),
        "in_radius_links": int(in_radius),
        "first_service_radius_step": "|".join(str(int(item)) for item in first_radius),
        "positive_service_steps": int(positive_steps),
        "max_ledger_gap_bits": float(max_gap),
        "max_accounting_error_bits": float(max_account),
        "seconds": float(time.perf_counter() - started),
        "decisions": decisions,
        "steps": steps,
    }


def _target_distance(env, association: np.ndarray, positions: np.ndarray) -> list[float]:
    values = []
    for uav, buoy in enumerate(association):
        if int(buoy) < 0:
            values.append(float("nan"))
        else:
            values.append(float(np.linalg.norm(positions[uav] - env.buoy_positions[int(buoy)])))
    return values


def _scheme_c(env, planner: FormalPlanner, tap: ScoreTap, position_prev, predecessor) -> dict:
    tap.chosen_score = None
    selection = planner.select_assignment(env, position_prev, predecessor)
    if selection.status != "selected" or selection.assignment is None:
        raise RuntimeError(f"scheme C did not select an assignment: {selection.reason or selection.status}")
    vector = np.asarray(selection.assignment, dtype=np.int64)
    return {
        "assignment": _text(vector),
        "vector": vector.copy(),
        "score_index": int(selection.score_index),
        "score": None if tap.chosen_score is None else float(tap.chosen_score),
        "candidate_count": int(selection.candidate_count),
        "pool_hash": str(selection.pool_hash),
        "called": int(selection.scheme_c_called),
        "pool_seconds": float(selection.pool_seconds),
        "score_seconds": float(selection.score_seconds),
    }


def _flush_subset(rows: list[dict]) -> None:
    """Persist the candidate list before any urgent-service label is computed."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "selected_candidates.csv"
    fields = (
        "scenario_id",
        "time_step",
        "candidate_id",
        "assignment",
        "sampling_score",
        "sampling_source",
        "utility_rank",
        "selection",
        "stratum",
        "filled",
        "pool_hash",
        "pool_count",
        "subset_hash",
    )
    new_file = not path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
        handle.flush()


def _fresh_tiebreak(env, ledger, actor, scenario_id: int) -> dict:
    horizon = label_horizon(int(env.t), int(env.config.episode_horizon))
    subset = sample_subset(env, int(scenario_id), int(env.t))
    _flush_subset(subset["rows"])
    env_snap = capture_state(env)
    ledger_snap = ledger.capture()
    scored = []
    for row in subset["rows"]:
        label = score_assignment(
            env_snap,
            ledger_snap,
            actor,
            row["assignment_vector"],
            scenario_id=int(scenario_id),
            horizon=int(horizon),
        )
        scored.append({**row, **label})
    payload_choice, tie_choice, executed = choose_executed(scored, "payload_first_urgency_tiebreak")
    return {
        "source": "fresh_mappo_labels",
        "state_hash": subset["state_hash"],
        "pool_hash": subset["pool_hash"],
        "subset_hash": subset["subset_hash"],
        "subset_count": int(len(subset["rows"])),
        "pool_count": int(subset["pool_count"]),
        "sampling_source": subset["source"],
        "payload_tie_count": int(sum(abs(float(row["payload_bits"]) - max(float(item["payload_bits"]) for item in scored)) <= float(EPS_BITS) for row in scored)),
        "payload_first_assignment": payload_choice["assignment"],
        "payload_first_vector": np.asarray(payload_choice["assignment_vector"], dtype=np.int64).copy(),
        "tiebreak_assignment": tie_choice["assignment"],
        "tiebreak_vector": np.asarray(tie_choice["assignment_vector"], dtype=np.int64).copy(),
        "executed_vector": np.asarray(executed["assignment_vector"], dtype=np.int64).copy(),
        "label_horizon": int(horizon),
        "label_payload_gap_bits": float(tie_choice["payload_bits"]) - float(payload_choice["payload_bits"]),
        "label_urgent_gain_bits": float(tie_choice["urgent_service_bits"]) - float(payload_choice["urgent_service_bits"]),
        "max_label_ledger_gap_bits": float(max(float(row["ledger_gap_bits"]) for row in scored)),
        "max_label_accounting_error_bits": float(max(float(row["accounting_error_bits"]) for row in scored)),
        "env_snap": env_snap,
        "ledger_snap": ledger_snap,
    }


def _choose_high_level(env, ledger, planner, tap, scenario_id: int, arm: str, intervention_time: int | None, position_prev, predecessor) -> dict:
    live_hash = state_hash(int(scenario_id), state_arrays(env))
    scheme = _scheme_c(env, planner, tap, position_prev, predecessor)
    if state_hash(int(scenario_id), state_arrays(env)) != live_hash:
        raise RuntimeError("scheme C changed the live state hash")
    is_intervention = intervention_time is not None and int(env.t) == int(intervention_time)
    record = {
        "scenario_id": int(scenario_id),
        "arm": arm,
        "time_step": int(env.t),
        "intervention_time": "" if intervention_time is None else int(intervention_time),
        "is_intervention": int(is_intervention),
        "state_hash": live_hash,
        "pool_hash": scheme["pool_hash"],
        "candidate_count": int(scheme["candidate_count"]),
        "subset_hash": "",
        "subset_count": "",
        "sampling_source": "",
        "scheme_c_assignment": scheme["assignment"],
        "scheme_c_score_index": int(scheme["score_index"]),
        "scheme_c_score": scheme["score"],
        "scheme_c_called": int(scheme["called"]),
        "payload_first_assignment": "",
        "tiebreak_assignment": "",
        "executed_assignment": scheme["assignment"],
        "executed_vector": scheme["vector"],
        "assignment_changed_vs_scheme_c": 0,
        "assignment_changed_vs_payload_first": "",
        "label_horizon": "",
        "label_payload_gap_bits": "",
        "label_urgent_gain_bits": "",
        "label_gap_within_one_bit": "",
        "real_payload_first_bits": "",
        "real_tiebreak_payload_bits": "",
        "real_payload_gap_bits": "",
        "real_urgent_gain_bits": "",
        "real_generated_gap_bits": "",
        "arrival_pairing": "",
        "arrival_paired": "",
        "diagnostic_previous_action": "",
        "label_source": "",
        "max_label_ledger_gap_bits": "",
        "max_label_accounting_error_bits": "",
        "pool_seconds": float(scheme["pool_seconds"]),
        "score_seconds": float(scheme["score_seconds"]),
        "selection_rule": "scheme_c",
    }
    if not is_intervention:
        return record
    if int(env.t) == 0:
        cached = load_t0(scenario_id)
        if cached["state_hash"] != live_hash:
            choice = _fresh_tiebreak(env, ledger, planner.actor, scenario_id)
            choice["source"] = "fresh_mappo_labels_after_d3_hash_mismatch"
        else:
            choice = {
                "source": "d3_t0_reuse",
                "state_hash": cached["state_hash"],
                "pool_hash": cached["pool_hash"],
                "subset_hash": cached["subset_hash"],
                "subset_count": cached["subset_count"],
                "pool_count": cached["pool_count"],
                "sampling_source": cached["sampling_source"],
                "payload_first_assignment": cached["payload_first_assignment"],
                "payload_first_vector": cached["payload_first_vector"],
                "tiebreak_assignment": cached["tiebreak_assignment"],
                "tiebreak_vector": cached["tiebreak_vector"],
                "executed_vector": cached["tiebreak_vector"],
                "label_horizon": 15,
                "label_payload_gap_bits": cached["label_payload_gap_bits"],
                "label_urgent_gain_bits": cached["label_urgent_gain_bits"],
                "max_label_ledger_gap_bits": cached["max_label_ledger_gap_bits"],
                "max_label_accounting_error_bits": cached["max_label_accounting_error_bits"],
                "env_snap": capture_state(env),
                "ledger_snap": ledger.capture(),
            }
    else:
        choice = _fresh_tiebreak(env, ledger, planner.actor, scenario_id)
    if abs(float(choice["label_payload_gap_bits"])) > float(EPS_BITS):
        raise RuntimeError("tie-break label payload gap exceeds 1 bit")
    if float(choice["max_label_ledger_gap_bits"]) > float(EPS_BITS) or float(choice["max_label_accounting_error_bits"]) > float(EPS_BITS):
        raise RuntimeError("tie-break label ledger or payload account exceeds 1 bit")
    real_first = real_horizon(
        choice["env_snap"],
        choice["ledger_snap"],
        planner.actor,
        choice["payload_first_vector"],
        scenario_id=scenario_id,
        horizon=int(choice["label_horizon"]),
    )
    real_tie = real_horizon(
        choice["env_snap"],
        choice["ledger_snap"],
        planner.actor,
        choice["tiebreak_vector"],
        scenario_id=scenario_id,
        horizon=int(choice["label_horizon"]),
    )
    record.update(
        {
            "subset_hash": choice["subset_hash"],
            "subset_count": int(choice["subset_count"]),
            "sampling_source": choice["sampling_source"],
            "payload_first_assignment": choice["payload_first_assignment"],
            "tiebreak_assignment": choice["tiebreak_assignment"],
            "executed_assignment": choice["tiebreak_assignment"],
            "executed_vector": np.asarray(choice["executed_vector"], dtype=np.int64).copy(),
            "assignment_changed_vs_scheme_c": int(choice["tiebreak_assignment"] != scheme["assignment"]),
            "assignment_changed_vs_payload_first": int(choice["tiebreak_assignment"] != choice["payload_first_assignment"]),
            "label_horizon": int(choice["label_horizon"]),
            "label_payload_gap_bits": float(choice["label_payload_gap_bits"]),
            "label_urgent_gain_bits": float(choice["label_urgent_gain_bits"]),
            "label_gap_within_one_bit": 1,
            "real_payload_first_bits": float(real_first["payload_bits"]),
            "real_tiebreak_payload_bits": float(real_tie["payload_bits"]),
            "real_payload_gap_bits": float(real_tie["payload_bits"]) - float(real_first["payload_bits"]),
            "real_urgent_gain_bits": float(real_tie["urgent_service_bits"]) - float(real_first["urgent_service_bits"]),
            "real_generated_gap_bits": float(real_tie["generated_arrival_bits"]) - float(real_first["generated_arrival_bits"]),
            "arrival_pairing": PAIRING,
            "arrival_paired": int(abs(float(real_tie["generated_arrival_bits"]) - float(real_first["generated_arrival_bits"])) <= float(EPS_BITS)),
            "diagnostic_previous_action": "zeroed_to_match_label_scorer",
            "label_source": choice["source"],
            "max_label_ledger_gap_bits": float(choice["max_label_ledger_gap_bits"]),
            "max_label_accounting_error_bits": float(choice["max_label_accounting_error_bits"]),
            "selection_rule": "payload_first_urgency_tiebreak",
        }
    )
    return record
