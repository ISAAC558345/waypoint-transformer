"""One closed-loop arm. Each decision rebuilds its own 128-candidate subset."""
from __future__ import annotations

import time

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6d.shadow import capture_state
from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import D_SAFE_M, DECISION_TIMES
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info, make_option
from wt.benchmark_v2.phase6i_urgency_stage_a.ledger import AdmissionAgeLedger, snap_vector
from wt.benchmark_v2.phase6i_urgency_stage_a.runner import communication_at_service, frozen_mappo_action
from wt.lowlevel_v2.contracts import minimum_pairwise_distance

from .labels import score_assignment
from .pool import sample_subset
from .protocol import EPS_BITS, label_horizon
from .rules import choose_executed


def _join(values) -> str:
    return "|".join(str(float(item)) for item in np.asarray(values, dtype=np.float64).reshape(-1))


class BudgetStop(RuntimeError):
    """The first state is too slow to finish the registered pilot."""

    def __init__(self, first_seconds: float, estimate_seconds: float):
        super().__init__(f"first state {first_seconds:.3f}s, estimate {estimate_seconds:.3f}s")
        self.first_seconds = float(first_seconds)
        self.estimate_seconds = float(estimate_seconds)


class LabelClock:
    """Time the first 128-candidate state and stop before the estimate exceeds 30 minutes."""

    def __init__(self, planned_states: int, max_state_seconds: float, max_estimated_seconds: float):
        self.planned_states = int(planned_states)
        self.max_state_seconds = float(max_state_seconds)
        self.max_estimated_seconds = float(max_estimated_seconds)
        self.first_seconds = None
        self.estimate_seconds = None
        self.calls = 0
        self.label_seconds = 0.0

    def observe(self, seconds: float) -> None:
        self.calls += 1
        self.label_seconds += float(seconds)
        if self.first_seconds is None:
            self.first_seconds = float(seconds)
            self.estimate_seconds = float(seconds) * float(self.planned_states)
            if self.first_seconds > self.max_state_seconds or self.estimate_seconds > self.max_estimated_seconds:
                raise BudgetStop(self.first_seconds, self.estimate_seconds)


def run_arm(scenario_id: int, arm: str, actor, clock: LabelClock, cache: dict, sink) -> dict:
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=int(scenario_id))
    if int(env.config.episode_horizon) != 48:
        raise RuntimeError("episode horizon is not 48")
    ledger = AdmissionAgeLedger(int(env.config.num_buoys))
    ledger.reset_from_queue(env.q_bits)
    ledger.assert_matches(env.q_bits, float(env.tol_bits))
    last = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    option = None
    pending = 0
    decisions = []
    steps = []
    served_sum = 0.0
    urgent_sum = 0.0
    dropped_sum = 0.0
    path = 0.0
    sinr_sum = 0.0
    sinr_count = 0
    interference_sum = 0.0
    projector_steps = 0
    projection_norm_sum = 0.0
    safety_steps = 0
    max_gap = 0.0
    max_account = 0.0
    max_sinr_gap = 0.0
    oldest_max = 0.0
    oldest_sum = 0.0
    oldest_count = 0
    mean_wait_max = 0.0
    mean_wait_sum = 0.0
    mean_wait_count = 0
    first_service = -1
    positive_steps = 0
    started = time.perf_counter()
    while int(env.t) < int(env.config.episode_horizon):
        if pending <= 0:
            if int(env.t) not in DECISION_TIMES:
                raise RuntimeError(f"live clock left the decision grid at t={int(env.t)}")
            decision = _decide_at(env, ledger, actor, int(scenario_id), arm, clock, cache, sink)
            decisions.append(decision)
            option = make_option(env, decision["executed_vector"])
            pending = int(label_horizon(int(env.t), int(env.config.episode_horizon)))
            option.remaining_hold = int(pending)
        t_before = int(env.t)
        step_start = np.asarray(env.uav_positions, dtype=np.float64).copy()
        queue_before = np.asarray(env.q_bits, dtype=np.float64).copy()
        snap_before = np.asarray(env.cumulative_snap_bits, dtype=np.float64).copy()
        view = ledger.view(t_before, queue_before, float(env.tol_bits))
        action = frozen_mappo_action(env, actor, option, last)
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
        served_now = np.asarray(info["served_bits"], dtype=np.float64)
        urgency = np.asarray(view["urgency"], dtype=np.float64)
        served_sum += float(np.sum(served_now))
        urgent_sum += float(np.sum(urgency * served_now))
        dropped_sum += float(np.sum(info["dropped_bits_step"]))
        path += float(np.sum(np.linalg.norm(positions_after - step_start, axis=1)))
        oldest = np.asarray(view["oldest_wait_age"], dtype=np.float64)
        mean_age = np.asarray(view["mean_wait_age"], dtype=np.float64)
        valid = np.asarray(view["urgency_valid"], dtype=np.int64) == 1
        oldest_max = max(oldest_max, float(np.max(oldest)))
        mean_wait_max = max(mean_wait_max, float(np.max(mean_age)))
        if np.any(valid):
            oldest_sum += float(np.sum(oldest[valid]))
            oldest_count += int(np.sum(valid))
            mean_wait_sum += float(np.sum(mean_age[valid]))
            mean_wait_count += int(np.sum(valid))
        if float(np.sum(served_now)) > float(env.tol_bits):
            positive_steps += 1
            if first_service < 0:
                first_service = int(t_before)
        projection_norm_sum += float(action.projection_norm)
        if action.constraint_status != "ok" or float(action.projection_norm) > 1.0e-3:
            projector_steps += 1
        if minimum_pairwise_distance(positions_after) < float(D_SAFE_M):
            safety_steps += 1
        for uav, buoy in enumerate(np.asarray(option.association, dtype=np.int64)):
            if int(buoy) < 0:
                continue
            sinr_sum += float(info["sinr"][uav])
            interference_sum += float(comm["total_interference"][uav])
            sinr_count += 1
        max_gap = max(max_gap, float(np.max(np.abs(gap))))
        max_account = max(max_account, float(np.max(np.abs(account))))
        max_sinr_gap = max(max_sinr_gap, float(np.max(np.abs(np.asarray(info["sinr"]) - comm["sinr"]))))
        steps.append(
            {
                "scenario_id": int(scenario_id),
                "arm": arm,
                "time_step": int(t_before),
                "assignment": decision["executed_assignment"] if pending else "",
                "served_bits": _join(served_now),
                "urgency": _join(urgency),
                "oldest_wait_age": _join(oldest),
                "q_bits_before": _join(queue_before),
                "q_bits_after": _join(env.q_bits),
                "dropped_bits": _join(info["dropped_bits_step"]),
                "payload_increment_bits": float(info["payload_increment_bits"]),
                "sinr": _join(info["sinr"]),
                "total_interference": _join(comm["total_interference"]),
                "projection_norm": float(action.projection_norm),
                "constraint_status": str(action.constraint_status),
                "ledger_gap_bits": float(np.max(np.abs(gap))),
                "accounting_error_bits": float(np.max(np.abs(account))),
            }
        )
        # The assignment string above used the pre-decrement pending. Keep the executed assignment explicitly.
        steps[-1]["assignment"] = decision["executed_assignment"]
        pending -= 1
        option.remaining_hold = int(pending)
        last = np.asarray(action.executed, dtype=np.float64).copy()
    if int(env.t) != 48 or len(steps) != 48 or len(decisions) != 4:
        raise RuntimeError(f"arm {arm} scenario {scenario_id} did not finish 48 steps and 4 decisions")
    payload_gap = abs(float(env.payload_bits) - float(served_sum))
    status = "ok"
    if payload_gap > float(env.tol_bits) or max_account > float(env.tol_bits):
        status = "accounting"
    if max_gap > float(env.tol_bits):
        status = "ledger"
    return {
        "scenario_id": int(scenario_id),
        "arm": arm,
        "status": status,
        "final_t": int(env.t),
        "payload_bits": float(env.payload_bits),
        "served_bits_sum": float(served_sum),
        "payload_accounting_gap_bits": float(payload_gap),
        "urgent_service_bits": float(urgent_sum),
        "dropped_bits": float(dropped_sum),
        "final_queue_bits": float(np.sum(env.q_bits)),
        "final_queue_vector": _join(env.q_bits),
        "oldest_wait_max": float(oldest_max),
        "oldest_wait_mean": float(oldest_sum / oldest_count) if oldest_count else 0.0,
        "mean_wait_max": float(mean_wait_max),
        "mean_wait_mean": float(mean_wait_sum / mean_wait_count) if mean_wait_count else 0.0,
        "path_length_m": float(path),
        "mean_sinr": float(sinr_sum / sinr_count) if sinr_count else 0.0,
        "mean_total_interference": float(interference_sum / sinr_count) if sinr_count else 0.0,
        "max_sinr_recompute_gap": float(max_sinr_gap),
        "projector_trigger_steps": int(projector_steps),
        "projector_trigger_rate": float(projector_steps / len(steps)),
        "mean_projection_norm": float(projection_norm_sum / len(steps)),
        "safety_violation_steps": int(safety_steps),
        "first_service_step": int(first_service),
        "positive_service_steps": int(positive_steps),
        "max_ledger_gap_bits": float(max_gap),
        "max_accounting_error_bits": float(max_account),
        "seconds": float(time.perf_counter() - started),
        "decisions": decisions,
        "steps": steps,
        "tiebreak_changed_decisions": int(sum(int(row["assignment_changed"]) for row in decisions)),
        "max_abs_choice_payload_gap_bits": float(max(abs(float(row["payload_gap_bits"])) for row in decisions)),
    }


def _decide_at(env, ledger, actor, scenario_id: int, arm: str, clock: LabelClock, cache: dict, sink) -> dict:
    horizon = label_horizon(int(env.t), int(env.config.episode_horizon))
    subset = sample_subset(env, scenario_id, int(env.t))
    for row in subset["rows"]:
        sink["selected"].append(
            {
                "scenario_id": int(scenario_id),
                "arm": arm,
                "time_step": int(env.t),
                "candidate_id": int(row["candidate_id"]),
                "assignment": row["assignment"],
                "sampling_score": float(row["sampling_score"]),
                "sampling_source": row["sampling_source"],
                "utility_rank": int(row["utility_rank"]),
                "selection": row["selection"],
                "stratum": int(row["stratum"]),
                "filled": int(row["filled"]),
                "pool_hash": row["pool_hash"],
                "pool_count": int(row["pool_count"]),
                "subset_hash": row["subset_hash"],
                "label_horizon": int(horizon),
            }
        )
    sink["flush"]()
    print(f"label scenario {scenario_id} arm {arm} t={int(env.t)} horizon {horizon} source {subset['source']}", flush=True)
    cache_key = (subset["state_hash"], subset["subset_hash"], int(horizon))
    started = time.perf_counter()
    if cache_key in cache:
        scored = cache[cache_key]
        reused = 1
    else:
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
        cache[cache_key] = scored
        clock.observe(float(time.perf_counter() - started))
        reused = 0
    payload_choice, tie_choice, executed = choose_executed(scored, arm)
    best_payload = max(float(row["payload_bits"]) for row in scored)
    payload_gap = float(tie_choice["payload_bits"]) - float(payload_choice["payload_bits"])
    if payload_gap < -float(EPS_BITS) or abs(float(executed["payload_bits"]) - best_payload) > float(EPS_BITS):
        raise RuntimeError("selection left the 1-bit payload tie")
    changed = int(payload_choice["assignment"] != tie_choice["assignment"])
    return {
        "scenario_id": int(scenario_id),
        "arm": arm,
        "time_step": int(env.t),
        "label_horizon": int(horizon),
        "pool_count": int(subset["pool_count"]),
        "pool_hash": subset["pool_hash"],
        "subset_hash": subset["subset_hash"],
        "subset_count": int(len(subset["rows"])),
        "sampling_source": subset["source"],
        "state_hash": subset["state_hash"],
        "payload_tie_count": int(sum(abs(float(row["payload_bits"]) - best_payload) <= float(EPS_BITS) for row in scored)),
        "payload_first_candidate_id": int(payload_choice["candidate_id"]),
        "tiebreak_candidate_id": int(tie_choice["candidate_id"]),
        "payload_first_assignment": payload_choice["assignment"],
        "tiebreak_assignment": tie_choice["assignment"],
        "assignment_changed": changed,
        "payload_first_payload_bits": float(payload_choice["payload_bits"]),
        "tiebreak_payload_bits": float(tie_choice["payload_bits"]),
        "payload_gap_bits": float(payload_gap),
        "payload_gap_within_one_bit": int(abs(payload_gap) <= float(EPS_BITS)),
        "payload_first_urgent_bits": float(payload_choice["urgent_service_bits"]),
        "tiebreak_urgent_bits": float(tie_choice["urgent_service_bits"]),
        "urgent_gain_bits": float(tie_choice["urgent_service_bits"]) - float(payload_choice["urgent_service_bits"]),
        "executed_candidate_id": int(executed["candidate_id"]),
        "executed_assignment": executed["assignment"],
        "executed_vector": np.asarray(executed["assignment_vector"], dtype=np.int64).copy(),
        "executed_payload_regret_bits": float(executed["payload_bits"]) - float(best_payload),
        "selection_reason": "payload_max_min_id" if arm == "payload_first" else "urgency_inside_one_bit_payload_tie",
        "label_reused": int(reused),
        "max_label_ledger_gap_bits": float(max(float(row["ledger_gap_bits"]) for row in scored)),
        "max_label_accounting_error_bits": float(max(float(row["accounting_error_bits"]) for row in scored)),
        "label_seconds": 0.0 if reused else float(time.perf_counter() - started),
    }
