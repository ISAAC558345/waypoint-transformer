"""One development-pool episode. Logging copies arrays and does not change the transition."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6e.generators import is_legal
from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import D_SAFE_M, DECISION_TIMES, HOLD
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info, make_option
from wt.benchmark_v2.phase6i_urgency_stage_a.ledger import AdmissionAgeLedger, snap_vector
from wt.benchmark_v2.phase6i_urgency_stage_a.runner import (
    STEP_FIELDS,
    ActionBundle,
    communication_at_service,
)
from wt.lowlevel_v2.contracts import minimum_pairwise_distance

from .protocol import (
    AGE_SCALE_STEPS,
    AGE_SCHEMA_VERSION,
    FIXED_URGENCY_THRESHOLD,
    HIGH_URGENCY_QUANTILE,
    require_dev_scenario,
)

EXTRA_FIELDS = (
    "condition_id",
    "replication_seed",
    "candidate_count",
    "unique_candidate_count",
)
ROW_FIELDS = EXTRA_FIELDS + STEP_FIELDS


@dataclass
class EpisodeRun:
    scenario_id: int
    condition_id: str
    replication_seed: int | None
    status: str
    anomaly: str
    final_t: int
    payload_bits: float
    served_bits_sum: float
    q_bits: np.ndarray
    assignment: np.ndarray
    positions: np.ndarray
    urgency: np.ndarray
    max_ledger_mismatch_bits: float
    max_conservation_error_bits: float
    max_sinr_recompute_gap: float
    max_rate_recompute_gap: float
    urgent_service_bits: float
    high_urgency_served_bits: float
    urgency_at_least_half_served_bits: float
    zero_urgency_threshold_steps: int
    oldest_wait_max: float
    oldest_wait_mean: float
    mean_wait_max: float
    mean_wait_mean: float
    age_weighted_backlog_mean: float
    final_queue_bits: float
    dropped_bits: float
    first_service_radius_step: int
    positive_service_steps: int
    in_radius_uav_steps: int
    path_length_m: float
    projector_trigger_steps: int
    mean_projection_norm: float
    safety_violation_steps: int
    mean_sinr: float
    mean_rate_bps: float
    mean_total_interference: float
    sinr_served_pearson: float | None
    mean_candidate_count: float
    scheme_c_calls: int
    mean_raw_candidate_count: float
    rows_written: int = 0
    notes: list[str] = field(default_factory=list)


def run_episode(
    scenario_id: int,
    *,
    condition_id: str,
    replication_seed: int | None,
    movement,
    decision,
    record: bool,
    writer: csv.DictWriter | None = None,
    hashes: dict | None = None,
    max_steps: int | None = None,
) -> EpisodeRun:
    """Run one episode. `record` only controls whether step rows are written."""
    require_dev_scenario(scenario_id)
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=int(scenario_id))
    if int(env.config.episode_horizon) != int(AGE_SCALE_STEPS):
        raise RuntimeError("episode horizon and the fixed age scale diverged")
    ledger = AdmissionAgeLedger(int(env.config.num_buoys), AGE_SCALE_STEPS)
    ledger.reset_from_queue(env.q_bits)
    ledger.assert_matches(env.q_bits, env.tol_bits)
    position_prev = np.asarray(env.uav_positions, dtype=np.float64).copy()
    predecessor = None
    last = np.zeros((int(env.config.num_uavs), 2), dtype=np.float64)
    option = None
    assignment = None
    hold = 0
    identity = hashes or {"policy_sha256": "", "config_sha256": "", "mappo_sha256": ""}
    radius = float(env.config.service_radius_m)
    limit = int(env.config.episode_horizon) if max_steps is None else int(max_steps)
    stats = _blank_stats()
    rows_written = 0
    steps = 0
    while int(env.t) < int(env.config.episode_horizon) and steps < limit:
        decision_meta = None
        if hold <= 0:
            if int(env.t) not in DECISION_TIMES:
                raise RuntimeError(f"decision time {int(env.t)} is outside the registered clock")
            chosen, decision_meta = decision(env, position_prev, predecessor)
            assignment = np.asarray(chosen, dtype=np.int64).copy()
            if not is_legal(assignment, int(env.config.num_uavs), int(env.config.num_buoys)):
                raise RuntimeError("selected assignment is illegal")
            option = make_option(env, assignment)
            hold = int(option.remaining_hold)
            if hold != int(HOLD):
                raise RuntimeError("new option hold is not 15")
            stats["candidate_counts"].append(float(decision_meta["candidate_count"]))
            stats["raw_counts"].append(float(decision_meta.get("raw_candidate_count", decision_meta["candidate_count"])))
            stats["scheme_c_calls"] += int(decision_meta.get("scheme_c_called", 0))
        if option is None or assignment is None:
            raise RuntimeError("episode has no assignment")
        t_before = int(env.t)
        step_start = np.asarray(env.uav_positions, dtype=np.float64).copy()
        queue_before = np.asarray(env.q_bits, dtype=np.float64).copy()
        snap_before = np.asarray(env.cumulative_snap_bits, dtype=np.float64).copy()
        view = ledger.view(t_before, queue_before, float(env.tol_bits))
        action = movement(env, option, last)
        info = env.step(action.executed, assignment)
        _require_accounting_keys(info)
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
        _accumulate(stats, env, assignment, view, info, action, positions_after, step_start, gap, account, comm, radius, t_before)
        if record:
            if writer is None:
                raise RuntimeError("a step-row writer is required when recording")
            for row in _rows(
                scenario_id=int(scenario_id),
                condition_id=condition_id,
                replication_seed=replication_seed,
                time_step=t_before,
                identity=identity,
                positions=step_start,
                positions_after=positions_after,
                assignment=assignment,
                remaining_hold=int(hold),
                action=action,
                queue_before=queue_before,
                view=view,
                decision_meta=decision_meta,
                info=info,
                comm=comm,
                queue_after=np.asarray(env.q_bits, dtype=np.float64).copy(),
                gap=gap,
                account_max=float(np.max(np.abs(account))),
                t_after=int(env.t),
            ):
                writer.writerow({key: _cell(row.get(key)) for key in ROW_FIELDS})
                rows_written += 1
        hold -= 1
        option.remaining_hold = int(hold)
        last = np.asarray(action.executed, dtype=np.float64).copy()
        position_prev = step_start
        predecessor = t_before
        steps += 1
    final_view = ledger.view(int(env.t), env.q_bits, float(env.tol_bits))
    ledger.assert_matches(env.q_bits, float(env.tol_bits))
    return _finish(
        scenario_id=int(scenario_id),
        condition_id=condition_id,
        replication_seed=replication_seed,
        env=env,
        final_view=final_view,
        stats=stats,
        rows_written=rows_written,
        expected_steps=limit if max_steps is not None else int(env.config.episode_horizon),
        steps=steps,
    )


def environment_match(left: EpisodeRun, right: EpisodeRun) -> bool:
    return (
        left.status == "ok"
        and right.status == "ok"
        and left.final_t == right.final_t
        and left.payload_bits == right.payload_bits
        and np.array_equal(left.q_bits, right.q_bits)
        and np.array_equal(left.assignment, right.assignment)
        and np.array_equal(left.positions, right.positions)
        and np.array_equal(left.urgency, right.urgency)
    )


def _blank_stats() -> dict:
    return {
        "served_sum": 0.0,
        "urgent": 0.0,
        "high_served": 0.0,
        "half_served": 0.0,
        "zero_threshold_steps": 0,
        "oldest_max": 0.0,
        "mean_wait_max": 0.0,
        "oldest_sum": 0.0,
        "oldest_count": 0,
        "mean_wait_sum": 0.0,
        "mean_wait_count": 0,
        "backlog_sum": 0.0,
        "dropped": 0.0,
        "first_radius": -1,
        "positive_steps": 0,
        "in_radius": 0,
        "path": 0.0,
        "projector_steps": 0,
        "projection_norm_sum": 0.0,
        "safety_steps": 0,
        "sinr_sum": 0.0,
        "sinr_count": 0,
        "rate_sum": 0.0,
        "interference_sum": 0.0,
        "paired_sinr": [],
        "paired_served": [],
        "max_gap": 0.0,
        "max_account": 0.0,
        "max_sinr_gap": 0.0,
        "max_rate_gap": 0.0,
        "candidate_counts": [],
        "raw_counts": [],
        "scheme_c_calls": 0,
        "step_count": 0,
    }


def _accumulate(stats, env, assignment, view, info, action: ActionBundle, positions_after, step_start, gap, account, comm, radius, t_before) -> None:
    served = np.asarray(info["served_bits"], dtype=np.float64)
    dropped = np.asarray(info["dropped_bits_step"], dtype=np.float64)
    urgency = np.asarray(view["urgency"], dtype=np.float64)
    valid = np.asarray(view["urgency_valid"], dtype=np.int64) == 1
    oldest = np.asarray(view["oldest_wait_age"], dtype=np.float64)
    mean_age = np.asarray(view["mean_wait_age"], dtype=np.float64)
    stats["served_sum"] += float(np.sum(served))
    stats["urgent"] += float(np.sum(urgency * served))
    stats["dropped"] += float(np.sum(dropped))
    stats["backlog_sum"] += float(np.sum(view["age_weighted_backlog"]))
    stats["oldest_max"] = max(stats["oldest_max"], float(np.max(oldest)))
    stats["mean_wait_max"] = max(stats["mean_wait_max"], float(np.max(mean_age)))
    if np.any(valid):
        stats["oldest_sum"] += float(np.sum(oldest[valid]))
        stats["oldest_count"] += int(np.sum(valid))
        stats["mean_wait_sum"] += float(np.sum(mean_age[valid]))
        stats["mean_wait_count"] += int(np.sum(valid))
        threshold = float(np.quantile(urgency[valid], HIGH_URGENCY_QUANTILE))
        if threshold <= 0.0:
            stats["zero_threshold_steps"] += 1
        stats["high_served"] += float(np.sum(served[valid & (urgency + 1.0e-15 >= threshold)]))
        stats["half_served"] += float(np.sum(served[valid & (urgency + 1.0e-15 >= FIXED_URGENCY_THRESHOLD)]))
    if float(info["payload_increment_bits"]) > 0.0:
        stats["positive_steps"] += 1
    stats["path"] += float(np.sum(np.linalg.norm(positions_after - step_start, axis=1)))
    stats["projection_norm_sum"] += float(action.projection_norm)
    if action.constraint_status != "ok" or float(action.projection_norm) > 1.0e-3:
        stats["projector_steps"] += 1
    if minimum_pairwise_distance(positions_after) < float(D_SAFE_M):
        stats["safety_steps"] += 1
    for uav, buoy in enumerate(np.asarray(assignment, dtype=np.int64)):
        if int(buoy) < 0:
            continue
        distance = float(np.linalg.norm(positions_after[uav] - env.buoy_positions[int(buoy)]))
        if distance <= radius:
            stats["in_radius"] += 1
            if int(stats["first_radius"]) < 0:
                stats["first_radius"] = int(t_before)
        stats["sinr_sum"] += float(info["sinr"][uav])
        stats["rate_sum"] += float(info["rate_bps"][uav])
        stats["interference_sum"] += float(comm["total_interference"][uav])
        stats["sinr_count"] += 1
        stats["paired_sinr"].append(float(info["sinr"][uav]))
        stats["paired_served"].append(float(served[int(buoy)]))
    stats["max_gap"] = max(stats["max_gap"], float(np.max(np.abs(gap))))
    stats["max_account"] = max(stats["max_account"], float(np.max(np.abs(account))))
    stats["max_sinr_gap"] = max(stats["max_sinr_gap"], float(np.max(np.abs(np.asarray(info["sinr"]) - comm["sinr"]))))
    stats["max_rate_gap"] = max(stats["max_rate_gap"], float(np.max(np.abs(np.asarray(info["rate_bps"]) - comm["rate"]))))
    stats["step_count"] += 1


def _finish(*, scenario_id, condition_id, replication_seed, env, final_view, stats, rows_written, expected_steps, steps) -> EpisodeRun:
    count = max(int(stats["sinr_count"]), 1)
    steps_done = max(int(stats["step_count"]), 1)
    notes = []
    status = "ok"
    if int(steps) != int(expected_steps) or int(env.t) != int(steps):
        status = "incomplete"
        notes.append(f"completed {int(steps)} steps at t={int(env.t)}")
    payload_gap = abs(float(env.payload_bits) - float(stats["served_sum"]))
    if payload_gap > float(env.tol_bits):
        status = "accounting"
        notes.append(f"payload and served bits differ by {payload_gap}")
    if float(stats["max_gap"]) > float(env.tol_bits):
        status = "ledger"
        notes.append(f"ledger mismatch {stats['max_gap']}")
    if float(stats["max_account"]) > float(env.tol_bits):
        status = "accounting"
        notes.append(f"conservation error {stats['max_account']}")
    if float(stats["max_sinr_gap"]) > 1.0e-6 or float(stats["max_rate_gap"]) > 1.0e-4:
        status = "link"
        notes.append(f"sinr gap {stats['max_sinr_gap']} rate gap {stats['max_rate_gap']}")
    return EpisodeRun(
        scenario_id=int(scenario_id),
        condition_id=str(condition_id),
        replication_seed=None if replication_seed is None else int(replication_seed),
        status=status,
        anomaly="; ".join(notes),
        final_t=int(env.t),
        payload_bits=float(env.payload_bits),
        served_bits_sum=float(stats["served_sum"]),
        q_bits=np.asarray(env.q_bits, dtype=np.float64).copy(),
        assignment=np.asarray(env.assignment, dtype=np.int64).copy(),
        positions=np.asarray(env.uav_positions, dtype=np.float64).copy(),
        urgency=np.asarray(final_view["urgency"], dtype=np.float64).copy(),
        max_ledger_mismatch_bits=float(stats["max_gap"]),
        max_conservation_error_bits=float(stats["max_account"]),
        max_sinr_recompute_gap=float(stats["max_sinr_gap"]),
        max_rate_recompute_gap=float(stats["max_rate_gap"]),
        urgent_service_bits=float(stats["urgent"]),
        high_urgency_served_bits=float(stats["high_served"]),
        urgency_at_least_half_served_bits=float(stats["half_served"]),
        zero_urgency_threshold_steps=int(stats["zero_threshold_steps"]),
        oldest_wait_max=float(stats["oldest_max"]),
        oldest_wait_mean=float(stats["oldest_sum"] / stats["oldest_count"]) if stats["oldest_count"] else 0.0,
        mean_wait_max=float(stats["mean_wait_max"]),
        mean_wait_mean=float(stats["mean_wait_sum"] / stats["mean_wait_count"]) if stats["mean_wait_count"] else 0.0,
        age_weighted_backlog_mean=float(stats["backlog_sum"] / steps_done),
        final_queue_bits=float(np.sum(env.q_bits)),
        dropped_bits=float(stats["dropped"]),
        first_service_radius_step=int(stats["first_radius"]),
        positive_service_steps=int(stats["positive_steps"]),
        in_radius_uav_steps=int(stats["in_radius"]),
        path_length_m=float(stats["path"]),
        projector_trigger_steps=int(stats["projector_steps"]),
        mean_projection_norm=float(stats["projection_norm_sum"] / steps_done),
        safety_violation_steps=int(stats["safety_steps"]),
        mean_sinr=float(stats["sinr_sum"] / count) if stats["sinr_count"] else 0.0,
        mean_rate_bps=float(stats["rate_sum"] / count) if stats["sinr_count"] else 0.0,
        mean_total_interference=float(stats["interference_sum"] / count) if stats["sinr_count"] else 0.0,
        sinr_served_pearson=_pearson(stats["paired_sinr"], stats["paired_served"]),
        mean_candidate_count=float(np.mean(stats["candidate_counts"])) if stats["candidate_counts"] else 0.0,
        scheme_c_calls=int(stats["scheme_c_calls"]),
        mean_raw_candidate_count=float(np.mean(stats["raw_counts"])) if stats["raw_counts"] else 0.0,
        rows_written=int(rows_written),
    )


def _pearson(left, right) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _require_accounting_keys(info: dict) -> None:
    required = ("served_bits", "generated_arrival_bits", "admitted_arrival_bits", "dropped_bits_step", "sinr", "rate_bps", "payload_increment_bits")
    missing = [key for key in required if key not in info]
    if missing:
        raise RuntimeError(f"step result is missing accounting fields: {missing}")


def _rows(
    *,
    scenario_id: int,
    condition_id: str,
    replication_seed: int | None,
    time_step: int,
    identity: dict,
    positions: np.ndarray,
    positions_after: np.ndarray,
    assignment: np.ndarray,
    remaining_hold: int,
    action: ActionBundle,
    queue_before: np.ndarray,
    view: dict,
    decision_meta: dict | None,
    info: dict,
    comm: dict,
    queue_after: np.ndarray,
    gap: np.ndarray,
    account_max: float,
    t_after: int,
) -> list[dict]:
    decision_values = {
        "candidate_pool_hash": "" if decision_meta is None else decision_meta["candidate_pool_hash"],
        "candidate_index": "" if decision_meta is None else decision_meta["candidate_index"],
        "candidate_count": "" if decision_meta is None else decision_meta["candidate_count"],
        "unique_candidate_count": "" if decision_meta is None else decision_meta["unique_candidate_count"],
        "scheme_c_score": "" if decision_meta is None else decision_meta["scheme_c_score"],
    }
    shared = {
        "scenario_id": int(scenario_id),
        "condition_id": str(condition_id),
        "replication_seed": "" if replication_seed is None else int(replication_seed),
        "time_step": int(time_step),
        "schema_version": AGE_SCHEMA_VERSION,
        "policy_sha256": identity["policy_sha256"],
        "config_sha256": identity["config_sha256"],
        "mappo_sha256": identity["mappo_sha256"],
        "payload_increment_bits": float(info["payload_increment_bits"]),
        "conservation_error_bits": float(account_max),
        "t_after": int(t_after),
        **decision_values,
    }
    rows = []
    interference = np.asarray(comm["cross_link_interference"], dtype=np.float64)
    for uav in range(int(positions.shape[0])):
        row = dict(shared)
        row.update(
            {
                "record_kind": "uav",
                "index": int(uav),
                "uav_x_m": float(positions[uav, 0]),
                "uav_y_m": float(positions[uav, 1]),
                "uav_x_after_m": float(positions_after[uav, 0]),
                "uav_y_after_m": float(positions_after[uav, 1]),
                "assignment_buoy": int(assignment[uav]),
                "remaining_hold": int(remaining_hold),
                "a_des_x": float(action.desired[uav, 0]),
                "a_des_y": float(action.desired[uav, 1]),
                "projected_x": float(action.projected[uav, 0]),
                "projected_y": float(action.projected[uav, 1]),
                "executed_x": float(action.executed[uav, 0]),
                "executed_y": float(action.executed[uav, 1]),
                "constraint_status": action.constraint_status,
                "projection_norm": float(action.projection_norm),
                "action_delta_norm": float(action.per_uav_delta[uav]),
                "sinr": float(info["sinr"][uav]),
                "rate_bps": float(info["rate_bps"][uav]),
                "total_interference": float(comm["total_interference"][uav]),
                "ledger_mismatch_bits": float(np.max(np.abs(gap))),
            }
        )
        for other in range(6):
            row[f"interference_{other}"] = float(interference[uav, other])
        rows.append(row)
    served = np.asarray(info["served_bits"], dtype=np.float64)
    generated = np.asarray(info["generated_arrival_bits"], dtype=np.float64)
    admitted = np.asarray(info["admitted_arrival_bits"], dtype=np.float64)
    dropped = np.asarray(info["dropped_bits_step"], dtype=np.float64)
    for buoy in range(int(queue_before.shape[0])):
        row = dict(shared)
        row.update(
            {
                "record_kind": "buoy",
                "index": int(buoy),
                "q_bits_before": float(queue_before[buoy]),
                "oldest_wait_age": float(view["oldest_wait_age"][buoy]),
                "mean_wait_age": float(view["mean_wait_age"][buoy]),
                "age_weighted_backlog": float(view["age_weighted_backlog"][buoy]),
                "urgency_valid": int(view["urgency_valid"][buoy]),
                "urgency": float(view["urgency"][buoy]),
                "served_bits": float(served[buoy]),
                "generated_arrival_bits": float(generated[buoy]),
                "admitted_arrival_bits": float(admitted[buoy]),
                "dropped_bits": float(dropped[buoy]),
                "q_bits_after": float(queue_after[buoy]),
                "ledger_mismatch_bits": float(gap[buoy]),
            }
        )
        rows.append(row)
    return rows


def _cell(value) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)
