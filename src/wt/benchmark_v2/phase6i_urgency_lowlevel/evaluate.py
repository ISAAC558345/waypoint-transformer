"""Closed-loop and fixed-hold evaluation. Scheme C stays frozen for every candidate."""
from __future__ import annotations

import csv
from dataclasses import asdict

import numpy as np
import torch

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6e.generators import is_legal
from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import ALPHA_RESIDUAL, DECISION_TIMES, HOLD, SERVICE_SAFETY_MARGIN_M
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info, make_option
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import load_actor
from wt.benchmark_v2.phase6i_urgency_stage_a.ledger import AdmissionAgeLedger, snap_vector
from wt.benchmark_v2.phase6i_urgency_stage_a.protocol import AGE_SCALE_STEPS
from wt.benchmark_v2.phase6i_urgency_stage_a.runner import (
    ActionBundle,
    ScoreTap,
    _require_accounting_keys,
    communication_at_service,
    frozen_mappo_action,
)
from wt.benchmark_v2.phase6i_urgency_stage_b.conditions import scheme_c_decision
from wt.benchmark_v2.phase6i_urgency_stage_b.rollout import _accumulate, _blank_stats, _finish
from wt.benchmark_v2.phase6iaa_formal_planner.planner import FormalPlanner
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.contracts import CollisionConfig, clip_unit_step
from wt.lowlevel_v2.safety_projector import SafetyProjector

from .models import UrgencyActor, load_checkpoint
from .observation import build_urgency_actor_observation
from .protocol import FIXED_URGENCY_THRESHOLD, require_dev_scenario, require_train_scenario

STEP_FIELDS = (
    "scenario_id",
    "variant",
    "model_seed",
    "protocol",
    "time_step",
    "record_kind",
    "index",
    "uav_x_m",
    "uav_y_m",
    "uav_x_after_m",
    "uav_y_after_m",
    "assignment_buoy",
    "remaining_hold",
    "executed_x",
    "executed_y",
    "constraint_status",
    "projection_norm",
    "sinr",
    "rate_bps",
    "total_interference",
    "q_bits_before",
    "oldest_wait_age",
    "urgency",
    "urgency_valid",
    "served_bits",
    "dropped_bits",
    "q_bits_after",
    "payload_increment_bits",
    "ledger_mismatch_bits",
    "candidate_count",
    "scheme_c_score",
    "idle_slots",
    "policy_sha256",
    "config_sha256",
    "mappo_sha256",
    "lowlevel_sha256",
)


def load_l0_actor():
    return load_actor(0)


def load_trained_actor(path):
    actor, _critic, payload = load_checkpoint(path)
    actor.eval()
    return actor, payload


def movement_for(variant: str, actor):
    if variant == "L0":

        def movement(env, option, last, _view):
            return frozen_mappo_action(env, actor, option, last)

        return movement

    def movement(env, option, last, view):
        return _urgency_action(actor, env, option, last, view)

    return movement


def run_episode(
    scenario_id: int,
    *,
    variant: str,
    model_seed: int,
    movement,
    decision,
    protocol: str,
    split: str,
    lowlevel_sha256: str,
    hashes: dict,
    writer: csv.DictWriter | None = None,
    max_steps: int | None = None,
    fixed_assignment=None,
) -> dict:
    if split == "dev":
        require_dev_scenario(scenario_id)
    elif split == "train":
        require_train_scenario(scenario_id)
    else:
        raise RuntimeError(f"unknown split {split}")
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=int(scenario_id))
    ledger = AdmissionAgeLedger(int(env.config.num_buoys), AGE_SCALE_STEPS)
    ledger.reset_from_queue(env.q_bits)
    ledger.assert_matches(env.q_bits, env.tol_bits)
    position_prev = np.asarray(env.uav_positions, dtype=np.float64).copy()
    predecessor = None
    last = np.zeros((int(env.config.num_uavs), 2), dtype=np.float64)
    option = None
    assignment = None
    hold = 0
    stats = _blank_stats()
    steps = 0
    rows_written = 0
    idle_slots = []
    first_high = -1
    limit = int(env.config.episode_horizon) if max_steps is None else int(max_steps)
    radius = float(env.config.service_radius_m)
    while int(env.t) < int(env.config.episode_horizon) and steps < limit:
        decision_meta = None
        if hold <= 0:
            if int(env.t) not in DECISION_TIMES:
                raise RuntimeError(f"decision time {int(env.t)} is outside the registered clock")
            if fixed_assignment is None:
                chosen, decision_meta = decision(env, position_prev, predecessor)
            else:
                chosen = np.asarray(fixed_assignment, dtype=np.int64)
                decision_meta = decision
            assignment = np.asarray(chosen, dtype=np.int64).copy()
            if not is_legal(assignment, int(env.config.num_uavs), int(env.config.num_buoys)):
                raise RuntimeError("selected assignment is illegal")
            option = make_option(env, assignment)
            hold = int(option.remaining_hold)
            stats["candidate_counts"].append(float(decision_meta["candidate_count"]))
            stats["raw_counts"].append(float(decision_meta.get("raw_candidate_count", decision_meta["candidate_count"])))
            stats["scheme_c_calls"] += int(decision_meta.get("scheme_c_called", 0))
            idle_slots.append(int(np.sum(assignment < 0)))
        t_before = int(env.t)
        step_start = np.asarray(env.uav_positions, dtype=np.float64).copy()
        queue_before = np.asarray(env.q_bits, dtype=np.float64).copy()
        snap_before = np.asarray(env.cumulative_snap_bits, dtype=np.float64).copy()
        view = ledger.view(t_before, queue_before, float(env.tol_bits))
        action = movement(env, option, last, view)
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
        served = np.asarray(info["served_bits"], dtype=np.float64)
        urgency = np.asarray(view["urgency"], dtype=np.float64)
        if first_high < 0 and np.any((served > 0.0) & (urgency + 1.0e-15 >= float(FIXED_URGENCY_THRESHOLD))):
            first_high = int(t_before)
        if writer is not None:
            rows_written += _write_step(
                writer,
                scenario_id=int(scenario_id),
                variant=variant,
                model_seed=int(model_seed),
                protocol=protocol,
                time_step=t_before,
                positions=step_start,
                positions_after=positions_after,
                assignment=assignment,
                remaining_hold=int(hold),
                action=action,
                queue_before=queue_before,
                view=view,
                info=info,
                comm=comm,
                queue_after=np.asarray(env.q_bits, dtype=np.float64),
                gap=gap,
                decision_meta=decision_meta,
                idle_slots=0 if not idle_slots else int(idle_slots[-1]) if decision_meta is not None else "",
                hashes=hashes,
                lowlevel_sha256=lowlevel_sha256,
            )
        hold -= 1
        option.remaining_hold = int(hold)
        last = np.asarray(action.executed, dtype=np.float64).copy()
        position_prev = step_start
        predecessor = t_before
        steps += 1
    final_view = ledger.view(int(env.t), env.q_bits, float(env.tol_bits))
    finished = _finish(
        scenario_id=int(scenario_id),
        condition_id=str(variant),
        replication_seed=int(model_seed),
        env=env,
        final_view=final_view,
        stats=stats,
        rows_written=rows_written,
        expected_steps=limit,
        steps=steps,
    )
    row = {key: _plain(value) for key, value in asdict(finished).items()}
    row.update(
        {
            "variant": variant,
            "model_seed": int(model_seed),
            "protocol": protocol,
            "lowlevel_sha256": lowlevel_sha256,
            "first_high_urgency_service_step": int(first_high),
            "mean_idle_slots": float(np.mean(idle_slots)) if idle_slots else 0.0,
            "decision_count": int(len(idle_slots)),
            "final_urgency": [float(item) for item in np.asarray(final_view["urgency"]).reshape(-1)],
        }
    )
    return row


def initial_scheme_c(planner: FormalPlanner, tap: ScoreTap, scenario_id: int) -> tuple[np.ndarray, dict]:
    require_dev_scenario(scenario_id)
    env = L3BBitQueueEnvV2()
    install_light_info(env)
    env.reset(seed=int(scenario_id))
    chosen, meta = scheme_c_decision(planner, tap)(env, np.asarray(env.uav_positions, dtype=np.float64).copy(), None)
    if int(meta.get("scheme_c_called", 0)) != 1:
        raise RuntimeError("fixed-hold diagnostic did not call scheme C")
    return np.asarray(chosen, dtype=np.int64), meta


def _urgency_action(actor: UrgencyActor, env, option, last, view) -> ActionBundle:
    obs = build_urgency_actor_observation(env, option, view, last)
    with torch.no_grad():
        mu, _log_std = actor(
            torch.as_tensor(obs["self"], dtype=torch.float32),
            torch.as_tensor(obs["task"], dtype=torch.float32),
            torch.as_tensor(obs["communication"], dtype=torch.float32),
            torch.as_tensor(obs["neighbor_set"], dtype=torch.float32),
            torch.as_tensor(obs["d_nearest"], dtype=torch.float32),
        )
    pre = mu.detach().cpu().numpy().astype(np.float64)
    desired = clip_unit_step(fly_toward(env, option.association) + float(ALPHA_RESIDUAL) * np.tanh(pre))
    projected = SafetyProjector(
        CollisionConfig(0.4, 0.2, placeholder=False),
        safety_margin=float(SERVICE_SAFETY_MARGIN_M),
    ).project(env, desired, option)
    failure = projected.constraint_status != "ok" or not projected.finite
    executed = np.zeros_like(desired) if failure else np.asarray(projected.projected_actions, dtype=np.float64)
    executed = clip_unit_step(executed)
    return ActionBundle(
        executed=executed,
        desired=np.asarray(desired, dtype=np.float64).copy(),
        projected=np.asarray(projected.projected_actions, dtype=np.float64).copy(),
        constraint_status=str(projected.constraint_status),
        projection_norm=float(projected.intervention_norm),
        per_uav_delta=np.linalg.norm(executed - desired, axis=1),
    )


def _write_step(writer, **payload) -> int:
    written = 0
    decision_meta = payload["decision_meta"] or {}
    shared = {
        "scenario_id": int(payload["scenario_id"]),
        "variant": payload["variant"],
        "model_seed": int(payload["model_seed"]),
        "protocol": payload["protocol"],
        "time_step": int(payload["time_step"]),
        "payload_increment_bits": float(payload["info"]["payload_increment_bits"]),
        "candidate_count": "" if payload["decision_meta"] is None else decision_meta.get("candidate_count", ""),
        "scheme_c_score": "" if payload["decision_meta"] is None else decision_meta.get("scheme_c_score", ""),
        "idle_slots": payload["idle_slots"],
    }
    positions = payload["positions"]
    after = payload["positions_after"]
    assignment = payload["assignment"]
    action = payload["action"]
    interference = np.asarray(payload["comm"]["total_interference"], dtype=np.float64)
    for uav in range(int(positions.shape[0])):
        row = dict(shared)
        row.update(
            {
                "record_kind": "uav",
                "index": int(uav),
                "uav_x_m": float(positions[uav, 0]),
                "uav_y_m": float(positions[uav, 1]),
                "uav_x_after_m": float(after[uav, 0]),
                "uav_y_after_m": float(after[uav, 1]),
                "assignment_buoy": int(assignment[uav]),
                "remaining_hold": int(payload["remaining_hold"]),
                "executed_x": float(action.executed[uav, 0]),
                "executed_y": float(action.executed[uav, 1]),
                "constraint_status": action.constraint_status,
                "projection_norm": float(action.projection_norm),
                "sinr": float(payload["info"]["sinr"][uav]),
                "rate_bps": float(payload["info"]["rate_bps"][uav]),
                "total_interference": float(interference[uav]),
                "ledger_mismatch_bits": float(np.max(np.abs(payload["gap"]))),
            }
        )
        if uav == 0:
            row["policy_sha256"] = payload["hashes"].get("policy_sha256", "")
            row["config_sha256"] = payload["hashes"].get("config_sha256", "")
            row["mappo_sha256"] = payload["hashes"].get("mappo_sha256", "")
            row["lowlevel_sha256"] = payload["lowlevel_sha256"]
        writer.writerow({key: _cell(row.get(key)) for key in STEP_FIELDS})
        written += 1
    served = np.asarray(payload["info"]["served_bits"], dtype=np.float64)
    dropped = np.asarray(payload["info"]["dropped_bits_step"], dtype=np.float64)
    queue_before = payload["queue_before"]
    queue_after = payload["queue_after"]
    view = payload["view"]
    for buoy in range(int(queue_before.shape[0])):
        row = dict(shared)
        row.update(
            {
                "record_kind": "buoy",
                "index": int(buoy),
                "q_bits_before": float(queue_before[buoy]),
                "oldest_wait_age": float(view["oldest_wait_age"][buoy]),
                "urgency": float(view["urgency"][buoy]),
                "urgency_valid": int(view["urgency_valid"][buoy]),
                "served_bits": float(served[buoy]),
                "dropped_bits": float(dropped[buoy]),
                "q_bits_after": float(queue_after[buoy]),
                "ledger_mismatch_bits": float(payload["gap"][buoy]),
            }
        )
        writer.writerow({key: _cell(row.get(key)) for key in STEP_FIELDS})
        written += 1
    return written


def _plain(value):
    if isinstance(value, np.ndarray):
        return [float(item) if np.ndim(item) == 0 else item.tolist() for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _cell(value) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)
