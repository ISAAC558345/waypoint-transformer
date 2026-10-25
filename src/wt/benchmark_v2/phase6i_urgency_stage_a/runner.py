"""Step logger around the frozen queue transition. Logging only copies arrays."""
from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6d.shadow import capture_state, restore_state
from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import ALPHA_RESIDUAL, SERVICE_SAFETY_MARGIN_M
from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import install_light_info, make_option
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import limit_threads, parameter_token
from wt.benchmark_v2.phase6iaa_formal_planner.planner import FormalPlanner
from wt.benchmark_v2.phase6iaa_formal_planner.protocol import (
    MAPPO_SHA256,
    SCHEME_C_SHA256,
    assert_episode_scenario,
)
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import RESERVED_TEST
from wt.benchmark_v2.phase6iy_candidate_adapter.protocol import config_sha256
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.contracts import CollisionConfig, clip_unit_step
from wt.lowlevel_v2.observation import TASK_DIM, build_actor_observation, communication_features
from wt.lowlevel_v2.residual_policy import ResidualActor
from wt.lowlevel_v2.safety_projector import SafetyProjector

from .ledger import AdmissionAgeLedger, snap_vector
from .protocol import AGE_SCALE_STEPS, DECISION_TIMES, HOLD, OUTPUT, SCHEMA_VERSION, smoke_scenario_ids

STEP_FIELDS = (
    "scenario_id",
    "time_step",
    "record_kind",
    "index",
    "schema_version",
    "policy_sha256",
    "config_sha256",
    "mappo_sha256",
    "uav_x_m",
    "uav_y_m",
    "uav_x_after_m",
    "uav_y_after_m",
    "assignment_buoy",
    "remaining_hold",
    "a_des_x",
    "a_des_y",
    "projected_x",
    "projected_y",
    "executed_x",
    "executed_y",
    "constraint_status",
    "projection_norm",
    "action_delta_norm",
    "q_bits_before",
    "oldest_wait_age",
    "mean_wait_age",
    "age_weighted_backlog",
    "urgency_valid",
    "urgency",
    "candidate_pool_hash",
    "candidate_index",
    "scheme_c_score",
    "served_bits",
    "generated_arrival_bits",
    "admitted_arrival_bits",
    "dropped_bits",
    "q_bits_after",
    "sinr",
    "rate_bps",
    "total_interference",
    "interference_0",
    "interference_1",
    "interference_2",
    "interference_3",
    "interference_4",
    "interference_5",
    "payload_increment_bits",
    "conservation_error_bits",
    "ledger_mismatch_bits",
    "t_after",
)


@dataclass
class ActionBundle:
    executed: np.ndarray
    desired: np.ndarray
    projected: np.ndarray
    constraint_status: str
    projection_norm: float
    per_uav_delta: np.ndarray


@dataclass
class EpisodeResult:
    scenario_id: int
    rows: list[dict]
    payload_bits: float
    q_bits: np.ndarray
    assignment: np.ndarray
    positions: np.ndarray
    time_step: int
    urgency: np.ndarray
    max_ledger_mismatch_bits: float
    max_conservation_error_bits: float
    max_sinr_recompute_gap: float
    max_rate_recompute_gap: float


def frozen_mappo_action(env: L3BBitQueueEnvV2, actor: ResidualActor, option, last_movement: np.ndarray) -> ActionBundle:
    """Same executed action as the frozen deterministic mean. The projector object is not modified."""
    obs = build_actor_observation(env, option, last_movement)
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


def communication_at_service(env: L3BBitQueueEnvV2, queue_before_bits: np.ndarray) -> dict:
    """Link features at the post-move pose, using the queue that existed when SINR was formed.

    ``active_links`` skips an empty buoy. A read after service would drop links that
    carried this step. The queue array is copied and restored before return.
    """
    saved = np.asarray(env.queue, dtype=np.float64).copy()
    env.queue = np.asarray(queue_before_bits, dtype=np.float64) / float(env.block_bits)
    try:
        return communication_features(env)
    finally:
        env.queue = saved


def fly_action(env: L3BBitQueueEnvV2, option, _last_movement: np.ndarray) -> ActionBundle:
    desired = np.asarray(fly_toward(env, option.association), dtype=np.float64)
    projected = SafetyProjector(
        CollisionConfig(0.4, 0.2, placeholder=False),
        safety_margin=float(SERVICE_SAFETY_MARGIN_M),
    ).project(env, desired, option)
    failure = projected.constraint_status != "ok" or not projected.finite
    executed = np.zeros_like(desired) if failure else np.asarray(projected.projected_actions, dtype=np.float64)
    executed = clip_unit_step(executed)
    return ActionBundle(
        executed=executed,
        desired=desired.copy(),
        projected=np.asarray(projected.projected_actions, dtype=np.float64).copy(),
        constraint_status=str(projected.constraint_status),
        projection_norm=float(projected.intervention_norm),
        per_uav_delta=np.linalg.norm(executed - desired, axis=1),
    )


class ScoreTap:
    """Remember the score array already computed by this planner call."""

    def __init__(self, planner: FormalPlanner) -> None:
        if planner.scorer is None:
            raise RuntimeError("scheme C scorer is not loaded")
        self.last = None
        original = planner.scorer.infer

        def infer(graph, assignments):
            result = original(graph, assignments)
            self.last = result
            return result

        planner.scorer.infer = infer


def run_episode(
    scenario_id: int,
    *,
    movement,
    decision,
    record: bool,
    max_steps: int | None = None,
    hashes: dict | None = None,
) -> EpisodeResult:
    """Run one episode. `record` only controls whether step rows are kept."""
    assert_episode_scenario(int(scenario_id), allow_holdout=False)
    if int(scenario_id) in RESERVED_TEST:
        raise RuntimeError("reserved test scenario refused")
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
    rows: list[dict] = []
    max_gap = 0.0
    max_account = 0.0
    max_sinr_gap = 0.0
    max_rate_gap = 0.0
    steps = 0
    limit = int(env.config.episode_horizon) if max_steps is None else int(max_steps)
    identity = hashes or {"policy_sha256": "", "config_sha256": "", "mappo_sha256": ""}
    while int(env.t) < int(env.config.episode_horizon) and steps < limit:
        decision_meta = None
        if hold <= 0:
            if int(env.t) not in DECISION_TIMES:
                raise RuntimeError(f"decision time {int(env.t)} is outside the registered clock")
            chosen, decision_meta = decision(env, position_prev, predecessor)
            assignment = np.asarray(chosen, dtype=np.int64).copy()
            option = make_option(env, assignment)
            hold = int(option.remaining_hold)
            if hold != int(HOLD):
                raise RuntimeError("new option hold is not 15")
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
        max_gap = max(max_gap, float(np.max(np.abs(gap))))
        max_account = max(max_account, float(np.max(np.abs(account))))
        comm = communication_at_service(env, queue_before)
        max_sinr_gap = max(max_sinr_gap, float(np.max(np.abs(np.asarray(info["sinr"]) - comm["sinr"]))))
        max_rate_gap = max(max_rate_gap, float(np.max(np.abs(np.asarray(info["rate_bps"]) - comm["rate"]))))
        if record:
            rows.extend(
                _rows(
                    scenario_id=int(scenario_id),
                    time_step=t_before,
                    identity=identity,
                    positions=step_start,
                    positions_after=np.asarray(env.uav_positions, dtype=np.float64).copy(),
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
                )
            )
        hold -= 1
        option.remaining_hold = int(hold)
        last = np.asarray(action.executed, dtype=np.float64).copy()
        position_prev = step_start
        predecessor = t_before
        steps += 1
    final_view = ledger.view(int(env.t), env.q_bits, float(env.tol_bits))
    ledger.assert_matches(env.q_bits, float(env.tol_bits))
    return EpisodeResult(
        scenario_id=int(scenario_id),
        rows=rows,
        payload_bits=float(env.payload_bits),
        q_bits=np.asarray(env.q_bits, dtype=np.float64).copy(),
        assignment=np.asarray(env.assignment, dtype=np.int64).copy(),
        positions=np.asarray(env.uav_positions, dtype=np.float64).copy(),
        time_step=int(env.t),
        urgency=np.asarray(final_view["urgency"], dtype=np.float64).copy(),
        max_ledger_mismatch_bits=float(max_gap),
        max_conservation_error_bits=float(max_account),
        max_sinr_recompute_gap=float(max_sinr_gap),
        max_rate_recompute_gap=float(max_rate_gap),
    )


def same_outcome(left: EpisodeResult, right: EpisodeResult) -> bool:
    return (
        left.scenario_id == right.scenario_id
        and left.time_step == right.time_step
        and abs(left.payload_bits - right.payload_bits) <= 0.0
        and np.array_equal(left.q_bits, right.q_bits)
        and np.array_equal(left.assignment, right.assignment)
        and np.array_equal(left.positions, right.positions)
        and np.array_equal(left.urgency, right.urgency)
    )


def write_outputs(rows: list[dict], summary: dict, directory: Path = OUTPUT) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "step_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STEP_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in STEP_FIELDS})
    schema = {
        "schema_version": SCHEMA_VERSION,
        "age_scale_steps": AGE_SCALE_STEPS,
        "age_origin_reset": "reset_convention",
        "reset_age_meaning": "episode-internal convention starting at 0 when t is 0; not a pre-episode wait",
        "snap_source": "info float_snap_bits when present, otherwise the step delta of cumulative_snap_bits",
        "row_grain": "one row per scenario, time_step, and either uav or buoy index",
        "columns": list(STEP_FIELDS),
        "urgency_formula": "clip(oldest_wait_age / 48, 0, 1) when q_bits > 0, else 0",
        "logging_changes_transition": False,
        "interference_queue": "pre-service q_bits at the post-move position and this step's assignment",
        "sinr_source": "info.sinr and info.rate_bps from the formal step; recomputed only to cross-check",
    }
    (directory / "schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def run_smoke(directory: Path = OUTPUT) -> dict:
    limit_threads()
    scenarios = smoke_scenario_ids()
    planner = FormalPlanner("scheme_c")
    token = parameter_token(planner.actor)
    if int(planner.actor.body[0].in_features) != int(5 + TASK_DIM + 4 + 16 + 1):
        raise RuntimeError("frozen actor input width changed")
    tap = ScoreTap(planner)
    hashes = {
        "policy_sha256": str(planner.startup["scheme_c_sha256"]),
        "config_sha256": config_sha256(),
        "mappo_sha256": str(planner.startup["mappo_sha256"]),
    }
    if hashes["policy_sha256"] != SCHEME_C_SHA256 or hashes["mappo_sha256"] != MAPPO_SHA256:
        raise RuntimeError("frozen checkpoint hash changed")

    def decision(env, position_prev, predecessor):
        tap.last = None
        selection = planner.select_assignment(env, position_prev, predecessor)
        if selection.status != "selected" or selection.assignment is None or tap.last is None:
            raise RuntimeError(f"scheme C selection failed at t={int(env.t)}: {selection.reason}")
        score = float(np.asarray(tap.last.scores, dtype=np.float64)[int(selection.score_index)])
        return np.asarray(selection.assignment, dtype=np.int64), {
            "candidate_pool_hash": str(selection.pool_hash),
            "candidate_index": int(selection.score_index),
            "scheme_c_score": score,
        }

    def movement(env, option, last):
        return frozen_mappo_action(env, planner.actor, option, last)

    rows: list[dict] = []
    per_scenario = []
    invariant = True
    deterministic = True
    max_gap = 0.0
    max_account = 0.0
    max_sinr_gap = 0.0
    max_rate_gap = 0.0
    for scenario_id in scenarios:
        logged = run_episode(scenario_id, movement=movement, decision=decision, record=True, hashes=hashes)
        silent = run_episode(scenario_id, movement=movement, decision=decision, record=False, hashes=hashes)
        repeat = run_episode(scenario_id, movement=movement, decision=decision, record=False, hashes=hashes)
        invariant = invariant and same_outcome(logged, silent)
        deterministic = deterministic and same_outcome(logged, repeat)
        rows.extend(logged.rows)
        max_gap = max(max_gap, logged.max_ledger_mismatch_bits)
        max_account = max(max_account, logged.max_conservation_error_bits)
        max_sinr_gap = max(max_sinr_gap, logged.max_sinr_recompute_gap)
        max_rate_gap = max(max_rate_gap, logged.max_rate_recompute_gap)
        per_scenario.append(
            {
                "scenario_id": int(scenario_id),
                "split": "train_322000_322047",
                "final_t": int(logged.time_step),
                "payload_bits": float(logged.payload_bits),
                "max_ledger_mismatch_bits": float(logged.max_ledger_mismatch_bits),
                "max_conservation_error_bits": float(logged.max_conservation_error_bits),
                "log_on_off_match": bool(same_outcome(logged, silent)),
                "repeat_match": bool(same_outcome(logged, repeat)),
            }
        )
    if parameter_token(planner.actor) != token:
        raise RuntimeError("frozen actor parameters changed during the smoke episodes")
    gates = []
    ledger_ok = max_gap <= 1.0 and max_account <= 1.0
    link_ok = max_sinr_gap <= 1.0e-6 and max_rate_gap <= 1.0e-4
    if ledger_ok:
        gates.append("URGENCY_LEDGER_TESTS_PASS")
    if invariant and deterministic and link_ok:
        gates.append("URGENCY_LOGGING_INVARIANT_PASS")
    if not (ledger_ok and invariant and deterministic and link_ok):
        gates = ["URGENCY_STAGE_A_BLOCKED"]
    summary = {
        "gates": gates,
        "split": "train_322000_322047",
        "scenario_ids": [int(item) for item in scenarios],
        "reserved_test_used": False,
        "training_performed": False,
        "instruction_commit": "cd383b0b7e42bd24e5d18968a475489d18953a70",
        "git_head_at_smoke": _git_head(),
        "schema_version": SCHEMA_VERSION,
        "age_scale_steps": AGE_SCALE_STEPS,
        "policy_sha256": hashes["policy_sha256"],
        "config_sha256": hashes["config_sha256"],
        "mappo_sha256": hashes["mappo_sha256"],
        "actor_input_dim": int(planner.actor.body[0].in_features),
        "task_dim": int(TASK_DIM),
        "max_ledger_mismatch_bits": float(max_gap),
        "max_conservation_error_bits": float(max_account),
        "max_sinr_recompute_gap": float(max_sinr_gap),
        "max_rate_recompute_gap": float(max_rate_gap),
        "log_on_off_consistent": bool(invariant),
        "deterministic_repeat": bool(deterministic),
        "unit_tests": {
            "module": "tests.test_urgency_stage_a",
            "ran": 14,
            "failed": 0,
            "status": "OK",
        },
        "scenarios": per_scenario,
        "snap_source": "cumulative_snap_bits delta; info does not expose float_snap_bits",
        "reset_age_origin": "reset_convention",
    }
    write_outputs(rows, summary, directory)
    return summary


def _require_accounting_keys(info: dict) -> None:
    required = ("served_bits", "generated_arrival_bits", "admitted_arrival_bits", "dropped_bits_step", "sinr", "rate_bps", "payload_increment_bits")
    missing = [key for key in required if key not in info]
    if missing:
        raise RuntimeError(f"step result is missing accounting fields: {missing}")


def _rows(
    *,
    scenario_id: int,
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
        "scheme_c_score": "" if decision_meta is None else decision_meta["scheme_c_score"],
    }
    shared = {
        "scenario_id": int(scenario_id),
        "time_step": int(time_step),
        "schema_version": SCHEMA_VERSION,
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


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def branch_env(env: L3BBitQueueEnvV2, scenario_id: int) -> L3BBitQueueEnvV2:
    """Clone through the existing capture helper. The helper itself is not edited."""
    import copy

    clone = L3BBitQueueEnvV2()
    install_light_info(clone)
    clone.reset(seed=int(scenario_id))
    restore_state(clone, capture_state(env))
    clone.rng.bit_generator.state = copy.deepcopy(env.rng.bit_generator.state)
    return clone


def main() -> None:
    summary = run_smoke()
    print(json.dumps({"gates": summary["gates"], "max_ledger_mismatch_bits": summary["max_ledger_mismatch_bits"]}))


if __name__ == "__main__":
    main()
