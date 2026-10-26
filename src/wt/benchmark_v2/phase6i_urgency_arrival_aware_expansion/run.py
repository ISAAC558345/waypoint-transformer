"""Compare zero-arrival and real-arrival labels at four decision times."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import limit_threads
from wt.benchmark_v2.phase6ib_highlevel_dataset.jobs import _frozen_actor
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import PRIMARY_TIMES
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.protocol import OUTPUT as D0_OUTPUT
from wt.benchmark_v2.phase6i_urgency_highlevel_pilot.shards import dataset_fingerprint
from wt.benchmark_v2.phase6i_urgency_highlevel_t30_pilot.protocol import OUTPUT as D1_OUTPUT
from wt.benchmark_v2.phase6i_urgency_stage_a.ledger import AdmissionAgeLedger
from wt.benchmark_v2.phase6i_urgency_tiebreak_closedloop.protocol import (
    ADAPTER_CONFIG_SHA256,
    EPOCH6_CKPT,
    EPOCH6_SHA256,
    MAPPO_SHA256,
    SCHEME_C_CKPT,
    SCHEME_C_SHA256,
    OUTPUT as D3_OUTPUT,
    config_sha256,
    primary_checkpoint,
)
from wt.benchmark_v2.phase6i_urgency_utility_rule.protocol import OUTPUT as D2_OUTPUT

from .analyze import quartile_labels, rank_state, stratum_summary
from .labels import expected_generated_bits, score_label
from .protocol import (
    DECISION_TIMES,
    D5_GATE,
    EPS_BITS,
    FROZEN_SHARD_COMMIT,
    GATE_BLOCKED,
    GATE_COMPLETE,
    GENERATOR,
    INSTRUCTION_COMMIT,
    LABEL_SEED,
    MAX_CANDIDATES,
    MAX_SECONDS,
    MAX_STATE_SECONDS,
    MAX_STUDY_ROLLOUTS,
    MAX_VERIFICATION_ROLLOUTS,
    OUTPUT,
    SELECTED_K,
    assert_same_horizon,
    choose_gate,
    horizon_for,
    project_seconds,
    repeat_state,
    scenario_effect,
    scenarios,
    sum_tolerance,
    time_effect,
)
from .replay import replay_scenario
from .sampling import select_state

D4_OUTPUT = Path(__file__).resolve().parents[4] / "results" / "phase6i_urgency_tiebreak_single_intervention"
D5_OUTPUT = Path(__file__).resolve().parents[4] / "results" / "phase6i_urgency_arrival_aware_labels"
PLANNER_PATH = Path(__file__).resolve().parents[1] / "phase6iaa_formal_planner" / "planner.py"

SELECTED_FIELDS = (
    "scenario_id",
    "time_step",
    "horizon",
    "candidate_id",
    "assignment",
    "j15_bits",
    "sampling_score",
    "sampling_source",
    "utility_rank",
    "tie_group_id",
    "selection",
    "stratum",
    "filled",
    "idle_count",
    "urgency_start_assigned",
    "state_hash",
    "pool_hash",
    "subset_hash",
    "row_index",
)
LABEL_FIELDS = SELECTED_FIELDS + (
    "zero_payload_bits",
    "zero_urgent_service_bits",
    "zero_dropped_bits",
    "zero_admitted_bits",
    "zero_final_queue_bits",
    "zero_initial_queue_bits",
    "zero_served_bits",
    "zero_generated_bits",
    "zero_snap_bits",
    "zero_final_age_weighted_backlog_bits",
    "zero_ledger_gap_bits",
    "zero_accounting_error_bits",
    "zero_payload_served_gap_bits",
    "zero_payload_gap_vs_j15_bits",
    "real_payload_bits",
    "real_urgent_service_bits",
    "real_dropped_bits",
    "real_admitted_bits",
    "real_final_queue_bits",
    "real_initial_queue_bits",
    "real_served_bits",
    "real_generated_bits",
    "real_snap_bits",
    "real_final_age_weighted_backlog_bits",
    "real_ledger_gap_bits",
    "real_accounting_error_bits",
    "real_payload_served_gap_bits",
    "payload_delta_bits",
    "urgent_delta_bits",
    "dropped_delta_bits",
    "final_queue_delta_bits",
    "arrival_account_gap_bits",
    "payload_quartile",
    "urgency_start_quartile",
)
COMPARE_FIELDS = (
    "candidate_id",
    "assignment",
    "zero_payload_bits",
    "zero_urgent_service_bits",
    "real_payload_bits",
    "real_urgent_service_bits",
    "zero_generated_bits",
    "real_generated_bits",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _join(values) -> str:
    return "|".join(str(int(item)) if float(item).is_integer() else f"{float(item):.8g}" for item in values)


def _csv(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.16g}"
    return value


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv(row.get(key, "")) for key in fields})
        handle.flush()


def _watched() -> dict:
    paths = {
        "d0_labels": D0_OUTPUT / "labels.csv",
        "d1_labels": D1_OUTPUT / "labels.csv",
        "d2_summary": D2_OUTPUT / "summary.json",
        "d3_summary": D3_OUTPUT / "summary.json",
        "d4_summary": D4_OUTPUT / "summary.json",
        "d5_summary": D5_OUTPUT / "summary.json",
        "formal_planner": PLANNER_PATH,
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _checkpoint_hashes() -> dict:
    found = {
        "mappo_v2_seed0": _sha256(primary_checkpoint()),
        "scheme_c": _sha256(SCHEME_C_CKPT),
        "epoch6": _sha256(EPOCH6_CKPT),
        "adapter_config": config_sha256(),
    }
    expected = {
        "mappo_v2_seed0": MAPPO_SHA256,
        "scheme_c": SCHEME_C_SHA256,
        "epoch6": EPOCH6_SHA256,
        "adapter_config": ADAPTER_CONFIG_SHA256,
    }
    if found != expected:
        raise RuntimeError("frozen checkpoint or adapter config hash changed")
    return found


def _state_record(replay: dict, time_step: int) -> dict:
    state = next(item for item in replay["states"] if int(item["time_step"]) == int(time_step))
    if int(time_step) in PRIMARY_TIMES:
        if not state["shard_checked"] or not state["state_hash_match"] or not state["checksum_state_match"]:
            raise RuntimeError(f"t={int(time_step)} replay does not match the frozen shard for {state['scenario_id']}")
        if not state["ledger_restore_independent"]:
            raise RuntimeError(f"ledger snapshot is not independently restorable at t={int(time_step)}")
    else:
        if state["shard_checked"]:
            raise RuntimeError("t=45 capture tried to read a frozen shard")
        if not state["ledger_restore_independent"]:
            raise RuntimeError("t=45 ledger snapshot is not independently restorable")
    return state


def _score_rows(rows: list[dict], state: dict, actor) -> list[dict]:
    horizon = horizon_for(int(state["time_step"]))
    expected = expected_generated_bits(state["env_snap"], int(state["scenario_id"]), int(state["time_step"]), horizon)
    tol = float(state["tol_bits"])
    total_tol = sum_tolerance(tol, int(np.asarray(state["env_snap"]["q_bits"]).size))
    scored = []
    for row in rows:
        if int(row["horizon"]) != int(horizon):
            raise RuntimeError("candidate horizon does not match the decision time")
        zero = score_label(
            state["env_snap"],
            state["ledger_snap"],
            actor,
            row["assignment_vector"],
            scenario_id=int(row["scenario_id"]),
            zero_arrivals=True,
            horizon=horizon,
        )
        real = score_label(
            state["env_snap"],
            state["ledger_snap"],
            actor,
            row["assignment_vector"],
            scenario_id=int(row["scenario_id"]),
            zero_arrivals=False,
            horizon=horizon,
        )
        if abs(float(zero["initial_queue_bits"]) - float(real["initial_queue_bits"])) > tol:
            raise RuntimeError("the two arms did not start from the same queue")
        gap = None
        if row["sampling_source"] == "stored_j15":
            if row["j15_bits"] is None:
                raise RuntimeError("stored J15 sampling is missing j15_bits")
            gap = float(zero["payload_bits"]) - float(row["j15_bits"])
            if abs(gap) > tol:
                raise RuntimeError(
                    f"zero-arrival payload does not reproduce j15 for scenario {row['scenario_id']} "
                    f"t={row['time_step']} candidate {row['candidate_id']}: gap {gap}"
                )
        elif row["j15_bits"] is not None:
            raise RuntimeError("additive sampling stored a legacy j15 value")
        account = (
            (float(real["generated_bits"]) - float(zero["generated_bits"]))
            - (float(real["payload_bits"]) - float(zero["payload_bits"]))
            - (float(real["dropped_bits"]) - float(zero["dropped_bits"]))
            - (float(real["final_queue_bits"]) - float(zero["final_queue_bits"]))
            - (float(real["snap_bits"]) - float(zero["snap_bits"]))
        )
        scored.append(
            {
                **{key: value for key, value in row.items() if key != "assignment_vector"},
                "expected_generated_bits": float(expected),
                "zero_payload_bits": float(zero["payload_bits"]),
                "zero_urgent_service_bits": float(zero["urgent_service_bits"]),
                "zero_dropped_bits": float(zero["dropped_bits"]),
                "zero_admitted_bits": float(zero["admitted_bits"]),
                "zero_final_queue_bits": float(zero["final_queue_bits"]),
                "zero_initial_queue_bits": float(zero["initial_queue_bits"]),
                "zero_served_bits": _join(zero["served_bits"]),
                "zero_generated_bits": float(zero["generated_bits"]),
                "zero_snap_bits": float(zero["snap_bits"]),
                "zero_final_age_weighted_backlog_bits": float(zero["final_age_weighted_backlog_bits"]),
                "zero_ledger_gap_bits": float(zero["ledger_gap_bits"]),
                "zero_accounting_error_bits": float(zero["accounting_error_bits"]),
                "zero_payload_served_gap_bits": float(zero["payload_served_gap_bits"]),
                "zero_payload_gap_vs_j15_bits": None if gap is None else float(gap),
                "real_payload_bits": float(real["payload_bits"]),
                "real_urgent_service_bits": float(real["urgent_service_bits"]),
                "real_dropped_bits": float(real["dropped_bits"]),
                "real_admitted_bits": float(real["admitted_bits"]),
                "real_final_queue_bits": float(real["final_queue_bits"]),
                "real_initial_queue_bits": float(real["initial_queue_bits"]),
                "real_served_bits": _join(real["served_bits"]),
                "real_generated_bits": float(real["generated_bits"]),
                "real_snap_bits": float(real["snap_bits"]),
                "real_final_age_weighted_backlog_bits": float(real["final_age_weighted_backlog_bits"]),
                "real_ledger_gap_bits": float(real["ledger_gap_bits"]),
                "real_accounting_error_bits": float(real["accounting_error_bits"]),
                "real_payload_served_gap_bits": float(real["payload_served_gap_bits"]),
                "payload_delta_bits": float(real["payload_bits"]) - float(zero["payload_bits"]),
                "urgent_delta_bits": float(real["urgent_service_bits"]) - float(zero["urgent_service_bits"]),
                "dropped_delta_bits": float(real["dropped_bits"]) - float(zero["dropped_bits"]),
                "final_queue_delta_bits": float(real["final_queue_bits"]) - float(zero["final_queue_bits"]),
                "arrival_account_gap_bits": float(account),
                "sum_tolerance_bits": float(total_tol),
            }
        )
    return scored


def _annotate(rows: list[dict]) -> None:
    payload_bands = quartile_labels([float(row["zero_payload_bits"]) for row in rows])
    urgency_bands = quartile_labels([float(row["urgency_start_assigned"]) for row in rows])
    for row, payload_band, urgency_band in zip(rows, payload_bands, urgency_bands):
        row["payload_quartile"] = payload_band
        row["urgency_start_quartile"] = urgency_band


def _problems(rows: list[dict], state: dict) -> list[str]:
    reasons = []
    tol = float(state["tol_bits"])
    total_tol = sum_tolerance(tol, int(np.asarray(state["env_snap"]["q_bits"]).size))
    time_step = int(state["time_step"])
    source = str(rows[0]["sampling_source"]) if rows else ""
    if time_step in PRIMARY_TIMES and source != "stored_j15":
        reasons.append("a 15-step state did not sample from stored j15")
    if time_step not in PRIMARY_TIMES and source != "additive_g1_rate":
        reasons.append("t=45 did not sample from the additive G1_RATE rank")
    if any(row["zero_payload_gap_vs_j15_bits"] is not None and abs(float(row["zero_payload_gap_vs_j15_bits"])) > tol for row in rows):
        reasons.append("a zero-arrival payload label does not reproduce j15_bits")
    if time_step not in PRIMARY_TIMES and any(row["zero_payload_gap_vs_j15_bits"] is not None for row in rows):
        reasons.append("t=45 was compared with a legacy j15 label")
    if any(float(row["zero_generated_bits"]) > tol for row in rows):
        reasons.append("a zero-arrival label still generated arrivals")
    generated = [float(row["real_generated_bits"]) for row in rows]
    if generated and max(generated) - min(generated) > tol:
        reasons.append("real-arrival copies did not share the exogenous arrival amount")
    if rows and any(abs(float(row["real_generated_bits"]) - float(row["expected_generated_bits"])) > tol for row in rows):
        reasons.append("real generated bits do not match arrivals_at over the horizon")
    if float(rows[0]["expected_generated_bits"]) <= tol if rows else False:
        if any(abs(float(row["payload_delta_bits"])) > tol or abs(float(row["urgent_delta_bits"])) > tol for row in rows):
            reasons.append("arms diverged in a window with no exogenous arrivals")
    for key in (
        "zero_ledger_gap_bits",
        "real_ledger_gap_bits",
        "zero_accounting_error_bits",
        "real_accounting_error_bits",
        "zero_payload_served_gap_bits",
        "real_payload_served_gap_bits",
    ):
        if any(float(row[key]) > tol for row in rows):
            reasons.append(f"{key} exceeded 1 bit")
            break
    if any(abs(float(row["arrival_account_gap_bits"])) > float(row["sum_tolerance_bits"]) for row in rows):
        reasons.append("arrival bits do not split into served, dropped, final queue, and snap")
    if any(abs(float(row["real_generated_bits"]) - float(row["real_admitted_bits"]) - float(row["real_dropped_bits"])) > total_tol for row in rows):
        reasons.append("real generated bits do not equal admitted plus dropped")
    if any(abs(float(row["zero_generated_bits"]) - float(row["zero_admitted_bits"]) - float(row["zero_dropped_bits"])) > total_tol for row in rows):
        reasons.append("zero generated bits do not equal admitted plus dropped")
    if float(state["ledger_gap_bits"]) > tol:
        reasons.append("the replay ledger exceeded 1 bit")
    if len({int(row["candidate_id"]) for row in rows}) != len(rows):
        reasons.append("candidate ids are not unique")
    if any(str(row["subset_hash"]) != str(rows[0]["subset_hash"]) or str(row["state_hash"]) != str(rows[0]["state_hash"]) for row in rows):
        reasons.append("rows inside one state do not share the candidate subset and snapshot")
    return reasons


def _effect_name(rows: list[dict], problems: list[str]) -> str:
    if any("arms diverged" in item for item in problems):
        return "arms_diverged"
    if rows and float(rows[0]["expected_generated_bits"]) <= float(EPS_BITS):
        return "arms_match"
    ranking = rank_state(rows)
    return scenario_effect(
        urgent_reversal=float(ranking["urgent_inversion"]["inversion_rate"]),
        payload_reversal=float(ranking["payload_inversion"]["inversion_rate"]),
        payload_top1_within_one_bit=bool(ranking["payload_top1_within_one_bit"]),
        urgent_top1_within_one_bit=bool(ranking["urgent_top1_within_one_bit"]),
    )


def _state_summary(scenario: int, replay: dict, state: dict, pool: dict, scored: list[dict], elapsed: float, score_seconds: float) -> dict:
    ranking = rank_state(scored)
    effect = _effect_name(scored, _problems(scored, state))
    q_bits = np.asarray(state["env_snap"]["q_bits"], dtype=np.float64)
    ledger = AdmissionAgeLedger(int(q_bits.size))
    ledger.restore(state["ledger_snap"])
    view = ledger.view(int(state["time_step"]), q_bits, float(state["tol_bits"]))
    return {
        "scenario_id": int(scenario),
        "time_step": int(state["time_step"]),
        "horizon": horizon_for(int(state["time_step"])),
        "seconds": float(elapsed),
        "score_seconds": float(score_seconds),
        "state_hash": str(state["state_hash"]),
        "shard_checked": bool(state["shard_checked"]),
        "pool_hash": pool["pool_hash"],
        "pool_count": int(pool["pool_count"]),
        "subset_hash": pool["subset_hash"],
        "checksum_candidates": pool["checksum_candidates"],
        "sampling_source": pool["sampling_source"],
        "expected_generated_bits": float(scored[0]["expected_generated_bits"]),
        "q_bits": [float(item) for item in q_bits],
        "urgency": [float(item) for item in view["urgency"]],
        "oldest_wait_age": [float(item) for item in view["oldest_wait_age"]],
        "replay_seconds": float(replay["seconds"]),
        "replay_ledger_gap_bits": float(replay["ledger_gap_bits"]),
        "replay_accounting_error_bits": float(replay["accounting_error_bits"]),
        "ranking": ranking,
        "effect": effect,
        "payload_strata": stratum_summary(scored, "payload_quartile"),
        "urgency_start_strata": stratum_summary(scored, "urgency_start_quartile"),
        "idle_strata": stratum_summary(scored, "idle_count"),
        "problems": _problems(scored, state),
    }


def _time_rollup(state_rows: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = {}
    for row in state_rows:
        grouped.setdefault(int(row["time_step"]), []).append(row)
    output = []
    for time_step in DECISION_TIMES:
        chosen = grouped.get(int(time_step), [])
        if not chosen:
            continue
        horizon = assert_same_horizon([int(row["time_step"]) for row in chosen])
        effects = [str(row["effect"]) for row in chosen]

        def _mean(name: str) -> float | None:
            values = [row["ranking"][name] for row in chosen if row["ranking"][name] is not None]
            if not values:
                return None
            return float(np.mean(values))

        output.append(
            {
                "time_step": int(time_step),
                "horizon": int(horizon),
                "states": int(len(chosen)),
                "effect": time_effect(effects),
                "scenario_effects": effects,
                "expected_generated_bits": float(chosen[0]["expected_generated_bits"]),
                "payload_spearman_mean": _mean("payload_spearman"),
                "urgent_spearman_mean": _mean("urgent_spearman"),
                "payload_urgent_spearman_mean": _mean("payload_urgent_spearman"),
                "payload_inversion_mean": float(np.mean([row["ranking"]["payload_inversion"]["inversion_rate"] for row in chosen])),
                "urgent_inversion_mean": float(np.mean([row["ranking"]["urgent_inversion"]["inversion_rate"] for row in chosen])),
                "payload_top1_changed_count": int(sum(bool(row["ranking"]["payload_top1_changed"]) for row in chosen)),
                "urgent_top1_changed_count": int(sum(bool(row["ranking"]["urgent_top1_changed"]) for row in chosen)),
                "payload_top1_regret_mean_bits": float(np.mean([row["ranking"]["payload_top1_regret_bits"] for row in chosen])),
                "urgent_top1_regret_mean_bits": float(np.mean([row["ranking"]["urgent_top1_regret_bits"] for row in chosen])),
                "payload_delta_mean_bits": float(np.mean([row["ranking"]["payload_delta_mean_bits"] for row in chosen])),
                "urgent_delta_mean_bits": float(np.mean([row["ranking"]["urgent_delta_mean_bits"] for row in chosen])),
                "dropped_delta_mean_bits": float(np.mean([row["ranking"]["dropped_delta_mean_bits"] for row in chosen])),
                "final_queue_delta_mean_bits": float(np.mean([row["ranking"]["final_queue_delta_mean_bits"] for row in chosen])),
                "pareto_count_min": int(min(row["ranking"]["pareto"]["count"] for row in chosen)),
                "pareto_count_max": int(max(row["ranking"]["pareto"]["count"] for row in chosen)),
                "pooled_with_other_horizons": False,
            }
        )
    return output


def _verify(actor, label_rows: list[dict]) -> dict:
    scenario, time_step = repeat_state()
    stored = [row for row in label_rows if int(row["scenario_id"]) == int(scenario) and int(row["time_step"]) == int(time_step)]
    if len(stored) != int(MAX_CANDIDATES):
        return {"matched": False, "reason": "pre-registered state is missing from the study labels", "rollouts": 0}
    replay = replay_scenario(int(scenario))
    state = _state_record(replay, int(time_step))
    chosen, pool = select_state(int(scenario), state)
    if [int(row["candidate_id"]) for row in chosen] != [int(row["candidate_id"]) for row in stored]:
        return {"matched": False, "reason": "candidate order changed", "rollouts": 0, "scenario_id": int(scenario), "time_step": int(time_step)}
    if str(pool["subset_hash"]) != str(stored[0]["subset_hash"]) or str(state["state_hash"]) != str(stored[0]["state_hash"]):
        return {"matched": False, "reason": "state or subset hash changed", "rollouts": 0, "scenario_id": int(scenario), "time_step": int(time_step)}
    scored = _score_rows(chosen, state, actor)
    gap = 0.0
    for left, right in zip(stored, scored):
        for key in COMPARE_FIELDS:
            if key in {"candidate_id", "assignment"}:
                if str(left[key]) != str(right[key]):
                    gap = max(gap, float(EPS_BITS) + 1.0)
            else:
                gap = max(gap, abs(float(left[key]) - float(right[key])))
    return {
        "matched": bool(gap <= float(state["tol_bits"])),
        "max_abs_label_gap_bits": float(gap),
        "rollouts": int(2 * len(scored)),
        "scenario_id": int(scenario),
        "time_step": int(time_step),
        "state_hash": str(state["state_hash"]),
        "subset_hash": str(pool["subset_hash"]),
        "reason": "" if gap <= float(state["tol_bits"]) else "repeated labels differ by more than 1 bit",
    }


def execute() -> dict:
    limit_threads()
    before = _watched()
    fingerprint_before = dataset_fingerprint(scenarios())
    hashes = _checkpoint_hashes()
    actor = _frozen_actor()
    selected_rows: list[dict] = []
    label_rows: list[dict] = []
    state_rows: list[dict] = []
    budget_ok = True
    crashed = False
    failure = ""
    failure_trace = ""
    first_seconds = None
    first_replay = None
    first_score = None
    estimate = None
    rollouts = 0
    verification = {"matched": False, "rollouts": 0, "reason": "not run"}
    started_all = time.perf_counter()
    try:
        halt = False
        for scenario in scenarios():
            if halt:
                break
            print(f"scenario {scenario} replay", flush=True)
            replay_started = time.perf_counter()
            replay = replay_scenario(int(scenario))
            replay_seconds = float(time.perf_counter() - replay_started)
            print(f"scenario {scenario} replay {replay_seconds:.3f}s", flush=True)
            for time_step in DECISION_TIMES:
                spent_now = float(time.perf_counter() - started_all)
                if spent_now > float(MAX_SECONDS):
                    budget_ok = False
                    failure = f"elapsed {spent_now:.3f}s exceeds {MAX_SECONDS:.0f}s before t={int(time_step)}"
                    halt = True
                    break
                if first_seconds is not None and estimate is not None and float(estimate) > float(MAX_SECONDS):
                    budget_ok = False
                    failure = f"projected total {estimate:.3f}s exceeds {MAX_SECONDS:.0f}s"
                    halt = True
                    break
                state = _state_record(replay, int(time_step))
                chosen, pool = select_state(int(scenario), state)
                for row in chosen:
                    row["horizon"] = horizon_for(int(time_step))
                selected_rows.extend({key: row.get(key, "") for key in SELECTED_FIELDS} for row in chosen)
                _write_csv(OUTPUT / "selected_candidates.csv", selected_rows, SELECTED_FIELDS)
                print(f"scenario {scenario} t={time_step} scoring {len(chosen)} candidates", flush=True)
                score_started = time.perf_counter()
                scored = _score_rows(chosen, state, actor)
                score_seconds = float(time.perf_counter() - score_started)
                _annotate(scored)
                label_rows.extend(scored)
                _write_csv(OUTPUT / "labels.csv", label_rows, LABEL_FIELDS)
                rollouts += 2 * len(scored)
                wall = score_seconds + (replay_seconds if int(time_step) == int(DECISION_TIMES[0]) else 0.0)
                print(f"scenario {scenario} t={time_step} score {score_seconds:.3f}s wall {wall:.3f}s", flush=True)
                state_rows.append(_state_summary(int(scenario), replay, state, pool, scored, wall, score_seconds))
                if state_rows[-1]["problems"]:
                    raise RuntimeError("; ".join(state_rows[-1]["problems"]))
                if first_seconds is None:
                    first_seconds = float(wall)
                    first_replay = float(replay_seconds)
                    first_score = float(score_seconds)
                    estimate = project_seconds(first_replay, first_score)
                    print(f"projected total {estimate:.3f}s", flush=True)
                    if float(wall) > float(MAX_STATE_SECONDS) or float(score_seconds) > float(MAX_STATE_SECONDS):
                        budget_ok = False
                        failure = f"first state {wall:.3f}s exceeds {MAX_STATE_SECONDS:.0f}s"
                        halt = True
                        break
                    if float(estimate) > float(MAX_SECONDS):
                        budget_ok = False
                        failure = f"projected total {estimate:.3f}s exceeds {MAX_SECONDS:.0f}s"
                        halt = True
                        break
                elif float(wall) > float(MAX_STATE_SECONDS):
                    budget_ok = False
                    failure = f"state {scenario} t={time_step} took {wall:.3f}s"
                    halt = True
                    break
        if budget_ok and not failure and len(state_rows) == len(scenarios()) * len(DECISION_TIMES):
            spent = float(time.perf_counter() - started_all)
            repeat_cost = float(first_replay or 0.0) + float(first_score or 0.0)
            if spent + repeat_cost > float(MAX_SECONDS):
                budget_ok = False
                failure = f"verification rerun would exceed {MAX_SECONDS:.0f}s"
            else:
                print("verification rerun", flush=True)
                verification = _verify(actor, label_rows)
                if int(verification["rollouts"]) > int(MAX_VERIFICATION_ROLLOUTS):
                    raise RuntimeError("verification used more than one state")
    except Exception as exc:
        crashed = True
        budget_ok = False
        failure = f"{type(exc).__name__}: {exc}"
        failure_trace = traceback.format_exc()
    elapsed = float(time.perf_counter() - started_all)
    after = _watched()
    fingerprint_after = dataset_fingerprint(scenarios())
    reasons = []
    if failure:
        reasons.append(failure)
    if before != after:
        reasons.append("D0-D5 results or the formal planner file changed")
    if fingerprint_before != fingerprint_after:
        reasons.append("the frozen dataset files changed")
    if rollouts > int(MAX_STUDY_ROLLOUTS):
        reasons.append("study rollout count exceeded 8192")
    if int(verification.get("rollouts", 0)) > int(MAX_VERIFICATION_ROLLOUTS):
        reasons.append("verification rollout count exceeded one state")
    for row in state_rows:
        reasons.extend(str(item) for item in row["problems"])
    if verification.get("reason") and str(verification.get("reason")) != "not run":
        reasons.append(str(verification["reason"]))
    complete = len(state_rows) == len(scenarios()) * len(DECISION_TIMES) and all(
        int(row["ranking"]["count"]) == int(MAX_CANDIDATES) for row in state_rows
    )
    reproducible = bool(verification.get("matched"))
    by_time = _time_rollup(state_rows)
    time_effects = {int(row["time_step"]): str(row["effect"]) for row in by_time}
    mechanical_ok = complete and not reasons and not failure and before == after and fingerprint_before == fingerprint_after
    gate = choose_gate(
        mechanical_ok=mechanical_ok,
        budget_ok=budget_ok and not failure,
        reproducible=reproducible,
        time_effects=time_effects,
    )
    if failure or not budget_ok or crashed:
        gate = GATE_BLOCKED
    if gate == GATE_COMPLETE:
        next_step = "下一步只提出如何将双目标标签转为高层训练目标的决策文件。仍需研究负责人明确 payload 损失预算，不能擅自选择 lambda_H。不训练 HGNN，不构建全量数据集。"
    elif gate == GATE_BLOCKED:
        next_step = "不要扩大预算，也不要训练。先处理完整性或时间失败。"
    else:
        next_step = "不要把四个时刻合成一个训练目标，也不要训练 HGNN。t=15 与 t=30 的排序变化可以写入下一份决策文件分开讨论，但 lambda_H 和 payload 损失预算仍未确定。"
    return {
        "gate": gate,
        "reasons": reasons,
        "failure_trace": failure_trace,
        "time_effects": time_effects,
        "by_time": by_time,
        "next_step": next_step,
        "learned_policy": False,
        "trained_hgnn": False,
        "trained_mappo": False,
        "formal_system_modified": False,
        "lambda_H": None,
        "delta_payload_max": None,
        "d5_gate_unchanged": D5_GATE,
        "instruction_commit": INSTRUCTION_COMMIT,
        "frozen_shard_commit": FROZEN_SHARD_COMMIT,
        "label_seed": int(LABEL_SEED),
        "seed_sequence": "[605, scenario_id, time_step]",
        "decision_times": list(DECISION_TIMES),
        "horizons": {str(item): horizon_for(item) for item in DECISION_TIMES},
        "horizons_pooled": False,
        "generator": GENERATOR,
        "selected_k": int(SELECTED_K),
        "candidates_per_state": int(MAX_CANDIDATES),
        "study_rollouts": int(rollouts),
        "verification_rollouts": int(verification.get("rollouts", 0)),
        "max_study_rollouts": int(MAX_STUDY_ROLLOUTS),
        "max_verification_rollouts": int(MAX_VERIFICATION_ROLLOUTS),
        "first_state_seconds": first_seconds,
        "first_replay_seconds": first_replay,
        "first_score_seconds": first_score,
        "estimated_seconds": estimate,
        "elapsed_seconds": elapsed,
        "max_seconds": float(MAX_SECONDS),
        "max_state_seconds": float(MAX_STATE_SECONDS),
        "scenarios": list(scenarios()),
        "repeat_state": {"scenario_id": int(repeat_state()[0]), "time_step": int(repeat_state()[1])},
        "verification": verification,
        "states": state_rows,
        "checkpoint_sha256": hashes,
        "watched_sha256": before,
        "watched_unchanged": before == after,
        "dataset_unchanged": fingerprint_before == fingerprint_after,
        "arrival_pairing": "arrivals_at is a deterministic function of time; zero-arrival disables it only on the scratch copy",
        "selected_rows": selected_rows,
        "label_rows": label_rows,
    }


def write_results(summary: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT / "selected_candidates.csv", summary["selected_rows"], SELECTED_FIELDS)
    _write_csv(OUTPUT / "labels.csv", summary["label_rows"], LABEL_FIELDS)
    public = {key: value for key, value in summary.items() if key not in {"selected_rows", "label_rows"}}
    (OUTPUT / "summary.json").write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")
    schema = {
        "meaning": "Each selected candidate receives one zero-arrival label and one real-arrival label from the same snapshot. t=45 uses 3 steps and is not pooled with the 15-step labels.",
        "arms": {
            "zero_arrival": "scratch copy disables arrivals and arrivals_at. At t=0, 15, and 30 the payload is checked against frozen j15_bits. At t=45 this is a 3-step diagnostic, not a legacy J15 label.",
            "real_arrival": "scratch copy keeps formal arrivals_at(t). The result is an offline label and is not fed back into the observation.",
        },
        "sampling": {
            "seed": int(LABEL_SEED),
            "seed_sequence": "[605, scenario_id, time_step]",
            "t0_t15_t30": "32 stable top stored J15 labels plus 96 utility-rank strata",
            "t45": "32 plus 96 strata from the deterministic additive G1_RATE score. That score is not a label.",
            "saved_before_rollout": True,
        },
        "horizons": {str(item): horizon_for(item) for item in DECISION_TIMES},
        "horizons_pooled": False,
        "eps_bits": float(EPS_BITS),
        "summed_identity_tolerance": "per-buoy tolerance times the buoy count",
        "top1": "maximum label, with a 1-bit tie broken by the smaller candidate id",
        "inversion": "a pair counts only when both labels differ by more than 1 bit and their order disagrees",
        "kendall_tau": "(concordant - discordant) / compared pairs, after the same 1-bit exclusion",
        "quiet_window": "a decision whose formal arrivals_at sum is within 1 bit of zero is reported separately and is not an arrival-ranking result",
        "effect_thresholds": {
            "material_urgent_reversal": 0.05,
            "small_reversal": 0.01,
        },
        "verification_state": summary["repeat_state"],
        "strata_are_descriptive": True,
        "lambda_H": None,
        "delta_payload_max": None,
        "held_out_splits_read": False,
        "instruction_commit": INSTRUCTION_COMMIT,
        "fields": {
            "selected_candidates.csv": list(SELECTED_FIELDS),
            "labels.csv": list(LABEL_FIELDS),
        },
    }
    (OUTPUT / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_labels(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            row["scenario_id"] = int(row["scenario_id"])
            row["time_step"] = int(row["time_step"])
            row["candidate_id"] = int(row["candidate_id"])
            for key in COMPARE_FIELDS:
                if key in {"candidate_id", "assignment"}:
                    continue
                row[key] = float(row[key])
            rows.append(row)
    return rows


def _next_step(gate: str) -> str:
    if gate == GATE_COMPLETE:
        return "下一步只提出如何将双目标标签转为高层训练目标的决策文件。仍需研究负责人明确 payload 损失预算，不能擅自选择 lambda_H。不训练 HGNN，不构建全量数据集。"
    if gate == GATE_BLOCKED:
        return "不要扩大预算，也不要训练。先处理完整性或时间失败。"
    return "不要把四个时刻合成一个训练目标，也不要训练 HGNN。t=15 与 t=30 的排序变化可以写入下一份决策文件分开讨论，但 lambda_H 和 payload 损失预算仍未确定。"


def refresh_verification() -> dict:
    """Score the pre-registered state again. The 8,192 study rollouts stay on disk."""
    limit_threads()
    started = time.perf_counter()
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    labels = _load_labels(OUTPUT / "labels.csv")
    actor = _frozen_actor()
    verification = _verify(actor, labels)
    verification["seconds"] = float(time.perf_counter() - started)
    verification["note"] = "The first call stopped before scoring because the sampled row had no horizon. This pass scores the same 128 candidates once."
    summary["verification"] = verification
    summary["verification_rollouts"] = int(verification["rollouts"])
    summary["elapsed_seconds"] = float(summary["elapsed_seconds"]) + float(verification["seconds"])
    reasons = []
    if not verification.get("matched"):
        reasons.append(str(verification.get("reason") or "verification failed"))
    reasons.extend(str(item) for state in summary["states"] for item in state["problems"])
    if not summary.get("watched_unchanged"):
        reasons.append("D0-D5 results or the formal planner file changed")
    if not summary.get("dataset_unchanged"):
        reasons.append("the frozen dataset files changed")
    summary["reasons"] = reasons
    summary["failure_trace"] = ""
    time_effects = {int(key): str(value) for key, value in summary["time_effects"].items()}
    mechanical_ok = not reasons and len(summary["states"]) == len(scenarios()) * len(DECISION_TIMES)
    gate = choose_gate(
        mechanical_ok=mechanical_ok,
        budget_ok=True,
        reproducible=bool(verification.get("matched")),
        time_effects=time_effects,
    )
    summary["gate"] = gate
    summary["time_effects"] = {str(key): value for key, value in time_effects.items()}
    summary["next_step"] = _next_step(gate)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(gate, verification["matched"], verification.get("max_abs_label_gap_bits"), verification["rollouts"], verification["seconds"], flush=True)
    return summary


def main() -> dict:
    summary = execute()
    write_results(summary)
    print(summary["gate"], summary["time_effects"], summary["first_state_seconds"], summary["study_rollouts"], summary["elapsed_seconds"], flush=True)
    if summary["failure_trace"]:
        print(summary["failure_trace"], flush=True)
    return summary


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--verify-only":
        refresh_verification()
    else:
        main()
