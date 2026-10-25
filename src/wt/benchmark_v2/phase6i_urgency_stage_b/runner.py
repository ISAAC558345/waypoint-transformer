"""Cross-task development evaluation of the frozen MAPPO-V2 actor. No training."""
from __future__ import annotations

import csv
import dataclasses
import json
import subprocess
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import (
    ALPHA_RESIDUAL,
    D_SAFE_M,
    DECISION_TIMES,
    HOLD,
    LAMBDA_PROJECT,
    LAMBDA_RESIDUAL,
    SERVICE_SAFETY_MARGIN_M,
)
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import limit_threads, parameter_token
from wt.benchmark_v2.phase6iaa_formal_planner.planner import FormalPlanner
from wt.benchmark_v2.phase6iaa_formal_planner.protocol import MAPPO_SHA256, SCHEME_C_SHA256
from wt.benchmark_v2.phase6i_urgency_stage_a.runner import ScoreTap, frozen_mappo_action
from wt.benchmark_v2.phase6iy_candidate_adapter.protocol import config_sha256
from wt.l3b.config import L3BConfig
from wt.lowlevel_v2.observation import TASK_DIM

from .analyze import (
    ALL_CONTRASTS,
    PRIMARY_CONTRASTS,
    SCENARIO_METRICS,
    choose_gate,
    condition_summary,
    paired_contrast,
    replication_variance,
    scenario_level,
    write_figures,
)
from .conditions import g1_rate_decision, scheme_c_decision, training_pool_decision, wide_pool_decision
from .protocol import (
    AGE_SCALE_STEPS,
    AGE_SCHEMA_VERSION,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CONDITION_SPECS,
    FIXED_URGENCY_THRESHOLD,
    HIGH_URGENCY_QUANTILE,
    INSTRUCTION_COMMIT,
    LOG_SCHEMA_VERSION,
    OUTPUT,
    POSITIVE_BIT,
    REPLICATION_SEEDS,
    condition_code,
    dev_scenario_ids,
    replication_rng,
)
from .rollout import ROW_FIELDS, environment_match, run_episode

EPISODE_FIELDS = (
    "scenario_id",
    "condition_id",
    "replication_seed",
    "status",
    "anomaly",
    "log_on_off_match",
    "final_t",
    "payload_bits",
    "served_bits_sum",
    "max_ledger_mismatch_bits",
    "max_conservation_error_bits",
    "max_sinr_recompute_gap",
    "max_rate_recompute_gap",
    "urgent_service_bits",
    "high_urgency_served_bits",
    "urgency_at_least_half_served_bits",
    "zero_urgency_threshold_steps",
    "oldest_wait_max",
    "oldest_wait_mean",
    "mean_wait_max",
    "mean_wait_mean",
    "age_weighted_backlog_mean",
    "final_queue_bits",
    "dropped_bits",
    "first_service_radius_step",
    "positive_service_steps",
    "in_radius_uav_steps",
    "path_length_m",
    "projector_trigger_steps",
    "projector_trigger_rate",
    "mean_projection_norm",
    "safety_violation_steps",
    "mean_sinr",
    "mean_rate_bps",
    "mean_total_interference",
    "sinr_served_pearson",
    "mean_candidate_count",
    "mean_raw_candidate_count",
    "scheme_c_calls",
)


def run_evaluation(scenarios: tuple[int, ...] | None = None, directory: Path = OUTPUT) -> dict:
    limit_threads()
    chosen = dev_scenario_ids() if scenarios is None else tuple(int(item) for item in scenarios)
    for scenario_id in chosen:
        if scenario_id not in dev_scenario_ids():
            raise RuntimeError(f"scenario {scenario_id} is outside the phase6g dev pool")
    scheme = FormalPlanner("scheme_c")
    baseline = FormalPlanner("g1_rate")
    if parameter_token(scheme.actor) != parameter_token(baseline.actor):
        raise RuntimeError("frozen actor copies diverged")
    if int(scheme.actor.body[0].in_features) != int(5 + TASK_DIM + 4 + 16 + 1):
        raise RuntimeError("frozen actor input width changed")
    token = parameter_token(scheme.actor)
    tap = ScoreTap(scheme)
    hashes = {
        "policy_sha256": str(scheme.startup["scheme_c_sha256"]),
        "config_sha256": config_sha256(),
        "mappo_sha256": str(scheme.startup["mappo_sha256"]),
    }
    if hashes["policy_sha256"] != SCHEME_C_SHA256 or hashes["mappo_sha256"] != MAPPO_SHA256:
        raise RuntimeError("frozen checkpoint hash changed")
    directory.mkdir(parents=True, exist_ok=True)
    episodes: list[dict] = []
    step_path = directory / "step_rows.csv"
    with step_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for scenario_id in chosen:
            for spec in CONDITION_SPECS:
                seeds = REPLICATION_SEEDS if spec["random"] else (None,)
                for seed in seeds:
                    record = _one_pair(
                        scenario_id=int(scenario_id),
                        spec=spec,
                        seed=seed,
                        scheme=scheme,
                        baseline=baseline,
                        tap=tap,
                        writer=writer,
                        hashes=hashes,
                    )
                    episodes.append(record)
                    handle.flush()
                    print(
                        f"{scenario_id} {spec['condition_id']} seed={seed} status={record['status']} payload={record['payload_bits']}",
                        flush=True,
                    )
    if parameter_token(scheme.actor) != token:
        raise RuntimeError("frozen actor parameters changed during the evaluation")
    summary = _summarize(episodes, hashes, directory)
    _write_episodes(episodes, directory / "episode_rows.csv")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (directory / "schema.json").write_text(json.dumps(_schema(), indent=2) + "\n", encoding="utf-8")
    return summary


def _one_pair(*, scenario_id, spec, seed, scheme, baseline, tap, writer, hashes) -> dict:
    try:
        logged = run_episode(
            scenario_id,
            condition_id=spec["condition_id"],
            replication_seed=seed,
            movement=lambda env, option, last: frozen_mappo_action(env, scheme.actor, option, last),
            decision=_decision(spec["condition_id"], scenario_id, seed, scheme, baseline, tap),
            record=True,
            writer=writer,
            hashes=hashes,
        )
        silent = run_episode(
            scenario_id,
            condition_id=spec["condition_id"],
            replication_seed=seed,
            movement=lambda env, option, last: frozen_mappo_action(env, scheme.actor, option, last),
            decision=_decision(spec["condition_id"], scenario_id, seed, scheme, baseline, tap),
            record=False,
            hashes=hashes,
        )
    except Exception as exc:
        return _failed(scenario_id, spec["condition_id"], seed, f"{type(exc).__name__}: {exc}")
    row = _public_row(logged)
    row["log_on_off_match"] = bool(environment_match(logged, silent))
    if logged.status != "ok":
        row["status"] = logged.status
    elif not row["log_on_off_match"]:
        row["status"] = "log_mismatch"
        row["anomaly"] = "log on/off changed the environment outcome"
    return row


def _decision(condition_id: str, scenario_id: int, seed: int | None, scheme, baseline, tap):
    if condition_id == "A":
        return scheme_c_decision(scheme, tap)
    if condition_id == "B":
        return g1_rate_decision(baseline)
    if seed is None:
        raise RuntimeError("a random condition requires a replication seed")
    rng = replication_rng(scenario_id, condition_code(condition_id), seed)
    if condition_id == "C":
        return training_pool_decision(rng)
    if condition_id == "D":
        return wide_pool_decision(rng)
    raise RuntimeError(f"unknown condition {condition_id}")


def _public_row(run) -> dict:
    rate = float(run.projector_trigger_steps) / float(run.final_t) if int(run.final_t) else 0.0
    return {
        "scenario_id": int(run.scenario_id),
        "condition_id": str(run.condition_id),
        "replication_seed": "" if run.replication_seed is None else int(run.replication_seed),
        "status": str(run.status),
        "anomaly": str(run.anomaly),
        "log_on_off_match": True,
        "final_t": int(run.final_t),
        "payload_bits": float(run.payload_bits),
        "served_bits_sum": float(run.served_bits_sum),
        "max_ledger_mismatch_bits": float(run.max_ledger_mismatch_bits),
        "max_conservation_error_bits": float(run.max_conservation_error_bits),
        "max_sinr_recompute_gap": float(run.max_sinr_recompute_gap),
        "max_rate_recompute_gap": float(run.max_rate_recompute_gap),
        "urgent_service_bits": float(run.urgent_service_bits),
        "high_urgency_served_bits": float(run.high_urgency_served_bits),
        "urgency_at_least_half_served_bits": float(run.urgency_at_least_half_served_bits),
        "zero_urgency_threshold_steps": int(run.zero_urgency_threshold_steps),
        "oldest_wait_max": float(run.oldest_wait_max),
        "oldest_wait_mean": float(run.oldest_wait_mean),
        "mean_wait_max": float(run.mean_wait_max),
        "mean_wait_mean": float(run.mean_wait_mean),
        "age_weighted_backlog_mean": float(run.age_weighted_backlog_mean),
        "final_queue_bits": float(run.final_queue_bits),
        "dropped_bits": float(run.dropped_bits),
        "first_service_radius_step": int(run.first_service_radius_step),
        "positive_service_steps": int(run.positive_service_steps),
        "in_radius_uav_steps": int(run.in_radius_uav_steps),
        "path_length_m": float(run.path_length_m),
        "projector_trigger_steps": int(run.projector_trigger_steps),
        "projector_trigger_rate": float(rate),
        "mean_projection_norm": float(run.mean_projection_norm),
        "safety_violation_steps": int(run.safety_violation_steps),
        "mean_sinr": float(run.mean_sinr),
        "mean_rate_bps": float(run.mean_rate_bps),
        "mean_total_interference": float(run.mean_total_interference),
        "sinr_served_pearson": run.sinr_served_pearson,
        "mean_candidate_count": float(run.mean_candidate_count),
        "mean_raw_candidate_count": float(run.mean_raw_candidate_count),
        "scheme_c_calls": int(run.scheme_c_calls),
    }


def _failed(scenario_id: int, condition_id: str, seed: int | None, message: str) -> dict:
    row = {key: "" for key in EPISODE_FIELDS}
    row.update(
        {
            "scenario_id": int(scenario_id),
            "condition_id": str(condition_id),
            "replication_seed": "" if seed is None else int(seed),
            "status": "failed",
            "anomaly": message,
            "log_on_off_match": False,
        }
    )
    return row


def _summarize(episodes: list[dict], hashes: dict, directory: Path) -> dict:
    scenarios = dev_scenario_ids()
    expected = 0
    for spec in CONDITION_SPECS:
        expected += len(scenarios) * (len(REPLICATION_SEEDS) if spec["random"] else 1)
    failures = [row for row in episodes if row["status"] != "ok"]
    log_mismatches = [row for row in episodes if row["status"] == "log_mismatch" or row.get("log_on_off_match") is False]
    level = scenario_level(episodes)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    contrasts = []
    for left, right in ALL_CONTRASTS:
        for metric in ("payload_bits", "urgent_service_bits", "oldest_wait_mean", "positive_service_steps", "projector_trigger_rate", "dropped_bits", "final_queue_bits"):
            contrasts.append(paired_contrast(level, left, right, metric, scenarios, rng))
    primary = [item for item in contrasts if item["metric"] == "payload_bits" and (item["contrast"].split("-")[0], item["contrast"].split("-")[1]) in PRIMARY_CONTRASTS]
    excludes = any(bool(item.get("ci_excludes_zero")) for item in primary)
    gate = choose_gate(
        expected_episodes=expected,
        complete_episodes=len(episodes),
        failures=len(failures),
        log_mismatches=len([row for row in episodes if row.get("log_on_off_match") is False or row["status"] != "ok"]),
        primary_payload_ci_excludes_zero=excludes,
    )
    by_condition = []
    for spec in CONDITION_SPECS:
        rows = [row for row in episodes if row["condition_id"] == spec["condition_id"] and row["status"] == "ok"]
        counts = [float(row["mean_candidate_count"]) for row in rows if row["mean_candidate_count"] != ""]
        raw_counts = [float(row["mean_raw_candidate_count"]) for row in rows if row["mean_raw_candidate_count"] != ""]
        by_condition.append(
            {
                **condition_summary(level, spec["condition_id"], scenarios),
                "condition_name": spec["condition_name"],
                "assignment_rule": spec["assignment_rule"],
                "random_replications": list(REPLICATION_SEEDS) if spec["random"] else [],
                "episodes": int(len(rows)),
                "failures": int(sum(row["condition_id"] == spec["condition_id"] and row["status"] != "ok" for row in episodes)),
                "scheme_c_calls": int(sum(int(row["scheme_c_calls"]) for row in rows if row["scheme_c_calls"] != "")),
                "candidate_count_min": float(np.min(counts)) if counts else None,
                "candidate_count_max": float(np.max(counts)) if counts else None,
                "candidate_count_mean": float(np.mean(counts)) if counts else None,
                "raw_candidate_count_mean": float(np.mean(raw_counts)) if raw_counts else None,
                "max_ledger_mismatch_bits": float(np.max([float(row["max_ledger_mismatch_bits"]) for row in rows])) if rows else None,
                "max_conservation_error_bits": float(np.max([float(row["max_conservation_error_bits"]) for row in rows])) if rows else None,
            }
        )
    figure_path = directory / "figures" / "task_distribution_comparison.png"
    figure_written = write_figures(level, scenarios, figure_path)
    config = dataclasses.asdict(L3BConfig())
    config["step_length_m"] = float(L3BConfig().step_length_m)
    return {
        "gate": gate,
        "split": "phase6g_dev_321096_321127",
        "scenario_ids": list(scenarios),
        "episodes_expected": int(expected),
        "episodes_recorded": int(len(episodes)),
        "failures": int(len(failures)),
        "log_mismatches": int(len(log_mismatches)),
        "failure_rows": failures,
        "conditions": by_condition,
        "paired_contrasts": contrasts,
        "replication_variance": [
            replication_variance(episodes, "C"),
            replication_variance(episodes, "D"),
        ],
        "scheme_c_on_low_level_dev_pool": "diagnostic inference, not a formal high-level benchmark",
        "urgency_used_as_reward_or_observation": False,
        "training_performed": False,
        "reserved_test_used": False,
        "high_level_validation_322048_322063_used": False,
        "instruction_commit": INSTRUCTION_COMMIT,
        "git_head_at_run": _git_head(),
        "age_schema_version": AGE_SCHEMA_VERSION,
        "log_schema_version": LOG_SCHEMA_VERSION,
        "age_scale_steps": AGE_SCALE_STEPS,
        "policy_sha256": hashes["policy_sha256"],
        "config_sha256": hashes["config_sha256"],
        "mappo_sha256": hashes["mappo_sha256"],
        "actor_input_dim": 34,
        "task_dim": int(TASK_DIM),
        "decision_times": list(DECISION_TIMES),
        "hold": int(HOLD),
        "replication_seeds": list(REPLICATION_SEEDS),
        "replication_rng": "SeedSequence([scenario_id, condition_code, replication_seed])",
        "high_urgency_quantile": HIGH_URGENCY_QUANTILE,
        "fixed_urgency_threshold": FIXED_URGENCY_THRESHOLD,
        "tie_bits": float(POSITIVE_BIT),
        "bootstrap_draws": int(BOOTSTRAP_DRAWS),
        "bootstrap_seed": int(BOOTSTRAP_SEED),
        "environment": config,
        "safety": {
            "d_safe_m": float(D_SAFE_M),
            "service_safety_margin_m": float(SERVICE_SAFETY_MARGIN_M),
            "alpha_residual": float(ALPHA_RESIDUAL),
            "lambda_project": float(LAMBDA_PROJECT),
            "lambda_residual": float(LAMBDA_RESIDUAL),
            "projector_trigger": "constraint_status != ok or team projection norm > 1e-3",
        },
        "metrics_recorded": list(SCENARIO_METRICS),
        "figure": "results/phase6i_urgency_stage_b/figures/task_distribution_comparison.png" if figure_written else "",
        "recommendation": "",
    }


def _schema() -> dict:
    return {
        "log_schema_version": LOG_SCHEMA_VERSION,
        "age_schema_version": AGE_SCHEMA_VERSION,
        "age_scale_steps": AGE_SCALE_STEPS,
        "reset_age_origin": "reset_convention",
        "reset_age_meaning": "episode-internal convention starting at 0 when t is 0; not a pre-episode wait",
        "urgency_formula": "clip(oldest_wait_age / 48, 0, 1) when q_bits > 0, else 0",
        "urgency_in_observation_or_reward": False,
        "high_urgency_group": "within each step, valid buoys at or above the 0.75 quantile of urgency",
        "fixed_urgency_threshold": FIXED_URGENCY_THRESHOLD,
        "interference_queue": "pre-service q_bits at the post-move position and this step's assignment",
        "row_grain": "one step row per scenario, condition, replication, time_step, and uav or buoy index",
        "step_columns": list(ROW_FIELDS),
        "episode_columns": list(EPISODE_FIELDS),
        "logging_changes_transition": False,
    }


def _write_episodes(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPISODE_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in EPISODE_FIELDS})


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


def main() -> None:
    summary = run_evaluation()
    print(json.dumps({"gate": summary["gate"], "failures": summary["failures"]}))


if __name__ == "__main__":
    main()
