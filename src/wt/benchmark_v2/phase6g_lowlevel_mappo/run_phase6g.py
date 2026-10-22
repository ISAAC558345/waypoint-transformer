"""Train and evaluate Benchmark-V2 residual MAPPO."""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from wt.benchmark_v2.phase6g_lowlevel_mappo.evaluate import (
    bootstrap_mean,
    build_eval_set,
    execute_option,
    positive_fraction,
    run_full_episode,
    safety_regressed,
    scenario_means,
)
from wt.benchmark_v2.phase6g_lowlevel_mappo.io_utils import finish, freeze_inputs, write_csv, write_json
from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import (
    CASE_SEED,
    DEV_POOL,
    LEGACY_ACTOR,
    LOG_STD_INIT,
    MODEL_SEEDS,
    PHASE,
    choose_gate,
)
from wt.benchmark_v2.phase6g_lowlevel_mappo.train import probe_initialization, train_seed
from wt.lowlevel_v2.residual_policy import ResidualActor


def _job(args: tuple[int, str, str]) -> dict:
    return train_seed(args[0], args[1], args[2])


def _load_actor(path) -> ResidualActor:
    actor = ResidualActor(log_std_init=float(LOG_STD_INIT))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["actor"] if isinstance(payload, dict) and "actor" in payload else payload
    actor.load_state_dict(state, strict=True)
    actor.eval()
    return actor


def _paired(eval_set: dict) -> dict:
    legacy = _load_actor(LEGACY_ACTOR)
    actors = {seed: _load_actor(PHASE / "checkpoints" / f"phase6g_seed{seed}.pt") for seed in MODEL_SEEDS}
    rows = []
    cases = []
    for state in eval_set["states"]:
        for assignment_index, assignment in enumerate(state["assignments"]):
            fly = execute_option(state, assignment, None)
            results = {"fly_toward": fly, "legacy_mappo": execute_option(state, assignment, legacy)}
            for seed, actor in actors.items():
                results[f"mappo_v2_seed{seed}"] = execute_option(state, assignment, actor)
            fly_bits = float(fly["served_bits"])
            for name, result in results.items():
                seed = -1 if "seed" not in name else int(name.rsplit("seed", 1)[1])
                rows.append(
                    {
                        "scenario_id": int(state["scenario_id"]),
                        "decision_time": int(state["decision_time"]),
                        "assignment_index": int(assignment_index),
                        "executor": name,
                        "model_seed": seed,
                        "served_bits": result["served_bits"],
                        "delta_bits_vs_fly": float(result["served_bits"] - fly_bits),
                        "mean_sinr": result["mean_sinr"],
                        "mean_rate": result["mean_rate"],
                        "mean_interference": result["mean_interference"],
                        "distance_reduction": result["distance_reduction"],
                        "path_length": result["path_length"],
                        "service_utilization": result["service_utilization"],
                        "projection_rate": result["projection_rate"],
                        "residual_action_norm": result["residual_action_norm"],
                        "safety_violation_rate": result["safety_violation_rate"],
                        "final_distance": result["final_distance"],
                        "first_service_step": result["first_service_step"],
                        "illegal": result["illegal"],
                        "accounting_failures": result["accounting_failures"],
                    }
                )
            seed_result = results[f"mappo_v2_seed{CASE_SEED}"]
            cases.append(
                {
                    "scenario_id": int(state["scenario_id"]),
                    "decision_time": int(state["decision_time"]),
                    "assignment_index": int(assignment_index),
                    "delta": float(seed_result["served_bits"] - fly_bits),
                    "series": {"fly": fly["series"], "mappo_v2": seed_result["series"]},
                }
            )
    return {"rows": rows, "cases": cases}


def _empty_summary(initialization: str) -> dict:
    summary = {key: 0.0 for key in ("mean_delta", "ci_low", "ci_high", "positive_fraction", "legacy_delta", "legacy_low", "legacy_high", "sinr_delta", "interference_delta", "path_delta", "entry_delta", "distance_delta", "full_fly", "full_legacy", "full_v2", "safety_v2", "safety_fly", "projection_v2")}
    summary["initialization"] = initialization
    summary["positive_seeds"] = 0
    summary["safety_regression"] = False
    return summary


def main() -> None:
    started = time.perf_counter()
    initialization = probe_initialization()
    freeze_inputs(initialization)
    eval_started = time.perf_counter()
    eval_set = build_eval_set()
    write_json(PHASE / "phase6g_execution_eval_set.json", eval_set)
    eval_set_seconds = time.perf_counter() - eval_started
    checkpoint_dir = PHASE / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(int(seed), initialization, str(checkpoint_dir / f"phase6g_seed{seed}.pt")) for seed in MODEL_SEEDS]
    train_started = time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=4) as pool:
            trained = list(pool.map(_job, jobs))
    except Exception as exc:
        message = str(exc)
        gate = choose_gate(
            accounting_failures=int("ACCOUNTING" in message),
            interface_failures=int("INTERFACE" in message),
            nan_collapse="ACCOUNTING" not in message and "INTERFACE" not in message,
            projection_rate=0.0,
            safety_regression=False,
            mean_delta=0.0,
            ci_low=0.0,
            positive_fraction=0.0,
            positive_seeds=0,
        )
        finish(gate, _empty_summary(initialization), [], [], [], [], [], [], [{"stage": "train_failure", "seconds": time.perf_counter() - started, "error": message}], [])
        raise
    train_seconds = time.perf_counter() - train_started
    curves = [row for item in trained for row in item["history"]]
    (PHASE / "logs").mkdir(parents=True, exist_ok=True)
    write_csv(PHASE / "logs" / "training_curves.csv", curves)
    compare_started = time.perf_counter()
    compared = _paired(eval_set)
    rows = compared["rows"]
    v2_names = {f"mappo_v2_seed{seed}" for seed in MODEL_SEEDS}
    per_seed = []
    positive_seeds = 0
    for seed in MODEL_SEEDS:
        ordered = np.asarray([scenario_means(rows, f"mappo_v2_seed{seed}")[scenario] for scenario in DEV_POOL], dtype=np.float64)
        mean, low, high = bootstrap_mean(ordered)
        positive_seeds += int(mean > 0.0)
        subset = [row for row in rows if row["executor"] == f"mappo_v2_seed{seed}"]
        per_seed.append(
            {
                "model_seed": int(seed),
                "mean_delta_bits": mean,
                "ci_low": low,
                "ci_high": high,
                "positive_fraction": positive_fraction(ordered),
                "mean_sinr": float(np.mean([row["mean_sinr"] for row in subset])),
                "mean_interference": float(np.mean([row["mean_interference"] for row in subset])),
                "projection_rate": float(np.mean([row["projection_rate"] for row in subset])),
                "residual_action_norm": float(np.mean([row["residual_action_norm"] for row in subset])),
            }
        )
    pooled = []
    legacy_pool = []
    legacy_rows = []
    for scenario in DEV_POOL:
        seed_means = [float(np.mean([row["delta_bits_vs_fly"] for row in rows if row["executor"] == f"mappo_v2_seed{seed}" and int(row["scenario_id"]) == int(scenario)])) for seed in MODEL_SEEDS]
        pooled.append(float(np.mean(seed_means)))
        pair_deltas = []
        for row in rows:
            if int(row["scenario_id"]) != int(scenario) or row["executor"] not in v2_names:
                continue
            legacy = next(item for item in rows if item["executor"] == "legacy_mappo" and item["scenario_id"] == row["scenario_id"] and item["decision_time"] == row["decision_time"] and item["assignment_index"] == row["assignment_index"])
            pair_deltas.append(float(row["served_bits"] - legacy["served_bits"]))
        legacy_pool.append(float(np.mean(pair_deltas)))
        legacy_rows.append({"scenario_id": int(scenario), "delta_bits_v2_minus_legacy": legacy_pool[-1]})
    mean_delta, ci_low, ci_high = bootstrap_mean(np.asarray(pooled))
    legacy_mean, legacy_low, legacy_high = bootstrap_mean(np.asarray(legacy_pool))
    fly_rows = [row for row in rows if row["executor"] == "fly_toward"]
    v2_rows = [row for row in rows if row["executor"] in v2_names]

    def gap(key: str) -> float:
        return float(np.mean([row[key] for row in v2_rows]) - np.mean([row[key] for row in fly_rows]))

    full_rows = []
    for scenario in DEV_POOL:
        full_rows.append({"executor": "fly_toward", "model_seed": -1, **run_full_episode(int(scenario), None)})
        full_rows.append({"executor": "legacy_mappo", "model_seed": -1, **run_full_episode(int(scenario), _load_actor(LEGACY_ACTOR))})
        for seed in MODEL_SEEDS:
            full_rows.append({"executor": "mappo_v2", "model_seed": int(seed), **run_full_episode(int(scenario), _load_actor(PHASE / "checkpoints" / f"phase6g_seed{seed}.pt"))})

    def full_mean(name: str) -> float:
        return float(np.mean([row["cumulative_payload_bits"] for row in full_rows if row["executor"] == name]))

    safety_v2 = float(np.mean([row["safety_violation_rate"] for row in v2_rows]))
    safety_fly = float(np.mean([row["safety_violation_rate"] for row in fly_rows]))
    projection_v2 = float(np.mean([row["projection_rate"] for row in v2_rows]))
    regressed = safety_regressed(safety_v2, safety_fly)
    gate = choose_gate(
        accounting_failures=int(sum(row["accounting_failures"] for row in rows) + sum(row["accounting_failures"] for row in full_rows)),
        interface_failures=int(sum(row["illegal"] for row in rows) + sum(row["illegal"] for row in full_rows)),
        nan_collapse=bool(any(not np.isfinite(row["actor_loss"]) for row in curves)),
        projection_rate=float(np.mean([item["projection_rate"] for item in trained])),
        safety_regression=regressed,
        mean_delta=mean_delta,
        ci_low=ci_low,
        positive_fraction=positive_fraction(np.asarray(pooled)),
        positive_seeds=positive_seeds,
    )
    ordered = sorted(compared["cases"], key=lambda item: item["delta"])
    selected = [
        {"label": "max_negative_delta", **ordered[0]},
        {"label": "median_delta", **ordered[len(ordered) // 2]},
        {"label": "max_positive_delta", **ordered[-1]},
    ]
    mechanism = [{"metric": key, "v2_minus_fly": gap(key)} for key in ("mean_sinr", "mean_interference", "path_length", "distance_reduction", "first_service_step", "service_utilization")]
    summary = _empty_summary(initialization)
    summary.update(
        {
            "mean_delta": mean_delta,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "positive_fraction": positive_fraction(np.asarray(pooled)),
            "positive_seeds": positive_seeds,
            "legacy_delta": legacy_mean,
            "legacy_low": legacy_low,
            "legacy_high": legacy_high,
            "sinr_delta": gap("mean_sinr"),
            "interference_delta": gap("mean_interference"),
            "path_delta": gap("path_length"),
            "entry_delta": gap("first_service_step"),
            "distance_delta": gap("distance_reduction"),
            "full_fly": full_mean("fly_toward"),
            "full_legacy": full_mean("legacy_mappo"),
            "full_v2": full_mean("mappo_v2"),
            "safety_v2": safety_v2,
            "safety_fly": safety_fly,
            "projection_v2": projection_v2,
            "safety_regression": regressed,
        }
    )
    runtime = [
        {"stage": "eval_set", "seconds": eval_set_seconds},
        {"stage": "training", "seconds": train_seconds},
        {"stage": "comparison", "seconds": time.perf_counter() - compare_started},
        {"stage": "total", "seconds": time.perf_counter() - started},
    ]
    finish(gate, summary, curves, rows, per_seed, legacy_rows, full_rows, mechanism, runtime, selected)


if __name__ == "__main__":
    main()
