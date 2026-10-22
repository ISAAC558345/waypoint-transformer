"""Direction checks, smoke, then one potential-shaping run. The official actor is not overwritten."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import load_primary

from .evaluate import evaluate_residual, load_shaped_actor, paired_summary
from .protocol import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CHECKPOINT_DIR,
    EVAL_POOL,
    EXPECTED_SOURCE_SHA256,
    MIN_SUPPORTED_SEEDS,
    MODEL_SEEDS,
    OUTPUT,
    SAFETY_MARGIN,
    SOURCE_ACTOR,
    UPDATES,
    assert_phase_output,
)
from .shaping import direction_report, directions_pass
from .train import load_critic, load_source, load_warm_actor, train_seed
from .worker import ShapedTrainWorker


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _gain_ok(pair: dict, *, top_share: float, mean_without_best: float) -> bool:
    concentrated = top_share > 0.50 and mean_without_best <= 0.0
    return (
        float(pair["mean"]) > 0.0
        and float(pair["ci_low"]) > 0.0
        and float(pair["candidate_safety"]) <= float(pair["baseline_safety"]) + float(SAFETY_MARGIN)
        and int(pair["incomplete"]) == 0
        and not concentrated
    )


def _curve(histories: list[list[dict]], path: Path) -> None:
    header = [
        "model_seed",
        "update",
        "mean_reward",
        "mean_task_reward",
        "mean_shaping",
        "mean_served_bits",
        "actor_loss",
        "critic_loss",
        "gradient_norm_preclip",
        "explained_variance",
        "projection_rate",
        "safety_violation_rate",
    ]
    lines = [",".join(header)]
    for history in histories:
        for row in history:
            lines.append(",".join(str(row[key]) for key in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _failures(baseline: list[dict], rows: list[dict]) -> list[dict]:
    left = {int(row["scenario_id"]): row for row in baseline}
    cases = []
    for row in rows:
        other = left[int(row["scenario_id"])]
        cases.append(
            {
                "scenario_id": int(row["scenario_id"]),
                "delta_payload": float(row["payload_bits"]) - float(other["payload_bits"]),
                "payload_bits": float(row["payload_bits"]),
                "baseline_payload_bits": float(other["payload_bits"]),
                "entry_steps": float(row["mean_first_entry_steps"]),
                "service_steps": int(row["service_steps"]),
                "path_length_m": float(row["path_length_m"]),
                "sinr_mean": float(row["sinr_mean"]),
                "interference_mean": float(row["interference_mean"]),
                "projection_rate": float(row["projection_rate"]),
                "task_reward_sum": float(row["task_reward_sum"]),
                "shaping_sum": float(row["shaping_sum"]),
                "safety_violation_rate": float(row["safety_violation_rate"]),
                "complete": bool(row["complete"]),
            }
        )
    cases.sort(key=lambda item: item["delta_payload"])
    return cases


def _smoke(payload: dict) -> dict:
    actor = load_warm_actor(payload)
    actor.eval()
    critic, mode = load_critic(payload)
    worker = ShapedTrainWorker(0, 0, actor, critic)
    worker.reset()
    horizon = int(worker.env.config.episode_horizon)
    task_sum = 0.0
    shaping_sum = 0.0
    for _ in range(horizon):
        record = worker.step()
        if record["obs"]["self"].shape[-1] != 5 or record["pre_tanh"].shape[-1] != 2:
            raise RuntimeError("INTERFACE")
        if abs(float(record["reward"]) - float(record["task_reward"]) - float(record["shaping"])) > 1.0e-5:
            raise RuntimeError("INTERFACE")
        if not np.isfinite(record["reward"]):
            raise RuntimeError("NONFINITE")
        task_sum += float(record["task_reward"])
        shaping_sum += float(record["shaping"])
    return {
        "accounting_ok": True,
        "action_dim": 2,
        "alpha": 0.25,
        "critic_mode": mode,
        "safety_projector": "unchanged",
        "task_reward_sum": float(task_sum),
        "shaping_sum": float(shaping_sum),
    }


def _write_summary(payload: dict) -> None:
    assert_phase_output(OUTPUT / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=int(UPDATES))
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in MODEL_SEEDS))
    args = parser.parse_args()
    seeds = tuple(int(item) for item in args.seeds.split(",") if item)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    before = _sha256(SOURCE_ACTOR)
    if before != EXPECTED_SOURCE_SHA256:
        _write_summary(
            {
                "gates": ["POTENTIAL_SHAPING_BLOCKED"],
                "reason": "source checkpoint hash mismatch",
                "source_sha256": before,
                "replaced_official_low_level": False,
            }
        )
        raise RuntimeError("POTENTIAL_SHAPING_BLOCKED")
    report = direction_report()
    assert_phase_output(OUTPUT / "direction.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not directions_pass(report):
        _write_summary(
            {
                "gates": ["POTENTIAL_SHAPING_BLOCKED"],
                "reason": "direction check failed",
                "direction": report,
                "replaced_official_low_level": False,
                "source_sha256": before,
            }
        )
        raise RuntimeError("POTENTIAL_SHAPING_BLOCKED")
    source = load_source()
    try:
        smoke = _smoke(source)
    except RuntimeError as exc:
        _write_summary(
            {
                "gates": ["POTENTIAL_SHAPING_UNIT_TEST_PASS", "POTENTIAL_SHAPING_BLOCKED"],
                "reason": str(exc),
                "direction": report,
                "replaced_official_low_level": False,
                "source_sha256": before,
            }
        )
        raise
    started = time.perf_counter()
    histories = []
    checkpoints = []
    critic_mode = ""
    try:
        for seed in seeds:
            path = assert_phase_output(CHECKPOINT_DIR / f"phase6ip_potential_seed{seed}.pt")
            result = train_seed(int(seed), int(args.updates), path, source)
            histories.append(result["history"])
            checkpoints.append(path)
            critic_mode = result["critic_mode"]
    except RuntimeError as exc:
        _curve(histories, OUTPUT / "training_curve.csv")
        _write_summary(
            {
                "gates": ["POTENTIAL_SHAPING_UNIT_TEST_PASS", "POTENTIAL_SHAPING_BLOCKED"],
                "reason": str(exc),
                "direction": report,
                "smoke": smoke,
                "replaced_official_low_level": False,
                "source_sha256": before,
                "source_unchanged": int(_sha256(SOURCE_ACTOR) == before),
            }
        )
        raise
    elapsed = time.perf_counter() - started
    _curve(histories, OUTPUT / "training_curve.csv")
    print("evaluating frozen residual", flush=True)
    baseline = evaluate_residual(load_primary(), EVAL_POOL)
    trained = []
    for path in checkpoints:
        print(f"evaluating {path.name}", flush=True)
        actor = load_shaped_actor(path)
        rows = evaluate_residual(actor, EVAL_POOL)
        pair = paired_summary(baseline, rows, draws=int(BOOTSTRAP_DRAWS), seed=int(BOOTSTRAP_SEED))
        left = {int(row["scenario_id"]): row for row in baseline}
        delta = np.asarray(
            [row["payload_bits"] - left[int(row["scenario_id"])]["payload_bits"] for row in rows],
            dtype=np.float64,
        )
        positive = delta[delta > 1.0]
        top_share = float(np.max(positive) / np.sum(positive)) if positive.size else 0.0
        mean_without = float(np.mean(np.delete(delta, int(np.argmax(delta))))) if len(delta) > 1 else float(delta[0])
        trained.append(
            {
                "checkpoint": str(path),
                "sha256": _sha256(path),
                "paired": pair,
                "task_reward_sum": float(np.mean([row["task_reward_sum"] for row in rows])),
                "shaping_sum": float(np.mean([row["shaping_sum"] for row in rows])),
                "baseline_task_reward_sum": float(np.mean([row["task_reward_sum"] for row in baseline])),
                "baseline_shaping_sum": float(np.mean([row["shaping_sum"] for row in baseline])),
                "top_positive_share": top_share,
                "mean_without_best": mean_without,
                "rows": rows,
                "failures": _failures(baseline, rows)[:8],
            }
        )
    supported = sum(
        _gain_ok(item["paired"], top_share=float(item["top_positive_share"]), mean_without_best=float(item["mean_without_best"]))
        for item in trained
    ) >= int(MIN_SUPPORTED_SEEDS)
    blocked = any(int(item["paired"]["incomplete"]) > 0 for item in trained)
    outcome = "POTENTIAL_SHAPING_BLOCKED" if blocked else (
        "POTENTIAL_SHAPING_GAIN_SUPPORTED" if supported else "POTENTIAL_SHAPING_GAIN_INCONCLUSIVE"
    )
    summary = {
        "gates": ["POTENTIAL_SHAPING_UNIT_TEST_PASS", outcome],
        "factor": "distance_potential_shaping",
        "gamma": float(report["gamma"]),
        "distance_norm_m": float(report["distance_norm_m"]),
        "service_radius_m": float(report["service_radius_m"]),
        "replaced_official_low_level": False,
        "source_sha256": before,
        "source_unchanged": int(_sha256(SOURCE_ACTOR) == before),
        "critic_mode": critic_mode,
        "direction": report,
        "smoke": smoke,
        "train_seconds": float(elapsed),
        "seeds": [{key: value for key, value in item.items() if key != "rows"} for item in trained],
    }
    _write_summary(summary)
    assert_phase_output(OUTPUT / "scenario_rows.json").write_text(
        json.dumps({"baseline": baseline, "trained": trained}),
        encoding="utf-8",
    )
    assert_phase_output(OUTPUT / "failure_cases.json").write_text(
        json.dumps({Path(item["checkpoint"]).name: item["failures"] for item in trained}, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"gates": summary["gates"], "means": [item["paired"]["mean"] for item in trained], "seconds": elapsed},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
