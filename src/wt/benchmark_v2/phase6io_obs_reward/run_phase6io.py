"""Audit, smoke, then one distance-shaping training run. The official actor is not overwritten."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import load_primary

from .audit import audit_episodes
from .evaluate import evaluate_residual, load_shaped_actor, paired_summary
from .protocol import (
    AUDIT_SCENARIOS,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CHECKPOINT_DIR,
    EVAL_POOL,
    MODEL_SEEDS,
    OUTPUT,
    SAFETY_MARGIN,
    SOURCE_ACTOR,
    TRAIN_CHECK_POOL,
    UPDATES,
    assert_phase_output,
)
from .train import load_source, train_seed
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
                "invalid_motion_steps": int(row["invalid_motion_steps"]),
                "path_length_m": float(row["path_length_m"]),
                "projection_rate": float(row["projection_rate"]),
                "correction_norm": float(row["correction_norm"]),
                "safety_violation_rate": float(row["safety_violation_rate"]),
                "complete": bool(row["complete"]),
            }
        )
    cases.sort(key=lambda item: item["delta_payload"])
    return cases


def _smoke(payload: dict) -> dict:
    from .train import load_critic, load_warm_actor

    actor = load_warm_actor(payload)
    actor.eval()
    critic, mode = load_critic(payload)
    worker = ShapedTrainWorker(0, 0, actor, critic)
    worker.reset()
    horizon = int(worker.env.config.episode_horizon)
    for _ in range(horizon):
        record = worker.step()
        if record["obs"]["self"].shape[-1] != 5 or record["pre_tanh"].shape[-1] != 2:
            raise RuntimeError("observation or action width changed")
        if not np.isfinite(record["reward"]):
            raise RuntimeError("NONFINITE")
    return {"accounting_ok": True, "action_dim": 2, "alpha": 0.25, "critic_mode": mode, "safety_projector": "unchanged"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=int(UPDATES))
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in MODEL_SEEDS))
    args = parser.parse_args()
    seeds = tuple(int(item) for item in args.seeds.split(",") if item)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    before = _sha256(SOURCE_ACTOR)
    payload = load_source()
    audit = audit_episodes(load_primary(), AUDIT_SCENARIOS)
    assert_phase_output(OUTPUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    outside = audit["phases"]["outside"]["mean_r_comm"]
    serving = audit["phases"]["serving"]["mean_r_comm"]
    if outside > 0.05 * max(serving, 1.0e-6):
        raise RuntimeError("approach reward is not the gap this experiment targets")
    smoke = _smoke(payload)
    started = time.perf_counter()
    histories = []
    checkpoints = []
    critic_mode = ""
    for seed in seeds:
        path = assert_phase_output(CHECKPOINT_DIR / f"phase6io_shape_seed{seed}.pt")
        result = train_seed(int(seed), int(args.updates), path, payload)
        histories.append(result["history"])
        checkpoints.append(path)
        critic_mode = result["critic_mode"]
    elapsed = time.perf_counter() - started
    _curve(histories, OUTPUT / "training_curve.csv")
    print("evaluating frozen residual", flush=True)
    baseline = evaluate_residual(load_primary(), EVAL_POOL)
    train_base = evaluate_residual(load_primary(), TRAIN_CHECK_POOL)
    trained = []
    for path in checkpoints:
        print(f"evaluating {path.name}", flush=True)
        actor = load_shaped_actor(path)
        rows = evaluate_residual(actor, EVAL_POOL)
        train_rows = evaluate_residual(actor, TRAIN_CHECK_POOL)
        pair = paired_summary(baseline, rows, draws=int(BOOTSTRAP_DRAWS), seed=int(BOOTSTRAP_SEED))
        train_pair = paired_summary(train_base, train_rows, draws=int(BOOTSTRAP_DRAWS), seed=int(BOOTSTRAP_SEED))
        left = {int(row["scenario_id"]): row for row in baseline}
        delta = np.asarray([row["payload_bits"] - left[int(row["scenario_id"])]["payload_bits"] for row in rows], dtype=np.float64)
        positive = delta[delta > 1.0]
        top_share = float(np.max(positive) / np.sum(positive)) if positive.size else 0.0
        mean_without = float(np.mean(np.delete(delta, int(np.argmax(delta))))) if len(delta) > 1 else float(delta[0])
        trained.append(
            {
                "checkpoint": str(path),
                "sha256": _sha256(path),
                "paired": pair,
                "train_paired": train_pair,
                "invalid_motion_steps": float(np.mean([row["invalid_motion_steps"] for row in rows])),
                "service_steps": float(np.mean([row["service_steps"] for row in rows])),
                "correction_norm": float(np.mean([row["correction_norm"] for row in rows])),
                "top_positive_share": top_share,
                "mean_without_best": mean_without,
                "rows": rows,
                "failures": _failures(baseline, rows)[:8],
            }
        )
    supported = sum(
        _gain_ok(item["paired"], top_share=float(item["top_positive_share"]), mean_without_best=float(item["mean_without_best"]))
        for item in trained
    ) >= 2
    blocked = any(int(item["paired"]["incomplete"]) > 0 for item in trained)
    outcome = "LOW_LEVEL_SINGLE_FACTOR_BLOCKED" if blocked else (
        "LOW_LEVEL_SINGLE_FACTOR_GAIN_SUPPORTED" if supported else "LOW_LEVEL_SINGLE_FACTOR_GAIN_INCONCLUSIVE"
    )
    summary = {
        "gates": ["OBSERVABILITY_AUDIT_COMPLETE", "REWARD_ALIGNMENT_AUDIT_COMPLETE", outcome],
        "factor": "distance_potential_shaping",
        "replaced_official_low_level": False,
        "source_sha256": before,
        "source_unchanged": int(_sha256(SOURCE_ACTOR) == before),
        "critic_mode": critic_mode,
        "smoke": smoke,
        "train_seconds": float(elapsed),
        "seeds": [{key: value for key, value in item.items() if key != "rows"} for item in trained],
    }
    assert_phase_output(OUTPUT / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert_phase_output(OUTPUT / "scenario_rows.json").write_text(
        json.dumps({"baseline": baseline, "trained": trained, "train_baseline": train_base}),
        encoding="utf-8",
    )
    assert_phase_output(OUTPUT / "failure_cases.json").write_text(
        json.dumps({Path(item["checkpoint"]).name: item["failures"] for item in trained}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"gates": summary["gates"], "means": [item["paired"]["mean"] for item in trained], "seconds": elapsed}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
