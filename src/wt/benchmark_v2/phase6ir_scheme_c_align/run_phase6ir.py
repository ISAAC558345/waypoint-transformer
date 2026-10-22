"""Build the scheme-C task set, then train the original residual reward."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import load_primary
from wt.benchmark_v2.phase6iaa_formal_planner.planner import FormalPlanner

from .dataset import collect_split, distribution_summary, load_rows, repair_legacy_dataset, save_rows, validate_rows
from .evaluate import evaluate_closed_loop, evaluate_fixed_windows, fixed_summary, load_aligned_actor, paired_summary
from .protocol import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CHECKPOINT_DIR,
    DATASET_DIR,
    EVAL_POOL,
    EXPECTED_SOURCE_SHA256,
    MIN_SUPPORTED_SEEDS,
    MODEL_SEEDS,
    OUTPUT,
    SAFETY_MARGIN,
    SOURCE_ACTOR,
    TRAIN_SCENARIOS,
    UPDATES,
    assert_phase_output,
)
from .train import load_critic, load_source, load_warm_actor, train_seed
from .worker import SchemeCTrainWorker


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
        "mean_r_comm",
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
                "window_bits15": float(row["window_bits15"]),
                "entry_steps": float(row["mean_first_entry_steps"]),
                "service_steps": int(row["service_steps"]),
                "path_length_m": float(row["path_length_m"]),
                "sinr_mean": float(row["sinr_mean"]),
                "interference_mean": float(row["interference_mean"]),
                "projection_rate": float(row["projection_rate"]),
                "safety_violation_rate": float(row["safety_violation_rate"]),
                "complete": bool(row["complete"]),
            }
        )
    cases.sort(key=lambda item: item["delta_payload"])
    return cases


def _smoke(payload: dict, records: list[dict]) -> dict:
    actor = load_warm_actor(payload)
    actor.train()
    critic, mode = load_critic(payload)
    worker = SchemeCTrainWorker(0, 0, actor, critic)
    worker.bind_records(records[:8], 1)
    worker.reset()
    served = []
    rewards = []
    for _ in range(int(records[0]["window_steps"])):
        record = worker.step()
        if record["obs"]["self"].shape[-1] != 5 or record["pre_tanh"].shape[-1] != 2:
            raise RuntimeError("INTERFACE")
        if not np.isfinite(record["reward"]):
            raise RuntimeError("NONFINITE")
        served.append(float(record["served_bits"]))
        rewards.append(float(record["r_comm"]))
    if abs(sum(rewards) * 1.0e6 - sum(served)) > 1.0:
        raise RuntimeError("window return was not the sum of per-step served bits")
    return {
        "accounting_ok": True,
        "per_step_reward": True,
        "shaping": False,
        "action_dim": 2,
        "alpha": 0.25,
        "critic_mode": mode,
        "window_served_bits": float(sum(served)),
        "window_r_comm": float(sum(rewards)),
    }


def _write(name: str, payload: dict) -> None:
    assert_phase_output(OUTPUT / name).write_text(json.dumps(payload), encoding="utf-8")


def _manifest(planner: FormalPlanner, source_sha: str) -> dict:
    startup = planner.startup
    return {
        "task_source": "scheme_c select_assignment",
        "generator": "G1_RATE",
        "k": 6,
        "hold": 15,
        "episode_steps": 48,
        "alpha": 0.25,
        "reward": "served_bits / 1e6 - 0.05 * c_project - 0.01 * c_residual",
        "shaping": False,
        "scenario_seed": "reset seed equals scenario id",
        "scheme_c_checkpoint": startup["scheme_c_checkpoint"],
        "scheme_c_sha256": startup["scheme_c_sha256"],
        "mappo_checkpoint": startup["mappo_checkpoint"],
        "mappo_sha256": startup["mappo_sha256"],
        "source_sha256": source_sha,
        "candidate_config_sha256": startup["candidate_config_sha256"],
        "config_source": "phase6if scheme C epoch 1 + frozen G1_RATE k=6 + phase6g_seed0",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=int(UPDATES))
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in MODEL_SEEDS))
    args = parser.parse_args()
    seeds = tuple(int(item) for item in args.seeds.split(",") if item)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    source_sha = _sha256(SOURCE_ACTOR)
    if source_sha != EXPECTED_SOURCE_SHA256:
        _write("summary.json", {"gates": ["LOW_LEVEL_ALIGNMENT_BLOCKED"], "reason": "source hash mismatch", "replaced_official_low_level": False})
        raise RuntimeError("LOW_LEVEL_ALIGNMENT_BLOCKED")
    planner = FormalPlanner("scheme_c")
    manifest = _manifest(planner, source_sha)
    rng = np.random.default_rng(0)
    reused = False
    try:
        if (DATASET_DIR / "train.npz").exists() and (DATASET_DIR / "eval.npz").exists():
            repair_legacy_dataset("train", manifest)
            repair_legacy_dataset("eval", manifest)
            train_loaded = load_rows("train")
            dev_loaded = load_rows("eval")
            train_check = validate_rows(train_loaded, "train", planner)
            dev_check = validate_rows(dev_loaded, "eval", planner)
            reused = True
        else:
            raise RuntimeError("scheme C dataset is missing")
    except RuntimeError as first_error:
        print(f"dataset reuse stopped: {first_error}", flush=True)
        try:
            train_rows = collect_split(planner, TRAIN_SCENARIOS, "train", rng)
            dev_rows = collect_split(planner, EVAL_POOL, "eval", rng)
            save_rows(train_rows, "train", manifest)
            save_rows(dev_rows, "eval", manifest)
            train_loaded = load_rows("train")
            dev_loaded = load_rows("eval")
            train_check = validate_rows(train_loaded, "train", planner)
            dev_check = validate_rows(dev_loaded, "eval", planner)
            reused = False
        except RuntimeError as exc:
            _write(
                "summary.json",
                {
                    "gates": ["LOW_LEVEL_ALIGNMENT_BLOCKED"],
                    "reason": str(exc),
                    "replaced_official_low_level": False,
                    "training_started": False,
                    "source_sha256": source_sha,
                },
            )
            raise
    print(f"dataset ready reused={int(reused)}", flush=True)
    stats = {"train": distribution_summary(train_loaded), "eval": distribution_summary(dev_loaded)}
    _write("dataset_stats.json", {"manifest": manifest, "checks": {"train": train_check, "eval": dev_check}, "distribution": stats})
    source = load_source()
    replay_actor = load_warm_actor(source)
    replay_actor.eval()
    train_replay = evaluate_fixed_windows(replay_actor, train_loaded, split="train")
    train_gap = float(max(abs(row["served_bits"] - row["baseline_recorded_bits"]) for row in train_replay))
    if train_gap > 1.0:
        _write(
            "summary.json",
            {
                "gates": ["LOW_LEVEL_ALIGNMENT_BLOCKED"],
                "reason": "restored window replay does not match the recorded served bits",
                "replay_gap_bits": train_gap,
                "distribution": stats,
                "replaced_official_low_level": False,
                "training_started": False,
                "source_sha256": source_sha,
            },
        )
        raise RuntimeError("LOW_LEVEL_ALIGNMENT_BLOCKED")
    try:
        smoke = _smoke(source, train_loaded)
    except RuntimeError as exc:
        _write(
            "summary.json",
            {
                "gates": ["SCHEME_C_TASK_DATASET_READY", "LOW_LEVEL_ALIGNMENT_BLOCKED"],
                "reason": str(exc),
                "distribution": stats,
                "replaced_official_low_level": False,
                "source_sha256": source_sha,
            },
        )
        raise
    started = time.perf_counter()
    histories = []
    checkpoints = []
    critic_mode = ""
    try:
        for seed in seeds:
            path = assert_phase_output(CHECKPOINT_DIR / f"phase6ir_align_seed{seed}.pt")
            result = train_seed(int(seed), int(args.updates), path, source, train_loaded)
            histories.append(result["history"])
            checkpoints.append(path)
            critic_mode = result["critic_mode"]
    except RuntimeError as exc:
        _curve(histories, OUTPUT / "training_curve.csv")
        _write(
            "summary.json",
            {
                "gates": ["SCHEME_C_TASK_DATASET_READY", "LOW_LEVEL_ALIGNMENT_BLOCKED"],
                "reason": str(exc),
                "smoke": smoke,
                "replaced_official_low_level": False,
                "source_sha256": source_sha,
                "source_unchanged": int(_sha256(SOURCE_ACTOR) == source_sha),
            },
        )
        raise
    elapsed = time.perf_counter() - started
    _curve(histories, OUTPUT / "training_curve.csv")
    print("evaluating frozen closed loop", flush=True)
    baseline = evaluate_closed_loop(load_primary(), EVAL_POOL)
    print("evaluating frozen fixed windows", flush=True)
    fixed_base = evaluate_fixed_windows(load_primary(), dev_loaded)
    trained = []
    for path in checkpoints:
        print(f"evaluating {path.name}", flush=True)
        actor = load_aligned_actor(path)
        rows = evaluate_closed_loop(actor, EVAL_POOL)
        fixed_rows = evaluate_fixed_windows(actor, dev_loaded)
        pair = paired_summary(baseline, rows, draws=int(BOOTSTRAP_DRAWS), seed=int(BOOTSTRAP_SEED))
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
                "fixed_windows": fixed_summary(fixed_base, fixed_rows),
                "window_bits15": float(np.mean([row["window_bits15"] for row in rows])),
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
    outcome = "LOW_LEVEL_ALIGNMENT_BLOCKED" if blocked else (
        "LOW_LEVEL_ALIGNMENT_GAIN_SUPPORTED" if supported else "LOW_LEVEL_ALIGNMENT_GAIN_INCONCLUSIVE"
    )
    summary = {
        "gates": ["SCHEME_C_TASK_DATASET_READY", "LOW_LEVEL_ALIGNMENT_TRAINING_COMPLETE", outcome],
        "candidate_listed": bool(supported and not blocked),
        "replaced_official_low_level": False,
        "planner_modified": False,
        "joint_training": False,
        "source_sha256": source_sha,
        "source_unchanged": int(_sha256(SOURCE_ACTOR) == source_sha),
        "critic_mode": critic_mode,
        "manifest": manifest,
        "checks": {"train": train_check, "eval": dev_check},
        "distribution": stats,
        "smoke": smoke,
        "replay_gap_bits": train_gap,
        "train_seconds": float(elapsed),
        "baseline_window_bits15": float(np.mean([row["window_bits15"] for row in baseline])),
        "seeds": [{key: value for key, value in item.items() if key != "rows"} for item in trained],
    }
    _write("summary.json", summary)
    assert_phase_output(OUTPUT / "scenario_rows.json").write_text(
        json.dumps({"baseline_closed_loop": baseline, "trained": trained, "fixed_baseline_windows": fixed_base}),
        encoding="utf-8",
    )
    assert_phase_output(OUTPUT / "failure_cases.json").write_text(
        json.dumps({Path(item["checkpoint"]).name: item["failures"] for item in trained}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"gates": summary["gates"], "means": [item["paired"]["mean"] for item in trained], "seconds": elapsed}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
