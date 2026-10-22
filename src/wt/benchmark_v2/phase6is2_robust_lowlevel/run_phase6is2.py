"""Run mixed-source low-level training and cross-policy evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import load_primary
from wt.lowlevel_v2.critic_state import expected_critic_dim
from wt.lowlevel_v2.residual_policy import CentralizedCritic

from .audit import condition_span, gae_audit, observation_audit
from .evaluate import fixed_summary, load_robust_actor, paired_summary, run_episode
from .protocol import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CHECKPOINT_DIR,
    EVAL_POLICIES,
    EVAL_POOL,
    EXPECTED_SOURCE_SHA256,
    GAIN_RULE,
    HELD_OUT_SOURCE,
    MIN_SUPPORTED_SEEDS,
    MODEL_SEEDS,
    OUTPUT,
    SAFETY_MARGIN,
    SOURCE_ACTOR,
    TRAIN_SOURCES,
    UPDATES,
    assert_phase_output,
)
from .tasks import TaskSelector
from .train import load_source, load_warm_actor, train_seed
from .worker import RobustTrainWorker


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(name: str, payload: dict) -> None:
    assert_phase_output(OUTPUT / name).write_text(json.dumps(payload), encoding="utf-8")


def _blocked(reason: str, source_sha: str, extra: dict | None = None) -> None:
    payload = {
        "gates": ["ROBUST_TRAINING_BLOCKED"],
        "reason": reason,
        "replaced_official_low_level": False,
        "candidate_listed": False,
        "planner_modified": False,
        "source_sha256": source_sha,
        "gain_rule": GAIN_RULE,
    }
    if extra:
        payload.update(extra)
    _write("summary.json", payload)


def _seed_ok(pair: dict, *, top_share: float, mean_without_best: float) -> bool:
    concentrated = top_share > 0.50 and mean_without_best <= 0.0
    return (
        float(pair["mean"]) > 0.0
        and float(pair["ci_low"]) > 0.0
        and float(pair["candidate_safety"]) <= float(pair["baseline_safety"]) + float(SAFETY_MARGIN)
        and int(pair["incomplete"]) == 0
        and not concentrated
    )


def _extras(baseline: list[dict], rows: list[dict]) -> tuple[float, float]:
    left = {int(row["scenario_id"]): row for row in baseline}
    delta = np.asarray([float(row["payload_bits"]) - float(left[int(row["scenario_id"])]["payload_bits"]) for row in rows], dtype=np.float64)
    positive = delta[delta > 1.0]
    top_share = float(np.max(positive) / np.sum(positive)) if positive.size else 0.0
    mean_without = float(np.mean(np.delete(delta, int(np.argmax(delta))))) if len(delta) > 1 else float(delta[0])
    return top_share, mean_without


def _curve(histories: list[list[dict]], path: Path) -> None:
    header = ["model_seed", "update", "mean_reward", "mean_r_comm", "mean_served_bits", "actor_loss", "critic_loss", "entropy", "explained_variance", "projection_rate", "safety_violation_rate", "open_task_boundaries"]
    lines = [",".join(header)]
    for history in histories:
        for row in history:
            lines.append(",".join(str(row[key]) for key in header))
    assert_phase_output(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dataset_summary(decisions: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in decisions:
        grouped.setdefault(str(row["task_source"]), []).append(row)
    summary = {}
    for source, rows in grouped.items():
        summary[source] = {
            "decisions": int(len(rows)),
            "scenarios": int(len({int(row["scenario_id"]) for row in rows})),
            "mean_distance_m": float(np.mean([row["distance_m"] for row in rows])),
            "mean_queue_bits": float(np.mean([row["queue_bits"] for row in rows])),
            "mean_active": float(np.mean([row["n_active"] for row in rows])),
            "mean_realized_bits": float(np.mean([row["realized_served_bits"] for row in rows if row["realized_served_bits"] is not None])),
        }
    return summary


def _live_audit(selector: TaskSelector, actor) -> dict:
    critic, _mode = _critic(actor)
    worker = RobustTrainWorker(0, 0, actor, critic, selector, 1)
    actor.eval()
    episodes = []
    for _index in range(len(TRAIN_SOURCES)):
        worker.reset()
        served = 0.0
        flags = []
        for _step in range(int(worker.env.config.episode_horizon)):
            record = worker.step()
            served += float(record["served_bits"])
            flags.append(record)
        decisions = worker.decisions[-4:]
        times = [int(item["time"]) for item in decisions]
        if times != [0, 15, 30, 45]:
            raise RuntimeError(f"decision times drifted: {times}")
        realized = float(sum(float(item["realized_served_bits"]) for item in decisions))
        if abs(realized - served) > 1.0:
            raise RuntimeError("ACCOUNTING")
        if any(item["task_source"] not in TRAIN_SOURCES for item in decisions):
            raise RuntimeError("held-out source entered the training audit")
        if not any(bool(item["task_boundary"]) and not bool(item["done"]) for item in flags):
            raise RuntimeError("task boundary was treated as episode termination")
        if not any(int(item["time"]) == 15 and not bool(item["done"]) for item in flags):
            raise RuntimeError("t=15 terminated the episode")
        if not any(int(item["time"]) == 48 and bool(item["done"]) for item in flags):
            raise RuntimeError("episode horizon did not terminate")
        episodes.append({"source": decisions[0]["task_source"], "scenario_id": decisions[0]["scenario_id"], "realized_bits": realized})
    return {"episodes": episodes, "hold_is_terminal": False, "payload_reset_at_boundary": False}


def _critic(actor):
    _ = actor
    width = int(expected_critic_dim(6, 12))
    return CentralizedCritic(width), "audit"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=int(UPDATES))
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in MODEL_SEEDS))
    args = parser.parse_args()
    seeds = tuple(int(item) for item in args.seeds.split(",") if item)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    source_sha = _sha256(SOURCE_ACTOR)
    if source_sha != EXPECTED_SOURCE_SHA256:
        _blocked("source hash mismatch", source_sha)
        raise RuntimeError("ROBUST_TRAINING_BLOCKED")
    observation = observation_audit()
    temporal = gae_audit()
    conditions = condition_span()
    if observation["future_inputs"] != 0 or temporal["hold_is_terminal"]:
        _blocked("temporal or observation audit failed", source_sha, {"observation": observation, "temporal": temporal})
        raise RuntimeError("ROBUST_TRAINING_BLOCKED")
    print("auditing mixed task sources", flush=True)
    selector = TaskSelector()
    source = load_source()
    audit_actor = load_warm_actor(source)
    audit_actor.eval()
    try:
        live = _live_audit(selector, audit_actor)
        probe = run_episode(int(EVAL_POOL[0]), load_primary(), "scheme_c", selector)
        replay = run_episode(int(EVAL_POOL[0]), load_primary(), "scheme_c", selector, fixed_assignments=probe["assignments"])
        if abs(float(probe["payload_bits"]) - float(replay["payload_bits"])) > 1.0 or not probe["complete"] or not replay["complete"]:
            raise RuntimeError("fixed assignment replay diverged from the closed loop")
        held = run_episode(int(EVAL_POOL[0]), load_primary(), HELD_OUT_SOURCE, selector)
        if not held["complete"] or held["policy"] in TRAIN_SOURCES:
            raise RuntimeError("held-out policy did not stay out of training")
    except RuntimeError as exc:
        _blocked(str(exc), source_sha, {"observation": observation, "temporal": temporal, "conditions": conditions})
        raise
    _write(
        "audit.json",
        {
            "observation": observation,
            "temporal": temporal,
            "conditions": conditions,
            "live": live,
            "probe_replay_gap_bits": abs(float(probe["payload_bits"]) - float(replay["payload_bits"])),
            "config": selector.config,
            "config_sha256": selector.config_sha256,
        },
    )
    print("smoke update", flush=True)
    try:
        smoke = train_seed(0, 1, CHECKPOINT_DIR / "smoke_seed0.pt", source, selector)
    except RuntimeError as exc:
        _blocked(str(exc), source_sha, {"observation": observation, "temporal": temporal})
        raise
    smoke_sources = sorted({item["task_source"] for item in smoke["decisions"]})
    if smoke_sources != sorted(TRAIN_SOURCES) or any(item["realized_served_bits"] is None for item in smoke["decisions"]):
        _blocked("smoke test did not cover the mixed task sources", source_sha, {"sources": smoke_sources})
        raise RuntimeError("ROBUST_TRAINING_BLOCKED")
    started = time.perf_counter()
    histories = []
    checkpoints = []
    decisions = list(smoke["decisions"])
    critic_mode = smoke["critic_mode"]
    try:
        for seed in seeds:
            path = assert_phase_output(CHECKPOINT_DIR / f"phase6is2_robust_seed{seed}.pt")
            result = train_seed(int(seed), int(args.updates), path, source, selector)
            histories.append(result["history"])
            checkpoints.append(path)
            decisions.extend(result["decisions"])
            critic_mode = result["critic_mode"]
    except RuntimeError as exc:
        _curve(histories, OUTPUT / "training_curve.csv")
        _blocked(str(exc), source_sha, {"smoke_sources": smoke_sources, "source_unchanged": int(_sha256(SOURCE_ACTOR) == source_sha)})
        raise
    elapsed = time.perf_counter() - started
    _curve(histories, OUTPUT / "training_curve.csv")
    task_path = assert_phase_output(OUTPUT / "training_tasks.jsonl")
    task_path.write_text("\n".join(json.dumps(item) for item in decisions) + "\n", encoding="utf-8")
    dataset = _dataset_summary(decisions)
    _write("dataset_stats.json", {"config_sha256": selector.config_sha256, "sources": dataset, "conditions": conditions})
    frozen = load_primary()
    baselines = {}
    policies = {}
    worst = {}
    blocked = False
    for policy in EVAL_POLICIES:
        print(f"evaluating frozen {policy}", flush=True)
        baseline_rows = []
        fixed_base = []
        gaps = []
        for scenario in EVAL_POOL:
            closed = run_episode(int(scenario), frozen, policy, selector)
            fixed = run_episode(int(scenario), frozen, policy, selector, fixed_assignments=closed["assignments"])
            baseline_rows.append(closed)
            fixed_base.append(fixed)
            gaps.append(abs(float(closed["payload_bits"]) - float(fixed["payload_bits"])))
            print(f"closed {policy} {scenario} payload={closed['payload_bits']:.3f} complete={closed['complete']}", flush=True)
        if max(gaps) > 1.0 or any(not row["complete"] for row in baseline_rows):
            blocked = True
        baselines[policy] = {"closed": baseline_rows, "fixed": fixed_base, "replay_gap_bits": float(max(gaps))}
        policies[policy] = []
    if blocked:
        _blocked("baseline closed loop and fixed assignment replay diverged", source_sha, {"replay": {policy: baselines[policy]["replay_gap_bits"] for policy in EVAL_POLICIES}})
        raise RuntimeError("ROBUST_TRAINING_BLOCKED")
    for path in checkpoints:
        actor = load_robust_actor(path)
        for policy in EVAL_POLICIES:
            print(f"evaluating {path.name} {policy}", flush=True)
            rows = []
            fixed_rows = []
            for scenario in EVAL_POOL:
                rows.append(run_episode(int(scenario), actor, policy, selector))
                frozen_assignments = next(item["assignments"] for item in baselines[policy]["closed"] if int(item["scenario_id"]) == int(scenario))
                fixed_rows.append(run_episode(int(scenario), actor, policy, selector, fixed_assignments=frozen_assignments))
            pair = paired_summary(baselines[policy]["closed"], rows, draws=int(BOOTSTRAP_DRAWS), seed=int(BOOTSTRAP_SEED))
            top_share, mean_without = _extras(baselines[policy]["closed"], rows)
            left = {int(row["scenario_id"]): row for row in baselines[policy]["closed"]}
            ordered = sorted(rows, key=lambda item: float(item["payload_bits"]) - float(left[int(item["scenario_id"])]["payload_bits"]))
            policies[policy].append(
                {
                    "checkpoint": str(path),
                    "sha256": _sha256(path),
                    "paired": pair,
                    "fixed_windows": fixed_summary(baselines[policy]["fixed"], fixed_rows),
                    "window_bits15": float(np.mean([row["window_bits15"] for row in rows])),
                    "top_positive_share": top_share,
                    "mean_without_best": mean_without,
                    "worst_scenario": int(ordered[0]["scenario_id"]),
                    "worst_delta": float(ordered[0]["payload_bits"] - left[int(ordered[0]["scenario_id"])]["payload_bits"]),
                    "best_scenario": int(ordered[-1]["scenario_id"]),
                    "best_delta": float(ordered[-1]["payload_bits"] - left[int(ordered[-1]["scenario_id"])]["payload_bits"]),
                    "rows": rows,
                }
            )
    policy_supported = {}
    for policy, items in policies.items():
        policy_supported[policy] = sum(
            _seed_ok(item["paired"], top_share=float(item["top_positive_share"]), mean_without_best=float(item["mean_without_best"]))
            for item in items
        ) >= int(MIN_SUPPORTED_SEEDS)
    scheme_degraded = sum(
        float(item["paired"]["mean"]) < 0.0 and float(item["paired"]["ci_high"]) < 0.0 for item in policies["scheme_c"]
    ) >= int(MIN_SUPPORTED_SEEDS)
    incomplete = any(int(item["paired"]["incomplete"]) > 0 for items in policies.values() for item in items)
    if incomplete or blocked:
        outcome = "ROBUST_TRAINING_BLOCKED"
    elif all(policy_supported.values()) and not scheme_degraded:
        outcome = "CROSS_POLICY_GENERALIZATION_SUPPORTED"
    else:
        outcome = "CROSS_POLICY_GENERALIZATION_INCONCLUSIVE"
    for policy, items in policies.items():
        worst[policy] = [
            {"checkpoint": Path(item["checkpoint"]).name, "scenario_id": item["worst_scenario"], "delta_payload": item["worst_delta"]}
            for item in items
        ]
    summary = {
        "gates": ["ROBUST_TASK_DATASET_READY", "TEMPORAL_CREDIT_AUDIT_COMPLETE", "ROBUST_LOW_LEVEL_TRAINING_COMPLETE", outcome],
        "candidate_listed": bool(outcome == "CROSS_POLICY_GENERALIZATION_SUPPORTED"),
        "replaced_official_low_level": False,
        "planner_modified": False,
        "joint_training": False,
        "gain_rule": GAIN_RULE,
        "source_sha256": source_sha,
        "source_unchanged": int(_sha256(SOURCE_ACTOR) == source_sha),
        "critic_mode": critic_mode,
        "config_sha256": selector.config_sha256,
        "config": selector.config,
        "observation": observation,
        "temporal": temporal,
        "conditions": conditions,
        "dataset": dataset,
        "train_seconds": float(elapsed),
        "policy_supported": policy_supported,
        "scheme_c_degraded": bool(scheme_degraded),
        "replay_gap_bits": {policy: baselines[policy]["replay_gap_bits"] for policy in EVAL_POLICIES},
        "seeds": {
            policy: [{key: value for key, value in item.items() if key != "rows"} for item in items]
            for policy, items in policies.items()
        },
        "worst": worst,
    }
    _write("summary.json", summary)
    assert_phase_output(OUTPUT / "scenario_rows.json").write_text(
        json.dumps(
            {
                "baseline": {policy: baselines[policy]["closed"] for policy in EVAL_POLICIES},
                "trained": {policy: items for policy, items in policies.items()},
            }
        ),
        encoding="utf-8",
    )
    assert_phase_output(OUTPUT / "failure_cases.json").write_text(json.dumps(worst, indent=2), encoding="utf-8")
    print(json.dumps({"gates": summary["gates"], "supported": policy_supported, "seconds": elapsed}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
