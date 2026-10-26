"""Stage C entry points. Training and evaluation stay outside the formal trainer."""
from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6g_lowlevel_mappo.protocol import PPO
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import checkpoint_sha
from wt.benchmark_v2.phase6h_executor_conditioned_macro.protocol import primary_checkpoint
from wt.benchmark_v2.phase6i_urgency_stage_a.runner import ScoreTap, limit_threads
from wt.benchmark_v2.phase6iaa_formal_planner.planner import FormalPlanner
from wt.benchmark_v2.phase6iaa_formal_planner.protocol import ADAPTER_CONFIG_SHA256, MAPPO_SHA256, SCHEME_C_SHA256
from wt.benchmark_v2.phase6i_urgency_stage_b.conditions import scheme_c_decision

from .analyze import choose_gate, condition_level, contrast, retain_l2, seed_payload_spread
from .evaluate import STEP_FIELDS, initial_scheme_c, load_l0_actor, load_trained_actor, movement_for, run_episode
from .protocol import (
    CHECKPOINTS,
    INSTRUCTION_COMMIT,
    LAMBDA_URGENCY,
    MODEL_SEEDS_STAGE_C,
    OUTPUT,
    UPDATES,
    dev_scenario_ids,
    train_scenario_ids,
)
from .train import train_variant

EPISODE_FIELDS = (
    "scenario_id",
    "variant",
    "model_seed",
    "protocol",
    "status",
    "anomaly",
    "final_t",
    "payload_bits",
    "served_bits_sum",
    "urgent_service_bits",
    "high_urgency_served_bits",
    "urgency_at_least_half_served_bits",
    "oldest_wait_mean",
    "oldest_wait_max",
    "mean_wait_mean",
    "positive_service_steps",
    "in_radius_uav_steps",
    "path_length_m",
    "projector_trigger_steps",
    "mean_projection_norm",
    "safety_violation_steps",
    "dropped_bits",
    "final_queue_bits",
    "first_service_radius_step",
    "first_high_urgency_service_step",
    "age_weighted_backlog_mean",
    "mean_sinr",
    "mean_rate_bps",
    "mean_total_interference",
    "sinr_served_pearson",
    "mean_candidate_count",
    "scheme_c_calls",
    "mean_idle_slots",
    "decision_count",
    "max_ledger_mismatch_bits",
    "max_conservation_error_bits",
    "max_sinr_recompute_gap",
    "max_rate_recompute_gap",
    "lowlevel_sha256",
)


def reward_scale_audit(scenario_id: int = 321000) -> dict:
    """Compare one frozen episode's urgency term with its payload term before training."""
    limit_threads()
    started = time.perf_counter()
    planner = FormalPlanner("scheme_c")
    actor = load_l0_actor()
    row = run_episode(
        int(scenario_id),
        variant="L0",
        model_seed=0,
        movement=movement_for("L0", actor),
        decision=scheme_c_decision(planner, ScoreTap(planner)),
        protocol="scale_audit",
        split="train",
        lowlevel_sha256=MAPPO_SHA256,
        hashes=_hashes(planner),
        max_steps=None,
    )
    payload = float(row["payload_bits"])
    urgent = float(row["urgent_service_bits"])
    ratio = urgent / payload if payload else float("inf")
    accepted = bool(np.isfinite(ratio) and 0.1 <= ratio <= 10.0)
    result = {
        "scenario_id": int(scenario_id),
        "split": "train",
        "lambda_urgency": float(LAMBDA_URGENCY),
        "payload_bits": payload,
        "urgent_service_bits": urgent,
        "urgency_over_payload": ratio,
        "accepted_for_lambda_1": accepted,
        "status": row["status"],
        "seconds": time.perf_counter() - started,
        "mean_idle_slots": row["mean_idle_slots"],
        "scheme_c_calls": row["scheme_c_calls"],
    }
    _write_json(OUTPUT / "reward_scale.json", result)
    return result


def train_all() -> list[dict]:
    rows = []
    for variant in ("L1", "L2"):
        for seed in MODEL_SEEDS_STAGE_C:
            rows.append(train_variant(variant, int(seed), updates=int(UPDATES), ppo=PPO))
    return rows


def evaluate_all() -> list[dict]:
    limit_threads()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    planner = FormalPlanner("scheme_c")
    tap = ScoreTap(planner)
    hashes = _hashes(planner)
    if hashes["policy_sha256"] != SCHEME_C_SHA256 or hashes["mappo_sha256"] != MAPPO_SHA256:
        raise RuntimeError("frozen scheme C or MAPPO hash changed")
    if hashes["config_sha256"] != ADAPTER_CONFIG_SHA256:
        raise RuntimeError("candidate config hash changed")
    l0 = load_l0_actor()
    trained = {}
    for variant in ("L1", "L2"):
        for seed in MODEL_SEEDS_STAGE_C:
            path = CHECKPOINTS / f"{variant}_seed{int(seed)}.pt"
            actor, payload = load_trained_actor(path)
            if int(payload["updates"]) != int(UPDATES) or int(payload["model_seed"]) != int(seed):
                raise RuntimeError(f"checkpoint {path} is not a finished pre-registered run")
            trained[(variant, int(seed))] = (actor, _sha(path))
    closed_path = OUTPUT / "closed_loop_step_rows.csv"
    fixed_path = OUTPUT / "fixed_hold_step_rows.csv"
    episodes: list[dict] = []
    with closed_path.open("w", encoding="utf-8", newline="") as closed, fixed_path.open("w", encoding="utf-8", newline="") as fixed:
        closed_writer = csv.DictWriter(closed, fieldnames=STEP_FIELDS, extrasaction="ignore", lineterminator="\n")
        fixed_writer = csv.DictWriter(fixed, fieldnames=STEP_FIELDS, extrasaction="ignore", lineterminator="\n")
        closed_writer.writeheader()
        fixed_writer.writeheader()
        for scenario_id in dev_scenario_ids():
            episodes.append(
                run_episode(
                    int(scenario_id),
                    variant="L0",
                    model_seed=0,
                    movement=movement_for("L0", l0),
                    decision=scheme_c_decision(planner, tap),
                    protocol="closed48",
                    split="dev",
                    lowlevel_sha256=MAPPO_SHA256,
                    hashes=hashes,
                    writer=closed_writer,
                )
            )
            assignment, meta = initial_scheme_c(planner, tap, int(scenario_id))
            episodes.append(
                run_episode(
                    int(scenario_id),
                    variant="L0",
                    model_seed=0,
                    movement=movement_for("L0", l0),
                    decision=meta,
                    protocol="fixed15",
                    split="dev",
                    lowlevel_sha256=MAPPO_SHA256,
                    hashes=hashes,
                    writer=fixed_writer,
                    max_steps=15,
                    fixed_assignment=assignment,
                )
            )
            for variant in ("L1", "L2"):
                for seed in MODEL_SEEDS_STAGE_C:
                    actor, digest = trained[(variant, int(seed))]
                    episodes.append(
                        run_episode(
                            int(scenario_id),
                            variant=variant,
                            model_seed=int(seed),
                            movement=movement_for(variant, actor),
                            decision=scheme_c_decision(planner, tap),
                            protocol="closed48",
                            split="dev",
                            lowlevel_sha256=digest,
                            hashes=hashes,
                            writer=closed_writer,
                        )
                    )
                    episodes.append(
                        run_episode(
                            int(scenario_id),
                            variant=variant,
                            model_seed=int(seed),
                            movement=movement_for(variant, actor),
                            decision=meta,
                            protocol="fixed15",
                            split="dev",
                            lowlevel_sha256=digest,
                            hashes=hashes,
                            writer=fixed_writer,
                            max_steps=15,
                            fixed_assignment=assignment,
                        )
                    )
            print(f"evaluated scenario {scenario_id}", flush=True)
    _write_csv(OUTPUT / "episode_rows.csv", episodes)
    return episodes


def summarize(episodes: list[dict], scale: dict, smoke_ok: bool) -> dict:
    failures = [row for row in episodes if row.get("status") != "ok"]
    closed = [row for row in episodes if row.get("protocol") == "closed48"]
    fixed = [row for row in episodes if row.get("protocol") == "fixed15"]
    scenarios = dev_scenario_ids()
    level = condition_level(closed, "closed48")
    fixed_level = condition_level(fixed, "fixed15")
    rng = np.random.default_rng(0)
    contrasts = {}
    for protocol_name, table in (("closed48", level), ("fixed15", fixed_level)):
        contrasts[protocol_name] = {}
        for left, right in (("L1", "L0"), ("L2", "L0"), ("L2", "L1")):
            contrasts[protocol_name][f"{left}-{right}"] = {
                metric: contrast(table, left, right, metric, scenarios, rng)
                for metric in ("payload_bits", "urgent_service_bits", "positive_service_steps", "path_length_m", "mean_sinr", "safety_violation_steps")
            }
    l2_rule = retain_l2(
        contrasts["closed48"]["L2-L1"]["payload_bits"],
        contrasts["closed48"]["L2-L1"]["urgent_service_bits"],
    )
    spreads = {
        "L1": seed_payload_spread(closed, "L1"),
        "L2": seed_payload_spread(closed, "L2"),
    }
    payload_unclear = any(not contrasts["closed48"][name]["payload_bits"].get("ci_excludes_zero", False) for name in ("L1-L0", "L2-L0"))
    unstable = _unstable(spreads, contrasts["closed48"])
    inconclusive = bool(payload_unclear or l2_rule == "TRADEOFF" or unstable)
    training_complete = all((CHECKPOINTS / f"{variant}_seed{seed}.pt").exists() for variant in ("L1", "L2") for seed in MODEL_SEEDS_STAGE_C)
    gate = choose_gate(
        smoke_ok=smoke_ok,
        training_complete=training_complete,
        eval_complete=len(closed) == 32 * 7 and len(fixed) == 32 * 7,
        failures=len(failures),
        inconclusive=inconclusive,
    )
    summary = {
        "gate": gate,
        "component_gates": {
            "smoke": "URGENCY_LOWLEVEL_INPUT_SMOKE_PASS" if smoke_ok else "URGENCY_LOWLEVEL_BLOCKED",
            "training": "URGENCY_LOWLEVEL_TRAINING_COMPLETE" if training_complete else "URGENCY_LOWLEVEL_BLOCKED",
            "evaluation": "URGENCY_LOWLEVEL_EVAL_COMPLETE" if len(failures) == 0 and len(closed) == 32 * 7 else "URGENCY_LOWLEVEL_BLOCKED",
        },
        "instruction_commit": INSTRUCTION_COMMIT,
        "lambda_urgency": float(LAMBDA_URGENCY),
        "model_seeds": list(MODEL_SEEDS_STAGE_C),
        "updates": int(UPDATES),
        "ppo": PPO.to_dict(),
        "l2_retention": l2_rule,
        "seed_payload_spread": spreads,
        "reward_scale": scale,
        "contrasts": contrasts,
        "failures": len(failures),
        "train_pool": [int(item) for item in train_scenario_ids()],
        "dev_pool": [int(item) for item in scenarios],
        "closed_loop_step_rows_bytes": (OUTPUT / "closed_loop_step_rows.csv").stat().st_size,
        "fixed_hold_step_rows_bytes": (OUTPUT / "fixed_hold_step_rows.csv").stat().st_size,
    }
    _write_json(OUTPUT / "summary.json", summary)
    _write_schema()
    _figure(closed)
    return summary


def _unstable(spreads: dict, closed: dict) -> bool:
    for variant, pair in (("L1", "L1-L0"), ("L2", "L2-L0")):
        effect = abs(float(closed[pair]["payload_bits"].get("mean", 0.0)))
        spread = float(spreads[variant].get("range_bits", 0.0))
        if spread > max(effect, 1.0) and spread > 1.0e6:
            return True
    return False


def _figure(closed_rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["L0", "L1", "L2"]
    payload = []
    urgent = []
    for label in labels:
        rows = [row for row in closed_rows if row.get("status") == "ok" and row.get("variant") == label]
        payload.append(float(np.mean([float(row["payload_bits"]) for row in rows])) / 1.0e6 if rows else 0.0)
        urgent.append(float(np.mean([float(row["urgent_service_bits"]) for row in rows])) / 1.0e6 if rows else 0.0)
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].bar(labels, payload, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[1].bar(labels, urgent, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[0].set_title("Closed-loop payload")
    axes[1].set_title("Urgency-weighted service")
    axes[0].set_ylabel("Mbit")
    axes[1].set_ylabel("Mbit")
    figure.tight_layout()
    destination = OUTPUT / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination / "l0_l1_l2_closed_loop.png", dpi=140)
    plt.close(figure)


def _hashes(planner: FormalPlanner) -> dict:
    return {
        "policy_sha256": str(planner.startup["scheme_c_sha256"]),
        "config_sha256": str(planner.startup["candidate_config_sha256"]),
        "mappo_sha256": str(planner.startup["mappo_sha256"]),
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPISODE_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in EPISODE_FIELDS})


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_schema() -> None:
    payload = {
        "actor_in_dim": 36,
        "formal_actor_in_dim": 34,
        "task_dim": 10,
        "formal_task_dim": 8,
        "critic_in_dim": 200,
        "formal_critic_in_dim": 188,
        "critic_extra": "per-UAV target urgency and validity, fixed UAV order, appended to the formal critic",
        "actor_extra": "current target urgency and validity inserted after the formal task block",
        "new_columns": "zero at warm start",
        "lambda_urgency": float(LAMBDA_URGENCY),
        "seeds": list(MODEL_SEEDS_STAGE_C),
        "updates": int(UPDATES),
        "checkpoint_hash_rows": "policy, config, MAPPO, and low-level hashes are stored only on UAV index 0",
        "frozen_mappo": str(primary_checkpoint()),
        "frozen_mappo_sha256": checkpoint_sha(primary_checkpoint()),
    }
    _write_json(OUTPUT / "schema.json", payload)


if __name__ == "__main__":
    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else "audit"
    if command == "audit":
        print(json.dumps(reward_scale_audit(), indent=2), flush=True)
    elif command == "train":
        train_all()
    elif command == "train-one":
        train_variant(sys.argv[2], int(sys.argv[3]), updates=int(UPDATES), ppo=PPO)
    elif command == "eval":
        episodes = evaluate_all()
        scale = json.loads((OUTPUT / "reward_scale.json").read_text(encoding="utf-8"))
        summary = summarize(episodes, scale, smoke_ok=True)
        print(summary["gate"], summary["l2_retention"], flush=True)
    else:
        raise SystemExit(f"unknown command {command}")
