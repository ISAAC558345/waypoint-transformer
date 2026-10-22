"""Phase-6I-M adaptive residual training. Existing checkpoints are not overwritten."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import load_primary
from wt.benchmark_v2.phase6iaa_formal_planner.planner import FormalPlanner
from wt.benchmark_v2.phase6ik_lowlevel_opt.evaluate import evaluate_actor

from .calibrate import build_calibration
from .evaluate import evaluate_adaptive, load_adaptive_actor, paired_summary, run_adaptive_episode, smoke_episode
from .protocol import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CHECKPOINT_DIR,
    EVAL_POOL,
    MAX_SPEED_M_S,
    MAX_TURN_DEG,
    MODEL_SEEDS,
    OUTPUT,
    UPDATES,
    assert_phase_output,
)
from .train import load_source, load_warm_actor, source_sha256, train_seed


def _gain_ok(pair: dict) -> bool:
    return (
        float(pair["mean"]) > 0.0
        and float(pair["ci_low"]) > 0.0
        and float(pair["candidate_safety"]) <= float(pair["baseline_safety"]) + 0.02
        and int(pair["incomplete"]) == 0
    )


def _failure_cases(baseline: list[dict], rows: list[dict]) -> list[dict]:
    left = {int(row["scenario_id"]): row for row in baseline}
    cases = []
    for row in rows:
        other = left[int(row["scenario_id"])]
        delta = float(row["payload_bits"]) - float(other["payload_bits"])
        cases.append(
            {
                "scenario_id": int(row["scenario_id"]),
                "delta_payload": delta,
                "payload_bits": float(row["payload_bits"]),
                "baseline_payload_bits": float(other["payload_bits"]),
                "entry_steps": float(row["mean_first_entry_steps"]),
                "service_steps": int(row["service_steps"]),
                "path_length_m": float(row["path_length_m"]),
                "projection_rate": float(row["projection_rate"]),
                "correction_norm": float(row["correction_norm"]),
                "command_turn_deg": float(row["command_turn_deg"]),
                "executed_turn_deg": float(row["executed_turn_deg"]),
                "excessive_heading_rate": float(row["excessive_heading_rate"]),
                "excessive_speed_rate": float(row["excessive_speed_rate"]),
                "collision_risk_rate": float(row["collision_risk_rate"]),
                "envelope_risk_rate": float(row["envelope_risk_rate"]),
                "safety_violation_rate": float(row["safety_violation_rate"]),
                "complete": bool(row["complete"]),
            }
        )
    cases.sort(key=lambda item: item["delta_payload"])
    return cases


def _plot_curve(csv_path: Path, png_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = csv_path.read_text(encoding="utf-8").splitlines()
    if len(rows) < 2:
        return
    header = rows[0].split(",")
    served_index = header.index("mean_served_bits")
    turn_index = header.index("command_turn_deg")
    seed_index = header.index("model_seed")
    update_index = header.index("update")
    by_seed: dict[str, list[tuple[int, float, float]]] = {}
    for line in rows[1:]:
        cells = line.split(",")
        by_seed.setdefault(cells[seed_index], []).append(
            (int(cells[update_index]), float(cells[served_index]), float(cells[turn_index]))
        )
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for seed, series in sorted(by_seed.items()):
        axes[0].plot([item[0] for item in series], [item[1] for item in series], label=f"seed {seed}")
        axes[1].plot([item[0] for item in series], [item[2] for item in series], label=f"seed {seed}")
    axes[0].set_title("served bits / step")
    axes[1].set_title("command turn (deg)")
    for axis in axes:
        axis.set_xlabel("update")
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(assert_phase_output(png_path), dpi=120)
    plt.close(figure)


def _curve(histories: list[list[dict]], path: Path) -> None:
    header = [
        "model_seed",
        "update",
        "mean_reward",
        "mean_served_bits",
        "actor_loss",
        "critic_loss",
        "gradient_norm_preclip",
        "explained_variance",
        "projection_rate",
        "command_turn_deg",
        "executed_turn_deg",
        "excessive_heading_rate",
        "excessive_speed_rate",
        "collision_risk_rate",
        "envelope_risk_rate",
        "safety_violation_rate",
    ]
    lines = [",".join(header)]
    for history in histories:
        for row in history:
            lines.append(",".join(str(row[key]) for key in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=int(UPDATES))
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in MODEL_SEEDS))
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    seeds = tuple(int(item) for item in args.seeds.split(",") if item)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    before = source_sha256()
    payload, digest = load_source()
    probe_actor = load_warm_actor(payload)
    calibration = build_calibration(probe_actor)
    if float(calibration["init_speed_median"]) < 0.90 or float(calibration["init_turn_abs_median"]) > 8.0:
        raise RuntimeError(f"adaptive initialization is outside the loaded policy: {calibration}")
    smoke = smoke_episode(probe_actor, calibration)
    if float(smoke["max_step_m"]) > float(MAX_SPEED_M_S) + 1.0e-4 or float(smoke["max_command_turn_deg"]) > float(MAX_TURN_DEG) + 1.0e-4:
        raise RuntimeError(f"smoke kinematics failed: {smoke}")
    planner = FormalPlanner("scheme_c")
    planner.actor = probe_actor
    dev_smoke = run_adaptive_episode(int(EVAL_POOL[0]), planner, calibration)
    if not dev_smoke["complete"] or float(dev_smoke["command_turn_deg"]) > float(MAX_TURN_DEG) + 1.0e-4:
        raise RuntimeError(f"dev smoke failed: {dev_smoke}")
    config = {
        "official_config": False,
        "max_turn_deg": float(MAX_TURN_DEG),
        "max_speed_m_s": float(MAX_SPEED_M_S),
        "source_sha256": digest,
        "loaded_phase6il_actor": False,
        "loaded_phase6ik_actor": False,
        "calibration": calibration,
        "smoke": smoke,
        "dev_smoke": dev_smoke,
        "updates": int(args.updates),
        "seeds": list(seeds),
    }
    assert_phase_output(OUTPUT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "smoke": smoke,
                "dev_smoke_payload": dev_smoke["payload_bits"],
                "dev_smoke_turn": dev_smoke["command_turn_deg"],
                "dev_smoke_projection": dev_smoke["projection_rate"],
                "init_turn": calibration["init_turn_abs_median"],
                "init_speed": calibration["init_speed_median"],
                "old_speed": calibration["old_speed_median"],
                "temperature": calibration["steer_temperature"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.probe_only:
        return
    histories = []
    checkpoints = []
    if not args.skip_train:
        for seed in seeds:
            path = assert_phase_output(CHECKPOINT_DIR / f"phase6im_adaptive_seed{seed}.pt")
            result = train_seed(int(seed), int(args.updates), path, calibration, payload)
            histories.append(result["history"])
            checkpoints.append(path)
            config["critic_mode"] = result["critic_mode"]
    _curve(histories, OUTPUT / "training_curve.csv")
    if args.skip_train:
        assert_phase_output(OUTPUT / "summary.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        return
    scenarios = EVAL_POOL if int(args.eval_limit) <= 0 else EVAL_POOL[: int(args.eval_limit)]
    print("evaluating frozen residual MAPPO", flush=True)
    baseline = evaluate_actor(load_primary(), scenarios)
    print("evaluating calibrated initialization", flush=True)
    init_rows = evaluate_adaptive(load_warm_actor(payload), calibration, scenarios)
    init_pair = paired_summary(baseline, init_rows, draws=int(BOOTSTRAP_DRAWS), seed=int(BOOTSTRAP_SEED))
    trained = []
    for path in checkpoints:
        print(f"evaluating {path.name}", flush=True)
        actor, stored = load_adaptive_actor(path)
        rows = evaluate_adaptive(actor, stored, scenarios)
        pair = paired_summary(baseline, rows, draws=int(BOOTSTRAP_DRAWS), seed=int(BOOTSTRAP_SEED))
        extras = {
            "command_turn_deg": float(np.mean([row["command_turn_deg"] for row in rows])),
            "executed_turn_deg": float(np.mean([row["executed_turn_deg"] for row in rows])),
            "correction_norm": float(np.mean([row["correction_norm"] for row in rows])),
            "excessive_heading_rate": float(np.mean([row["excessive_heading_rate"] for row in rows])),
            "excessive_speed_rate": float(np.mean([row["excessive_speed_rate"] for row in rows])),
            "collision_risk_rate": float(np.mean([row["collision_risk_rate"] for row in rows])),
            "envelope_risk_rate": float(np.mean([row["envelope_risk_rate"] for row in rows])),
        }
        trained.append(
            {
                "checkpoint": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "paired": pair,
                "projection": extras,
                "rows": rows,
                "failures": _failure_cases(baseline, rows)[:8],
            }
        )
    supported = sum(_gain_ok(item["paired"]) for item in trained)
    summary = {
        "replaced_official_low_level": False,
        "gain_supported": bool(supported >= 2),
        "ablation_run": False,
        "source_unchanged": int(source_sha256() == before),
        "config": config,
        "initialization_paired": init_pair,
        "seeds": [{key: value for key, value in item.items() if key != "rows"} for item in trained],
    }
    assert_phase_output(OUTPUT / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert_phase_output(OUTPUT / "scenario_rows.json").write_text(
        json.dumps({"baseline": baseline, "initialization": init_rows, "trained": trained}),
        encoding="utf-8",
    )
    failures = {str(Path(item["checkpoint"]).name): item["failures"] for item in trained}
    assert_phase_output(OUTPUT / "failure_cases.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    _plot_curve(OUTPUT / "training_curve.csv", OUTPUT / "training_curve.png")
    print(
        json.dumps(
            {
                "gain_supported": summary["gain_supported"],
                "init_mean": init_pair["mean"],
                "trained_means": [item["paired"]["mean"] for item in trained],
                "source_unchanged": summary["source_unchanged"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
