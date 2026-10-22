"""Train the scaled residual actor and compare it with frozen MAPPO-V2 on the dev pool."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6g_lowlevel_mappo.worker import make_option, sample_assignment
from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.lowlevel_v2.observation import build_actor_observation

from .evaluate import evaluate_actor, load_folded_actor, load_primary, paired_summary, scale_only_actor
from .protocol import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CHECKPOINT_DIR,
    EVAL_POOL,
    MODEL_SEEDS,
    OUTPUT,
    PPO,
    SOURCE_ACTOR,
    TRAIN_POOL,
    UPDATES,
    assert_phase_output,
    physical_scales,
)
from .scale import first_layer_abs_mean, scale_actor_observation
from .train import load_warm_actor, source_sha256, train_seed


def _saturation_probe() -> dict:
    env = L3BBitQueueEnvV2()
    env.reset(seed=int(TRAIN_POOL[0]))
    rng = np.random.default_rng(0)
    option = make_option(env, sample_assignment(env, rng))
    raw = build_actor_observation(env, option, np.zeros((6, 2), dtype=np.float64))
    actor = load_warm_actor()
    actor.eval()
    return {
        "raw_first_layer_abs_mean": first_layer_abs_mean(actor, raw),
        "scaled_first_layer_abs_mean": first_layer_abs_mean(actor, scale_actor_observation(raw)),
    }


def _write_curve(histories: list[list[dict]], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [item for history in histories if history for item in history]
    header = [
        "model_seed",
        "update",
        "mean_reward",
        "mean_served_bits",
        "actor_loss",
        "critic_loss",
        "gradient_norm_preclip",
        "projection_rate",
        "safety_violation_rate",
        "explained_variance",
        "pre_tanh_abs",
        "tanh_saturation",
    ]
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row[key]) for key in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    figure = path.with_suffix(".png")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure_path = figure
        seeds = sorted({int(row["model_seed"]) for row in rows})
        fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
        fields = (
            ("mean_served_bits", "served bits / step"),
            ("mean_reward", "reward"),
            ("gradient_norm_preclip", "grad norm"),
            ("projection_rate", "projection rate"),
        )
        for ax, (key, title) in zip(axes.ravel(), fields):
            for seed in seeds:
                series = [row for row in rows if int(row["model_seed"]) == seed]
                ax.plot([row["update"] for row in series], [row[key] for row in series], label=f"seed {seed}")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figure_path, dpi=120)
        plt.close(fig)
        return str(figure_path)
    except Exception as exc:
        return f"csv_only:{type(exc).__name__}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=int(UPDATES))
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in MODEL_SEEDS))
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()
    seeds = tuple(int(item) for item in args.seeds.split(",") if item)
    for seed in seeds:
        if seed not in MODEL_SEEDS and seed not in range(0, 5):
            raise RuntimeError(f"unexpected seed {seed}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    before = source_sha256()
    probe = _saturation_probe()
    config = {
        "source_actor": str(SOURCE_ACTOR),
        "source_sha256": before,
        "updates": int(args.updates),
        "seeds": list(seeds),
        "train_pool": [int(TRAIN_POOL[0]), int(TRAIN_POOL[-1])],
        "eval_pool": [int(EVAL_POOL[0]), int(EVAL_POOL[-1])],
        "ppo": {
            "gamma": float(PPO.gamma),
            "gae_lambda": float(PPO.gae_lambda),
            "rollout_length": int(PPO.rollout_length),
            "num_envs": int(PPO.num_envs),
            "learning_rate": float(PPO.learning_rate),
        },
        "scales": physical_scales(),
        "reward_changed": False,
        "action_dim_changed": False,
        "safety_bounds_changed": False,
        "high_level_changed": False,
        "saturation_probe": probe,
    }
    config_path = assert_phase_output(OUTPUT / "config.json")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    histories = []
    checkpoints = []
    if not args.skip_train:
        for seed in seeds:
            path = assert_phase_output(CHECKPOINT_DIR / f"phase6ik_seed{seed}.pt")
            result = train_seed(int(seed), int(args.updates), path)
            histories.append(result["history"])
            checkpoints.append(path)
    else:
        checkpoints = [CHECKPOINT_DIR / f"phase6ik_seed{seed}.pt" for seed in seeds]
        for path in checkpoints:
            history_path = path.with_suffix(".history.json")
            histories.append(json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else [])
    curve = _write_curve(histories, OUTPUT / "training_curve.csv")
    summary = {"config": config, "curve": curve, "source_sha256_after_train": source_sha256(), "seeds": []}
    if args.skip_eval:
        (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"trained": True, "saturation": probe, "curve": curve}, sort_keys=True), flush=True)
        return
    scenarios = EVAL_POOL if int(args.eval_limit) <= 0 else EVAL_POOL[: int(args.eval_limit)]
    print("evaluating frozen MAPPO-V2", flush=True)
    baseline_rows = evaluate_actor(load_primary(), scenarios)
    print("evaluating scale-only actor", flush=True)
    scale_rows = evaluate_actor(scale_only_actor(), scenarios)
    scale_pair = paired_summary(baseline_rows, scale_rows, draws=int(BOOTSTRAP_DRAWS), seed=int(BOOTSTRAP_SEED))
    summary["scale_only_vs_frozen"] = scale_pair
    trained_pairs = []
    for path in checkpoints:
        print(f"evaluating {path.name}", flush=True)
        rows = evaluate_actor(load_folded_actor(path), scenarios)
        pair = paired_summary(baseline_rows, rows, draws=int(BOOTSTRAP_DRAWS), seed=int(BOOTSTRAP_SEED))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        trained_pairs.append({"checkpoint": str(path), "sha256": digest, "paired": pair, "rows": rows})
        summary["seeds"].append({"checkpoint": str(path), "sha256": digest, "paired": pair})
    (OUTPUT / "baseline_rows.json").write_text(json.dumps(baseline_rows), encoding="utf-8")
    (OUTPUT / "scale_only_rows.json").write_text(json.dumps(scale_rows), encoding="utf-8")
    (OUTPUT / "trained_rows.json").write_text(json.dumps(trained_pairs), encoding="utf-8")
    summary["source_sha256_after_eval"] = source_sha256()
    summary["source_unchanged"] = int(summary["source_sha256_after_eval"] == before)
    assert_phase_output(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"source_unchanged": summary["source_unchanged"], "scale_only_mean": scale_pair["mean"], "trained": [item["paired"]["mean"] for item in summary["seeds"]]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
