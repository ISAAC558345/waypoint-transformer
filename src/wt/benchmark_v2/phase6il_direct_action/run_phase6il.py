"""Phase-6I-L residual audit, direct-action smoke, and dev training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from wt.benchmark_v2.phase6ik_lowlevel_opt.evaluate import evaluate_actor
from wt.benchmark_v2.phase6h_executor_conditioned_macro.executor import load_primary

from .audit import write_audit
from .evaluate import evaluate_direct, load_direct_actor, paired_summary
from .protocol import (
    AUDIT_OUTPUT,
    BC_COSINE_MIN,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CHECKPOINT_DIR,
    EVAL_POOL,
    MODEL_SEEDS,
    OUTPUT,
    PHASE6IK_SUMMARY,
    UPDATES,
    assert_phase_output,
)
from .train import source_sha256, train_seed


def _gate(smoke_ok: bool, pairs: list[dict]) -> str:
    if not smoke_ok:
        return "LOW_LEVEL_ACTION_BLOCKED"
    supported = 0
    for pair in pairs:
        gain = float(pair["mean"]) > 0.0 and float(pair["ci_low"]) > 0.0
        safety_ok = float(pair["candidate_safety"]) <= float(pair["baseline_safety"]) + 0.02
        projection_ok = float(pair["candidate_projection"]) <= 0.85
        if gain and safety_ok and projection_ok and int(pair["incomplete"]) == 0:
            supported += 1
    if supported >= 2:
        return "LOW_LEVEL_ACTION_GAIN_SUPPORTED"
    return "LOW_LEVEL_ACTION_GAIN_INCONCLUSIVE"


def _curve(histories: list[list[dict]], path: Path) -> None:
    header = ["model_seed", "update", "mean_reward", "mean_served_bits", "actor_loss", "critic_loss", "gradient_norm_preclip", "projection_rate", "explained_variance", "bc_cosine"]
    lines = [",".join(header)]
    for history in histories:
        for row in history:
            lines.append(",".join(str(row[key]) for key in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=int(UPDATES))
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in MODEL_SEEDS))
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    seeds = tuple(int(item) for item in args.seeds.split(",") if item)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    before = source_sha256()
    audit = write_audit()
    print("residual audit written", flush=True)
    histories = []
    checkpoints = []
    clones = []
    smoke_ok = True
    if not args.skip_train:
        for seed in seeds:
            path = assert_phase_output(CHECKPOINT_DIR / f"phase6il_direct_seed{seed}.pt")
            result = train_seed(int(seed), int(args.updates), path)
            histories.append(result["history"])
            checkpoints.append(path)
            clones.append({"model_seed": int(seed), "bc_cosine": result["bc_cosine"], "bc_loss": result["bc_loss"]})
            if float(result["bc_cosine"]) < float(BC_COSINE_MIN):
                smoke_ok = False
    else:
        checkpoints = [CHECKPOINT_DIR / f"phase6il_direct_seed{seed}.pt" for seed in seeds]
    _curve(histories, OUTPUT / "training_curve.csv")
    scenarios = EVAL_POOL if int(args.eval_limit) <= 0 else EVAL_POOL[: int(args.eval_limit)]
    summary = {
        "audit_gate": "ACTION_MECHANISM_AUDIT_COMPLETE",
        "smoke_gate": "DIRECT_ACTION_SMOKE_PASS" if smoke_ok else "LOW_LEVEL_ACTION_BLOCKED",
        "residual_limits": audit["limits"],
        "bc": clones,
        "source_sha256": before,
        "loaded_phase6ik_as_policy": False,
        "seeds": [],
    }
    if not smoke_ok or args.skip_train:
        summary["gate"] = "LOW_LEVEL_ACTION_BLOCKED" if not smoke_ok else "DIRECT_ACTION_SMOKE_PASS"
        assert_phase_output(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"gate": summary["gate"], "bc": clones}, sort_keys=True), flush=True)
        return
    print("evaluating frozen residual MAPPO", flush=True)
    baseline = evaluate_actor(load_primary(), scenarios)
    pairs = []
    for path in checkpoints:
        print(f"evaluating {path.name}", flush=True)
        rows = evaluate_direct(load_direct_actor(path), scenarios)
        pair = paired_summary(baseline, rows, draws=int(BOOTSTRAP_DRAWS), seed=int(BOOTSTRAP_SEED))
        pair["candidate_correction_norm"] = float(np.mean([row["correction_norm"] for row in rows]))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        pairs.append(pair)
        summary["seeds"].append({"checkpoint": str(path), "sha256": digest, "paired": pair})
    phase6ik = json.loads(Path(PHASE6IK_SUMMARY).read_text(encoding="utf-8")) if Path(PHASE6IK_SUMMARY).is_file() else {}
    summary["phase6ik_reference"] = [
        {"sha256": item.get("sha256"), "mean": item.get("paired", {}).get("mean"), "ci_low": item.get("paired", {}).get("ci_low"), "ci_high": item.get("paired", {}).get("ci_high")}
        for item in phase6ik.get("seeds", [])
    ]
    summary["gate"] = _gate(smoke_ok, pairs)
    summary["source_unchanged"] = int(source_sha256() == before)
    summary["audit_file"] = str(AUDIT_OUTPUT / "residual_action_audit.json")
    assert_phase_output(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"gate": summary["gate"], "means": [item["paired"]["mean"] for item in summary["seeds"]], "source_unchanged": summary["source_unchanged"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
