"""LL-V2.3 formal 5-seed independent validation. Frozen Phase-2D method."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from wt.lowlevel_v2.phase1b.controller import ResidualWorker, load_frozen_high, make_networks
from wt.lowlevel_v2.phase2.stats import bootstrap_deltas
from wt.lowlevel_v2.phase2b.train import make_worker as make_trust_worker
from wt.lowlevel_v2.phase3.eval import ValWorker, evaluate_validation
from wt.lowlevel_v2.phase3.protocol import (
    ALPHA_RESIDUAL,
    BETA_RESIDUAL,
    BOOT_SEED,
    D_SAFE_M,
    EPSILON_BASE_FRACTION,
    EXPECTED_HIGH_SHA,
    LOW_SEEDS,
    MANIFEST_FILE,
    N_BOOT,
    NEW_SEEDS,
    OUTPUT,
    PHASE2D,
    PHASE2D_CKPT,
    PHASE3,
    PILOT_UPDATES_N,
    PPO,
    PREFREEZE_FILE,
    REUSED_SEEDS,
    ROOT,
    SERVICE_SAFETY_MARGIN_M,
    action_max,
    env_sha,
    epsilon_base,
    high_sha,
    load_phase2d_gate,
    load_pools,
    load_val_seeds,
    projector_sha,
)
from wt.lowlevel_v2.phase3.scenarios import freeze_validation_manifest
from wt.lowlevel_v2.phase3.train import train_seed


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_prefreeze() -> dict:
    train, design = load_pools()
    p2d = load_phase2d_gate()
    payload = {
        "method": "ActiveSampleAwareServiceLockedResidualMAPPO",
        "P2D": p2d["P2D"],
        "active_sample_optimization_supported": True,
        "ServiceResidualLock": True,
        "BaseZeroResidualFreeze": True,
        "BaseRelativeResidualCap": True,
        "residual_norm_cap_empirically_active": False,
        "alpha_residual": ALPHA_RESIDUAL,
        "beta_residual": BETA_RESIDUAL,
        "epsilon_base_fraction": EPSILON_BASE_FRACTION,
        "epsilon_base": epsilon_base(),
        "a_max": action_max(),
        "d_safe_m": D_SAFE_M,
        "service_safety_margin_m": SERVICE_SAFETY_MARGIN_M,
        "gamma": PPO.gamma,
        "gae_lambda": PPO.gae_lambda,
        "clip_eps": PPO.clip_eps,
        "learning_rate": PPO.learning_rate,
        "value_coef": PPO.value_coef,
        "entropy_coef": PPO.entropy_coef,
        "max_grad_norm": PPO.max_grad_norm,
        "rollout_steps": int(PPO.rollout_length),
        "num_envs": int(PPO.num_envs),
        "update_epochs": int(PPO.update_epochs),
        "updates": PILOT_UPDATES_N,
        "reused_seeds": list(REUSED_SEEDS),
        "new_seeds": list(NEW_SEEDS),
        "high_checkpoint_sha": EXPECTED_HIGH_SHA,
        "safety_projector_sha": projector_sha(),
        "environment_sha": env_sha(),
        "train_seeds": [int(s) for s in train],
        "design_seeds": [int(s) for s in design],
        "no_method_change": True,
        "relation_enhancement": False,
        "phase2_validation_blocked": True,
    }
    _dump(PREFREEZE_FILE, payload)
    _dump(OUTPUT / "LLV2_PHASE3_PREFREEZE.json", payload)
    return payload


def load_ckpt(path: Path, *, allow_val: bool) -> ResidualWorker:
    high, _ = load_frozen_high()
    actor, critic = make_networks()
    payload = torch.load(path, map_location="cpu")
    actor.load_state_dict(payload["actor"])
    critic.load_state_dict(payload["critic"])
    actor.eval()
    critic.eval()
    if allow_val:
        return ValWorker(high, actor, critic, service_lock=True, trust_region=True)
    return make_trust_worker(high, actor, critic)


def mark_consumed() -> None:
    payload = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    payload["consumed"] = True
    _dump(MANIFEST_FILE, payload)


def plot_figures(curve, hist, per_seed, boot, l0_mean, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    names = ["L0"] + [f"s{s}" for s in LOW_SEEDS]
    vals = [l0_mean / 1e6] + [per_seed[s]["payload"] / 1e6 for s in LOW_SEEDS]
    ax.bar(names, vals, color=["#777"] + ["#1f77b4"] * 5)
    ax.axhline(l0_mean / 1e6, color="k", lw=0.6)
    ax.set_ylabel("mean payload / 1e6")
    fig.tight_layout()
    fig.savefig(out / "llv2_phase3_payload_comparison.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar([f"s{s}" for s in LOW_SEEDS], [per_seed[s]["delta"] / 1e6 for s in LOW_SEEDS], color="#2ca02c")
    ax.axhline(0, color="k")
    ax.set_ylabel("Delta vs L0 / 1e6")
    fig.tight_layout()
    fig.savefig(out / "llv2_phase3_per_seed_delta.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.errorbar(
        [0, 1, 2],
        [boot["overall_mean_delta"] / 1e6, boot["hierarchical_ci95"]["low"] / 1e6, boot["seed_cluster_ci95"]["low"] / 1e6],
        fmt="o",
    )
    ax.set_xticks([0, 1, 2], ["mean", "hier lo", "seed lo"])
    ax.axhline(0, color="k")
    ax.set_ylabel("bits / 1e6")
    fig.tight_layout()
    fig.savefig(out / "llv2_phase3_hierarchical_ci.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for seed in NEW_SEEDS:
        xs = [r["update"] for r in curve if int(r["model_seed"]) == seed]
        ys = [r["payload"] / 1e6 for r in curve if int(r["model_seed"]) == seed]
        ax.plot(xs, ys, marker="o", label=f"seed{seed}")
    ax.axhline(10.375, color="#333", ls="--", label="design L0")
    ax.legend()
    ax.set_xlabel("update")
    ax.set_ylabel("design payload / 1e6")
    fig.tight_layout()
    fig.savefig(out / "llv2_phase3_learning_curves.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for seed in LOW_SEEDS:
        xs = [r["update"] for r in hist if int(r["model_seed"]) == seed]
        if not xs:
            continue
        ax.plot(xs, [float(r["active_fraction"]) for r in hist if int(r["model_seed"]) == seed], label=f"seed{seed}")
    ax.set_ylabel("active fraction")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "llv2_phase3_active_fraction.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    for seed in LOW_SEEDS:
        xs = [r["update"] for r in hist if int(r["model_seed"]) == seed]
        if not xs:
            continue
        gkey = "actor_gradient_norm" if "actor_gradient_norm" in hist[0] else "gradient_norm_preclip"
        axes[0].plot(xs, [float(r.get(gkey, 0)) for r in hist if int(r["model_seed"]) == seed], label=f"s{seed}")
        axes[1].plot(xs, [float(r["approx_kl"]) for r in hist if int(r["model_seed"]) == seed], label=f"s{seed}")
    axes[0].set_title("actor grad")
    axes[1].set_title("KL")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "llv2_phase3_actor_grad_kl.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar([f"s{s}" for s in LOW_SEEDS], [per_seed[s]["min_sep_worst"] for s in LOW_SEEDS])
    ax.axhline(1.0, color="k")
    ax.set_ylabel("min sep worst (m)")
    fig.tight_layout()
    fig.savefig(out / "llv2_phase3_min_sep.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar([f"s{s}" for s in LOW_SEEDS], [per_seed[s]["projector_intervention_rate"] for s in LOW_SEEDS])
    ax.set_ylabel("projector intervention")
    fig.tight_layout()
    fig.savefig(out / "llv2_phase3_projector.png", dpi=130)
    plt.close(fig)


def plot_relation(rel_csv: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rel_csv:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    variants = ["neighbor_shuffle", "interf_shuffle", "sinr_shuffle"]
    regimes = ["near", "high", "all"]
    xs = np.arange(len(variants))
    width = 0.25
    for i, regime in enumerate(regimes):
        ys = []
        for v in variants:
            vals = [float(r["mean_l2"]) for r in rel_csv if r["variant"] == v and r["regime"] == regime]
            ys.append(float(np.mean(vals)) if vals else 0.0)
        ax.bar(xs + (i - 1) * width, ys, width, label=regime)
    ax.set_xticks(xs, variants)
    ax.set_ylabel("mean Actor pre-tanh L2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "llv2_phase3_relation_sensitivity.png", dpi=130)
    plt.close(fig)


def relation_gate(rel_csv: list[dict]) -> str:
    by_seed = {}
    for row in rel_csv:
        by_seed.setdefault(int(row["model_seed"]), []).append(row)
    if not by_seed:
        return "0"

    def mean_l2(rows, variant, regime):
        xs = [float(r["mean_l2"]) for r in rows if r["variant"] == variant and r["regime"] == regime]
        return float(np.mean(xs)) if xs else 0.0

    strong = 0
    weak = 0
    none = 0
    for seed, rows in by_seed.items():
        near = mean_l2(rows, "neighbor_shuffle", "near")
        high_i = mean_l2(rows, "interf_shuffle", "high")
        all_n = mean_l2(rows, "neighbor_shuffle", "all")
        if near > 0.05 or high_i > 0.05:
            strong += 1
        elif near > 2.0 * max(all_n, 1e-8) or high_i > 0.01:
            weak += 1
        else:
            none += 1
    n = max(len(by_seed), 1)
    if strong >= (n + 1) // 2 and strong >= 3:
        return "2"
    if none >= n - 1:
        return "0"
    return "1"


def gate_decision(per_seed, boot, l0, hist, rel2) -> dict:
    deltas = [per_seed[s]["delta"] for s in LOW_SEEDS]
    n_pos = int(sum(d > 0 for d in deltas))
    mean_d = float(np.mean(deltas))
    med_d = float(np.median(deltas))
    hier_lo = boot["hierarchical_ci95"]["low"]
    seed_lo = boot["seed_cluster_ci95"]["low"]
    safe = all(
        per_seed[s]["collision_count"] == 0
        and per_seed[s]["projector_failure_count"] == 0
        and per_seed[s]["service_lock_violation_count"] == 0
        and per_seed[s]["min_sep_worst"] + 1e-6 >= 1.0
        for s in LOW_SEEDS
    )
    blocks_ok = float(np.mean([per_seed[s]["completed_blocks"] for s in LOW_SEEDS])) + 1e-9 >= l0["completed_blocks"]
    zero_mb = any(float(r.get("min_actor_minibatch_size", 1)) <= 0 and float(r.get("active_samples", 0)) > 0 for r in hist)
    adv_ok = all(abs(float(r["active_advantage_mean_after"])) < 0.05 for r in hist if float(r.get("active_samples", 0)) > 1)
    opt_ok = (not zero_mb) and adv_ok
    if mean_d > 0 and hier_lo > 0 and seed_lo > 0 and n_pos >= 4 and med_d >= 0 and safe and blocks_ok and opt_ok:
        letter, stable = "A", True
        text = "5-seed independent validation supports a stable residual advantage over L0."
    elif mean_d > 0 and safe and (hier_lo <= 0 or seed_lo <= 0 or n_pos == 3):
        letter, stable = "B", False
        text = "Positive mean trend but CI crosses 0 or only 3/5 seeds positive. Do not add relation modules yet."
    else:
        letter, stable = "C", False
        text = "Residual MAPPO does not stably beat L0 on independent validation, or safety failed."
    next_step = "STOP"
    if letter == "A" and rel2 in {"0", "1"}:
        next_step = "optional dedicated collision/interference relation utilization (not automatic)"
    elif letter == "A" and rel2 == "2":
        next_step = "no extra relation complexity; proceed to more formal system validation if needed"
    elif letter == "B":
        next_step = "explain remaining seed variance; do not add relation modules"
    else:
        next_step = "stop further RL complexity"
    return {
        "LLV2_MS2": letter,
        "stable_residual_advantage": stable,
        "REL2": rel2,
        "REL": 1,
        "n_positive_seeds": n_pos,
        "overall_mean_delta": mean_d,
        "overall_median_delta": med_d,
        "hierarchical_ci95": boot["hierarchical_ci95"],
        "seed_cluster_ci95": boot["seed_cluster_ci95"],
        "all_safe": safe,
        "completed_blocks_not_worse": blocks_ok,
        "active_opt_ok": opt_ok,
        "residual_norm_cap_redundant": all(per_seed[s]["cap_activation_rate"] < 1e-6 for s in LOW_SEEDS),
        "next_step": next_step,
        "text": text,
        "trained_new_seeds": list(NEW_SEEDS),
        "reused_phase2d_seeds": list(REUSED_SEEDS),
        "did_not_modify_relation": True,
        "did_not_retrain_seed0_seed1": True,
        "phase2_validation_blocked": True,
    }


def write_report(pre, manifest, per_seed, l0, boot, gate, hist) -> str:
    def af(seed):
        xs = [float(r["active_fraction"]) for r in hist if int(r["model_seed"]) == seed]
        return float(np.mean(xs)) if xs else float("nan")

    lines = [
        "# HTW Low-Level V2 Phase-3：正式五种子独立验证（LL-V2.3）",
        "",
        "方法完全冻结自 Phase-2D（P2D-A）。未改 Actor 优化、ServiceResidualLock、BaseZeroResidualFreeze、reward、网络、超参、高层。seed0/seed1 复用 Phase-2D u100，仅新训 seed2/3/4。未访问 222000–222063 与 106000–106063。Primary 仅 u100。",
        "正式结论在本阶段由 MS2 gate 更新；relation 模块未修改。beta cap 若仍≈0 则标记 redundant，不作为主要贡献。",
        "",
        f"independent validation `{manifest['name']}` n={manifest['n']}，environment SHA={manifest['environment_sha']}。",
        f"L0 mean payload={l0['payload']:.1f}。",
        *[f"seed{s} payload={per_seed[s]['payload']:.1f} Delta={per_seed[s]['delta']:+.1f}" for s in LOW_SEEDS],
        f"positive seeds={gate['n_positive_seeds']}/5。overall mean delta={gate['overall_mean_delta']:+.1f}。median={gate['overall_median_delta']:+.1f}。",
        f"hierarchical 95% CI={gate['hierarchical_ci95']}。seed-cluster 95% CI={gate['seed_cluster_ci95']}。",
        f"collision={[per_seed[s]['collision_count'] for s in LOW_SEEDS]}。projector_failure={[per_seed[s]['projector_failure_count'] for s in LOW_SEEDS]}。",
        f"cap={[per_seed[s]['cap_activation_rate'] for s in LOW_SEEDS]}。",
        f"Gate **LLV2-MS2-{gate['LLV2_MS2']}**。stable_residual_advantage={gate['stable_residual_advantage']}。REL2={gate['REL2']}。",
        f"下一步：{gate['next_step']}",
        "未自动修改 relation 模块。",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    p2d = load_phase2d_gate()
    pre = write_prefreeze()
    if MANIFEST_FILE.exists():
        manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        if manifest.get("consumed") is True:
            raise RuntimeError("Phase-3 validation already consumed; refusing to regenerate")
    else:
        manifest = freeze_validation_manifest(MANIFEST_FILE)
    _dump(OUTPUT / "LLV2_PHASE3_VALIDATION_MANIFEST.json", manifest)
    protocol = {
        "name": "LL-V2.3",
        "method_frozen_from": "LL-V2.2D",
        "P2D": p2d["P2D"],
        "reused_seeds": list(REUSED_SEEDS),
        "new_seeds": list(NEW_SEEDS),
        "validation": manifest["name"],
        "phase2_validation": "blocked",
        "high_sha": high_sha(),
        "no_method_change": True,
    }
    _dump(OUTPUT / "LLV2_PHASE3_PROTOCOL.json", protocol)
    train_seeds, design_seeds = load_pools()
    sha_before = high_sha()
    all_hist, all_curve = [], []
    p2d_hist = list(csv.DictReader((PHASE2D / "LLV2_PHASE2D_TRAINING_HISTORY.csv").open(encoding="utf-8")))
    p2d_curve = list(csv.DictReader((PHASE2D / "LLV2_PHASE2D_EVAL_CURVE.csv").open(encoding="utf-8")))
    all_hist.extend(p2d_hist)
    for row in p2d_curve:
        if int(row["model_seed"]) in REUSED_SEEDS:
            all_curve.append(row)
    ckpt_manifest = []
    for seed in REUSED_SEEDS:
        src = PHASE2D_CKPT / f"ckpt_seed{seed}_update100.pt"
        dest = OUTPUT / f"ckpt_seed{seed}_update100.pt"
        shutil.copy2(src, dest)
        ckpt_manifest.append({"model_seed": int(seed), "update": 100, "path": str(dest), "reused_phase2d": True, "primary": True})
    for seed in NEW_SEEDS:
        result = train_seed(int(seed), train_seeds, design_seeds, output_dir=OUTPUT)
        all_hist.extend(result["history"])
        all_curve.extend(result["curve"])
        ckpt_manifest.append({"model_seed": int(seed), "update": 100, "path": result["final"], "reused_phase2d": False, "primary": True})
    if high_sha() != sha_before or high_sha() != EXPECTED_HIGH_SHA:
        raise RuntimeError("high SHA changed during Phase-3 training")
    _csv(OUTPUT / "LLV2_PHASE3_TRAINING_HISTORY.csv", all_hist)
    _csv(OUTPUT / "LLV2_PHASE3_EVAL_CURVE.csv", all_curve)
    _dump(OUTPUT / "LLV2_PHASE3_TRAINING_SEEDS.json", {"reused": list(REUSED_SEEDS), "trained": list(NEW_SEEDS), "budget_env": 19200, "budget_agent": 115200})
    _dump(OUTPUT / "LLV2_PHASE3_CHECKPOINT_MANIFEST.json", {"primary": "update100", "items": ckpt_manifest})
    val_seeds = load_val_seeds()
    mark_consumed()
    consumed_manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    _dump(OUTPUT / "LLV2_PHASE3_VALIDATION_MANIFEST.json", consumed_manifest)
    high, _ = load_frozen_high()
    actor, critic = make_networks()
    l0w = ValWorker(high, actor, critic, service_lock=True, trust_region=False)
    l0 = evaluate_validation(l0w, val_seeds, zero_residual=True, diagnose=False)
    summaries = {}
    per_seed = {}
    rel_csv, sep_csv, interf_csv = [], [], []
    per_scen_rows = []
    for rec in l0["scenarios"]:
        per_scen_rows.append({"model_seed": "L0", **rec, "delta": 0.0})
    l0_pay = {int(r["scenario"]): r["payload"] for r in l0["scenarios"]}
    delta_mat = np.zeros((5, len(val_seeds)), dtype=np.float64)
    for i, seed in enumerate(LOW_SEEDS):
        w = load_ckpt(OUTPUT / f"ckpt_seed{seed}_update100.pt", allow_val=True)
        summary = evaluate_validation(w, val_seeds, zero_residual=False, diagnose=True)
        summaries[seed] = summary
        pays = {int(r["scenario"]): r["payload"] for r in summary["scenarios"]}
        deltas = [pays[s] - l0_pay[s] for s in val_seeds]
        delta_mat[i] = np.asarray(deltas)
        per_seed[seed] = {
            "payload": summary["payload"],
            "delta": float(np.mean(deltas)),
            "median_delta": float(np.median(deltas)),
            "positive_scenario_fraction": float(np.mean(np.asarray(deltas) > 0)),
            "relative_gain": float(np.mean(deltas) / max(l0["payload"], 1.0)),
            **{k: summary[k] for k in summary if k not in {"scenarios", "relation", "separation", "interference_response"}},
        }
        for rec in summary["scenarios"]:
            per_scen_rows.append({"model_seed": int(seed), **rec, "delta": rec["payload"] - l0_pay[int(rec["scenario"])]})
        for row in summary.get("relation", []):
            rel_csv.append({"model_seed": int(seed), **row})
        sep_csv.append({"model_seed": int(seed), **summary.get("separation", {})})
        interf_csv.append({"model_seed": int(seed), **summary.get("interference_response", {})})
    boot = bootstrap_deltas(delta_mat, n_boot=N_BOOT, seed=BOOT_SEED)
    seed_metric_rows = [{"model_seed": int(s), **per_seed[s]} for s in LOW_SEEDS]
    seed_metric_rows.append({"model_seed": "L0", "payload": l0["payload"], "delta": 0.0, **{k: l0[k] for k in l0 if k not in {"scenarios", "relation"}}})
    _csv(OUTPUT / "LLV2_PHASE3_PER_SEED_METRICS.csv", seed_metric_rows)
    _csv(OUTPUT / "LLV2_PHASE3_PER_SCENARIO_METRICS.csv", per_scen_rows)
    active_rows = []
    opt_rows = []
    for r in all_hist:
        active_rows.append(
            {
                "model_seed": r["model_seed"],
                "update": r["update"],
                "total_agent_samples": r.get("total_agent_samples"),
                "active_samples": r.get("active_samples"),
                "active_fraction": r.get("active_fraction", r.get("policy_active_fraction")),
                "actor_minibatch_count": r.get("actor_minibatch_count"),
                "min_actor_minibatch_size": r.get("min_actor_minibatch_size"),
                "max_actor_minibatch_size": r.get("max_actor_minibatch_size"),
                "mean_actor_minibatch_size": r.get("mean_actor_minibatch_size"),
                "actor_optimizer_steps": r.get("actor_optimizer_steps"),
                "active_advantage_mean_before": r.get("active_advantage_mean_before"),
                "active_advantage_std_before": r.get("active_advantage_std_before"),
                "active_advantage_mean_after": r.get("active_advantage_mean_after"),
                "active_advantage_std_after": r.get("active_advantage_std_after"),
            }
        )
        opt_rows.append(
            {
                "model_seed": r["model_seed"],
                "update": r["update"],
                "policy_loss": r.get("policy_loss"),
                "entropy": r.get("entropy"),
                "approx_kl": r.get("approx_kl"),
                "clip_fraction": r.get("clip_fraction"),
                "actor_gradient_norm": r.get("actor_gradient_norm", r.get("gradient_norm_preclip")),
                "value_loss": r.get("value_loss"),
                "critic_gradient_norm": r.get("critic_gradient_norm"),
                "explained_variance": r.get("explained_variance"),
            }
        )
    _csv(OUTPUT / "LLV2_PHASE3_ACTIVE_SAMPLE_STATS.csv", active_rows)
    _csv(OUTPUT / "LLV2_PHASE3_OPTIMIZATION.csv", opt_rows)
    _csv(
        OUTPUT / "LLV2_PHASE3_SAFETY.csv",
        [
            {
                "model_seed": int(s),
                "collision_count": per_seed[s]["collision_count"],
                "min_sep_worst": per_seed[s]["min_sep_worst"],
                "projector_failure_count": per_seed[s]["projector_failure_count"],
                "service_lock_violation_count": per_seed[s]["service_lock_violation_count"],
                "near_events": per_seed[s]["near_events"],
                "projector_intervention_rate": per_seed[s]["projector_intervention_rate"],
            }
            for s in LOW_SEEDS
        ],
    )
    _csv(
        OUTPUT / "LLV2_PHASE3_RESIDUAL_STATS.csv",
        [
            {
                "model_seed": int(s),
                "mean_raw_residual": per_seed[s]["mean_raw_residual"],
                "mean_used_residual": per_seed[s]["mean_used_residual"],
                "service_lock_rate": per_seed[s]["service_lock_rate"],
                "base_zero_freeze_rate": per_seed[s]["base_zero_freeze_rate"],
                "cap_activation_rate": per_seed[s]["cap_activation_rate"],
            }
            for s in LOW_SEEDS
        ],
    )
    _csv(OUTPUT / "LLV2_PHASE3_RELATION_UTILIZATION.csv", rel_csv)
    _csv(OUTPUT / "LLV2_PHASE3_SEPARATION_RESPONSE.csv", sep_csv)
    _csv(OUTPUT / "LLV2_PHASE3_INTERFERENCE_RESPONSE.csv", interf_csv)
    _dump(OUTPUT / "LLV2_PHASE3_BOOTSTRAP.json", boot)
    rel2 = relation_gate(rel_csv)
    gate = gate_decision(per_seed, boot, l0, all_hist, rel2)
    _dump(OUTPUT / "LLV2_PHASE3_GATE.json", gate)
    plot_figures(all_curve, all_hist, per_seed, boot, l0["payload"], OUTPUT / "figures")
    plot_relation(rel_csv, OUTPUT / "figures")
    report = write_report(pre, manifest, per_seed, l0, boot, gate, all_hist)
    name = "HTW_LowLevelV2_Phase3_Formal_Multiseed_CN.md"
    _dump(OUTPUT / name, report)
    for src in list(OUTPUT.glob("LLV2_PHASE3_*")) + [OUTPUT / name, PREFREEZE_FILE, MANIFEST_FILE]:
        if src.exists() and src.is_file():
            (PHASE3 / src.name).write_bytes(src.read_bytes())
            (ROOT / src.name).write_bytes(src.read_bytes())
    fig_dir = PHASE3 / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for png in (OUTPUT / "figures").glob("*.png"):
        (fig_dir / png.name).write_bytes(png.read_bytes())
    (ROOT / ("docs/archive/phase_reports" if str(name).endswith(".md") else "results/root_exports") / name).write_text(report, encoding="utf-8")
    return {
        "gate": gate["LLV2_MS2"],
        "REL2": rel2,
        "mean_delta": gate["overall_mean_delta"],
        "L0": l0["payload"],
        "n_pos": gate["n_positive_seeds"],
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
