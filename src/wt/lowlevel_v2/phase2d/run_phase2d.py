"""LL-V2.2D active-sample-aware Actor optimization pilot. Seeds 0 and 1 only."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from wt.lowlevel_v2.phase1b.controller import ResidualWorker, load_frozen_high, make_networks
from wt.lowlevel_v2.phase2b.eval import evaluate_design
from wt.lowlevel_v2.phase2b.train import make_worker
from wt.lowlevel_v2.phase2d.protocol import (
    ALPHA_RESIDUAL,
    BETA_RESIDUAL,
    DEV_SEEDS,
    D_SAFE_M,
    EPSILON_BASE_FRACTION,
    EXPECTED_HIGH_SHA,
    LOG_STD_INIT,
    OLD_SEED1_CKPT,
    OUTPUT,
    PHASE1B_CKPT0,
    PHASE2B,
    PHASE2D,
    PILOT_UPDATES_N,
    PPO,
    PREFREEZE_FILE,
    ROOT,
    SERVICE_SAFETY_MARGIN_M,
    action_max,
    epsilon_base,
    high_sha,
    load_phase2c_gate,
    load_pools,
    projector_sha,
)
from wt.lowlevel_v2.phase2d.tests import run_tests
from wt.lowlevel_v2.phase2d.train import train_seed


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
    p2c = load_phase2c_gate()
    payload = {
        "only_change": "ActiveSampleAwareActorOptimization",
        "ServiceResidualLock": True,
        "BaseZeroResidualFreeze": True,
        "BaseRelativeResidualCap": True,
        "beta_residual": BETA_RESIDUAL,
        "epsilon_base_fraction": EPSILON_BASE_FRACTION,
        "epsilon_base": epsilon_base(),
        "a_max": action_max(),
        "alpha_residual": ALPHA_RESIDUAL,
        "d_safe_m": D_SAFE_M,
        "service_safety_margin_m": SERVICE_SAFETY_MARGIN_M,
        "MASK": p2c["MASK"],
        "BASEZERO": p2c["BASEZERO"],
        "actor_active_only_advantage": True,
        "actor_active_only_minibatch": True,
        "critic_unchanged": True,
        "reward_unchanged": True,
        "network_unchanged": True,
        "lr_unchanged": True,
        "entropy_coef": PPO.entropy_coef,
        "clip_eps": PPO.clip_eps,
        "update_epochs": int(PPO.update_epochs),
        "max_actor_minibatches": int(PPO.num_minibatches),
        "no_oversample_active": True,
        "loaded_phase2b_weights": False,
        "beta_grid_search": False,
        "tuned_from_payload": False,
        "log_std_init": LOG_STD_INIT,
        "updates": PILOT_UPDATES_N,
        "num_envs": int(PPO.num_envs),
        "rollout_steps": int(PPO.rollout_length),
        "high_checkpoint_sha": EXPECTED_HIGH_SHA,
        "safety_projector_sha": projector_sha(),
        "train_seeds": [int(s) for s in train],
        "design_seeds": [int(s) for s in design],
        "dev_model_seeds": list(DEV_SEEDS),
        "phase2_validation_blocked": True,
        "relation_enhancement": False,
        "frozen_before_any_training": True,
    }
    _dump(PREFREEZE_FILE, payload)
    _dump(OUTPUT / "LLV2_PHASE2D_PREFREEZE.json", payload)
    return payload


def load_ckpt(path: Path, *, trust_region: bool) -> ResidualWorker:
    high, _ = load_frozen_high()
    actor, critic = make_networks()
    payload = torch.load(path, map_location="cpu")
    actor.load_state_dict(payload["actor"])
    critic.load_state_dict(payload["critic"])
    actor.eval()
    critic.eval()
    if trust_region:
        return make_worker(high, actor, critic)
    return ResidualWorker(high, actor, critic, service_lock=True, trust_region=False)


def plot_figures(curve: list[dict], hist: list[dict], rows: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    l0 = rows["L0"]["payload"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for seed, color in ((0, "#1f77b4"), (1, "#d62728")):
        xs = [r["update"] for r in curve if int(r["model_seed"]) == seed]
        ys = [r["payload"] / 1e6 for r in curve if int(r["model_seed"]) == seed]
        ax.plot(xs, ys, marker="o", color=color, label=f"new seed{seed}")
    ax.axhline(l0 / 1e6, color="#333", ls="--", label="L0")
    ax.set_xlabel("update")
    ax.set_ylabel("payload / 1e6")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "llv2_phase2d_payload_curves.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for seed, color in ((0, "#1f77b4"), (1, "#d62728")):
        xs = [r["update"] for r in hist if int(r["model_seed"]) == seed]
        ys = [float(r["active_fraction"]) for r in hist if int(r["model_seed"]) == seed]
        ax.plot(xs, ys, color=color, label=f"seed{seed}")
    ax.set_xlabel("update")
    ax.set_ylabel("active fraction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "llv2_phase2d_active_fraction.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for seed, color in ((0, "#1f77b4"), (1, "#d62728")):
        xs = [r["update"] for r in hist if int(r["model_seed"]) == seed]
        ax.plot(xs, [float(r["mean_actor_minibatch_size"]) for r in hist if int(r["model_seed"]) == seed], color=color, label=f"mean seed{seed}")
        ax.plot(xs, [float(r["min_actor_minibatch_size"]) for r in hist if int(r["model_seed"]) == seed], color=color, ls="--", alpha=0.6, label=f"min seed{seed}")
    ax.set_xlabel("update")
    ax.set_ylabel("actor minibatch size")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "llv2_phase2d_actor_minibatch_size.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.3))
    for seed, color in ((0, "#1f77b4"), (1, "#d62728")):
        xs = [r["update"] for r in hist if int(r["model_seed"]) == seed]
        axes[0].plot(xs, [float(r["approx_kl"]) for r in hist if int(r["model_seed"]) == seed], color=color, label=f"seed{seed}")
        axes[1].plot(xs, [float(r["clip_fraction"]) for r in hist if int(r["model_seed"]) == seed], color=color, label=f"seed{seed}")
        axes[2].plot(xs, [float(r["actor_gradient_norm"]) for r in hist if int(r["model_seed"]) == seed], color=color, label=f"seed{seed}")
    axes[0].set_title("Actor KL")
    axes[1].set_title("clip fraction")
    axes[2].set_title("actor grad")
    for ax in axes:
        ax.set_xlabel("update")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "llv2_phase2d_actor_kl_clip_grad.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for seed, color in ((0, "#1f77b4"), (1, "#d62728")):
        xs = [r["update"] for r in curve if int(r["model_seed"]) == seed]
        ax.plot(xs, [r["arrival_steps"] for r in curve if int(r["model_seed"]) == seed], marker="o", color=color, label=f"seed{seed}")
    ax.axhline(rows["L0"]["mean_arrival_steps"], color="#333", ls="--", label="L0")
    ax.set_xlabel("update")
    ax.set_ylabel("arrival steps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "llv2_phase2d_arrival_steps.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    names = ["L0", "new0", "new1", "p2b0", "p2b1"]
    vals = [rows[k]["payload"] / 1e6 for k in names]
    ax.bar(names, vals, color=["#777", "#1f77b4", "#d62728", "#9ecae1", "#fcae91"])
    ax.axhline(l0 / 1e6, color="k", lw=0.6)
    ax.set_ylabel("payload / 1e6")
    fig.tight_layout()
    fig.savefig(out / "llv2_phase2d_payload_vs_L0.png", dpi=130)
    plt.close(fig)


def gate_decision(rows: dict, hist: list[dict], tests: dict) -> dict:
    l0, n0, n1, b0, b1 = rows["L0"], rows["new0"], rows["new1"], rows["p2b0"], rows["p2b1"]

    def safe(r):
        return (
            r["collision_count"] == 0
            and r["projector_failure_count"] == 0
            and r["service_lock_violation_count"] == 0
            and r["min_sep_worst"] + 1.0e-6 >= 1.0
        )

    s0_ge = n0["payload"] + 1.0e-6 >= l0["payload"]
    s1_ge = n1["payload"] + 1.0e-6 >= l0["payload"]
    both_safe = safe(n0) and safe(n1)
    adv_ok = all(abs(float(r["active_advantage_mean_after"])) < 0.05 for r in hist if float(r["active_samples"]) > 1)
    tiny = any(float(r["min_actor_minibatch_size"]) <= 0 for r in hist)
    huge = any(float(r["actor_gradient_norm"]) > 50.0 for r in hist)
    s0_loss = n0["payload"] + 2.5e5 < b0["payload"]
    s1_keep = n1["payload"] + 1.0e-6 >= min(b1["payload"], l0["payload"]) - 1.0e5
    if s0_ge and s1_ge and both_safe and adv_ok and (not tiny) and (not huge) and (not s0_loss) and s1_keep:
        letter = "A"
        text = "both seeds >= L0, safety pass, active advantage normalized, no empty actor batch, seed1 improvement held."
        supported = True
    elif both_safe and (s1_ge or n1["payload"] > b1["payload"] - 1.0e5) and (not s0_ge or abs(n0["payload"] - n1["payload"]) > 5.0e5):
        letter = "B"
        text = "active-sample optimization helps but is not sufficient (seed0 below L0 or remaining seed gap)."
        supported = False
    else:
        letter = "C"
        text = "no improvement or safety/optimization anomaly."
        supported = False
    return {
        "P2D": letter,
        "active_sample_optimization_supported": supported,
        "new_seed0_ge_L0": s0_ge,
        "new_seed1_ge_L0": s1_ge,
        "both_safe": both_safe,
        "active_advantage_normalized": adv_ok,
        "zero_or_empty_actor_batch": tiny,
        "abnormal_actor_grad": huge,
        "seed0_no_obvious_loss_vs_p2b": (not s0_loss),
        "seed1_keeps_improvement": s1_keep,
        "reenter_5seed_independent_validation": bool(supported),
        "text": text,
        "trained_seeds": [0, 1],
        "did_not_train_seeds_2_3_4": True,
        "trained": True,
        "tests_pass": tests["all_pass"],
        "phase2_conclusion_unchanged": {"LLV2_MS": "B", "stable_residual_advantage": False, "REL": 1},
        "cap_attribution_forbidden": True,
        "mechanisms": ["ServiceResidualLock", "BaseZeroResidualFreeze", "ActiveSampleAwareActorOptimization"],
    }


def write_report(pre, tests, rows, gate, curve, hist) -> str:
    l0, n0, n1, b0, b1 = rows["L0"], rows["new0"], rows["new1"], rows["p2b0"], rows["p2b1"]

    def curve_txt(seed):
        pts = [r for r in curve if int(r["model_seed"]) == seed]
        return ", ".join(f"u{int(r['update'])}={r['payload']/1e6:.3f}e6" for r in pts)

    def mean_hist(seed, key):
        xs = [float(r[key]) for r in hist if int(r["model_seed"]) == seed]
        return float(np.mean(xs)) if xs else float("nan")

    lines = [
        "# HTW Low-Level V2 Phase-2D：Active-Sample-Aware Actor Optimization Pilot（LL-V2.2D）",
        "",
        "本阶段唯一方法改动是 Actor 的 active-sample-aware 优化：active-only advantage 归一化 + active-only minibatching。未改 reward / lr / clip / entropy 系数 / update epochs / alpha / beta / epsilon_base / d_safe / ServiceResidualLock / BaseZeroResidualFreeze / 网络 / Projector / 高层。beta cap 保留但不得写成贡献。未访问 222000–222063。只训练 seed0 与 seed1，从头初始化。Primary=update100。",
        "",
        "正式结论保持：LLV2-MS-B，stable_residual_advantage=false，REL-1。Phase-2C：MASK-B，BASEZERO-A。",
        "",
        f"测试 {tests['n_pass']}/{tests['n']} PASS。预算每 seed 100×48×4=19200 env / 115200 agent。entropy_coef={pre['entropy_coef']}。",
        f"L0={l0['payload']:.1f}。new0={n0['payload']:.1f} (Δ {n0['payload']-l0['payload']:+.1f})。new1={n1['payload']:.1f} (Δ {n1['payload']-l0['payload']:+.1f})。",
        f"Phase-2B seed0/1={b0['payload']:.1f}/{b1['payload']:.1f}。",
        f"seed0 曲线：{curve_txt(0)}",
        f"seed1 曲线：{curve_txt(1)}",
        f"训练 active fraction mean seed0/1={mean_hist(0,'active_fraction'):.4f}/{mean_hist(1,'active_fraction'):.4f}。",
        f"mean actor minibatch seed0/1={mean_hist(0,'mean_actor_minibatch_size'):.2f}/{mean_hist(1,'mean_actor_minibatch_size'):.2f}。",
        f"min actor minibatch seed0/1={min(float(r['min_actor_minibatch_size']) for r in hist if int(r['model_seed'])==0):.0f}/{min(float(r['min_actor_minibatch_size']) for r in hist if int(r['model_seed'])==1):.0f}。",
        f"Actor KL mean seed0/1={mean_hist(0,'approx_kl'):.4f}/{mean_hist(1,'approx_kl'):.4f}；clip={mean_hist(0,'clip_fraction'):.4f}/{mean_hist(1,'clip_fraction'):.4f}；grad={mean_hist(0,'actor_gradient_norm'):.3f}/{mean_hist(1,'actor_gradient_norm'):.3f}。",
        f"arrival new0/new1/L0={n0['mean_arrival_steps']:.3f}/{n1['mean_arrival_steps']:.3f}/{l0['mean_arrival_steps']:.3f}。",
        f"completed_blocks new0/new1/L0={n0['completed_blocks']:.4f}/{n1['completed_blocks']:.4f}/{l0['completed_blocks']:.4f}。",
        f"collision={n0['collision_count']}/{n1['collision_count']}。projector_failure={n0['projector_failure_count']}/{n1['projector_failure_count']}。cap={n0['cap_activation_rate']:.4f}/{n1['cap_activation_rate']:.4f}。",
        "",
        f"Gate **P2D-{gate['P2D']}**。active_sample_optimization_supported={gate['active_sample_optimization_supported']}。{gate['text']}",
        "未自动训练 seed2/3/4。",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pre = write_prefreeze()
    tests = run_tests()
    _dump(OUTPUT / "LLV2_PHASE2D_TESTS.json", tests)
    _dump(PHASE2D / "LLV2_PHASE2D_TESTS.json", tests)
    if not tests["all_pass"]:
        raise RuntimeError(f"Phase-2D tests failed: {tests}")
    train_seeds, design_seeds = load_pools()
    protocol = {
        "name": "LL-V2.2D",
        "only_change": "ActiveSampleAwareActorOptimization",
        "dev_seeds": list(DEV_SEEDS),
        "primary": "update100",
        "eval": "LLV2_design",
        "train": "LLV2_train",
        "phase2_validation": "blocked",
        "high_sha": high_sha(),
        "projector_sha": projector_sha(),
        "MASK": "B",
        "BASEZERO": "A",
        "critic_unchanged": True,
        "reward_unchanged": True,
    }
    _dump(OUTPUT / "LLV2_PHASE2D_PROTOCOL.json", protocol)
    _dump(PHASE2D / "LLV2_PHASE2D_PROTOCOL.json", protocol)
    all_hist, all_curve = [], []
    for seed in DEV_SEEDS:
        result = train_seed(int(seed), train_seeds, design_seeds)
        all_hist.extend(result["history"])
        all_curve.extend(result["curve"])
    _csv(OUTPUT / "LLV2_PHASE2D_TRAINING_HISTORY.csv", all_hist)
    batch_keys = [
        "model_seed",
        "update",
        "total_agent_samples",
        "active_samples",
        "active_fraction",
        "actor_minibatch_count",
        "min_actor_minibatch_size",
        "max_actor_minibatch_size",
        "mean_actor_minibatch_size",
        "actor_optimizer_steps",
        "active_advantage_mean_before",
        "active_advantage_std_before",
        "active_advantage_mean_after",
        "active_advantage_std_after",
    ]
    _csv(OUTPUT / "LLV2_PHASE2D_ACTIVE_BATCH_STATS.csv", [{k: r.get(k) for k in batch_keys} | {"model_seed": r["model_seed"], "update": r["update"]} for r in all_hist])
    _csv(
        OUTPUT / "LLV2_PHASE2D_ACTOR_OPTIMIZATION.csv",
        [
            {
                "model_seed": r["model_seed"],
                "update": r["update"],
                "policy_loss": r["policy_loss"],
                "entropy": r["entropy"],
                "approx_kl": r["approx_kl"],
                "clip_fraction": r["clip_fraction"],
                "actor_gradient_norm": r["actor_gradient_norm"],
            }
            for r in all_hist
        ],
    )
    _csv(
        OUTPUT / "LLV2_PHASE2D_CRITIC_OPTIMIZATION.csv",
        [
            {
                "model_seed": r["model_seed"],
                "update": r["update"],
                "value_loss": r["value_loss"],
                "critic_gradient_norm": r["critic_gradient_norm"],
                "explained_variance": r["explained_variance"],
            }
            for r in all_hist
        ],
    )
    _csv(OUTPUT / "LLV2_PHASE2D_EVAL_CURVE.csv", all_curve)
    high, _ = load_frozen_high()
    actor, critic = make_networks()
    l0w = ResidualWorker(high, actor, critic, service_lock=True, trust_region=False)
    l0 = evaluate_design(l0w, design_seeds, zero_residual=True)
    new0 = evaluate_design(load_ckpt(OUTPUT / "ckpt_seed0_update100.pt", trust_region=True), design_seeds, zero_residual=False)
    new1 = evaluate_design(load_ckpt(OUTPUT / "ckpt_seed1_update100.pt", trust_region=True), design_seeds, zero_residual=False)
    p2b0 = evaluate_design(load_ckpt(PHASE2B / "outputs" / "ckpt_seed0_update100.pt", trust_region=True), design_seeds, zero_residual=False)
    p2b1 = evaluate_design(load_ckpt(PHASE2B / "outputs" / "ckpt_seed1_update100.pt", trust_region=True), design_seeds, zero_residual=False)
    tagged = {"L0": l0, "new0": new0, "new1": new1, "p2b0": p2b0, "p2b1": p2b1}
    nav_rows, svc_rows, safe_rows = [], [], []
    for name, rec in tagged.items():
        nav_rows.append(
            {
                "tag": name,
                "mean_arrival_steps": rec["mean_arrival_steps"],
                "never_reach_fraction": rec["never_reach_fraction"],
                "time_in_range": rec["time_in_range"],
                "flight_distance": rec["flight_distance"],
                "idle_fraction": rec["idle_fraction"],
                "assignment_change_count": rec["assignment_change_count"],
                "target_switch_count": rec["target_switch_count"],
            }
        )
        svc_rows.append(
            {
                "tag": name,
                "completed_blocks": rec["completed_blocks"],
                "time_in_range": rec["time_in_range"],
                "service_lock_violation_count": rec["service_lock_violation_count"],
                "idle_fraction": rec["idle_fraction"],
            }
        )
        safe_rows.append(
            {
                "tag": name,
                "collision_count": rec["collision_count"],
                "min_sep_worst": rec["min_sep_worst"],
                "projector_failure_count": rec["projector_failure_count"],
                "service_lock_violation_count": rec["service_lock_violation_count"],
                "cap_activation_rate": rec["cap_activation_rate"],
                "base_zero_freeze_rate": rec["base_zero_freeze_rate"],
            }
        )
    gate = gate_decision(tagged, all_hist, tests)
    _csv(OUTPUT / "LLV2_PHASE2D_NAVIGATION.csv", nav_rows)
    _csv(OUTPUT / "LLV2_PHASE2D_SERVICE.csv", svc_rows)
    _csv(OUTPUT / "LLV2_PHASE2D_SAFETY.csv", safe_rows)
    _dump(OUTPUT / "LLV2_PHASE2D_GATE.json", gate)
    plot_figures(all_curve, all_hist, tagged, OUTPUT / "figures")
    report = write_report(pre, tests, tagged, gate, all_curve, hist=all_hist)
    report_name = "HTW_LowLevelV2_Phase2D_ActiveSampleOptimization_CN.md"
    _dump(OUTPUT / report_name, report)
    for src in list(OUTPUT.glob("LLV2_PHASE2D_*")) + [OUTPUT / report_name, PREFREEZE_FILE]:
        if src.exists() and src.is_file():
            (PHASE2D / src.name).write_bytes(src.read_bytes())
            (ROOT / src.name).write_bytes(src.read_bytes())
    fig_dir = PHASE2D / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for png in (OUTPUT / "figures").glob("*.png"):
        (fig_dir / png.name).write_bytes(png.read_bytes())
    (ROOT / report_name).write_text(report, encoding="utf-8")
    return {"tests": tests["all_pass"], "gate": gate["P2D"], "new0": new0["payload"], "new1": new1["payload"], "L0": l0["payload"]}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
