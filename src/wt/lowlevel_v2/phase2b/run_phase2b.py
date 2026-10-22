"""LL-V2.2B residual trust-region development pilot. Trains seed0 and seed1 only."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from wt.lowlevel_v2.phase1b.controller import ResidualWorker, load_frozen_high, make_networks
from wt.lowlevel_v2.phase2b.eval import evaluate_design
from wt.lowlevel_v2.phase2b.protocol import (
    ALPHA_RESIDUAL,
    BETA_RESIDUAL,
    D_SAFE_M,
    DEV_SEEDS,
    EPSILON_BASE_FRACTION,
    EXPECTED_HIGH_SHA,
    LOG_STD_INIT,
    OLD_SEED1_CKPT,
    OLD_SEED2_CKPT,
    OUTPUT,
    PHASE1B_CKPT0,
    PHASE2B,
    PILOT_UPDATES_N,
    PPO,
    PREFREEZE_FILE,
    ROOT,
    SERVICE_SAFETY_MARGIN_M,
    action_max,
    epsilon_base,
    high_sha,
    load_pools,
    projector_sha,
)
from wt.lowlevel_v2.phase2b.tests import run_tests
from wt.lowlevel_v2.phase2b.train import make_worker, train_seed


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
    payload = {
        "beta_residual": BETA_RESIDUAL,
        "epsilon_base_fraction": EPSILON_BASE_FRACTION,
        "epsilon_base": epsilon_base(),
        "a_max": action_max(),
        "a_max_source": "ACTION_SCALE / clip_unit_step unit disk",
        "alpha_residual": ALPHA_RESIDUAL,
        "d_safe_m": D_SAFE_M,
        "service_safety_margin_m": SERVICE_SAFETY_MARGIN_M,
        "ServiceResidualLock": True,
        "BaseZeroResidualFreeze": True,
        "BaseRelativeResidualCap": True,
        "policy_mask_base_zero": True,
        "beta_grid_search": False,
        "tuned_from_payload": False,
        "reward_equals_phase1b": True,
        "mappo_equals_phase1b": True,
        "log_std_init": LOG_STD_INIT,
        "updates": PILOT_UPDATES_N,
        "num_envs": int(PPO.num_envs),
        "rollout_steps": int(PPO.rollout_length),
        "minibatches": int(PPO.num_minibatches),
        "update_epochs": int(PPO.update_epochs),
        "high_checkpoint_sha": EXPECTED_HIGH_SHA,
        "safety_projector_sha": projector_sha(),
        "train_seeds": [int(s) for s in train],
        "design_seeds": [int(s) for s in design],
        "dev_model_seeds": list(DEV_SEEDS),
        "loaded_old_weights": False,
        "phase2_validation_blocked": True,
        "frozen_before_any_training": True,
    }
    _dump(PREFREEZE_FILE, payload)
    _dump(OUTPUT / "LLV2_PHASE2B_PREFREEZE.json", payload)
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
    w = ResidualWorker(high, actor, critic, service_lock=True, trust_region=False)
    return w


def plot_figures(curve: list[dict], rows: dict, out: Path) -> None:
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
    fig.savefig(out / "llv2_phase2b_learning_curves.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    names = ["L0", "new0", "new1", "old0", "old1"]
    payloads = [rows[k]["payload"] / 1e6 for k in names]
    axes[0].bar(names, payloads, color=["#777", "#1f77b4", "#d62728", "#9ecae1", "#fcae91"])
    axes[0].set_ylabel("payload / 1e6")
    axes[0].axhline(l0 / 1e6, color="k", lw=0.6)
    axes[1].bar(["new0 raw", "new0 used", "new1 raw", "new1 used"], [
        rows["new0"]["mean_raw_residual"],
        rows["new0"]["mean_used_residual"],
        rows["new1"]["mean_raw_residual"],
        rows["new1"]["mean_used_residual"],
    ])
    axes[1].set_ylabel("residual norm")
    fig.tight_layout()
    fig.savefig(out / "llv2_phase2b_payload_residual.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3))
    axes[0].bar(["new0", "new1"], [rows["new0"]["cap_activation_rate"], rows["new1"]["cap_activation_rate"]])
    axes[0].set_title("cap activation")
    axes[1].bar(["L0", "new1", "old1"], [rows["L0"]["mean_arrival_steps"], rows["new1"]["mean_arrival_steps"], rows["old1"]["mean_arrival_steps"]])
    axes[1].set_title("arrival steps")
    axes[2].bar(["new0", "new1", "old1"], [rows["new0"]["projector_intervention_rate"], rows["new1"]["projector_intervention_rate"], rows["old1"]["projector_intervention_rate"]])
    axes[2].set_title("projector")
    fig.tight_layout()
    fig.savefig(out / "llv2_phase2b_cap_arrival_projector.png", dpi=130)
    plt.close(fig)


def gate_decision(rows: dict) -> dict:
    l0, n0, n1, o1 = rows["L0"], rows["new0"], rows["new1"], rows["old1"]
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
    arr_better = n1["mean_arrival_steps"] <= o1["mean_arrival_steps"] - 0.3
    arr_close = abs(n1["mean_arrival_steps"] - l0["mean_arrival_steps"]) <= 1.0
    res_down = n1["mean_used_residual"] <= 0.75 * max(o1["mean_used_residual"], 1e-9)
    s0_ok = n0["payload"] + 1.0e-6 >= l0["payload"]
    improved = (n1["payload"] > o1["payload"] + 2.5e5) or arr_better
    if s0_ge and s1_ge and both_safe and arr_better and arr_close and res_down and s0_ok:
        letter, supported = "A", True
        text = "seed0/seed1 both >= L0, safety pass, seed1 arrival recovered and used residual suppressed."
    elif both_safe and improved and (not s1_ge or not s0_ge or not arr_close or not res_down):
        letter, supported = "B", False
        text = "ACTION/NAV partly supported; unified trust region is not enough. Do not search beta."
    else:
        letter, supported = "C", False
        text = "seed1 not improved or safety worse. Phase-2A attribution insufficient."
    return {
        "P2B": letter,
        "action_trust_region_supported": supported,
        "new_seed0_ge_L0": s0_ge,
        "new_seed1_ge_L0": s1_ge,
        "both_safe": both_safe,
        "seed1_arrival_better": arr_better,
        "seed1_arrival_close_L0": arr_close,
        "seed1_used_residual_down": res_down,
        "seed0_not_destroyed": s0_ok,
        "seed1_improved_vs_old": improved,
        "reenter_5seed_independent_validation": bool(supported),
        "text": text,
        "trained_seeds": [0, 1],
        "did_not_train_seeds_2_3_4": True,
        "trained": True,
        "phase2_conclusion_unchanged": {"LLV2_MS": "B", "stable_residual_advantage": False, "REL": 1},
    }


def write_report(pre, tests, rows, gate, curve, hist) -> str:
    l0, n0, n1, o0, o1 = rows["L0"], rows["new0"], rows["new1"], rows["old0"], rows["old1"]
    def curve_txt(seed):
        pts = [r for r in curve if int(r["model_seed"]) == seed]
        return ", ".join(f"u{int(r['update'])}={r['payload']/1e6:.3f}e6" for r in pts)
    lines = [
        "# HTW Low-Level V2 Phase-2B：Base-Relative Residual Trust-Region Pilot（LL-V2.2B）",
        "",
        "本阶段是 development pilot。唯一方法改动是 Base-Relative Residual Trust Region。未改 reward / lr / PPO / Actor-Critic 结构 / Neighbor / relation / alpha / d_safe / ServiceResidualLock / SafetyProjector / 高层。未访问 222000–222063。只训练 seed0 与 seed1，均从头初始化。Primary 是 update100。",
        "",
        f"正式结论保持：LLV2-MS-B，stable_residual_advantage=false，REL-1。Phase-2A：primary=ACTION，secondary=NAV，S1-B。",
        "",
        f"beta_residual={pre['beta_residual']}，epsilon_base_fraction={pre['epsilon_base_fraction']}，epsilon_base={pre['epsilon_base']}（a_max={pre['a_max']} from ACTION_SCALE）。",
        f"单元测试 {tests['n_pass']}/{tests['n']} PASS。预算：每 seed 100×48×4=19200 env / 115200 agent transitions。",
        "",
        f"L0 payload={l0['payload']:.1f}。new seed0={n0['payload']:.1f} (Δ {n0['payload']-l0['payload']:+.1f})。new seed1={n1['payload']:.1f} (Δ {n1['payload']-l0['payload']:+.1f})。old seed1={o1['payload']:.1f}。",
        f"seed0 曲线：{curve_txt(0)}",
        f"seed1 曲线：{curve_txt(1)}",
        "",
        f"new seed1 used residual={n1['mean_used_residual']:.4f}（old {o1['mean_used_residual']:.4f}），raw={n1['mean_raw_residual']:.4f}，cap={n1['cap_activation_rate']:.3f}，base-zero freeze={n1['base_zero_freeze_rate']:.3f}。",
        f"arrival new1={n1['mean_arrival_steps']:.3f}，old1={o1['mean_arrival_steps']:.3f}，L0={l0['mean_arrival_steps']:.3f}。idle new1={n1['idle_fraction']:.3f} vs old1={o1['idle_fraction']:.3f} vs L0={l0['idle_fraction']:.3f}。",
        f"collision new0/new1={n0['collision_count']}/{n1['collision_count']}。min_sep_worst={n0['min_sep_worst']:.9f}/{n1['min_sep_worst']:.9f}。projector_failure={n0['projector_failure_count']}/{n1['projector_failure_count']}。",
        "",
        f"Gate **P2B-{gate['P2B']}**。action_trust_region_supported={gate['action_trust_region_supported']}。{gate['text']}",
        "未自动训练 seed2/3/4，未开独立 validation。",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pre = write_prefreeze()
    tests = run_tests()
    _dump(OUTPUT / "LLV2_PHASE2B_TESTS.json", tests)
    _dump(PHASE2B / "LLV2_PHASE2B_TESTS.json", tests)
    if not tests["all_pass"]:
        raise RuntimeError(f"Phase-2B tests failed: {tests}")
    train_seeds, design_seeds = load_pools()
    protocol = {
        "name": "LL-V2.2B",
        "only_change": "BaseRelativeResidualTrustRegion",
        "dev_seeds": list(DEV_SEEDS),
        "primary": "update100",
        "eval": "LLV2_design",
        "train": "LLV2_train",
        "phase2_validation": "blocked",
        "high_sha": high_sha(),
        "projector_sha": projector_sha(),
    }
    _dump(OUTPUT / "LLV2_PHASE2B_PROTOCOL.json", protocol)
    _dump(PHASE2B / "LLV2_PHASE2B_PROTOCOL.json", protocol)
    all_hist, all_curve = [], []
    for seed in DEV_SEEDS:
        result = train_seed(int(seed), train_seeds, design_seeds)
        all_hist.extend(result["history"])
        all_curve.extend(result["curve"])
    _csv(OUTPUT / "LLV2_PHASE2B_TRAINING_HISTORY.csv", all_hist)
    _csv(OUTPUT / "LLV2_PHASE2B_EVAL_CURVE.csv", all_curve)
    high, _ = load_frozen_high()
    actor, critic = make_networks()
    l0w = ResidualWorker(high, actor, critic, service_lock=True, trust_region=False)
    l0 = evaluate_design(l0w, design_seeds, zero_residual=True)
    new0 = evaluate_design(load_ckpt(OUTPUT / "ckpt_seed0_update100.pt", trust_region=True), design_seeds, zero_residual=False)
    new1 = evaluate_design(load_ckpt(OUTPUT / "ckpt_seed1_update100.pt", trust_region=True), design_seeds, zero_residual=False)
    old0 = evaluate_design(load_ckpt(PHASE1B_CKPT0, trust_region=False), design_seeds, zero_residual=False)
    old1 = evaluate_design(load_ckpt(OLD_SEED1_CKPT, trust_region=False), design_seeds, zero_residual=False)
    old2 = evaluate_design(load_ckpt(OLD_SEED2_CKPT, trust_region=False), design_seeds, zero_residual=False)
    tagged = {"L0": l0, "new0": new0, "new1": new1, "old0": old0, "old1": old1, "old2": old2}
    nav_rows, svc_rows, hl_rows, safe_rows, res_rows, feat_rows = [], [], [], [], [], []
    for name, rec in tagged.items():
        nav_rows.append({"tag": name, "mean_arrival_steps": rec["mean_arrival_steps"], "median_arrival_steps": rec["median_arrival_steps"], "p90_arrival_steps": rec["p90_arrival_steps"], "never_reach_count": rec["never_reach_count"], "flight_distance": rec["flight_distance"], "time_in_range": rec["time_in_range"]})
        svc_rows.append({"tag": name, "time_in_range": rec["time_in_range"], "completed_blocks": rec["completed_blocks"], "unfinished_progress": rec["unfinished_progress"], "mean_active_sinr": rec["mean_active_sinr"], "mean_active_rate": rec["mean_active_rate"], "service_lock_violation_count": rec["service_lock_violation_count"]})
        hl_rows.append({"tag": name, "idle_fraction": rec["idle_fraction"], "assignment_change_count": rec["assignment_change_count"], "target_switch_count": rec["target_switch_count"], "high_level_decision_count": rec["high_level_decision_count"]})
        safe_rows.append({"tag": name, "collision_count": rec["collision_count"], "min_sep": rec["min_sep"], "min_sep_worst": rec["min_sep_worst"], "projector_failure_count": rec["projector_failure_count"], "service_lock_violation_count": rec["service_lock_violation_count"], "projector_intervention_rate": rec["projector_intervention_rate"], "PAIRWISE_COLLISION_among_intervened": rec["PAIRWISE_COLLISION_among_intervened"], "SERVICE_SAFE_ENVELOPE_among_intervened": rec["SERVICE_SAFE_ENVELOPE_among_intervened"], "MULTI_CONSTRAINT_among_intervened": rec["MULTI_CONSTRAINT_among_intervened"], "STEP_LIMIT_among_intervened": rec["STEP_LIMIT_among_intervened"], "OTHER_among_intervened": rec["OTHER_among_intervened"]})
        res_rows.append({"tag": name, "payload": rec["payload"], "mean_raw_residual": rec["mean_raw_residual"], "median_raw_residual": rec["median_raw_residual"], "p95_raw_residual": rec["p95_raw_residual"], "mean_used_residual": rec["mean_used_residual"], "median_used_residual": rec["median_used_residual"], "p95_used_residual": rec["p95_used_residual"], "cap_activation_rate": rec["cap_activation_rate"], "base_zero_freeze_rate": rec["base_zero_freeze_rate"], "mean_base_norm": rec["mean_base_norm"], "reverse_fraction": rec["reverse_fraction"], "mean_lateral": rec["mean_lateral"], "mean_forward": rec["mean_forward"]})
        feat_rows.append({"tag": name, "neighbor_shuffle": rec["neighbor_shuffle"], "interf_shuffle": rec["interf_shuffle"], "sinr_shuffle": rec["sinr_shuffle"]})
    gate = gate_decision(tagged)
    _csv(OUTPUT / "LLV2_PHASE2B_NAVIGATION.csv", nav_rows)
    _csv(OUTPUT / "LLV2_PHASE2B_SERVICE.csv", svc_rows)
    _csv(OUTPUT / "LLV2_PHASE2B_HIGHLEVEL_FEEDBACK.csv", hl_rows)
    _csv(OUTPUT / "LLV2_PHASE2B_SAFETY.csv", safe_rows)
    _csv(OUTPUT / "LLV2_PHASE2B_RESIDUAL_TRUST_REGION.csv", res_rows)
    _csv(OUTPUT / "LLV2_PHASE2B_FEATURE_AUDIT.csv", feat_rows)
    _dump(OUTPUT / "LLV2_PHASE2B_GATE.json", gate)
    plot_figures(all_curve, tagged, OUTPUT / "figures")
    report = write_report(pre, tests, tagged, gate, all_curve, all_hist)
    _dump(OUTPUT / "HTW_LowLevelV2_Phase2B_ResidualTrustRegion_CN.md", report)
    for src in list(OUTPUT.glob("LLV2_PHASE2B_*")) + [OUTPUT / "HTW_LowLevelV2_Phase2B_ResidualTrustRegion_CN.md", PREFREEZE_FILE]:
        if src.exists() and src.is_file():
            (PHASE2B / src.name).write_bytes(src.read_bytes())
            (ROOT / src.name).write_bytes(src.read_bytes())
    fig_dir = PHASE2B / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for png in (OUTPUT / "figures").glob("*.png"):
        (fig_dir / png.name).write_bytes(png.read_bytes())
    return {"tests": tests["all_pass"], "gate": gate, "new0": new0["payload"], "new1": new1["payload"], "L0": l0["payload"]}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
