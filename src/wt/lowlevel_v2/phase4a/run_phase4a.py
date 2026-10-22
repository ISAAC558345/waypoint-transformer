"""LL-V2.4A collision-relation distillation pilot. Frozen Phase-3 Residual MAPPO."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from wt.lowlevel_v2.phase1b.controller import load_frozen_high, make_networks
from wt.lowlevel_v2.phase4a.audit import teacher_signal_audit
from wt.lowlevel_v2.phase4a.eval import evaluate_design_mechanics, neighbor_l2
from wt.lowlevel_v2.phase4a.protocol import (
    ALPHA_RESIDUAL,
    BETA_RESIDUAL,
    D_SAFE_M,
    EXPECTED_HIGH_SHA,
    LAMBDA_COLLISION_DISTILL,
    LAMBDA_PROJECT,
    LAMBDA_RESIDUAL,
    OUTPUT,
    PAYLOAD_DROP_TOLERANCE,
    PHASE3_CKPT,
    PHASE4A,
    PILOT_SEEDS,
    PILOT_UPDATES_N,
    PPO,
    PREFREEZE_FILE,
    ROOT,
    SERVICE_SAFETY_MARGIN_M,
    TEACHER_MIN_ACTIVE_FRACTION,
    action_max,
    epsilon_base,
    high_sha,
    load_phase3_gate,
    load_pools,
    projector_sha,
)
from wt.lowlevel_v2.phase4a.train import train_seed
from wt.lowlevel_v2.phase4a.worker import make_worker


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
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
    p3 = load_phase3_gate()
    payload = {
        "method": "SafetyProjectorCollisionDistillation",
        "unique_new_mechanism": "Safety-Projector Collision Distillation",
        "lambda_collision_distill": LAMBDA_COLLISION_DISTILL,
        "payload_drop_tolerance": PAYLOAD_DROP_TOLERANCE,
        "teacher_min_active_fraction": TEACHER_MIN_ACTIVE_FRACTION,
        "no_lambda_grid": True,
        "reward_unchanged": True,
        "lambda_project": LAMBDA_PROJECT,
        "lambda_residual": LAMBDA_RESIDUAL,
        "mappo_lr": PPO.learning_rate,
        "clip": PPO.clip_eps,
        "entropy_coef": PPO.entropy_coef,
        "alpha_residual": ALPHA_RESIDUAL,
        "beta_residual": BETA_RESIDUAL,
        "epsilon_base": epsilon_base(),
        "a_max": action_max(),
        "d_safe_m": D_SAFE_M,
        "service_safety_margin_m": SERVICE_SAFETY_MARGIN_M,
        "updates": PILOT_UPDATES_N,
        "rollout_steps": int(PPO.rollout_length),
        "num_envs": int(PPO.num_envs),
        "pilot_seeds": list(PILOT_SEEDS),
        "phase3_gate": p3["LLV2_MS2"],
        "stable_residual_advantage": True,
        "REL2": "0",
        "high_sha": EXPECTED_HIGH_SHA,
        "safety_projector_sha": projector_sha(),
        "no_phase3_validation": True,
        "no_new_independent_validation": True,
        "no_interference_enhancement": True,
        "distill_events": ["PAIRWISE", "MULTI_containing_PAIRWISE"],
        "forbidden_distill_events": ["pure_SERVICE_SAFE_ENVELOPE", "pure_STEP_LIMIT"],
        "teacher": "a_projected - a_base from actual SafetyProjector, detached",
        "ppo_log_prob": "sampled raw residual only",
    }
    _dump(PREFREEZE_FILE, payload)
    _dump(OUTPUT / "LLV2_PHASE4A_PREFREEZE.json", payload)
    return payload


def load_ckpt(path: Path):
    high, _ = load_frozen_high()
    actor, critic = make_networks()
    payload = torch.load(path, map_location="cpu")
    actor.load_state_dict(payload["actor"])
    critic.load_state_dict(payload["critic"])
    actor.eval()
    critic.eval()
    return make_worker(high, actor, critic)


def plot_figures(c0, c1, hist, curve, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    for seed in PILOT_SEEDS:
        xs = [r["update"] for r in curve if int(r["model_seed"]) == seed]
        ys = [r["pairwise_intervention_rate"] for r in curve if int(r["model_seed"]) == seed]
        ax.plot(xs, ys, marker="o", label=f"C1 s{seed}")
        ax.axhline(c0[seed]["pairwise_intervention_rate"], ls="--", label=f"C0 s{seed}")
    ax.set_ylabel("PAIRWISE intervention rate")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "llv2_phase4a_pairwise_intervention.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    names = []
    vals = []
    for seed in PILOT_SEEDS:
        names += [f"C0s{seed}", f"C1s{seed}"]
        vals += [neighbor_l2(c0[seed]["neighbor"], "neighbor_shuffle", "near"), neighbor_l2(c1[seed]["neighbor"], "neighbor_shuffle", "near")]
    ax.bar(names, vals)
    ax.set_ylabel("near neighbor-shuffle L2")
    fig.tight_layout()
    fig.savefig(out / "llv2_phase4a_neighbor_sensitivity.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    x = np.arange(len(PILOT_SEEDS))
    ax.bar(x - 0.2, [c0[s]["separation"]["mean"] for s in PILOT_SEEDS], 0.4, label="C0")
    ax.bar(x + 0.2, [c1[s]["separation"]["mean"] for s in PILOT_SEEDS], 0.4, label="C1")
    ax.set_xticks(list(x), [f"s{s}" for s in PILOT_SEEDS])
    ax.axhline(0, color="k")
    ax.legend()
    ax.set_ylabel("near sep response mean")
    fig.tight_layout()
    fig.savefig(out / "llv2_phase4a_separation_response.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    for seed in PILOT_SEEDS:
        xs = [r["update"] for r in hist if int(r["model_seed"]) == seed]
        axes[0].plot(xs, [r["teacher_mse"] for r in hist if int(r["model_seed"]) == seed], label=f"s{seed}")
        axes[1].plot(xs, [r["teacher_cosine"] for r in hist if int(r["model_seed"]) == seed], label=f"s{seed}")
    axes[0].set_title("teacher MSE")
    axes[1].set_title("teacher cosine")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "llv2_phase4a_teacher_alignment.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    for seed in PILOT_SEEDS:
        xs = [r["update"] for r in curve if int(r["model_seed"]) == seed]
        ax.plot(xs, [r["payload"] / 1e6 for r in curve if int(r["model_seed"]) == seed], marker="o", label=f"C1 s{seed}")
        ax.axhline(c0[seed]["payload"] / 1e6, ls="--", label=f"C0 s{seed}")
    ax.legend(fontsize=7)
    ax.set_ylabel("design payload / 1e6")
    fig.tight_layout()
    fig.savefig(out / "llv2_phase4a_payload.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    for seed in PILOT_SEEDS:
        xs = [r["update"] for r in hist if int(r["model_seed"]) == seed]
        ax.plot(xs, [r["ppo_actor_gradient_norm"] for r in hist if int(r["model_seed"]) == seed], label=f"PPO s{seed}")
        ax.plot(xs, [r["distill_gradient_norm"] for r in hist if int(r["model_seed"]) == seed], ls="--", label=f"distill s{seed}")
    ax.legend(fontsize=7)
    ax.set_ylabel("grad norm")
    fig.tight_layout()
    fig.savefig(out / "llv2_phase4a_grad_norms.png", dpi=130)
    plt.close(fig)


def seed_pass(c0s, c1s, hist_s) -> dict:
    pair_down = c1s["pairwise_intervention_rate"] < c0s["pairwise_intervention_rate"]
    sens0 = neighbor_l2(c0s["neighbor"], "neighbor_shuffle", "near")
    sens1 = neighbor_l2(c1s["neighbor"], "neighbor_shuffle", "near")
    sep = c1s["separation"]
    mse = [float(r["teacher_mse"]) for r in hist_s if np.isfinite(float(r.get("teacher_mse", "nan")))]
    mse_down = bool(mse) and (mse[-1] < mse[0])
    pay_ok = c1s["payload"] + 1e-9 >= (1.0 - PAYLOAD_DROP_TOLERANCE) * c0s["payload"]
    safe = (
        c1s["collision_count"] == 0
        and c1s["projector_failure_count"] == 0
        and c1s["service_lock_violation_count"] == 0
        and c1s["min_sep_worst"] + 1e-6 >= 1.0
    )
    return {
        "pairwise_down": pair_down,
        "near_sensitivity_up": sens1 > sens0,
        "sep_mean_pos": float(sep["mean"]) > 0,
        "sep_posfrac": float(sep["positive_fraction"]) > 0.5,
        "mse_down": mse_down,
        "payload_ok": pay_ok,
        "safe": safe,
        "full": pair_down and (sens1 > sens0) and float(sep["mean"]) > 0 and float(sep["positive_fraction"]) > 0.5 and mse_down and pay_ok and safe,
        "sens0": sens0,
        "sens1": sens1,
        "mse0": mse[0] if mse else float("nan"),
        "mse1": mse[-1] if mse else float("nan"),
    }


def decide_gate(per, hist, l0, c0, c1) -> dict:
    flags = {s: seed_pass(c0[s], c1[s], [r for r in hist if int(r["model_seed"]) == s]) for s in PILOT_SEEDS}
    n_full = int(sum(flags[s]["full"] for s in PILOT_SEEDS))
    ratios = [float(r["distill_ppo_grad_ratio"]) for r in hist]
    mean_ratio = float(np.mean(ratios)) if ratios else 0.0
    dominates = bool(ratios) and (float(np.mean([r > 2.0 for r in ratios])) > 0.5)
    too_weak = bool(ratios) and (float(np.mean([r < 0.01 for r in ratios])) > 0.5)
    rel_up = any(flags[s]["near_sensitivity_up"] or flags[s]["sep_mean_pos"] for s in PILOT_SEEDS)
    if n_full == 2:
        letter = "A"
        internalized = True
        text = "Both seeds internalize collision relation under distillation."
        nxt = "optional 5-seed collision-aware validation (not automatic this round)"
    elif n_full == 1 or (rel_up and all(flags[s]["payload_ok"] or c1[s]["payload"] >= 0.9 * c0[s]["payload"] for s in PILOT_SEEDS)):
        letter = "B"
        internalized = False
        text = "Collision relation signal exists but not both seeds fully pass, or payload mildly degraded."
        nxt = "do not enter 5-seed collision-aware validation yet"
    else:
        letter = "C"
        internalized = False
        text = "Distillation did not make the Actor use collision relation."
        nxt = "do not add interference enhancement; stop this collision-distill line or redesign teacher"
    if not rel_up and not all(flags[s]["pairwise_down"] for s in PILOT_SEEDS):
        letter = "C"
        internalized = False
        nxt = "do not add interference enhancement"
    return {
        "COLREL": letter,
        "collision_relation_internalized": internalized,
        "n_full_seeds": n_full,
        "per_seed": {str(s): flags[s] for s in PILOT_SEEDS},
        "mean_distill_ppo_grad_ratio": mean_ratio,
        "auxiliary_dominates": dominates,
        "auxiliary_too_weak": too_weak,
        "support_5seed_collision_aware_validation": bool(letter == "A"),
        "did_not_touch_phase3_validation": True,
        "did_not_modify_base_mappo": True,
        "did_not_add_interference": True,
        "next_step": nxt,
        "text": text,
    }


def write_sparse_report(pre, audit, gate) -> str:
    return "\n".join(
        [
            "# HTW Low-Level V2 Phase-4A：碰撞关系内化（Safety-Projector Distillation）",
            "",
            "基础 Residual MAPPO 冻结自 Phase-3（LLV2-MS2-A，stable_residual_advantage=true，REL2-0）。本阶段唯一计划新增机制是 Safety-Projector Collision Distillation。训练前在 LLV2_train 上做冻结 rollout 审计，**未训练**。未改 d_safe、未制造碰撞场景、未访问 223000–223063、未进入 interference 增强。",
            "",
            "## 20 问正式回答",
            "",
            "1. 本轮唯一新增机制：Safety-Projector Collision Distillation（PAIRWISE / MULTI-含-PAIRWISE 的 projected residual 作为 detach teacher）。因 teacher 过稀，**该损失未被训练启用**。",
            f"2. lambda_collision_distill={pre['lambda_collision_distill']}（训练前写入 prefreeze；无 0.01/0.05/0.2/1.0 网格）。",
            "3. teacher 来自实际 SafetyProjector：r_teacher = a_projected − a_base，stop_gradient。不构造“远离最近邻”人工 teacher。",
            "4. 进入 distillation 的事件：PAIRWISE，以及 MULTI 中明确包含 PAIRWISE 的 agent-step。禁止蒸馏纯 SERVICE_SAFE_ENVELOPE 与纯 STEP_LIMIT。",
            f"5. teacher signal：active={audit['n_active']}，eligible active={audit['n_collision_distill_eligible_active']}，比例={audit['eligible_frac_of_active']:.6f}（阈值 {audit['threshold']}）。pairwise intervened agent-steps={audit['pairwise_intervened_agent_steps']} / {audit['n_agent_steps']}。分层：eligible near={audit['eligible_near']}，middle={audit['eligible_middle']}，far={audit['eligible_far']}。**collision_teacher_signal_too_sparse=true**。",
            "6. C0/C1 payload：未训练 C1，因此没有 C1；C0 未在本阶段重新训练（按协议禁止）。本阶段不报告 C1 payload。",
            "7. PAIRWISE intervention C0→C1：无 C1，无法比较。",
            "8. near-neighbor shuffle sensitivity C0→C1：无 C1，无法比较。",
            "9. separation_response C0→C1：无 C1，无法比较。",
            "10. teacher MSE 是否下降：未训练，无 u0→u100 曲线。",
            "11. teacher cosine 是否提高：未训练。",
            "12. distill/PPO gradient ratio：未训练。",
            "13. auxiliary dominates：未训练，故 false；亦非 too_weak 的训练后结论。",
            "14. collision 是否仍为 0：未跑 C1 closed-loop。审计 rollout 未改 d_safe。",
            "15. min separation：未做 C1 正式安全表。",
            "16. projector failure：未做 C1 正式安全表。",
            "17. service 机制：未训练，无回退证据；ServiceResidualLock 代码未改。",
            "18. COLREL Gate=**C**。",
            "19. collision_relation_internalized=**false**。",
            "20. 是否支持进入 5-seed collision-aware 验证：**否**。",
            "",
            f"下一步唯一建议：{gate['next_step']}",
            "不要自动进入 interference 增强。本阶段停止。",
            "",
        ]
    ) + "\n"


def write_report(pre, audit, l0, c0, c1, gate, hist) -> str:
    lines = [
        "# HTW Low-Level V2 Phase-4A：碰撞关系内化（Safety-Projector Distillation）",
        "",
        "基础 Residual MAPPO 冻结自 Phase-3（LLV2-MS2-A）。唯一新增：Safety-Projector Collision Distillation。未改 reward / MAPPO / lock / freeze / Active-Sample / projector 规则 / d_safe / 通信项。未访问 223000–223063。未创建新 independent validation。",
        f"lambda_collision_distill={pre['lambda_collision_distill']}（训练前冻结，无网格搜索）。payload 跌幅上限={pre['payload_drop_tolerance']}。",
        f"teacher audit eligible/active={audit['eligible_frac_of_active']:.6f}，too_sparse={audit['collision_teacher_signal_too_sparse']}。",
        f"L0 design payload={l0['payload']:.1f}。",
    ]
    for s in PILOT_SEEDS:
        lines.append(
            f"seed{s} C0 payload={c0[s]['payload']:.1f} PAIRWISE={c0[s]['pairwise_intervention_rate']:.6f}；"
            f"C1 payload={c1[s]['payload']:.1f} PAIRWISE={c1[s]['pairwise_intervention_rate']:.6f}。"
        )
    lines += [
        f"Gate **COLREL-{gate['COLREL']}**。collision_relation_internalized={gate['collision_relation_internalized']}。",
        f"distill/PPO grad ratio mean={gate['mean_distill_ppo_grad_ratio']:.4f}。dominates={gate['auxiliary_dominates']}。too_weak={gate['auxiliary_too_weak']}。",
        f"下一步：{gate['next_step']}",
        "未自动进入 interference 增强。",
        "",
    ]
    return "\n".join(lines) + "\n"


def publish(name: str, report: str) -> None:
    for src in list(OUTPUT.glob("LLV2_PHASE4A_*")) + [OUTPUT / name, PREFREEZE_FILE]:
        if src.exists() and src.is_file():
            (PHASE4A / src.name).write_bytes(src.read_bytes())
            (ROOT / src.name).write_bytes(src.read_bytes())
    fig_dir = PHASE4A / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for png in (OUTPUT / "figures").glob("*.png"):
        (fig_dir / png.name).write_bytes(png.read_bytes())
    (ROOT / ("docs/archive/phase_reports" if str(name).endswith(".md") else "results/root_exports") / name).write_text(report, encoding="utf-8")
    (PHASE4A / name).write_text(report, encoding="utf-8")


def main() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    p3 = load_phase3_gate()
    pre = write_prefreeze()
    train_seeds, design_seeds = load_pools()
    protocol = {
        "name": "LL-V2.4A",
        "phase3": "LLV2-MS2-A",
        "REL2": "0",
        "lambda_collision_distill": LAMBDA_COLLISION_DISTILL,
        "payload_drop_tolerance": PAYLOAD_DROP_TOLERANCE,
        "pilot_seeds": list(PILOT_SEEDS),
        "budget_env": 19200,
        "phase3_validation_blocked": True,
        "high_sha": high_sha(),
    }
    _dump(OUTPUT / "LLV2_PHASE4A_PROTOCOL.json", protocol)
    audit = teacher_signal_audit(train_seeds, model_seed=0)
    _csv(OUTPUT / "LLV2_PHASE4A_TEACHER_SIGNAL_AUDIT.csv", audit["rows"])
    _dump(OUTPUT / "LLV2_PHASE4A_TEACHER_SIGNAL_AUDIT.json", audit["summary"])
    if audit["summary"]["collision_teacher_signal_too_sparse"]:
        gate = {
            "COLREL": "C",
            "collision_relation_internalized": False,
            "collision_teacher_signal_too_sparse": True,
            "support_5seed_collision_aware_validation": False,
            "did_not_train": True,
            "did_not_touch_phase3_validation": True,
            "did_not_add_interference": True,
            "eligible_frac_of_active": audit["summary"]["eligible_frac_of_active"],
            "n_collision_distill_eligible_active": audit["summary"]["n_collision_distill_eligible_active"],
            "n_active": audit["summary"]["n_active"],
            "next_step": "stop; do not change d_safe or manufacture collisions; do not enter interference enhancement",
            "text": "Teacher signal too sparse on LLV2_train. Collision distillation was not trained.",
        }
        _dump(OUTPUT / "LLV2_PHASE4A_GATE.json", gate)
        stub = [{"status": "did_not_train", "reason": "collision_teacher_signal_too_sparse"}]
        for name in (
            "LLV2_PHASE4A_TRAINING_HISTORY.csv",
            "LLV2_PHASE4A_DISTILL_LOSS.csv",
            "LLV2_PHASE4A_GRADIENT_AUDIT.csv",
            "LLV2_PHASE4A_PROJECTOR_BREAKDOWN.csv",
            "LLV2_PHASE4A_NEIGHBOR_SENSITIVITY.csv",
            "LLV2_PHASE4A_SEPARATION_RESPONSE.csv",
            "LLV2_PHASE4A_TEACHER_ALIGNMENT.csv",
            "LLV2_PHASE4A_PAYLOAD.csv",
            "LLV2_PHASE4A_SAFETY.csv",
        ):
            _csv(OUTPUT / name, stub)
        report = write_sparse_report(pre, audit["summary"], gate)
        name = "HTW_LowLevelV2_Phase4A_CollisionRelation_Distillation_CN.md"
        _dump(OUTPUT / name, report)
        publish(name, report)
        return {"gate": "C", "sparse": True}

    sha_before = high_sha()
    high, _ = load_frozen_high()
    actor, critic = make_networks()
    l0w = make_worker(high, actor, critic)
    l0 = evaluate_design_mechanics(l0w, design_seeds, zero_residual=True)
    c0 = {}
    for seed in PILOT_SEEDS:
        w = load_ckpt(PHASE3_CKPT / f"ckpt_seed{seed}_update100.pt")
        c0[seed] = evaluate_design_mechanics(w, design_seeds, zero_residual=False)
    all_hist, all_curve = [], []
    for seed in PILOT_SEEDS:
        result = train_seed(int(seed), train_seeds, design_seeds, output_dir=OUTPUT)
        all_hist.extend(result["history"])
        all_curve.extend(result["curve"])
    if high_sha() != sha_before or high_sha() != EXPECTED_HIGH_SHA:
        raise RuntimeError("high SHA changed")
    c1 = {}
    for seed in PILOT_SEEDS:
        w = load_ckpt(OUTPUT / f"ckpt_c1_seed{seed}_update100.pt")
        c1[seed] = evaluate_design_mechanics(w, design_seeds, zero_residual=False)
    _csv(OUTPUT / "LLV2_PHASE4A_TRAINING_HISTORY.csv", all_hist)
    _csv(
        OUTPUT / "LLV2_PHASE4A_DISTILL_LOSS.csv",
        [{"model_seed": r["model_seed"], "update": r["update"], "collision_distill_loss": r["collision_distill_loss"], "n_teacher": r["collision_teacher_samples"]} for r in all_hist],
    )
    _csv(
        OUTPUT / "LLV2_PHASE4A_GRADIENT_AUDIT.csv",
        [
            {
                "model_seed": r["model_seed"],
                "update": r["update"],
                "ppo_actor_gradient_norm": r["ppo_actor_gradient_norm"],
                "distill_gradient_norm": r["distill_gradient_norm"],
                "total_actor_gradient_norm": r["actor_gradient_norm"],
                "distill_ppo_grad_ratio": r["distill_ppo_grad_ratio"],
            }
            for r in all_hist
        ],
    )
    proj_rows = []
    pay_rows = []
    safe_rows = []
    sep_rows = []
    neigh_rows = []
    for tag, store in (("L0", {0: l0}), ("C0", c0), ("C1", c1)):
        items = store.items() if tag != "L0" else [( "L0", l0)]
        for seed, rec in items:
            proj_rows.append(
                {
                    "method": tag,
                    "model_seed": seed,
                    "overall_projector_intervention": rec["projector_intervention_rate"],
                    "PAIRWISE": rec["pairwise_intervention_rate"],
                    "SERVICE_ONLY": rec["service_only_intervention_rate"],
                    "MULTI": rec["multi_intervention_rate"],
                    "STEP_ONLY": rec["step_only_intervention_rate"],
                }
            )
            pay_rows.append(
                {
                    "method": tag,
                    "model_seed": seed,
                    "payload": rec["payload"],
                    "completed_blocks": rec["completed_blocks"],
                    "time_in_range": rec["time_in_range"],
                    "service_lock_rate": rec["service_lock_rate"],
                }
            )
            safe_rows.append(
                {
                    "method": tag,
                    "model_seed": seed,
                    "collision_count": rec["collision_count"],
                    "min_sep_worst": rec["min_sep_worst"],
                    "projector_failure_count": rec["projector_failure_count"],
                    "service_lock_violation_count": rec["service_lock_violation_count"],
                }
            )
            sep_rows.append({"method": tag, "model_seed": seed, **rec["separation"]})
            for row in rec.get("neighbor", []):
                neigh_rows.append({"method": tag, "model_seed": seed, **row})
    _csv(OUTPUT / "LLV2_PHASE4A_PROJECTOR_BREAKDOWN.csv", proj_rows)
    _csv(OUTPUT / "LLV2_PHASE4A_PAYLOAD.csv", pay_rows)
    _csv(OUTPUT / "LLV2_PHASE4A_SAFETY.csv", safe_rows)
    _csv(OUTPUT / "LLV2_PHASE4A_SEPARATION_RESPONSE.csv", sep_rows)
    _csv(OUTPUT / "LLV2_PHASE4A_NEIGHBOR_SENSITIVITY.csv", neigh_rows)
    align = []
    for r in all_hist:
        if int(r["update"]) in {1, 20, 40, 60, 80, 100}:
            align.append(
                {
                    "model_seed": r["model_seed"],
                    "update": 0 if int(r["update"]) == 1 else r["update"],
                    "teacher_mse": r["teacher_mse"],
                    "teacher_cosine": r["teacher_cosine"],
                    "teacher_direction_agreement": r["teacher_direction_agreement"],
                }
            )
    for row in all_curve:
        if int(row["update"]) in {0, 20, 40, 60, 80, 100}:
            align.append({"model_seed": row["model_seed"], "update": row["update"], "design_payload": row["payload"], "pairwise": row["pairwise_intervention_rate"]})
    _csv(OUTPUT / "LLV2_PHASE4A_TEACHER_ALIGNMENT.csv", align)
    _csv(OUTPUT / "LLV2_PHASE4A_EVAL_CURVE.csv", all_curve)
    gate = decide_gate({s: True for s in PILOT_SEEDS}, all_hist, l0, c0, c1)
    gate["phase3"] = p3["LLV2_MS2"]
    _dump(OUTPUT / "LLV2_PHASE4A_GATE.json", gate)
    plot_figures(c0, c1, all_hist, all_curve, OUTPUT / "figures")
    name = "HTW_LowLevelV2_Phase4A_CollisionRelation_Distillation_CN.md"
    report = write_report(pre, audit["summary"], l0, c0, c1, gate, all_hist)
    _dump(OUTPUT / name, report)
    publish(name, report)
    return {"gate": gate["COLREL"], "internalized": gate["collision_relation_internalized"], "ratio": gate["mean_distill_ppo_grad_ratio"]}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
