"""LL-V2.4B communication controllability audit. training=false."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from wt.lowlevel_v2.phase1b.controller import load_frozen_high, make_networks
from wt.lowlevel_v2.phase1b.protocol import load_pools
from wt.lowlevel_v2.phase2b.train import make_worker
from wt.lowlevel_v2.phase4b.candidates import candidate_actions
from wt.lowlevel_v2.phase4b.collect import collect_active_states
from wt.lowlevel_v2.phase4b.physics import evaluate_h3, evaluate_joint_step
from wt.lowlevel_v2.phase4b.protocol import (
    ALPHA_RESIDUAL,
    CANDIDATE_OFFSET,
    EXPECTED_HIGH_SHA,
    HEADROOM_5PCT,
    HIGH_FRAC,
    H_SHADOW,
    LOW_FRAC,
    LOW_SEEDS,
    OUTPUT,
    PHASE3_CKPT,
    PHASE4B,
    TRAINING,
    high_sha,
    load_phase3_gate,
    load_phase4a_gate,
    refuse_phase4b,
)
from wt.lowlevel_v2.phase4b.snapshot import restore_env
from wt.lowlevel_v2.protocol import ROOT


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


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


def _pct(arr) -> dict:
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    empty = {k: float("nan") for k in ("min", "p1", "p10", "median", "p90", "p99", "max", "mean", "std")}
    if a.size == 0:
        empty["n"] = 0
        return empty
    qs = np.quantile(a, [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
    return {
        "min": float(qs[0]),
        "p1": float(qs[1]),
        "p10": float(qs[2]),
        "median": float(qs[3]),
        "p90": float(qs[4]),
        "p99": float(qs[5]),
        "max": float(qs[6]),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "n": int(a.size),
    }


def _spearman(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    if float(np.std(rx)) < 1e-12 or float(np.std(ry)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def load_rl_worker(seed: int):
    high, _ = load_frozen_high()
    actor, critic = make_networks()
    payload = torch.load(PHASE3_CKPT / f"ckpt_seed{seed}_update100.pt", map_location="cpu")
    actor.load_state_dict(payload["actor"])
    critic.load_state_dict(payload["critic"])
    actor.eval()
    critic.eval()
    return make_worker(high, actor, critic)


def rl_desired(worker, state: dict) -> np.ndarray:
    with torch.no_grad():
        pre, _, _ = worker.act_tensors(state["obs"], deterministic=True)
    pre_np = pre.detach().cpu().numpy().astype(np.float64)
    a_des, _, _ = worker._compose_desired(state["a_base"], pre_np, state["lock"])
    return a_des


def split_high_low(states: list[dict]):
    interf = np.asarray([s["team_interf"] for s in states], dtype=np.float64)
    hi_q = float(np.quantile(interf, 1.0 - HIGH_FRAC))
    lo_q = float(np.quantile(interf, LOW_FRAC))
    high = [s for s in states if s["team_interf"] >= hi_q]
    low = [s for s in states if s["team_interf"] <= lo_q]
    return high, low, lo_q, hi_q


def landscape_for_states(env, states, tag: str, rl_lookup=None) -> list[dict]:
    rows = []
    for st in states:
        u = int(st["uav"])
        a_rl_u = None if rl_lookup is None else rl_lookup(st)[u]
        cands = candidate_actions(st["a_base"][u], a_rl_u)
        base_move = np.asarray(st["a_base"], dtype=np.float64).copy()
        base_out = evaluate_joint_step(env, st["snap"], st["option"], base_move, u)
        scored = []
        for name, au in cands.items():
            move = base_move.copy()
            move[u] = au
            out = evaluate_joint_step(env, st["snap"], st["option"], move, u)
            rec = {
                "tag": tag,
                "pool": st["pool"],
                "scenario": st["scenario"],
                "t": st["t"],
                "uav": u,
                "candidate": name,
                "feasible": out["feasible"],
                "team_rate": out["team_rate"],
                "delta_team_rate": out["team_rate"] - base_out["team_rate"],
                "delta_self_rate": out["rate_u"] - base_out["rate_u"],
                "delta_sinr": out["sinr_u"] - base_out["sinr_u"],
                "delta_payload": out["payload_inc"] - base_out["payload_inc"],
                "delta_others_rate": out["others_rate"] - base_out["others_rate"],
                "base_team_rate": base_out["team_rate"],
            }
            scored.append(rec)
            rows.append(rec)
        feas = [r for r in scored if r["feasible"]]
        best = max(feas, key=lambda r: r["delta_team_rate"]) if feas else None
        span = float(max(r["team_rate"] for r in feas) - min(r["team_rate"] for r in feas)) if feas else 0.0
        rel = 0.0
        if best is not None and abs(base_out["team_rate"]) > 1e-18:
            rel = best["delta_team_rate"] / abs(base_out["team_rate"])
        rows.append(
            {
                "tag": tag,
                "pool": st["pool"],
                "scenario": st["scenario"],
                "t": st["t"],
                "uav": u,
                "candidate": "SUMMARY",
                "feasible_count": len(feas),
                "span_team_rate": span,
                "best_candidate": None if best is None else best["candidate"],
                "best_delta_team_rate": 0.0 if best is None else best["delta_team_rate"],
                "best_delta_self_rate": 0.0 if best is None else best["delta_self_rate"],
                "best_delta_SINR": 0.0 if best is None else best["delta_sinr"],
                "best_delta_payload": 0.0 if best is None else best["delta_payload"],
                "base_team_rate": base_out["team_rate"],
                "rel_best_team": rel,
            }
        )
    return rows


def headroom_from_summaries(rows: list[dict], tag: str) -> dict:
    sm = [r for r in rows if r.get("candidate") == "SUMMARY" and r.get("tag") == tag]
    if not sm:
        return {"tag": tag, "n": 0}
    rel = np.asarray([float(r["rel_best_team"]) for r in sm], dtype=np.float64)
    dteam = np.asarray([float(r["best_delta_team_rate"]) for r in sm], dtype=np.float64)
    dpay = np.asarray([float(r["best_delta_payload"]) for r in sm], dtype=np.float64)
    feas = np.asarray([float(r["feasible_count"]) for r in sm], dtype=np.float64)
    return {
        "tag": tag,
        "n": len(sm),
        "mean_feasible": float(np.mean(feas)),
        "frac_gt_1pct": float(np.mean(rel > 0.01)),
        "frac_gt_5pct": float(np.mean(rel > HEADROOM_5PCT)),
        "frac_gt_10pct": float(np.mean(rel > 0.10)),
        "mean_rel": float(np.mean(rel)),
        "median_rel": float(np.median(rel)),
        "p75_rel": float(np.quantile(rel, 0.75)),
        "p90_rel": float(np.quantile(rel, 0.90)),
        "mean_delta_team": float(np.mean(dteam)),
        "median_delta_team": float(np.median(dteam)),
        "frac_payload_pos": float(np.mean(dpay > 0)),
        "mean_span": float(np.mean([float(r["span_team_rate"]) for r in sm])),
    }


def jacobian_rows(states: list[dict], seeds: tuple[int, ...]) -> list[dict]:
    rows = []
    interf = np.asarray([s["raw_interf"] for s in states], dtype=np.float64)
    q1, q2 = (np.quantile(interf, [1 / 3, 2 / 3]) if interf.size else (0.0, 0.0))
    names = ("sinr", "rate", "interference", "inr")
    for seed in seeds:
        w = load_rl_worker(int(seed))
        for st in states:
            u = int(st["uav"])
            obs = st["obs"]
            self_t = torch.as_tensor(obs["self"][u : u + 1], dtype=torch.float32)
            task_t = torch.as_tensor(obs["task"][u : u + 1], dtype=torch.float32)
            comm_t = torch.tensor(np.asarray(obs["communication"][u : u + 1], dtype=np.float32), requires_grad=True)
            neigh_t = torch.as_tensor(obs["neighbor_set"][u : u + 1], dtype=torch.float32)
            near_t = torch.as_tensor(np.asarray(obs["d_nearest"][u : u + 1], dtype=np.float32)).reshape(1, 1)
            mu, _ = w.actor(self_t, task_t, comm_t, neigh_t, near_t)
            mu = mu.reshape(2)
            band = "low" if st["raw_interf"] <= q1 else ("high" if st["raw_interf"] > q2 else "medium")
            rec = {"model_seed": int(seed), "scenario": st["scenario"], "uav": u, "band": band, "raw_interf": st["raw_interf"]}
            for k, name in enumerate(names):
                grads = []
                for i in range(2):
                    g = torch.autograd.grad(mu[i], comm_t, retain_graph=True, allow_unused=True)[0]
                    grads.append(0.0 if g is None else float(g.reshape(-1)[k].detach().cpu()))
                rec[f"jac_{name}"] = float(np.linalg.norm(grads))
            rows.append(rec)
    return rows


def plot_figures(head_hi, head_lo, align, jac, regret, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    labels = [">1%", ">5%", ">10%", "median rel"]
    hi = [head_hi.get("frac_gt_1pct", 0), head_hi.get("frac_gt_5pct", 0), head_hi.get("frac_gt_10pct", 0), head_hi.get("median_rel", 0)]
    lo = [head_lo.get("frac_gt_1pct", 0), head_lo.get("frac_gt_5pct", 0), head_lo.get("frac_gt_10pct", 0), head_lo.get("median_rel", 0)]
    x = np.arange(4)
    ax.bar(x - 0.2, hi, 0.4, label="HIGH")
    ax.bar(x + 0.2, lo, 0.4, label="LOW")
    ax.set_xticks(list(x), labels)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "llv2_phase4b_high_low_headroom.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    if regret:
        ax.bar([str(r["model_seed"]) for r in regret], [r["mean_regret_high"] for r in regret])
    ax.set_ylabel("mean team-rate regret HIGH")
    fig.tight_layout()
    fig.savefig(out / "llv2_phase4b_rl_regret.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    if jac:
        names = ["sinr", "rate", "interference", "inr"]
        ax.bar(names, [float(np.mean([r[f"jac_{n}"] for r in jac])) for n in names])
    ax.set_ylabel("mean ||d mu / d feat||")
    fig.tight_layout()
    fig.savefig(out / "llv2_phase4b_jacobian.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    if align:
        ax.bar([str(r["model_seed"]) for r in align], [r["mean_cosine"] for r in align])
        ax.axhline(0, color="k")
    ax.set_ylabel("RL vs best-comm cosine")
    fig.tight_layout()
    fig.savefig(out / "llv2_phase4b_direction_alignment.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(["HIGH span", "LOW span"], [head_hi.get("mean_span", 0), head_lo.get("mean_span", 0)])
    fig.tight_layout()
    fig.savefig(out / "llv2_phase4b_candidate_utility_span.png", dpi=130)
    plt.close(fig)


def decide_gates(head_hi, head_lo, h3, feat_notes, jac) -> dict:
    n_feas = float(head_hi.get("mean_feasible", 0) or 0)
    frac5 = float(head_hi.get("frac_gt_5pct", 0) or 0)
    med = float(head_hi.get("median_rel", 0) or 0)
    h3_med = float(h3.get("median_rel_h3_team", 0) or 0)
    h3_pay = float(h3.get("frac_payload_pos", 0) or 0)
    high_gt_low = (med > float(head_lo.get("median_rel", 0) or 0) + 1e-12) and (frac5 > float(head_lo.get("frac_gt_5pct", 0) or 0))
    feas_ok = n_feas >= 2.0
    jac_i = float(np.mean([r["jac_interference"] for r in jac])) if jac else 0.0
    jac_s = float(np.mean([r["jac_sinr"] for r in jac])) if jac else 0.0
    interf_dead = bool(feat_notes.get("interference_almost_zero") or jac_i < 0.01 * max(jac_s, 1e-8))
    if feas_ok and frac5 >= 0.25 and med > 0 and (h3_med > 0 or h3_pay >= 0.25) and high_gt_low:
        comm, headroom, ctext = "A", True, "Safe alternatives exist and HIGH-interference states show learnable team-rate headroom."
    elif (frac5 > 0.05 or med > 0) and feas_ok:
        comm, headroom, ctext = "B", False, "Some communication headroom exists but it is small or H=3 payload impact is weak."
    else:
        comm, headroom, ctext = "C", False, "Safe alternatives do not stably improve communication utility, or HIGH≈LOW."
    if comm == "C":
        feat, ftext = "C", "Communication trajectory headroom itself is insufficient; do not change the Actor."
        nxt = "do not force interference-aware trajectory learning on this benchmark"
    elif interf_dead:
        feat, ftext = "A", "Headroom exists but interference/INR representation or Jacobian is effectively dead."
        nxt = "only repair communication feature normalization; do not add a communication auxiliary yet"
    else:
        feat, ftext = "B", "Feature scale is usable; residual communication headroom is limited and not payload-stable."
        nxt = "do not train now; do not add an interference auxiliary in this phase"
    if comm == "B":
        nxt = "do not train now; communication rate headroom exists but next-step/H3 payload impact is weak; no interference module this phase"
    return {
        "COMM": comm,
        "communication_trajectory_headroom": headroom,
        "COMFEAT": feat,
        "training": False,
        "collision_module_touched": False,
        "COLREL": "C",
        "text_comm": ctext,
        "text_feat": ftext,
        "next_step": nxt,
        "high_mean_feasible": n_feas,
        "high_frac_gt_5pct": frac5,
        "high_median_rel": med,
        "high_gt_low": high_gt_low,
        "interf_jacobian_mean": jac_i,
        "sinr_jacobian_mean": jac_s,
    }


def write_report(p3, p4a, feat_rows, head_hi, head_lo, h3, regret, align, abl, corr, gate, n_high, n_low) -> str:
    def row(name, stage):
        hits = [r for r in feat_rows if r["feature"] == name and r["stage"] == stage]
        return hits[0] if hits else {}

    si, ai = row("interference", "raw"), row("interference", "actor_input")
    sinr_r = row("sinr", "raw")
    rate_r = row("rate", "raw")
    inr_r, inr_a = row("inr", "raw"), row("inr", "actor_input")
    return "\n".join(
        [
            "# HTW Low-Level V2 Phase-4B：通信干扰可控性与动作相关性审计",
            "",
            "本阶段 **training=false**。未改方法、未加 penalty/reward、未碰 collision distillation、未访问 223000–223063。仅 LLV2_train + LLV2_design。候选幅度冻结 `0.5 * alpha_residual = 0.125`。H=3 唯一。",
            f"Phase-3 LLV2-MS2-{p3['LLV2_MS2']}，stable={p3['stable_residual_advantage']}，REL2={p3['REL2']}。Phase-4A COLREL-{p4a['COLREL']}，collision_relation_internalized={p4a['collision_relation_internalized']}。碰撞仍由 SafetyProjector 硬保证。",
            "",
            "## 22 问正式回答",
            "",
            f"1. Interference/INR：raw interf mean={si.get('mean')} std={si.get('std')} min={si.get('min')} max={si.get('max')}；actor_input mean={ai.get('mean')} std={ai.get('std')}。INR raw mean={inr_r.get('mean')} actor={inr_a.get('mean')}。无 log/clip/norm。",
            f"2. SINR/rate：SINR mean={sinr_r.get('mean')} std={sinr_r.get('std')}；rate mean={rate_r.get('mean')} std={rate_r.get('std')}（bps，未缩放）。actor_input 为同一数值。",
            f"3. Jacobian 均值 interf={gate['interf_jacobian_mean']:.6g} SINR={gate['sinr_jacobian_mean']:.6g}。",
            f"4. HIGH states={n_high}；LOW={n_low}。",
            f"5. HIGH 平均 safe feasible（含 C0）={head_hi.get('mean_feasible')}",
            f"6. >1% team-rate={head_hi.get('frac_gt_1pct')}",
            f"7. >5%={head_hi.get('frac_gt_5pct')}",
            f"8. >10%={head_hi.get('frac_gt_10pct')}",
            f"9. HIGH median rel={head_hi.get('median_rel')}",
            f"10. LOW median rel={head_lo.get('median_rel')}",
            f"11. H=3 median rel team={h3.get('median_rel_h3_team')}；frac>0={h3.get('frac_h3_team_pos')}",
            f"12. next-step payload+ HIGH={head_hi.get('frac_payload_pos')}；H3 payload+={h3.get('frac_payload_pos')}",
            f"13. RL HIGH mean regret={ [r.get('mean_regret_high') for r in regret] }",
            f"14. cosine mean={ [r.get('mean_cosine') for r in align] }；pos frac={ [r.get('positive_fraction') for r in align] }",
            f"15. 特征消融={abl}",
            f"16. Spearman interf-d_nearest={corr.get('interf_d_nearest')}；interf-d_target={corr.get('interf_d_target')}",
            f"17. COMM=**{gate['COMM']}**",
            f"18. communication_trajectory_headroom=**{gate['communication_trajectory_headroom']}**",
            f"19. COMFEAT=**{gate['COMFEAT']}**",
            f"20. 下一步：{gate['next_step']}",
            "21. 本阶段是否训练？**否**。",
            "22. 是否重新碰 collision 模块？**否**。",
            "",
            "本阶段停止。",
            "",
        ]
    ) + "\n"


def main() -> dict:
    if TRAINING:
        raise RuntimeError("Phase-4B forbids training")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    p3 = load_phase3_gate()
    p4a = load_phase4a_gate()
    if high_sha() != EXPECTED_HIGH_SHA:
        raise RuntimeError("high SHA mismatch")
    train_seeds, design_seeds = load_pools()
    for s in list(train_seeds) + list(design_seeds):
        refuse_phase4b(int(s))
    _dump(
        OUTPUT / "LLV2_PHASE4B_PROTOCOL.json",
        {
            "name": "LL-V2.4B",
            "training": False,
            "collision_distillation": False,
            "candidate_offset": CANDIDATE_OFFSET,
            "alpha_residual": ALPHA_RESIDUAL,
            "offset_rule": "0.5 * alpha_residual, single value, no grid",
            "H": H_SHADOW,
            "high_frac": HIGH_FRAC,
            "low_frac": LOW_FRAC,
            "phase3_validation_blocked": True,
            "actor_comm_transform": "identity copy of sinr, rate, total_interference, INR; idle/invalid target -> zeros",
        },
    )
    _dump(
        OUTPUT / "LLV2_PHASE4B_ACTION_CANDIDATES.json",
        {
            "offset": CANDIDATE_OFFSET,
            "C0": "base",
            "C1": "Phase-3 RL desired",
            "C2": "base + offset * left",
            "C3": "base - offset * left",
            "C4": "base - offset * forward",
            "C5": "base + offset * forward then clip_unit_step",
        },
    )
    high_mod, _ = load_frozen_high()
    actor, critic = make_networks()
    l0 = make_worker(high_mod, actor, critic)
    train_states = collect_active_states(l0, train_seeds, pool="train", zero_residual=True)
    design_states = collect_active_states(l0, design_seeds, pool="design", zero_residual=True)
    states = train_states + design_states
    feat_rows = []
    for name, raw_key, col in (("sinr", "raw_sinr", 0), ("rate", "raw_rate", 1), ("interference", "raw_interf", 2), ("inr", "raw_inr", 3)):
        raw = [s[raw_key] for s in states]
        act = [float(s["actor_comm"][col]) for s in states]
        for stage, st in (("raw", _pct(raw)), ("actor_input", _pct(act))):
            feat_rows.append({"feature": name, "stage": stage, **st})
    _csv(OUTPUT / "LLV2_PHASE4B_COMM_FEATURE_SCALE.csv", feat_rows)
    interf_act = np.asarray([float(s["actor_comm"][2]) for s in states], dtype=np.float64)
    feat_notes = {
        "interference_almost_zero": bool(np.nanmean(np.abs(interf_act)) < 1e-6) if interf_act.size else True,
        "no_log_clip_scale": True,
        "duplicated_sinr_rate": bool(abs(_spearman([s["raw_sinr"] for s in states], [s["raw_rate"] for s in states])) > 0.999),
    }
    high, low, lo_q, hi_q = split_high_low(states)
    _dump(
        OUTPUT / "LLV2_PHASE4B_STATE_MANIFEST.json",
        {
            "n_active": len(states),
            "n_train": len(train_states),
            "n_design": len(design_states),
            "n_high": len(high),
            "n_low": len(low),
            "low_q": lo_q,
            "high_q": hi_q,
            "generator": "L0 gated fly-toward zero_residual",
            "training": False,
        },
    )
    env = l0.env
    land_hi = landscape_for_states(env, high, "HIGH")
    land_lo = landscape_for_states(env, low, "LOW")
    _csv(OUTPUT / "LLV2_PHASE4B_ACTION_UTILITY_LANDSCAPE.csv", land_hi + land_lo)
    head_hi = headroom_from_summaries(land_hi, "HIGH")
    head_lo = headroom_from_summaries(land_lo, "LOW")
    _csv(OUTPUT / "LLV2_PHASE4B_HIGH_LOW_HEADROOM.csv", [head_hi, head_lo])

    h3_rows = []
    for st in high:
        u = int(st["uav"])
        cands = candidate_actions(st["a_base"][u], None)
        base_move = np.asarray(st["a_base"], dtype=np.float64).copy()
        h0 = evaluate_h3(env, st["snap"], st["option"], base_move, l0.base.act, u)
        best_rel, best_pay = 0.0, 0.0
        for name, au in cands.items():
            move = base_move.copy()
            move[u] = au
            flags = evaluate_joint_step(env, st["snap"], st["option"], move, u)
            if not flags["feasible"]:
                continue
            h = evaluate_h3(env, st["snap"], st["option"], move, l0.base.act, u)
            rel = (h["h3_team_rate_integral"] - h0["h3_team_rate_integral"]) / max(abs(h0["h3_team_rate_integral"]), 1e-18)
            d_pay = h["h3_payload_inc"] - h0["h3_payload_inc"]
            best_rel = max(best_rel, rel)
            best_pay = max(best_pay, d_pay)
            h3_rows.append({"scenario": st["scenario"], "uav": u, "candidate": name, "rel_h3_team": rel, "delta_h3_payload": d_pay})
        h3_rows.append({"scenario": st["scenario"], "uav": u, "candidate": "BEST", "rel_h3_team": best_rel, "delta_h3_payload": best_pay})
    _csv(OUTPUT / "LLV2_PHASE4B_H3_COMM_HEADROOM.csv", h3_rows)
    best_h3 = [r for r in h3_rows if r.get("candidate") == "BEST"]
    h3 = {
        "median_rel_h3_team": float(np.median([r["rel_h3_team"] for r in best_h3])) if best_h3 else float("nan"),
        "mean_rel_h3_team": float(np.mean([r["rel_h3_team"] for r in best_h3])) if best_h3 else float("nan"),
        "frac_h3_team_pos": float(np.mean([r["rel_h3_team"] > 0 for r in best_h3])) if best_h3 else float("nan"),
        "frac_payload_pos": float(np.mean([r["delta_h3_payload"] > 0 for r in best_h3])) if best_h3 else float("nan"),
    }

    jac = jacobian_rows(design_states, LOW_SEEDS) if design_states else []
    _csv(OUTPUT / "LLV2_PHASE4B_COMM_JACOBIAN.csv", jac)

    hi_sum = {(r["scenario"], r["t"], r["uav"]): r for r in land_hi if r.get("candidate") == "SUMMARY"}
    lo_sum = {(r["scenario"], r["t"], r["uav"]): r for r in land_lo if r.get("candidate") == "SUMMARY"}
    regret_rows, align_rows = [], []
    for seed in LOW_SEEDS:
        w = load_rl_worker(int(seed))
        regs_h, regs_l, cosines = [], [], []
        for st, store in ((high, hi_sum), (low, lo_sum)):
            bucket = regs_h if store is hi_sum else regs_l
            for rec in st:
                a_des = rl_desired(w, rec)
                u = int(rec["uav"])
                move = np.asarray(rec["a_base"], dtype=np.float64).copy()
                move[u] = a_des[u]
                base_move = np.asarray(rec["a_base"], dtype=np.float64).copy()
                base_out = evaluate_joint_step(env, rec["snap"], rec["option"], base_move, u)
                rl_out = evaluate_joint_step(env, rec["snap"], rec["option"], move, u)
                sm = store.get((rec["scenario"], rec["t"], u))
                u_best = base_out["team_rate"] + (0.0 if sm is None else float(sm["best_delta_team_rate"]))
                bucket.append(u_best - rl_out["team_rate"])
                if store is hi_sum and sm is not None and sm.get("best_candidate") not in {None, "C0_base"}:
                    cands = candidate_actions(rec["a_base"][u], a_des[u])
                    a_best = cands.get(sm["best_candidate"], rec["a_base"][u])
                    r_rl = a_des[u] - rec["a_base"][u]
                    r_best = np.asarray(a_best) - rec["a_base"][u]
                    n1, n2 = np.linalg.norm(r_rl), np.linalg.norm(r_best)
                    if n1 >= 1e-9 and n2 >= 1e-9:
                        cosines.append(float(np.dot(r_rl, r_best) / (n1 * n2)))
        regret_rows.append(
            {
                "model_seed": int(seed),
                "mean_regret_high": float(np.mean(regs_h)) if regs_h else float("nan"),
                "median_regret_high": float(np.median(regs_h)) if regs_h else float("nan"),
                "mean_regret_low": float(np.mean(regs_l)) if regs_l else float("nan"),
            }
        )
        arr = np.asarray(cosines, dtype=np.float64)
        align_rows.append(
            {
                "model_seed": int(seed),
                "mean_cosine": float(np.mean(arr)) if arr.size else float("nan"),
                "median_cosine": float(np.median(arr)) if arr.size else float("nan"),
                "positive_fraction": float(np.mean(arr > 0)) if arr.size else float("nan"),
                "n": int(arr.size),
            }
        )
    _csv(OUTPUT / "LLV2_PHASE4B_RL_COMM_REGRET.csv", regret_rows)
    _csv(OUTPUT / "LLV2_PHASE4B_DIRECTION_ALIGNMENT.csv", align_rows)

    rng = np.random.default_rng(22401)
    abl_rows = []
    w0 = load_rl_worker(0)
    for st in high[: min(80, len(high))]:
        u = int(st["uav"])
        a0 = rl_desired(w0, st)
        move0 = np.asarray(st["a_base"], dtype=np.float64).copy()
        move0[u] = a0[u]
        u0 = evaluate_joint_step(env, st["snap"], st["option"], move0, u)["team_rate"]
        comm = np.array(st["obs"]["communication"], copy=True)
        variants = {
            "interf_zero": np.array(comm, copy=True),
            "interf_shuffle": np.array(comm, copy=True),
            "sinr_zero": np.array(comm, copy=True),
            "sinr_shuffle": np.array(comm, copy=True),
        }
        variants["interf_zero"][:, 2:4] = 0.0
        variants["interf_shuffle"][:, 2] = rng.permutation(variants["interf_shuffle"][:, 2])
        variants["interf_shuffle"][:, 3] = rng.permutation(variants["interf_shuffle"][:, 3])
        variants["sinr_zero"][:, 0:2] = 0.0
        variants["sinr_shuffle"][:, 0] = rng.permutation(variants["sinr_shuffle"][:, 0])
        variants["sinr_shuffle"][:, 1] = rng.permutation(variants["sinr_shuffle"][:, 1])
        for vname, mat in variants.items():
            obs = dict(st["obs"], communication=mat)
            a1 = rl_desired(w0, dict(st, obs=obs))
            dact = float(np.linalg.norm(a1[u] - a0[u]))
            move = np.asarray(st["a_base"], dtype=np.float64).copy()
            move[u] = a1[u]
            u1 = evaluate_joint_step(env, st["snap"], st["option"], move, u)["team_rate"]
            abl_rows.append({"variant": vname, "action_l2": dact, "delta_team_rate": u1 - u0, "utility_improved": bool(u1 > u0)})
    _csv(OUTPUT / "LLV2_PHASE4B_FEATURE_ABLATION.csv", abl_rows)
    abl_sum = {}
    for v in ("interf_zero", "interf_shuffle", "sinr_zero", "sinr_shuffle"):
        sub = [r for r in abl_rows if r["variant"] == v]
        abl_sum[v] = {
            "mean_action_l2": float(np.mean([r["action_l2"] for r in sub])) if sub else float("nan"),
            "mean_delta_team": float(np.mean([r["delta_team_rate"] for r in sub])) if sub else float("nan"),
            "frac_utility_up": float(np.mean([r["utility_improved"] for r in sub])) if sub else float("nan"),
        }
    corr = {
        "interf_d_nearest": _spearman([s["raw_interf"] for s in states], [s["d_nearest"] for s in states]),
        "interf_d_target": _spearman([s["raw_interf"] for s in states], [s["d_target"] for s in states]),
        "inr_d_nearest": _spearman([s["raw_inr"] for s in states], [s["d_nearest"] for s in states]),
        "sinr_d_target": _spearman([s["raw_sinr"] for s in states], [s["d_target"] for s in states]),
    }
    _csv(OUTPUT / "LLV2_PHASE4B_GEOMETRY_CORRELATION.csv", [{"pair": k, "spearman": v} for k, v in corr.items()])
    gate = decide_gates(head_hi, head_lo, h3, feat_notes, jac)
    gate.update({"phase3": p3["LLV2_MS2"], "phase4a_COLREL": p4a["COLREL"], "feat_notes": feat_notes})
    _dump(OUTPUT / "LLV2_PHASE4B_GATE.json", gate)
    plot_figures(head_hi, head_lo, align_rows, jac, regret_rows, OUTPUT / "figures")
    name = "HTW_LowLevelV2_Phase4B_CommunicationControllability_CN.md"
    report = write_report(p3, p4a, feat_rows, head_hi, head_lo, h3, regret_rows, align_rows, abl_sum, corr, gate, len(high), len(low))
    _dump(OUTPUT / name, report)
    for src in list(OUTPUT.glob("LLV2_PHASE4B_*")) + [OUTPUT / name]:
        if src.exists() and src.is_file():
            (PHASE4B / src.name).write_bytes(src.read_bytes())
            (ROOT / src.name).write_bytes(src.read_bytes())
    fig_dir = PHASE4B / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for png in (OUTPUT / "figures").glob("*.png"):
        (fig_dir / png.name).write_bytes(png.read_bytes())
    (ROOT / ("docs/archive/phase_reports" if str(name).endswith(".md") else "results/root_exports") / name).write_text(report, encoding="utf-8")
    (PHASE4B / name).write_text(report, encoding="utf-8")
    return {"COMM": gate["COMM"], "COMFEAT": gate["COMFEAT"], "training": False, "n_high": len(high), "n_active": len(states)}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
