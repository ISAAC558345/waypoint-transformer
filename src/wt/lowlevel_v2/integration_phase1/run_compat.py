"""HL-LL COMPAT runner. training=false. Frozen DeepSets + five Phase-3 Residual MAPPO seeds."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from wt.htw_phase92.run_phase92 import _batch_from_env
from wt.htw_phase943.evaluation import encode_env
from wt.htw_phase943.set_context import make_snap
from wt.htw_phase945.candidate_pool import assignment_hash, is_legal
from wt.htw_phase945.utility_loss import candidate_logprobs
from wt.lowlevel_v2.phase1b.controller import load_frozen_high, make_networks
from wt.lowlevel_v2.phase2b.eval import evaluate_design
from wt.lowlevel_v2.phase4b.snapshot import restore_env

from .collect import collect_hold_states
from .features import candidate_features
from .protocol import (
    CKPT_UPDATE,
    EXPECTED_HIGH_SHA,
    HOLD,
    HOLD_STEPS,
    INTEG1,
    LOW_SEEDS,
    NEAR_OPT_FRAC,
    N_STATES,
    OUTPUT,
    PHASE106_DIR,
    POOL_SIZE,
    ROOT,
    TRAINING,
    UTILITY_EPS,
    ckpt_path,
    high_sha,
    load_phase3_gate,
    load_phase4a_gate,
    load_phase4b_gate,
    load_pools,
    refuse_compat,
)
from .shadow import shadow_h15
from .stats import jaccard, kendall_tau, near_opt_set, spearman, summarize, topk_overlap
from .worker import CompatWorker


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n", encoding="utf-8")


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


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
            writer.writerow({k: _jsonable(v) for k, v in row.items()})


def load_compat_worker(model_seed: int | None) -> CompatWorker:
    high, digest = load_frozen_high()
    if digest != EXPECTED_HIGH_SHA:
        raise RuntimeError("high SHA mismatch")
    actor, critic = make_networks()
    seed = int(LOW_SEEDS[0] if model_seed is None else model_seed)
    payload = torch.load(ckpt_path(seed), map_location="cpu")
    actor.load_state_dict(payload["actor"])
    critic.load_state_dict(payload["critic"])
    actor.eval()
    critic.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return CompatWorker(high, actor, critic, service_lock=True, trust_region=True)


def policy_logprobs(worker, state: dict) -> np.ndarray:
    restore_env(worker.env, state["env"])
    info = worker.env.info()
    batch = _batch_from_env(info, worker.env, state["last_arrival"], state["last_positions"])
    encoded = encode_env(worker.high, batch)
    snap = make_snap(worker.env.uav_positions, worker.env.buoy_positions, worker.env.config, env=worker.env)
    with torch.no_grad():
        logp = candidate_logprobs(worker.high, encoded, snap, state["candidates"], batch["buoy_mask"][0])
    return logp.detach().cpu().numpy().astype(np.float64)


def unique_needed(state: dict) -> list[np.ndarray]:
    rows = [np.asarray(row, dtype=np.int64) for row in state["candidates"]]
    policy = np.asarray(state["policy_assignment"], dtype=np.int64)
    keys = {tuple(row.tolist()) for row in rows}
    if tuple(policy.tolist()) not in keys:
        rows.append(policy)
    return rows


def run() -> dict:
    if TRAINING:
        raise RuntimeError("training must stay false")
    p3 = load_phase3_gate()
    p4a = load_phase4a_gate()
    p4b = load_phase4b_gate()
    if PHASE106_DIR.joinpath("outputs", "untouched", "phase106_per_scenario_metrics.csv").is_file():
        # File exists on disk; we must not open it.
        pass
    train, design = load_pools()
    for seed in list(train) + list(design):
        refuse_compat(int(seed))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "name": "HL-LL COMPAT",
        "training": False,
        "hold": int(HOLD),
        "pool_size": int(POOL_SIZE),
        "n_states": int(N_STATES),
        "low_seeds": list(LOW_SEEDS),
        "ckpt_update": int(CKPT_UPDATE),
        "primary_rl_utility": "U_RL_mean over five frozen Phase-3 seeds",
        "l0": "gated fly-toward ResidualWorker zero_residual with SafetyProjector",
        "high": "frozen DeepSets-util seed0",
        "high_sha": high_sha(),
        "phase3": p3.get("LLV2_MS2"),
        "stable_residual_advantage": p3.get("stable_residual_advantage"),
        "REL2": p3.get("REL2"),
        "COLREL": p4a.get("COLREL"),
        "collision_relation_internalized": p4a.get("collision_relation_internalized"),
        "COMM": p4b.get("COMM"),
        "communication_trajectory_headroom": p4b.get("communication_trajectory_headroom"),
        "collision_enhancement": False,
        "interference_enhancement": False,
        "design_seeds": design,
        "blocked": ["106000-106063", "222000-222063", "223000-223063", "confirmation", "final untouched"],
    }
    _dump(OUTPUT / "HL_LL_COMPAT_PROTOCOL.json", protocol)

    collector = load_compat_worker(None)
    states = collect_hold_states(collector, design)
    if len(states) != N_STATES:
        raise RuntimeError("state count mismatch")
    manifest = []
    cand_manifest = []
    for state in states:
        manifest.append(
            {
                "state_id": state["state_id"],
                "scenario": state["scenario"],
                "t": state["t"],
                "hold_timing": state["hold_timing"],
                "uav_positions": state["env"]["uav_positions"],
                "queue": state["queue"],
                "progress": state["progress"],
                "association_context": state["association_context"],
                "communication": state["communication"],
                "payload_bits": state["env"]["payload_bits"],
                "policy_hash": state["policy_hash"],
                "saved_future_arrivals": False,
            }
        )
        for slot, (row, digest) in enumerate(zip(state["candidates"], state["candidate_hashes"])):
            cand_manifest.append(
                {
                    "state_id": state["state_id"],
                    "slot": slot,
                    "hash": digest,
                    "assignment": np.asarray(row, dtype=np.int64),
                    "legal": bool(is_legal(row, collector.env.config.num_buoys)),
                }
            )
    _dump(OUTPUT / "HL_LL_COMPAT_STATE_MANIFEST.json", {"n": len(states), "states": manifest})
    _dump(OUTPUT / "HL_LL_COMPAT_CANDIDATE_MANIFEST.json", {"pool_size": POOL_SIZE, "items": cand_manifest})

    rl_workers = {int(seed): load_compat_worker(int(seed)) for seed in LOW_SEEDS}
    cache: dict[tuple, dict] = {}

    def cached_shadow(tag: str, worker, state, assignment, zero_residual: bool) -> float:
        key = (tag, int(state["state_id"]), assignment_hash(assignment))
        if key not in cache:
            cache[key] = shadow_h15(worker, state, assignment, zero_residual=zero_residual)
        return float(cache[key]["payload_increment"])

    l0_rows = []
    rl_seed_rows = []
    rl_mean_rows = []
    for state in states:
        print(f"utility state {state['state_id']+1}/{len(states)} scenario={state['scenario']} t={state['t']}", flush=True)
        needed = unique_needed(state)
        u_map_l0 = {}
        u_map_rl = {seed: {} for seed in LOW_SEEDS}
        for assignment in needed:
            digest = assignment_hash(assignment)
            u_map_l0[digest] = cached_shadow("L0", collector, state, assignment, True)
            for seed, worker in rl_workers.items():
                u_map_rl[seed][digest] = cached_shadow(f"RL{seed}", worker, state, assignment, False)
        for slot, row in enumerate(state["candidates"]):
            digest = state["candidate_hashes"][slot]
            u_l0 = u_map_l0[digest]
            per = [u_map_rl[seed][digest] for seed in LOW_SEEDS]
            u_mean = float(np.mean(per))
            l0_rows.append({"state_id": state["state_id"], "slot": slot, "hash": digest, "U_L0": u_l0})
            for seed, val in zip(LOW_SEEDS, per):
                rl_seed_rows.append({"state_id": state["state_id"], "slot": slot, "hash": digest, "model_seed": int(seed), "U_RL": val})
            rl_mean_rows.append({"state_id": state["state_id"], "slot": slot, "hash": digest, "U_RL_mean": u_mean, **{f"U_RL_{s}": v for s, v in zip(LOW_SEEDS, per)}})
        state["u_l0"] = np.asarray([u_map_l0[h] for h in state["candidate_hashes"]], dtype=np.float64)
        state["u_rl_seed"] = np.stack([np.asarray([u_map_rl[seed][h] for h in state["candidate_hashes"]], dtype=np.float64) for seed in LOW_SEEDS], axis=0)
        state["u_rl_mean"] = np.mean(state["u_rl_seed"], axis=0)
        ph = assignment_hash(state["policy_assignment"])
        state["U_L0_policy"] = u_map_l0[ph]
        state["U_RL_policy"] = float(np.mean([u_map_rl[seed][ph] for seed in LOW_SEEDS]))
        state["logp"] = policy_logprobs(collector, state)
        state["prob"] = np.exp(state["logp"] - np.max(state["logp"]))
        state["prob"] = state["prob"] / np.clip(np.sum(state["prob"]), 1.0e-18, None)

    _csv(OUTPUT / "HL_LL_COMPAT_UTILITY_L0.csv", l0_rows)
    _csv(OUTPUT / "HL_LL_COMPAT_UTILITY_RL_PER_SEED.csv", rl_seed_rows)
    _csv(OUTPUT / "HL_LL_COMPAT_UTILITY_RL_MEAN.csv", rl_mean_rows)

    rank_rows = []
    near_rows = []
    policy_rows = []
    align_rows = []
    shift_rows = []
    inter_rows = []
    agree_rows = []
    old_nr = []
    pol_nr = []
    for state in states:
        u0 = state["u_l0"]
        ur = state["u_rl_mean"]
        k25 = max(int(round(0.25 * POOL_SIZE)), 1)
        rank_rows.append(
            {
                "state_id": state["state_id"],
                "spearman": spearman(u0, ur),
                "kendall": kendall_tau(u0, ur),
                "top1": float(int(np.argmax(u0) == np.argmax(ur))),
                "top3": topk_overlap(u0, ur, 3),
                "top25": topk_overlap(u0, ur, k25),
            }
        )
        s0 = near_opt_set(u0, NEAR_OPT_FRAC)
        sr = near_opt_set(ur, NEAR_OPT_FRAC)
        near_rows.append(
            {
                "state_id": state["state_id"],
                "jaccard": jaccard(s0, sr),
                "n_l0": len(s0),
                "n_rl": len(sr),
            }
        )
        i_l0 = int(np.argmax(u0))
        best_rl = float(np.max(ur))
        regret_old = best_rl - float(ur[i_l0])
        nr_old = regret_old / (abs(best_rl) + UTILITY_EPS)
        old_nr.append(nr_old)
        regret_pol = best_rl - float(state["U_RL_policy"])
        nr_pol = regret_pol / (abs(best_rl) + UTILITY_EPS)
        pol_nr.append(nr_pol)
        best_l0 = float(np.max(u0))
        policy_rows.append(
            {
                "state_id": state["state_id"],
                "U_L0_policy": state["U_L0_policy"],
                "U_RL_policy": state["U_RL_policy"],
                "regret_policy_L0": best_l0 - float(state["U_L0_policy"]),
                "regret_policy_RL": regret_pol,
                "NR_policy_RL": nr_pol,
                "NR_old_best_RL": nr_old,
                "in_pool": assignment_hash(state["policy_assignment"]) in set(state["candidate_hashes"]),
            }
        )
        align_rows.append(
            {
                "state_id": state["state_id"],
                "spearman_prob_L0": spearman(state["prob"], u0),
                "spearman_prob_RL": spearman(state["prob"], ur),
            }
        )
        delta = ur - u0
        shift_rows.append(
            {
                "state_id": state["state_id"],
                "mean_delta": float(np.mean(delta)),
                "std_delta": float(np.std(delta)),
                "range_delta": float(np.max(delta) - np.min(delta)),
            }
        )
        restore_env(collector.env, state["env"])
        for slot, row in enumerate(state["candidates"]):
            feat = candidate_features(collector.env, row, state["communication"])
            inter_rows.append({"state_id": state["state_id"], "slot": slot, "delta_exec": float(delta[slot]), **feat})
        mats = []
        seed_u = state["u_rl_seed"]
        for i, si in enumerate(LOW_SEEDS):
            for j, sj in enumerate(LOW_SEEDS):
                rho = 1.0 if i == j else spearman(seed_u[i], seed_u[j])
                mats.append({"state_id": state["state_id"], "seed_a": int(si), "seed_b": int(sj), "spearman": rho})
        agree_rows.extend(mats)
        cand_std = np.std(seed_u, axis=0)
        agree_rows.append({"state_id": state["state_id"], "seed_a": "std", "seed_b": "mean", "spearman": float(np.mean(cand_std))})

    _csv(OUTPUT / "HL_LL_UTILITY_RANK_COMPAT.csv", rank_rows)
    _csv(OUTPUT / "HL_LL_NEAR_OPTIMAL_OVERLAP.csv", near_rows)
    _csv(OUTPUT / "HL_LL_POLICY_REGRET.csv", policy_rows)
    _csv(OUTPUT / "HL_LL_UTILITY_PROB_ALIGNMENT.csv", align_rows)
    _csv(OUTPUT / "HL_LL_EXECUTOR_SHIFT.csv", shift_rows)
    _csv(OUTPUT / "HL_LL_EXECUTOR_INTERACTION.csv", inter_rows)
    _csv(OUTPUT / "HL_LL_LOWLEVEL_SEED_AGREEMENT.csv", agree_rows)

    pair_rho = [r["spearman"] for r in agree_rows if r.get("seed_a") != "std" and int(r["seed_a"]) < int(r["seed_b"])]
    feat_names = [
        "mean_uav_target_dist",
        "max_uav_target_dist",
        "queue_pressure",
        "idle_count",
        "current_mean_sinr",
        "current_interference",
        "uav_pairwise_proximity",
        "competition_indicator",
    ]
    inter_corr = {name: spearman([r["delta_exec"] for r in inter_rows], [r[name] for r in inter_rows]) for name in feat_names}

    print("closed-loop LLV2_design", flush=True)
    loop_rows = []
    l0_metrics = evaluate_design(collector, design, zero_residual=True)
    loop_rows.append({"executor": "L0", "model_seed": "L0", **{k: v for k, v in l0_metrics.items() if k != "scenarios"}})
    rl_payloads = []
    for seed, worker in rl_workers.items():
        metrics = evaluate_design(worker, design, zero_residual=False)
        loop_rows.append({"executor": "RL", "model_seed": int(seed), **{k: v for k, v in metrics.items() if k != "scenarios"}})
        rl_payloads.append(float(metrics["payload"]))
    _csv(OUTPUT / "HL_LL_CLOSED_LOOP_DESIGN.csv", loop_rows)

    rank_sum = summarize([r["spearman"] for r in rank_rows])
    ken_sum = summarize([r["kendall"] for r in rank_rows])
    top1_sum = summarize([r["top1"] for r in rank_rows])
    top3_sum = summarize([r["top3"] for r in rank_rows])
    top25_sum = summarize([r["top25"] for r in rank_rows])
    jac_sum = summarize([r["jaccard"] for r in near_rows])
    old_sum = summarize(old_nr)
    pol_sum = summarize(pol_nr)
    prob_rl = summarize([r["spearman_prob_RL"] for r in align_rows])
    seed_sum = summarize(pair_rho)
    shift_range = summarize([r["range_delta"] for r in shift_rows])
    assignment_dependent = float(shift_range["median"]) > 0.10 * max(abs(float(np.median([r["mean_delta"] for r in shift_rows]))), UTILITY_EPS)

    med_s = rank_sum["median"]
    med_k = ken_sum["median"]
    med_25 = top25_sum["median"]
    med_j = jac_sum["median"]
    med_old = old_sum["median"]
    med_pol = pol_sum["median"]
    med_pr = prob_rl["median"]
    med_seed = seed_sum["median"]
    no_flip = bool(med_seed >= 0.50)
    if (
        med_s >= 0.80
        and med_k >= 0.65
        and med_25 >= 0.75
        and med_j >= 0.60
        and med_old <= 0.05
        and med_pol <= 0.10
        and med_pr > 0
        and no_flip
    ):
        gate, retr = "A", False
        text = "Frozen DeepSets high-level ranking remains compatible with Residual MAPPO utilities."
    elif med_s < 0.50 or med_pol > 0.25:
        gate, retr = "C", True
        text = "Original high-level utility semantics do not transfer to the Residual MAPPO executor."
    else:
        gate, retr = "B", True
        text = "New low-level executor changes part of the high-level utility structure."
    if gate == "A":
        next_step = "Do not retrain the high-level. Next: full stacked-system benchmark of frozen DeepSets-util + Residual MAPPO."
    else:
        next_step = "Do not train in this phase. Next: regenerate U_new(Gamma|s)=mean of five frozen RL seeds H=15 payload, then retrain DeepSets high-level."

    gate_payload = {
        "COMPAT": gate,
        "high_level_retraining_needed": retr,
        "training": False,
        "collision_enhancement": False,
        "interference_enhancement": False,
        "COLREL": "C",
        "COMM": "B",
        "communication_trajectory_headroom": False,
        "median_spearman": med_s,
        "median_kendall": med_k,
        "median_top1": top1_sum["median"],
        "median_top3": top3_sum["median"],
        "median_top25": med_25,
        "median_near_opt_jaccard": med_j,
        "median_NR_old_best_RL": med_old,
        "median_NR_policy_RL": med_pol,
        "median_prob_vs_U_RL": med_pr,
        "median_seed_pair_spearman": med_seed,
        "assignment_dependent_executor_gain": assignment_dependent,
        "interaction_spearman": inter_corr,
        "closed_loop_L0_payload": l0_metrics["payload"],
        "closed_loop_RL_payloads": rl_payloads,
        "closed_loop_RL_mean_payload": float(np.mean(rl_payloads)),
        "n_states": len(states),
        "pool_size": POOL_SIZE,
        "text": text,
        "next_step": next_step,
        "summaries": {
            "spearman": rank_sum,
            "kendall": ken_sum,
            "top1": top1_sum,
            "top3": top3_sum,
            "top25": top25_sum,
            "jaccard": jac_sum,
            "NR_old": old_sum,
            "NR_policy": pol_sum,
            "prob_RL": prob_rl,
            "seed_pair": seed_sum,
        },
    }
    _dump(OUTPUT / "HL_LL_COMPAT_GATE.json", gate_payload)
    plot_figures(states, rank_rows, near_rows, policy_rows, shift_rows, agree_rows, OUTPUT / "figures")
    report = write_report(gate_payload, inter_corr, assignment_dependent)
    _dump(OUTPUT / "HTW_HighLow_Utility_Compatibility_Audit_CN.md", report)
    publish(report)
    return gate_payload


def plot_figures(states, rank_rows, near_rows, policy_rows, shift_rows, agree_rows, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    u0 = np.concatenate([s["u_l0"] for s in states])
    ur = np.concatenate([s["u_rl_mean"] for s in states])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(u0 / 1e6, ur / 1e6, s=12, alpha=0.6)
    ax.set_xlabel("U_L0 (Mbit)")
    ax.set_ylabel("U_RL_mean (Mbit)")
    ax.set_title("L0 vs RL utility")
    fig.tight_layout()
    fig.savefig(out / "hl_ll_utility_scatter.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.hist([r["spearman"] for r in rank_rows], bins=12)
    ax.set_title("Spearman distribution")
    fig.tight_layout()
    fig.savefig(out / "hl_ll_spearman.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.hist([r["kendall"] for r in rank_rows], bins=12)
    ax.set_title("Kendall distribution")
    fig.tight_layout()
    fig.savefig(out / "hl_ll_kendall.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.boxplot([[r["top1"] for r in rank_rows], [r["top3"] for r in rank_rows], [r["top25"] for r in rank_rows]], tick_labels=["top-1", "top-3", "top-25%"])
    ax.set_title("top-k overlap")
    fig.tight_layout()
    fig.savefig(out / "hl_ll_topk.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.hist([r["jaccard"] for r in near_rows], bins=12)
    ax.set_title("near-optimal Jaccard")
    fig.tight_layout()
    fig.savefig(out / "hl_ll_jaccard.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.hist([r["NR_policy_RL"] for r in policy_rows], bins=12)
    ax.set_title("policy regret under RL")
    fig.tight_layout()
    fig.savefig(out / "hl_ll_policy_regret.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.hist([r["range_delta"] / 1e6 for r in shift_rows], bins=12)
    ax.set_title("executor shift range")
    fig.tight_layout()
    fig.savefig(out / "hl_ll_executor_shift.png", dpi=130)
    plt.close(fig)

    pair = [r["spearman"] for r in agree_rows if r.get("seed_a") != "std" and int(r["seed_a"]) < int(r["seed_b"])]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.hist(pair, bins=12)
    ax.set_title("low-level seed rank agreement")
    fig.tight_layout()
    fig.savefig(out / "hl_ll_seed_agreement.png", dpi=130)
    plt.close(fig)


def write_report(gate: dict, inter_corr: dict, assignment_dependent: bool) -> str:
    s = gate["summaries"]
    corr_txt = ", ".join(f"{k}={v:.3f}" for k, v in inter_corr.items())
    rl_txt = ", ".join(f"{p:.3e}" for p in gate["closed_loop_RL_payloads"])
    return f"""# HTW Integration Phase-1：高层效用兼容性审计（冻结 Residual MAPPO 低层）

本阶段 **training=false**。未重新训练高层或低层。未改 reward / Actor / Critic / SafetyProjector / Task Option / hold。未开启 collision 或 interference 增强。未加 HGNN / Transformer。未访问 `106000–106063`、`222000–222063`、`223000–223063` 或 confirmation / final untouched。

恢复：高层 DeepSets-util + Order-Free + Utility-Aligned + Task Option + hold=15。旧低层 L0 gated fly-toward。新低层 Residual MAPPO + ServiceResidualLock + BaseZeroResidualFreeze + Active-Sample-Aware + SafetyProjector + ServiceSafeEnvelope。

确认：LLV2-MS2-A，`stable_residual_advantage=true`，REL2-0，COLREL-C，`collision_relation_internalized=false`，COMM-B，`communication_trajectory_headroom=false`。碰撞继续由 SafetyProjector 硬保证。不能写成 Actor 已学会主动避碰或 interference-aware trajectory。

开发数据：仅 `LLV2_design`。64 个高层决策 state = 16 场景 × hold 边界，预注册，不按结果挑选。Candidate 使用 Phase-9.4.5 冻结 model-independent generator，L=24。U_L0 与 U_RL 都是同一 state、同一 assignment、H=15、不重调高层下的累计 payload increment。Primary RL utility 是五 seed 均值。

---

## 24 问正式回答

1. **用了多少 state？** {gate['n_states']}
2. **每 state 多少 candidate？** {gate['pool_size']}
3. **L0 utility？** 冻结 assignment 后 H=15，L0 gated fly-toward（zero residual + SafetyProjector）累计 payload increment。
4. **RL utility？** 同样 H=15 shadow，冻结 Phase-3 Residual MAPPO。
5. **是否使用 5 低层 seed mean？** 是。Primary=`U_RL_mean`。未挑最好 seed。
6. **median Spearman？** {s['spearman']['median']:.4f}（mean={s['spearman']['mean']:.4f}）
7. **median Kendall tau？** {s['kendall']['median']:.4f}
8. **top-1 agreement？** median={s['top1']['median']:.4f}，mean={s['top1']['mean']:.4f}
9. **top-3 overlap？** median={s['top3']['median']:.4f}
10. **top-25% overlap？** median={s['top25']['median']:.4f}
11. **near-optimal Jaccard？** median={s['jaccard']['median']:.4f}
12. **old L0-best 在 RL 下 normalized regret？** median={s['NR_old']['median']:.4f}，mean={s['NR_old']['mean']:.4f}，p90={s['NR_old']['p90']:.4f}
13. **当前 DeepSets policy 在 RL 下 normalized regret？** median={s['NR_policy']['median']:.4f}，mean={s['NR_policy']['mean']:.4f}，p90={s['NR_policy']['p90']:.4f}
14. **高层 probability 和 RL utility Spearman？** median={s['prob_RL']['median']:.4f}
15. **executor gain 是否 assignment-dependent？** {'是' if assignment_dependent else '否（各 candidate 的 Delta_exec 差异相对小）'}
16. **哪些 assignment 特征与 executor gain 相关？** {corr_txt}。只作描述，不作因果。
17. **五低层 seed ranking agreement？** median pairwise Spearman={s['seed_pair']['median']:.4f}
18. **Frozen high-level + L0 closed-loop payload？** {gate['closed_loop_L0_payload']:.6e}
19. **Frozen high-level + 五 RL seed closed-loop payload？** {rl_txt}；mean={gate['closed_loop_RL_mean_payload']:.6e}
20. **COMPAT Gate？** {gate['COMPAT']}
21. **high_level_retraining_needed？** {str(gate['high_level_retraining_needed']).lower()}
22. **下一阶段唯一建议？** {gate['next_step']}
23. **本阶段是否训练？否。**
24. **是否重新开启 collision/interference 增强？否。**

{gate['text']}

本阶段停止。
"""


def publish(report: str) -> None:
    names = [
        "HL_LL_COMPAT_PROTOCOL.json",
        "HL_LL_COMPAT_STATE_MANIFEST.json",
        "HL_LL_COMPAT_CANDIDATE_MANIFEST.json",
        "HL_LL_COMPAT_UTILITY_L0.csv",
        "HL_LL_COMPAT_UTILITY_RL_PER_SEED.csv",
        "HL_LL_COMPAT_UTILITY_RL_MEAN.csv",
        "HL_LL_UTILITY_RANK_COMPAT.csv",
        "HL_LL_NEAR_OPTIMAL_OVERLAP.csv",
        "HL_LL_POLICY_REGRET.csv",
        "HL_LL_UTILITY_PROB_ALIGNMENT.csv",
        "HL_LL_EXECUTOR_SHIFT.csv",
        "HL_LL_EXECUTOR_INTERACTION.csv",
        "HL_LL_LOWLEVEL_SEED_AGREEMENT.csv",
        "HL_LL_CLOSED_LOOP_DESIGN.csv",
        "HL_LL_COMPAT_GATE.json",
        "HTW_HighLow_Utility_Compatibility_Audit_CN.md",
    ]
    for name in names:
        src = OUTPUT / name
        if src.is_file():
            shutil.copy2(src, INTEG1 / name)
            shutil.copy2(src, ROOT / ("docs/archive/phase_reports" if str(name).endswith(".md") else "results/root_exports") / name)
    fig_src = OUTPUT / "figures"
    fig_dst = INTEG1 / "figures"
    fig_dst.mkdir(parents=True, exist_ok=True)
    if fig_src.is_dir():
        for png in fig_src.glob("*.png"):
            shutil.copy2(png, fig_dst / png.name)


if __name__ == "__main__":
    gate = run()
    print(json.dumps({"COMPAT": gate["COMPAT"], "retrain": gate["high_level_retraining_needed"]}, indent=2), flush=True)
