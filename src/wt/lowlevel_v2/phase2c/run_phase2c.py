"""LL-V2.2C frozen base-zero / mask audit. No optimizer.step. No training."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from wt.lowlevel_v2.contracts import clip_unit_step, env_target, minimum_pairwise_distance
from wt.lowlevel_v2.phase1a.diagnose import predict_sinr_rate, predicted_next
from wt.lowlevel_v2.phase1b.controller import ResidualWorker, load_frozen_high, make_networks
from wt.lowlevel_v2.phase1b.mappo import ICARMAPPOLocked, tanh_log_prob_uav
from wt.lowlevel_v2.phase1b.run_phase1b import SeedCursor, attach_cursor, collect_rollout
from wt.lowlevel_v2.phase1.protocol import D_SAFE_M
from wt.lowlevel_v2.phase2b.protocol import OLD_SEED1_CKPT, PHASE1B_CKPT0
from wt.lowlevel_v2.phase2b.train import make_worker
from wt.lowlevel_v2.phase2b.trust_region import BETA_RESIDUAL, epsilon_base
from wt.lowlevel_v2.phase2c.protocol import CKPT, OUTPUT, PHASE2C, PPO, ROOT, UPDATES, load_pools, refuse_phase2_validation
from wt.round30.ppo import compute_gae

REASON_NAMES = ("SERVICE_LOCK", "IDLE_OPTION", "INVALID_TARGET", "EMPTY_QUEUE", "TARGET_ALREADY_REACHED", "BASE_EXECUTOR_ZERO_OTHER")
PRIORITY = list(REASON_NAMES)


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


def classify_uav(env, option, uav: int) -> dict:
    k_count = int(env.config.num_buoys)
    assoc = int(option.association[uav])
    tgt = env_target(assoc, k_count)
    idle = assoc < 0
    invalid = (not idle) and tgt < 0
    valid = tgt >= 0
    empty_q = bool(valid and float(env.queue[tgt]) <= 1.0e-9)
    in_range = bool(valid and env._in_range(uav, tgt))
    service = bool(valid and (not empty_q) and in_range)
    already = bool(in_range and valid and (not service))
    bits = {
        "SERVICE_LOCK": service,
        "IDLE_OPTION": idle,
        "INVALID_TARGET": invalid,
        "EMPTY_QUEUE": empty_q,
        "TARGET_ALREADY_REACHED": already,
    }
    n_true = sum(int(v) for k, v in bits.items() if k != "BASE_EXECUTOR_ZERO_OTHER")
    exclusive = "BASE_EXECUTOR_ZERO_OTHER"
    for name in ("SERVICE_LOCK", "IDLE_OPTION", "INVALID_TARGET", "EMPTY_QUEUE", "TARGET_ALREADY_REACHED"):
        if bits[name]:
            exclusive = name
            break
    if exclusive == "BASE_EXECUTOR_ZERO_OTHER":
        bits["BASE_EXECUTOR_ZERO_OTHER"] = True
    else:
        bits["BASE_EXECUTOR_ZERO_OTHER"] = False
    mask = 0
    for i, name in enumerate(REASON_NAMES):
        if bits.get(name):
            mask |= 1 << i
    return {**bits, "bitmask": mask, "exclusive": exclusive, "n_reasons": n_true, "tgt": tgt, "assoc": assoc, "in_range": in_range}


def collect_steps(worker: ResidualWorker, seeds: list[int], *, zero: bool) -> list[dict]:
    rows = []
    for seed in seeds:
        refuse_phase2_validation(int(seed))
        info = worker.reset(int(seed))
        while True:
            pos0 = np.asarray(worker.env.uav_positions, dtype=np.float64)
            info, rec = worker.step(info, deterministic=True, zero_residual=zero)
            assoc = np.asarray(worker.option.association)
            for u in range(6):
                cls = classify_uav(worker.env if False else worker.env, worker.option, u)
                # classify on post-step env; lock/base were pre-step. Recompute on pos0.
            worker.env.uav_positions = pos0
            for u in range(6):
                cls = classify_uav(worker.env, worker.option, u)
                r_raw = np.asarray(rec.r_raw[u], dtype=np.float64)
                r_used = np.asarray(rec.delta_used[u], dtype=np.float64)
                a_base = np.asarray(rec.a_base[u], dtype=np.float64)
                rows.append(
                    {
                        "seed": int(seed),
                        "t": int(worker.env.t),
                        "uav": u,
                        "cap": bool(rec.cap_active[u]),
                        "base_zero": bool(rec.base_zero_lock[u]),
                        "service_lock": bool(rec.service_lock[u]),
                        "active": float(rec.policy_active_mask[u]) > 0.5,
                        "base_norm": float(np.linalg.norm(a_base)),
                        "r_raw_norm": float(np.linalg.norm(r_raw)),
                        "r_used_norm": float(np.linalg.norm(r_used)),
                        "ratio": float(np.linalg.norm(r_raw) / (np.linalg.norm(a_base) + 1e-12)),
                        "entropy": float(np.sum(np.exp(2.0 * rec.pre_tanh[u] * 0) + 0)),  # placeholder, filled below
                        "pre_norm": float(np.linalg.norm(rec.pre_tanh[u])),
                        "exclusive": cls["exclusive"],
                        "bitmask": cls["bitmask"],
                        "n_reasons": cls["n_reasons"],
                        "idle": cls["IDLE_OPTION"],
                        "empty_q": cls["EMPTY_QUEUE"],
                        "in_range": cls["in_range"],
                        "assoc": cls["assoc"],
                        "tgt": cls["tgt"],
                        "a_base": a_base,
                        "r_raw": r_raw,
                        "pos0": pos0.copy(),
                        "assoc_all": assoc.copy(),
                    }
                )
            worker.env.uav_positions = np.asarray(worker.env.uav_positions)  # already moved
            if rec.done or worker.env.t >= worker.env.config.episode_horizon:
                break
        # positions already advanced inside step; restore not needed
    return rows


def collect_steps_fixed(worker: ResidualWorker, seeds: list[int], *, zero: bool) -> list[dict]:
    """Classify using pre-action geometry (pos0), matching lock/base-zero."""
    rows = []
    log_std = float(worker.actor.log_std.detach().mean().cpu())
    ent_const = float(0.5 * np.log(2.0 * np.pi * np.e) * 2 + 2.0 * log_std)
    for seed in seeds:
        refuse_phase2_validation(int(seed))
        info = worker.reset(int(seed))
        while True:
            pos0 = np.asarray(worker.env.uav_positions, dtype=np.float64).copy()
            queue0 = np.asarray(worker.env.queue, dtype=np.float64).copy()
            info, rec = worker.step(info, deterministic=True, zero_residual=zero)
            saved = np.asarray(worker.env.uav_positions, dtype=np.float64).copy()
            saved_q = np.asarray(worker.env.queue, dtype=np.float64).copy()
            worker.env.uav_positions = pos0
            worker.env.queue = queue0
            for u in range(6):
                cls = classify_uav(worker.env, worker.option, u)
                a_base = np.asarray(rec.a_base[u], dtype=np.float64)
                r_raw = np.asarray(rec.r_raw[u], dtype=np.float64)
                bn = float(np.linalg.norm(a_base))
                rn = float(np.linalg.norm(r_raw))
                rows.append(
                    {
                        "scenario": int(seed),
                        "uav": u,
                        "cap": bool(np.asarray(rec.cap_active)[u]),
                        "base_zero": bool(np.asarray(rec.base_zero_lock)[u]),
                        "service_lock": bool(np.asarray(rec.service_lock)[u]),
                        "active": float(rec.policy_active_mask[u]) > 0.5,
                        "base_norm": bn,
                        "r_raw_norm": rn,
                        "r_used_norm": float(np.linalg.norm(rec.delta_used[u])),
                        "ratio": rn / (bn + 1e-12),
                        "entropy": ent_const,
                        "pre_norm": float(np.linalg.norm(rec.pre_tanh[u])),
                        "exclusive": cls["exclusive"],
                        "bitmask": int(cls["bitmask"]),
                        "n_reasons": int(cls["n_reasons"]),
                        "idle": bool(cls["IDLE_OPTION"]),
                        "empty_q": bool(cls["EMPTY_QUEUE"]),
                        "in_range": bool(cls["in_range"]),
                        "a_base": a_base,
                        "r_raw": r_raw,
                        "pos0": pos0,
                        "assoc": np.asarray(worker.option.association).copy(),
                    }
                )
            worker.env.uav_positions = saved
            worker.env.queue = saved_q
            if rec.done or worker.env.t >= worker.env.config.episode_horizon:
                break
    return rows


def summarize_cap(tag: str, rows: list[dict]) -> dict:
    caps = np.array([r["cap"] for r in rows], dtype=float)
    return {
        "tag": tag,
        "n": len(rows),
        "cap_activation_count": int(np.sum(caps)),
        "cap_activation_fraction": float(np.mean(caps) if len(caps) else 0.0),
        "base_zero_fraction": float(np.mean([r["base_zero"] for r in rows])) if rows else 0.0,
        "active_fraction": float(np.mean([r["active"] for r in rows])) if rows else 0.0,
    }


def decomp_rows(tag: str, pool: str, rows: list[dict]) -> list[dict]:
    bz = [r for r in rows if r["base_zero"]]
    n = max(len(bz), 1)
    counts = defaultdict(int)
    overlap = defaultdict(int)
    for r in bz:
        counts[r["exclusive"]] += 1
        overlap[r["n_reasons"] >= 2] += 1
        for name in REASON_NAMES:
            # bitmask
            idx = REASON_NAMES.index(name)
            if r["bitmask"] & (1 << idx):
                overlap[name] += 1
    out = []
    for name in list(PRIORITY) + ["MULTI_REASON"]:
        if name == "MULTI_REASON":
            c = sum(1 for r in bz if r["n_reasons"] >= 2)
            out.append({"tag": tag, "pool": pool, "kind": "exclusive_or_multi", "reason": name, "count": c, "fraction_of_basezero": c / n})
            continue
        out.append({"tag": tag, "pool": pool, "kind": "exclusive", "reason": name, "count": counts[name], "fraction_of_basezero": counts[name] / n})
    for name in REASON_NAMES:
        idx = REASON_NAMES.index(name)
        c = sum(1 for r in bz if r["bitmask"] & (1 << idx))
        out.append({"tag": tag, "pool": pool, "kind": "overlap_bit", "reason": name, "count": c, "fraction_of_basezero": c / n})
    out.append({"tag": tag, "pool": pool, "kind": "n_basezero", "reason": "ALL", "count": len(bz), "fraction_of_basezero": 1.0 if bz else 0.0})
    return out


def overlap_stats(tag: str, rows: list[dict]) -> dict:
    n = max(len(rows), 1)
    bz = np.array([r["base_zero"] for r in rows], dtype=bool)
    sl = np.array([r["service_lock"] for r in rows], dtype=bool)
    return {
        "tag": tag,
        "n": len(rows),
        "P_base_zero": float(np.mean(bz)),
        "P_service_lock": float(np.mean(sl)),
        "P_base_zero_AND_service_lock": float(np.mean(bz & sl)),
        "P_base_zero_AND_NOT_service_lock": float(np.mean(bz & ~sl)),
        "P_service_lock_AND_NOT_base_zero": float(np.mean(sl & ~bz)),
        "P_base_zero_given_service": float(np.mean(bz[sl])) if np.any(sl) else float("nan"),
        "P_service_given_base_zero": float(np.mean(sl[bz])) if np.any(bz) else None,
    }


def actor_grad_norm(actor, loss: torch.Tensor) -> float:
    actor.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    sq = 0.0
    for p in actor.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().square().sum().cpu())
    actor.zero_grad(set_to_none=True)
    return float(np.sqrt(sq))


def replay_minibatches(worker: ResidualWorker, agent: ICARMAPPOLocked, train_seeds: list[int], rng: np.random.Generator, n_mb: int = 10) -> tuple[list[dict], dict, list[dict]]:
    mb_rows = []
    grad_rows = []
    active_adv = []
    taken = 0
    workers = [make_worker(worker.high, worker.actor, worker.critic) for _ in range(int(PPO.num_envs))]
    cursor = SeedCursor(train_seeds, rng)
    attach_cursor(workers, cursor)
    for w in workers:
        w.reset(cursor.next())
    while taken < n_mb:
        storage = collect_rollout(workers, agent, int(PPO.rollout_length))
        steps, envs = storage["rewards"].shape
        advantages, returns = compute_gae(storage["rewards"], storage["values"], storage["dones"], storage["last_values"], PPO.gamma, PPO.gae_lambda)
        count = steps * envs
        minibatch = count // PPO.num_minibatches
        mask_np = storage["policy_active_mask"].reshape((count, 6))
        pre = storage["pre_tanh"].reshape((count, 6, 2))
        old_lp = storage["log_prob_uav"].reshape((count, 6))
        self_f = storage["self"].reshape((count, 6, 5))
        task_f = storage["task"].reshape((count, 6, 8))
        comm_f = storage["communication"].reshape((count, 6, 4))
        neigh = storage["neighbor_set"].reshape((count, 6, 5, 11))
        near = storage["d_nearest"].reshape((count, 6))
        glob = storage["global_state"].reshape((count, -1))
        adv_all = (advantages.reshape(-1) - advantages.mean()) / max(float(advantages.std()), 1e-6)
        order = rng.permutation(count)
        for start in range(0, count, minibatch):
            index = order[start : start + minibatch]
            mask = mask_np[index]
            n_tot = int(mask.size)
            n_act = int(np.sum(mask > 0.5))
            frac = n_act / max(n_tot, 1)
            mb_rows.append({"n_total": n_tot, "n_active": n_act, "active_fraction": frac})
            self_t = torch.as_tensor(self_f[index], dtype=torch.float32)
            task_t = torch.as_tensor(task_f[index], dtype=torch.float32)
            comm_t = torch.as_tensor(comm_f[index], dtype=torch.float32)
            neigh_t = torch.as_tensor(neigh[index], dtype=torch.float32)
            near_t = torch.as_tensor(near[index], dtype=torch.float32)
            pre_t = torch.as_tensor(pre[index], dtype=torch.float32)
            old = torch.as_tensor(old_lp[index], dtype=torch.float32)
            mask_t = torch.as_tensor(mask, dtype=torch.float32)
            advantage = torch.as_tensor(adv_all[index], dtype=torch.float32).unsqueeze(-1)
            mu, log_std = worker.actor(self_t, task_t, comm_t, neigh_t, near_t)
            log_prob, entropy = tanh_log_prob_uav(mu, log_std, pre_t)
            ratio = (log_prob - old).exp()
            clipped = ratio.clamp(1.0 - PPO.clip_eps, 1.0 + PPO.clip_eps)
            per_uav = -torch.minimum(ratio * advantage, clipped * advantage)
            denom = mask_t.sum().clamp_min(1.0)
            policy_active = (per_uav * mask_t).sum() / denom
            g_cur = actor_grad_norm(worker.actor, policy_active)
            mu, log_std = worker.actor(self_t, task_t, comm_t, neigh_t, near_t)
            log_prob, entropy = tanh_log_prob_uav(mu, log_std, pre_t)
            ratio = (log_prob - old).exp()
            clipped = ratio.clamp(1.0 - PPO.clip_eps, 1.0 + PPO.clip_eps)
            per_uav = -torch.minimum(ratio * advantage, clipped * advantage)
            policy_totalmean = (per_uav * mask_t).mean()
            g_tot = actor_grad_norm(worker.actor, policy_totalmean)
            grad_rows.append(
                {
                    "mb": taken,
                    "n_active": n_act,
                    "active_fraction": frac,
                    "G_current": g_cur,
                    "G_active_normalized": g_cur,
                    "G_totalmean": g_tot,
                    "ratio_current_over_active": g_cur / max(g_cur, 1e-12),
                    "ratio_current_over_totalmean": g_cur / max(g_tot, 1e-12),
                }
            )
            taken += 1
            for i, env_i in enumerate(index):
                for u in range(6):
                    if mask[i, u] > 0.5:
                        active_adv.append(float(adv_all[env_i]))
            if taken >= n_mb:
                break
    adv_arr = np.asarray(active_adv, dtype=np.float64) if active_adv else np.array([np.nan])
    # active-only normalize vs all-sample
    all_a = adv_all
    if np.any(mask_np.reshape(-1) > 0.5):
        # reconstruct active env-steps that have at least one active uav
        has_act = mask_np.reshape(count, 6).max(axis=1) > 0.5
        act_only = (advantages.reshape(-1)[has_act] - advantages.reshape(-1)[has_act].mean()) / max(float(advantages.reshape(-1)[has_act].std()), 1e-6)
        all_on_act = all_a[has_act]
        adv_cmp = {
            "all_norm_on_active_mean": float(np.mean(all_on_act)),
            "all_norm_on_active_std": float(np.std(all_on_act)),
            "all_norm_on_active_p10": float(np.quantile(all_on_act, 0.1)),
            "all_norm_on_active_p50": float(np.quantile(all_on_act, 0.5)),
            "all_norm_on_active_p90": float(np.quantile(all_on_act, 0.9)),
            "active_norm_mean": float(np.mean(act_only)),
            "active_norm_std": float(np.std(act_only)),
            "active_norm_p10": float(np.quantile(act_only, 0.1)),
            "active_norm_p50": float(np.quantile(act_only, 0.5)),
            "active_norm_p90": float(np.quantile(act_only, 0.9)),
        }
    else:
        adv_cmp = {}
    n_act_list = [r["n_active"] for r in mb_rows]
    frac_list = [r["active_fraction"] for r in mb_rows]
    summary = {
        "n_minibatches": len(mb_rows),
        "zero_active_count": int(sum(1 for r in mb_rows if r["n_active"] == 0)),
        "lt5pct_count": int(sum(1 for r in mb_rows if r["active_fraction"] < 0.05)),
        "lt10pct_count": int(sum(1 for r in mb_rows if r["active_fraction"] < 0.10)),
        "median_active_samples": float(np.median(n_act_list)),
        "min_active_samples": float(np.min(n_act_list)),
        "max_active_samples": float(np.max(n_act_list)),
        "mean_active_fraction": float(np.mean(frac_list)),
        **adv_cmp,
    }
    return mb_rows, summary, grad_rows


def counterfactual(worker: ResidualWorker, seeds: list[int], limit: int = 80) -> list[dict]:
    rows = []
    n = 0
    for seed in seeds:
        if n >= limit:
            break
        refuse_phase2_validation(int(seed))
        info = worker.reset(int(seed))
        while True:
            pos0 = np.asarray(worker.env.uav_positions, dtype=np.float64).copy()
            info, rec = worker.step(info, deterministic=True, zero_residual=False)
            saved = np.asarray(worker.env.uav_positions, dtype=np.float64).copy()
            worker.env.uav_positions = pos0
            assoc = np.asarray(worker.option.association)
            for u in range(6):
                if n >= limit:
                    break
                bz = bool(rec.base_zero_lock[u])
                sl = bool(rec.service_lock[u])
                idle = int(assoc[u]) < 0
                if not (bz and (not sl) and (not idle)):
                    continue
                tgt = env_target(int(assoc[u]), int(worker.env.config.num_buoys))
                a0 = rec.a_base.copy()
                a1 = rec.a_base.copy()
                a1[u] = clip_unit_step((rec.a_base[u] + rec.r_raw[u]).reshape(1, 2))[0]
                nxt0 = predicted_next(worker.env, a0)
                nxt1 = predicted_next(worker.env, a1)
                sep0 = minimum_pairwise_distance(nxt0)
                sep1 = minimum_pairwise_distance(nxt1)
                d0 = d1 = float("nan")
                if tgt >= 0:
                    d0 = float(np.linalg.norm(nxt0[u] - worker.env.buoy_positions[tgt]))
                    d1 = float(np.linalg.norm(nxt1[u] - worker.env.buoy_positions[tgt]))
                s0, r0 = predict_sinr_rate(worker.env, a0)
                s1, r1 = predict_sinr_rate(worker.env, a1)
                feas0 = bool(tgt >= 0 and np.linalg.norm(nxt0[u] - worker.env.buoy_positions[tgt]) <= float(worker.env.config.service_radius_m) + 1e-9)
                feas1 = bool(tgt >= 0 and np.linalg.norm(nxt1[u] - worker.env.buoy_positions[tgt]) <= float(worker.env.config.service_radius_m) + 1e-9)
                rows.append(
                    {
                        "scenario": int(seed),
                        "uav": u,
                        "r_raw_norm": float(np.linalg.norm(rec.r_raw[u])),
                        "delta_dist_to_target": float(d1 - d0) if np.isfinite(d0) else float("nan"),
                        "dist_improved": bool(np.isfinite(d0) and d1 < d0 - 1e-6),
                        "delta_min_sep": float(sep1 - sep0),
                        "sep_improved": bool(sep1 > sep0 + 1e-6),
                        "delta_sinr": float(s1[u] - s0[u]),
                        "delta_rate": float(r1[u] - r0[u]),
                        "feas0": feas0,
                        "feas1": feas1,
                        "feas_improved": bool(feas1 and not feas0),
                    }
                )
                n += 1
            worker.env.uav_positions = saved
            if rec.done or worker.env.t >= worker.env.config.episode_horizon or n >= limit:
                break
    return rows


def main() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train_seeds, design_seeds = load_pools()
    protocol = {
        "name": "LL-V2.2C",
        "restored": "LL-V2.2B / P2B-B",
        "beta_residual": BETA_RESIDUAL,
        "epsilon_base_fraction": 0.05,
        "epsilon_base": float(epsilon_base()),
        "a_max": 1.0,
        "trained": False,
        "optimizer_step": False,
        "phase2_validation": "blocked",
        "relation_enhancement": False,
        "training_cap_activation_resolved": False,
        "training_cap_note": "Phase-2B training history has no cap_activation field; no saved training rollouts.",
    }
    _dump(OUTPUT / "LLV2_PHASE2C_PROTOCOL.json", protocol)

    hist = list(csv.DictReader((PHASE2C.parent / "phase2b" / "LLV2_PHASE2B_TRAINING_HISTORY.csv").open(encoding="utf-8")))
    curve = list(csv.DictReader((PHASE2C.parent / "phase2b" / "LLV2_PHASE2B_EVAL_CURVE.csv").open(encoding="utf-8")))

    cap_rows = []
    for rec in curve:
        cap_rows.append(
            {
                "source": "design_eval_curve_phase2b",
                "model_seed": rec["model_seed"],
                "update": rec["update"],
                "cap_activation_fraction": rec["cap_activation_rate"],
                "cap_activation_count": "unavailable",
                "base_zero_freeze_rate": rec["base_zero_freeze_rate"],
            }
        )
    cap_rows.append({"source": "training_log", "model_seed": "all", "update": "1-100", "cap_activation_fraction": "unlogged", "cap_activation_count": "unlogged", "note": "training_cap_activation_resolved=false"})

    # frozen checkpoints on design
    design_by_tag = {}
    for seed in (0, 1):
        for upd in UPDATES:
            path = CKPT / f"ckpt_seed{seed}_update{upd}.pt"
            w = load_ckpt(path, trust_region=True)
            rows = collect_steps_fixed(w, design_seeds, zero=False)
            design_by_tag[f"new{seed}_u{upd}"] = rows
            s = summarize_cap(f"new{seed}_u{upd}_design", rows)
            cap_rows.append({"source": "frozen_design_replay", "model_seed": seed, "update": upd, **s})

    w0 = load_ckpt(CKPT / "ckpt_seed0_update100.pt", trust_region=True)
    w1 = load_ckpt(CKPT / "ckpt_seed1_update100.pt", trust_region=True)
    high, _ = load_frozen_high()
    actor, critic = make_networks()
    l0w = ResidualWorker(high, actor, critic, service_lock=True, trust_region=False)
    l0_design = collect_steps_fixed(l0w, design_seeds, zero=True)
    # L0 has trust_region False so base_zero_lock is always false; still record service_lock
    train16 = list(train_seeds)[:16]
    train0 = collect_steps_fixed(w0, train16, zero=False)
    train1 = collect_steps_fixed(w1, train16, zero=False)
    design0 = design_by_tag["new0_u100"]
    design1 = design_by_tag["new1_u100"]

    cap_rows.append({**summarize_cap("new0_u100_train16", train0), "source": "frozen_train16_replay", "model_seed": 0, "update": 100})
    cap_rows.append({**summarize_cap("new1_u100_train16", train1), "source": "frozen_train16_replay", "model_seed": 1, "update": 100})
    _csv(OUTPUT / "LLV2_PHASE2C_CAP_ACTIVATION.csv", cap_rows)

    decomp = []
    for tag, pool, rows in (
        ("new0_u100", "design", design0),
        ("new1_u100", "design", design1),
        ("new0_u100", "train16", train0),
        ("new1_u100", "train16", train1),
    ):
        decomp.extend(decomp_rows(tag, pool, rows))
    _csv(OUTPUT / "LLV2_PHASE2C_BASE_ZERO_DECOMPOSITION.csv", decomp)

    overlap = {
        "new0_design": overlap_stats("new0_design", design0),
        "new1_design": overlap_stats("new1_design", design1),
        "new0_train16": overlap_stats("new0_train16", train0),
        "new1_train16": overlap_stats("new1_train16", train1),
        "L0_design_service_lock_only": overlap_stats("L0_design", l0_design),
    }
    _dump(OUTPUT / "LLV2_PHASE2C_LOCK_OVERLAP.json", overlap)

    def active_stats(tag, rows, source):
        act = np.array([r["active"] for r in rows], dtype=float)
        return {
            "tag": tag,
            "source": source,
            "n_agent_steps": len(rows),
            "active_count": int(np.sum(act)),
            "active_fraction": float(np.mean(act)),
            "mean": float(np.mean(act)),
            "min": float(np.min(act)),
            "max": float(np.max(act)),
            "p10": float(np.quantile(act, 0.1)),
            "p50": float(np.quantile(act, 0.5)),
            "p90": float(np.quantile(act, 0.9)),
        }

    act_rows = [active_stats("new0_u100", design0, "design_frozen"), active_stats("new1_u100", design1, "design_frozen"), active_stats("new0_u100", train0, "train16_frozen"), active_stats("new1_u100", train1, "train16_frozen")]
    # training logged policy_active_fraction
    by_seed = {0: [], 1: []}
    for rec in hist:
        by_seed[int(rec["model_seed"])].append(float(rec["policy_active_fraction"]))
    for seed, xs in by_seed.items():
        arr = np.asarray(xs)
        act_rows.append(
            {
                "tag": f"seed{seed}_training_log",
                "source": "phase2b_training_history_policy_active_fraction",
                "n_updates": len(xs),
                "active_fraction": float(np.mean(arr)),
                "mean": float(np.mean(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "p10": float(np.quantile(arr, 0.1)),
                "p50": float(np.quantile(arr, 0.5)),
                "p90": float(np.quantile(arr, 0.9)),
            }
        )
    _csv(OUTPUT / "LLV2_PHASE2C_ACTIVE_SAMPLE_STATS.csv", act_rows)

    mask_audit = """# LL-V2.2C MAPPO mask reduction audit

Source: `src/wt/lowlevel_v2/phase1b/mappo.py` `ICARMAPPOLocked.update`.

## Policy surrogate

```
per_uav = -min(ratio * A, clip(ratio) * A)
denom = mask.sum().clamp_min(1)
policy_loss = (per_uav * mask).sum() / denom
```

This is **form B**: `sum(mask * loss_i) / max(sum(mask), 1)`.

`actor_mask_denominator = ACTIVE_SAMPLES` (UAV-steps with `policy_active_mask=1`).

Not form A `mean(mask * loss_i)` which would divide by total UAV-steps.

## Entropy

```
entropy_term = (entropy * mask).sum() / denom
```

Same **ACTIVE_SAMPLES** denominator.

## Value / GAE

`value_loss = 0.5 * (target - value).mean()` over **all env-steps**, unmasked.

GAE advantages are env-level `(T, n_env)`, then:

```
adv = (advantages.reshape(-1) - advantages.mean()) / advantages.std()
```

Advantage **normalization includes all env-steps**, including those whose UAVs are mostly locked. Then `advantage` is broadcast to 6 UAVs.

So:

- Actor loss denom: ACTIVE UAV-samples
- Advantage scale: ALL env-samples
"""
    _dump(OUTPUT / "LLV2_PHASE2C_MASK_REDUCTION_AUDIT.md", mask_audit)
    _dump(OUTPUT / "LLV2_PHASE2C_ADVANTAGE_NORMALIZATION.md", """# Advantage normalization

Implemented as **A**: normalize GAE over all env-steps in the rollout, then apply UAV `policy_active_mask` only inside the policy/entropy terms.

Not **B** (active-only advantage normalize).

See `LLV2_PHASE2C_MASK_REDUCTION_AUDIT.md` and minibatch replay stats.
""")

    mb_all = []
    grad_all = []
    adv_notes = {}
    for seed, w in ((0, w0), (1, w1)):
        agent = ICARMAPPOLocked(w.actor, w.critic, PPO)
        mb_rows, summary, grad_rows = replay_minibatches(w, agent, train_seeds, np.random.default_rng(22100 + seed), n_mb=10)
        adv_notes[f"seed{seed}"] = summary
        for r in mb_rows:
            mb_all.append({"model_seed": seed, **r})
        for r in grad_rows:
            grad_all.append({"model_seed": seed, **r})
    _csv(OUTPUT / "LLV2_PHASE2C_MINIBATCH_ACTIVE_STATS.csv", mb_all)
    _csv(OUTPUT / "LLV2_PHASE2C_GRADIENT_SCALING.csv", grad_all)
    _dump(OUTPUT / "LLV2_PHASE2C_ADVANTAGE_REPLAY.json", adv_notes)

    # residual ratio on active
    ratio_rows = []
    raw_bz_rows = []
    for tag, rows in (("new0_design", design0), ("new1_design", design1), ("new0_train16", train0), ("new1_train16", train1)):
        act = [r for r in rows if r["active"]]
        ratios = np.array([r["ratio"] for r in act], dtype=float) if act else np.array([np.nan])
        ratio_rows.append(
            {
                "tag": tag,
                "n_active": len(act),
                "mean": float(np.nanmean(ratios)),
                "median": float(np.nanmedian(ratios)),
                "p90": float(np.nanquantile(ratios, 0.9)) if act else float("nan"),
                "p95": float(np.nanquantile(ratios, 0.95)) if act else float("nan"),
                "p99": float(np.nanquantile(ratios, 0.99)) if act else float("nan"),
                "max": float(np.nanmax(ratios)),
                "frac_gt_beta": float(np.mean(ratios > BETA_RESIDUAL)) if act else float("nan"),
                "beta": BETA_RESIDUAL,
            }
        )
        for reason in PRIORITY + ["BASE_EXECUTOR_ZERO_OTHER"]:
            sub = [r for r in rows if r["base_zero"] and r["exclusive"] == reason]
            if not sub:
                continue
            raws = np.array([r["r_raw_norm"] for r in sub])
            raw_bz_rows.append({"tag": tag, "reason": reason, "n": len(sub), "mean_raw_norm": float(np.mean(raws)), "median_raw_norm": float(np.median(raws)), "p90_raw_norm": float(np.quantile(raws, 0.9)), "mean_entropy": float(np.mean([r["entropy"] for r in sub]))})
    _csv(OUTPUT / "LLV2_PHASE2C_RESIDUAL_BASE_RATIO.csv", ratio_rows)
    _csv(OUTPUT / "LLV2_PHASE2C_BASEZERO_RAW_RESIDUAL.csv", raw_bz_rows)

    cf0 = counterfactual(w0, design_seeds, limit=80)
    cf1 = counterfactual(w1, design_seeds, limit=80)
    cf_rows = [{"model_seed": 0, **r} for r in cf0] + [{"model_seed": 1, **r} for r in cf1]
    if not cf_rows:
        pop0 = sum(1 for r in design0 if r["base_zero"] and (not r["service_lock"]) and (not r["idle"]))
        pop1 = sum(1 for r in design1 if r["base_zero"] and (not r["service_lock"]) and (not r["idle"]))
        cf_rows = [
            {
                "model_seed": 0,
                "n_population": pop0,
                "note": "base_zero AND NOT rec.service_lock AND NOT idle is empty; leftover freeze is IDLE_OPTION",
            },
            {
                "model_seed": 1,
                "n_population": pop1,
                "note": "base_zero AND NOT rec.service_lock AND NOT idle is empty; leftover freeze is IDLE_OPTION",
            },
        ]
    _csv(OUTPUT / "LLV2_PHASE2C_BASEZERO_COUNTERFACTUAL.csv", cf_rows)

    def cf_summary(cfs):
        if not cfs:
            return {"n": 0}
        return {
            "n": len(cfs),
            "frac_dist_improved": float(np.mean([r["dist_improved"] for r in cfs])),
            "mean_delta_dist": float(np.nanmean([r["delta_dist_to_target"] for r in cfs])),
            "frac_sep_improved": float(np.mean([r["sep_improved"] for r in cfs])),
            "mean_delta_sep": float(np.mean([r["delta_min_sep"] for r in cfs])),
            "mean_delta_sinr": float(np.mean([r["delta_sinr"] for r in cfs])),
            "frac_feas_improved": float(np.mean([r["feas_improved"] for r in cfs])),
        }

    # training update intensity
    inten = []
    for rec in hist:
        inten.append(
            {
                "model_seed": rec["model_seed"],
                "update": rec["update"],
                "policy_active_fraction": rec["policy_active_fraction"],
                "gradient_norm_preclip": rec["gradient_norm_preclip"],
                "approx_kl": rec["approx_kl"],
                "clip_fraction": rec["clip_fraction"],
                "effective_actor_update_proxy": float(rec["policy_active_fraction"]) * float(rec["gradient_norm_preclip"]),
            }
        )
    _csv(OUTPUT / "LLV2_PHASE2C_UPDATE_INTENSITY.csv", inten)

    # mechanism gates from numbers
    cap_frozen = [r for r in cap_rows if r.get("source") == "frozen_design_replay"]
    cap_max = max(float(r["cap_activation_fraction"]) for r in cap_frozen)
    cap_train = max(float(summarize_cap("x", train0)["cap_activation_fraction"]), float(summarize_cap("y", train1)["cap_activation_fraction"]))
    cap_never = cap_max < 1e-6 and cap_train < 1e-6
    p99_0 = next(r for r in ratio_rows if r["tag"] == "new0_design")["p99"]
    p99_1 = next(r for r in ratio_rows if r["tag"] == "new1_design")["p99"]
    cap_redundant = (p99_0 < 0.25) and (p99_1 < 0.25)

    d0_ex = {r["reason"]: r["fraction_of_basezero"] for r in decomp if r["tag"] == "new0_u100" and r["pool"] == "design" and r["kind"] == "exclusive"}
    d1_ex = {r["reason"]: r["fraction_of_basezero"] for r in decomp if r["tag"] == "new1_u100" and r["pool"] == "design" and r["kind"] == "exclusive"}

    a0 = by_seed[0]
    a1 = by_seed[1]
    mean_a0, mean_a1 = float(np.mean(a0)), float(np.mean(a1))
    mb0 = [r for r in mb_all if int(r["model_seed"]) == 0]
    zero_mb = sum(1 for r in mb_all if r["n_active"] == 0)
    lt5 = sum(1 for r in mb_all if r["active_fraction"] < 0.05)

    g_ratio = float(np.mean([r["ratio_current_over_active"] for r in grad_all])) if grad_all else 1.0
    mask_gate = "C"
    if abs(g_ratio - 1.0) > 0.15:
        mask_gate = "A"
    elif mean_a0 < 0.15 and mean_a1 < 0.15:
        mask_gate = "B"

    cf_sum = {0: cf_summary(cf0), 1: cf_summary(cf1)}
    pop_nonidle = float(
        np.mean(
            [
                float(np.mean([r["base_zero"] and (not r["service_lock"]) and (not r["idle"]) for r in design0])),
                float(np.mean([r["base_zero"] and (not r["service_lock"]) and (not r["idle"]) for r in design1])),
            ]
        )
    )
    dist_imp = 0.0
    if cf_sum[0].get("n", 0) + cf_sum[1].get("n", 0):
        dist_imp = float(np.mean([cf_sum[s]["frac_dist_improved"] for s in (0, 1) if cf_sum[s].get("n", 0)]))
    bz_gate = "A"
    if pop_nonidle < 1e-6:
        bz_gate = "A"
    elif dist_imp > 0.4:
        bz_gate = "B"
    elif (cf_sum[0].get("n", 0) + cf_sum[1].get("n", 0)) < 10:
        bz_gate = "C"
    cf_payload = {
        "seed0": cf_sum[0],
        "seed1": cf_sum[1],
        "population_note": "non-service non-idle base-zero is empty on design u100; extra freeze is idle" if pop_nonidle < 1e-6 else "",
        "pop_nonidle_fraction": pop_nonidle,
    }

    if cap_never and cap_redundant:
        mech = "A"
        mech_text = "BaseZeroResidualFreeze is the operating constraint; residual norm cap did not bind."
    elif cap_never:
        mech = "A"
        mech_text = "Cap never fired on frozen design/train rollouts."
    else:
        mech = "C"
        mech_text = "mixed"

    next_change = "NO_METHOD_CHANGE"
    if mask_gate == "A":
        next_change = "normalize Actor surrogate and entropy by ACTIVE_SAMPLES only (already true; do not change other modules)"
    elif mask_gate == "B":
        next_change = "active-sample-aware minibatching (single change); do not retune beta"
    elif bz_gate == "B":
        next_change = "redesign base-zero condition only; do not retune beta"
    elif mask_gate == "C" and bz_gate == "A":
        next_change = "NO_METHOD_CHANGE; pre-register a larger official seed count for statistics"

    gate = {
        "MASK": mask_gate,
        "BASEZERO": bz_gate,
        "phase2b_mechanism": mech,
        "phase2b_mechanism_text": mech_text,
        "cap_attribution": "A" if cap_never else "B",
        "training_cap_activation_resolved": False,
        "frozen_cap_max_design": cap_max,
        "frozen_cap_max_train16": cap_train,
        "p99_ratio_design_seed0": p99_0,
        "p99_ratio_design_seed1": p99_1,
        "cap_redundant_vs_beta": cap_redundant,
        "mean_train_active_seed0": mean_a0,
        "mean_train_active_seed1": mean_a1,
        "zero_active_minibatches": zero_mb,
        "lt5pct_minibatches": lt5,
        "grad_ratio_current_over_active_mean": g_ratio,
        "exclusive_basezero_new0_design": d0_ex,
        "exclusive_basezero_new1_design": d1_ex,
        "counterfactual": cf_payload,
        "minimal_method_change": next_change,
        "trained": False,
        "relation_enhancement": False,
        "REL": 1,
        "P2B": "B",
    }
    _dump(OUTPUT / "LLV2_PHASE2C_MECHANISM_GATE.json", gate)

    for src in OUTPUT.glob("LLV2_PHASE2C_*"):
        (PHASE2C / src.name).write_bytes(src.read_bytes())
        (ROOT / src.name).write_bytes(src.read_bytes())
    return {
        "cap_never": cap_never,
        "mask": mask_gate,
        "basezero": bz_gate,
        "next": next_change,
        "active0": mean_a0,
        "active1": mean_a1,
        "overlap0": overlap["new0_design"],
        "overlap1": overlap["new1_design"],
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
