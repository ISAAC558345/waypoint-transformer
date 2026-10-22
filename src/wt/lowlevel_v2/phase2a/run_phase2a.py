"""LL-V2.2A frozen seed-1 causal audit. No training. No method changes."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from wt.lowlevel_v2.observation import communication_features
from wt.lowlevel_v2.phase1a.diagnose import desired_constraint_flags, intervention_label
from wt.lowlevel_v2.phase1b.controller import ResidualWorker, load_frozen_high, make_networks
from wt.lowlevel_v2.phase1b.protocol import ALPHA_RESIDUAL, D_SAFE_M, PPO, load_pools, refuse_untouched
from wt.lowlevel_v2.phase2.protocol import PHASE1B_CKPT0, ROOT
from wt.lowlevel_v2.service_safe_envelope import envelope_active

PHASE2 = Path(__file__).resolve().parents[1] / "phase2"
PHASE2A = Path(__file__).resolve().parent
OUT = PHASE2A / "outputs"
SEEDS = (0, 1, 2, 3, 4)
SAT = 0.98


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
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def ckpt_final(seed: int) -> Path:
    if int(seed) == 0:
        return PHASE1B_CKPT0
    return PHASE2 / "outputs" / f"ckpt_seed{seed}_update100.pt"


def ckpt_update(seed: int, update: int) -> Path:
    if int(seed) == 0:
        return PHASE2.parent / "phase1b" / "outputs" / f"ckpt_update{update}.pt"
    return PHASE2 / "outputs" / f"ckpt_seed{seed}_update{update}.pt"


def load_worker(path: Path) -> ResidualWorker:
    high, _ = load_frozen_high()
    actor, critic = make_networks()
    payload = torch.load(path, map_location="cpu")
    actor.load_state_dict(payload["actor"])
    critic.load_state_dict(payload["critic"])
    actor.eval()
    critic.eval()
    return ResidualWorker(high, actor, critic, service_lock=True)


def param_norm(module) -> float:
    return float(torch.sqrt(sum(p.detach().float().square().sum() for p in module.parameters())))


def rollout_design(worker: ResidualWorker, seeds: list[int], *, store_traj: set[int] | None, zero: bool) -> dict:
    steps = []
    episodes = []
    traj = defaultdict(list)
    gamma = float(PPO.gamma)
    for seed in seeds:
        refuse_untouched(int(seed))
        info = worker.reset(int(seed))
        payload0 = float(worker.env.payload_bits)
        assigns = []
        switches = 0
        idle_n = 0
        n_opt = 0
        arrive = []
        never = 0
        option_open: dict[int, dict] = {}
        inrange_blocks = []
        inrange_sinr = []
        inrange_rate = []
        time_in = 0
        hid_acc = []
        while True:
            pos0 = worker.env.uav_positions.copy()
            assoc0 = np.asarray(worker.option.association) if worker.option is not None else None
            pending_before = int(worker.pending)
            info, rec = worker.step(info, deterministic=True, zero_residual=zero)
            assoc = np.asarray(worker.option.association)
            if pending_before <= 0:
                n_opt += 1
                assigns.append(assoc.copy())
                if len(assigns) >= 2 and not np.array_equal(assigns[-1], assigns[-2]):
                    switches += 1
            idle_n += int(np.sum(assoc < 0))
            saved = worker.env.uav_positions.copy()
            worker.env.uav_positions = pos0
            flags = desired_constraint_flags(worker.env, worker.option, rec.a_des)
            comm = communication_features(worker.env)
            envl = envelope_active(worker.env, worker.option)
            worker.env.uav_positions = saved
            intervened = float(rec.intervention_norm) > 1e-6
            label = intervention_label(flags, intervened)
            d_near = np.asarray(rec.obs["d_nearest"], dtype=np.float64)
            interf = np.asarray(comm["total_interference"], dtype=np.float64)
            approaching = []
            in_svc = rec.service_lock.copy()
            for u in range(6):
                tgt_valid = int(assoc[u]) >= 0
                approaching.append(bool(tgt_valid and (not in_svc[u])))
                if tgt_valid and u not in option_open and not in_svc[u]:
                    option_open[u] = {"t0": worker.env.t, "entered": False}
                if u in option_open and in_svc[u] and not option_open[u]["entered"]:
                    option_open[u]["entered"] = True
                    arrive.append(worker.env.t - option_open[u]["t0"])
                if pending_before <= 0 and u in option_open and not option_open[u]["entered"]:
                    never += 1
                    option_open.pop(u, None)
                if in_svc[u]:
                    time_in += 1
                    inrange_sinr.append(float(comm["sinr"][u]))
                    inrange_rate.append(float(comm["rate"][u]))
            r = rec.delta_used
            a_base = rec.a_base
            fwd, lat, rev, zbase = [], [], [], []
            for u in range(6):
                bn = float(np.linalg.norm(a_base[u]))
                if bn < 1e-6:
                    zbase.append(float(np.linalg.norm(r[u])))
                    continue
                unit = a_base[u] / bn
                f = float(np.dot(r[u], unit))
                perp = r[u] - f * unit
                fwd.append(f)
                lat.append(float(np.linalg.norm(perp)))
                rev.append(float(f < -1e-6))
            with torch.no_grad():
                obs = rec.obs
                packed = worker.actor._pack(
                    torch.as_tensor(obs["self"], dtype=torch.float32),
                    torch.as_tensor(obs["task"], dtype=torch.float32),
                    torch.as_tensor(obs["communication"], dtype=torch.float32),
                    torch.as_tensor(obs["neighbor_set"], dtype=torch.float32),
                    torch.as_tensor(obs["d_nearest"], dtype=torch.float32),
                )
                hid = worker.actor.body(packed).cpu().numpy()
            hid_acc.append(hid)
            steps.append(
                {
                    "seed": int(seed),
                    "res_norm": np.linalg.norm(r, axis=-1),
                    "delta_tanh": rec.delta_tanh,
                    "pre": rec.pre_tanh,
                    "obs": rec.obs,
                    "a_base": a_base,
                    "a_des": rec.a_des,
                    "a_exec": rec.a_exec,
                    "delta_used": r,
                    "sat": float(np.mean(np.max(np.abs(rec.delta_tanh), axis=-1) >= SAT)),
                    "actor_mean": float(np.mean(np.linalg.norm(rec.pre_tanh, axis=-1))),
                    "des_minus": np.linalg.norm(rec.a_des - rec.a_base, axis=-1),
                    "exec_minus": np.linalg.norm(rec.a_exec - rec.a_base, axis=-1),
                    "lock": rec.service_lock,
                    "mask": rec.policy_active_mask,
                    "viol": rec.lock_violation,
                    "intervened": intervened,
                    "label": label,
                    "d_near": d_near,
                    "interf": interf,
                    "in_svc": in_svc,
                    "approaching": np.asarray(approaching, dtype=bool),
                    "fwd": fwd,
                    "lat": lat,
                    "rev": rev,
                    "zbase": zbase,
                    "value": rec.value,
                    "reward": rec.reward,
                    "done": rec.done,
                    "hid": hid,
                    "assoc": assoc.copy(),
                    "positions": pos0,
                }
            )
            if store_traj is not None and int(seed) in store_traj:
                traj[int(seed)].append(
                    {
                        "positions": pos0,
                        "assoc": assoc.copy(),
                        "intervened": intervened,
                        "a_des": rec.a_des.copy(),
                        "d_near": d_near.copy(),
                    }
                )
            if rec.done or worker.env.t >= 48:
                break
        by_seed_steps = [s for s in steps if s["seed"] == int(seed)]
        rewards = np.array([s["reward"] for s in by_seed_steps], dtype=np.float64)
        values = np.array([s["value"] for s in by_seed_steps], dtype=np.float64)
        dones = np.array([s["done"] for s in by_seed_steps], dtype=np.float64)
        ret = np.zeros_like(rewards)
        running = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            running = rewards[t] + gamma * (1.0 - dones[t]) * running
            ret[t] = running
        episodes.append(
            {
                "seed": int(seed),
                "payload": float(worker.env.payload_bits - payload0),
                "completed_blocks": float((worker.env.payload_bits - payload0) / 1e6),
                "unfinished": float(np.mean(worker.env.progress)),
                "n_opt": n_opt,
                "assignment_changes": switches,
                "idle_fraction": idle_n / max(6 * max(len(by_seed_steps), 1), 1),
                "arrive_steps": arrive,
                "never_enter": never,
                "time_in_range": time_in,
                "inrange_sinr": float(np.mean(inrange_sinr) if inrange_sinr else np.nan),
                "inrange_rate": float(np.mean(inrange_rate) if inrange_rate else np.nan),
                "values": values,
                "returns": ret,
            }
        )
        _ = hid_acc
    return {"steps": steps, "episodes": episodes, "traj": traj}


def summarize_residual(tag: str, bundle: dict, log_std: float) -> dict:
    res = np.concatenate([s["res_norm"] for s in bundle["steps"]])
    return {
        "checkpoint": tag,
        "mean_residual_norm": float(np.mean(res)),
        "median_residual_norm": float(np.median(res)),
        "p95_residual_norm": float(np.quantile(res, 0.95)),
        "max_residual_norm": float(np.max(res)),
        "tanh_saturation_fraction": float(np.mean([s["sat"] for s in bundle["steps"]])),
        "actor_mean_norm": float(np.mean([s["actor_mean"] for s in bundle["steps"]])),
        "actor_log_std": float(log_std),
        "mean_desired_minus_base": float(np.mean(np.concatenate([s["des_minus"] for s in bundle["steps"]]))),
        "mean_executed_minus_base": float(np.mean(np.concatenate([s["exec_minus"] for s in bundle["steps"]]))),
    }


def _traj_frames(bundle: dict, seed: int) -> list:
    frames = list(bundle.get("traj", {}).get(int(seed), []))
    if frames:
        return frames
    return [{"positions": s["positions"]} for s in bundle.get("steps", []) if int(s["seed"]) == int(seed)]


def plot_traj(map_bundles: dict, buoys: np.ndarray, out: Path, ids: list[int] | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    if ids is None:
        found = set()
        for bundle in map_bundles.values():
            found.update(int(k) for k in bundle.get("traj", {}).keys())
        ids = sorted(found)[:8]
    ids = [int(x) for x in ids]
    if not ids:
        raise RuntimeError("plot_traj received empty scenario id list")
    for seed in ids:
        fig, axes = plt.subplots(1, len(map_bundles), figsize=(4 * len(map_bundles), 4), sharex=True, sharey=True)
        if len(map_bundles) == 1:
            axes = [axes]
        for ax, (name, bundle) in zip(axes, map_bundles.items()):
            frames = _traj_frames(bundle, seed)
            if not frames:
                ax.set_title(f"{name} {seed} empty")
                continue
            for uav in range(6):
                xy = np.array([fr["positions"][uav] for fr in frames])
                ax.plot(xy[:, 0], xy[:, 1], color=colors[uav], lw=1.1)
            ax.scatter(buoys[:, 0], buoys[:, 1], marker="x", c="red", s=14)
            for buoy in np.asarray(buoys, dtype=np.float64):
                ax.add_patch(plt.Circle((float(buoy[0]), float(buoy[1])), 16.0, fill=False, lw=0.35, color="#888"))
            ax.set_title(f"{name} {seed}")
            ax.set_aspect("equal")
        fig.tight_layout()
        dest = out / f"seed1_vs_seed2_traj_{seed}.png"
        fig.savefig(dest, dpi=130)
        plt.close(fig)
        if not dest.exists():
            raise RuntimeError(f"failed to write {dest}")


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    train_seeds, design_seeds = load_pools()
    design_seeds = list(design_seeds)
    traj_ids = set(sorted(design_seeds)[:8])
    # learning curves
    curve_eval = list(csv.DictReader((PHASE2 / "outputs" / "LLV2_PHASE2_LEARNING_CURVES.csv").open(encoding="utf-8")))
    hist1 = {s: [] for s in (1, 2, 3, 4)}
    for rec in csv.DictReader((PHASE2 / "outputs" / "LLV2_PHASE2_TRAINING_HISTORY.csv").open(encoding="utf-8")):
        hist1[int(rec["model_seed"])].append(rec)
    hist0 = list(csv.DictReader((PHASE2.parent / "phase1b" / "outputs" / "LLV2_PHASE1B_TRAINING_HISTORY.csv").open(encoding="utf-8")))
    learn_rows = []
    for rec in curve_eval:
        u = int(float(rec["update"]))
        s = int(rec["model_seed"])
        row = {k: rec.get(k) for k in rec}
        src_hist = hist0 if s == 0 else hist1.get(s, [])
        match = None
        for h in src_hist:
            if int(float(h["update"])) == u:
                match = h
                break
        keys = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "explained_variance", "advantage_mean", "advantage_std")
        for key in keys:
            row[key] = match.get(key) if match else "missing"
        row["gradient_norm"] = (match.get("gradient_norm_preclip") if match else "missing")
        row["return_mean"] = "missing"
        row["return_std"] = "missing"
        learn_rows.append(row)
    # design rollouts
    high, _ = load_frozen_high()
    actor0, critic0 = make_networks()
    l0w = ResidualWorker(high, actor0, critic0, service_lock=True)
    env0 = l0w.env
    env0.reset(seed=int(design_seeds[0]))
    buoys = env0.buoy_positions.copy()
    l0 = rollout_design(l0w, design_seeds, store_traj=traj_ids, zero=True)
    bundles = {}
    residual_rows = []
    nav_rows = []
    svc_rows = []
    hl_rows = []
    proj_ctx = []
    feat_rows = []
    dir_rows = []
    crit_rows = []
    repr_rows = []
    init_rows = []
    lock_rows = []
    rng = np.random.default_rng(22101)
    for seed in SEEDS:
        w = load_worker(ckpt_final(seed))
        bundle = rollout_design(w, design_seeds, store_traj=traj_ids if seed in {1, 2} else set(), zero=False)
        bundles[seed] = bundle
        log_std = float(w.actor.log_std.detach().cpu().mean())
        residual_rows.append(summarize_residual(f"seed{seed}", bundle, log_std))
        viol = float(np.sum([s["viol"] for s in bundle["steps"]]))
        locked = np.concatenate([s["lock"] for s in bundle["steps"]])
        used = np.concatenate([np.linalg.norm(s["delta_used"], axis=-1) for s in bundle["steps"]])
        lock_mask = locked
        lock_rows.append(
            {
                "model_seed": seed,
                "service_lock_violation": viol,
                "locked_residual_nonzero_fraction": float(np.mean(used[lock_mask] > 1e-8)) if np.any(lock_mask) else 0.0,
                "envelope_intervention_rate": float(np.mean([s["intervened"] and s["label"] == "SERVICE_SAFE_ENVELOPE" for s in bundle["steps"]])),
                "policy_active_mean": float(np.mean([np.mean(s["mask"]) for s in bundle["steps"]])),
            }
        )
        d_all = np.concatenate([s["d_near"] for s in bundle["steps"]])
        i_all = np.concatenate([s["interf"] for s in bundle["steps"]])
        q1, q2 = np.quantile(d_all, [1 / 3, 2 / 3])
        iq1, iq2 = np.quantile(i_all, [1 / 3, 2 / 3])
        ctx = defaultdict(int)
        n_int = 0
        for s in bundle["steps"]:
            if not s["intervened"]:
                continue
            n_int += 1
            ctx[s["label"]] += 1
            near = bool(np.any(s["d_near"] <= q1))
            far = bool(np.all(s["d_near"] > q2))
            hi = bool(np.any(s["interf"] > iq2))
            lo = bool(np.all(s["interf"] <= iq1))
            svc = bool(np.any(s["in_svc"]))
            ap = bool(np.any(s["approaching"]))
            if near:
                ctx["near_neighbor"] += 1
            if far:
                ctx["far_neighbor"] += 1
            if hi:
                ctx["high_interference"] += 1
            if lo:
                ctx["low_interference"] += 1
            if ap:
                ctx["target_approach"] += 1
            if svc:
                ctx["service_phase"] += 1
        proj_ctx.append({"model_seed": seed, "n_intervened": n_int, **{k: ctx[k] / max(n_int, 1) for k in ("PAIRWISE_COLLISION", "SERVICE_SAFE_ENVELOPE", "MULTI_CONSTRAINT", "STEP_LIMIT", "OTHER", "near_neighbor", "far_neighbor", "high_interference", "low_interference", "target_approach", "service_phase")}})
        fw = np.concatenate([np.asarray(s["fwd"], dtype=np.float64) for s in bundle["steps"] if s["fwd"]])
        lt = np.concatenate([np.asarray(s["lat"], dtype=np.float64) for s in bundle["steps"] if s["lat"]])
        rv = np.concatenate([np.asarray(s["rev"], dtype=np.float64) for s in bundle["steps"] if s["rev"]])
        zb = np.concatenate([np.asarray(s["zbase"], dtype=np.float64) for s in bundle["steps"] if len(s["zbase"])])
        dir_rows.append(
            {
                "model_seed": seed,
                "mean_forward_component": float(np.mean(fw)) if fw.size else float("nan"),
                "mean_lateral_magnitude": float(np.mean(lt)) if lt.size else float("nan"),
                "reverse_fraction": float(np.mean(rv)) if rv.size else float("nan"),
                "base_zero_mean_residual": float(np.mean(zb)) if zb.size else float("nan"),
            }
        )
        arr = np.concatenate([np.asarray(e["arrive_steps"], dtype=np.float64) for e in bundle["episodes"] if e["arrive_steps"]] or [np.array([np.nan])])
        nav_rows.append(
            {
                "model_seed": seed,
                "mean_arrival_steps": float(np.nanmean(arr)),
                "median_arrival_steps": float(np.nanmedian(arr)),
                "p90_arrival_steps": float(np.nanquantile(arr[~np.isnan(arr)], 0.90)) if np.any(~np.isnan(arr)) else float("nan"),
                "never_enter_count": float(np.sum([e["never_enter"] for e in bundle["episodes"]])),
            }
        )
        svc_rows.append(
            {
                "model_seed": seed,
                "time_in_range": float(np.mean([e["time_in_range"] for e in bundle["episodes"]])),
                "completed_blocks": float(np.mean([e["completed_blocks"] for e in bundle["episodes"]])),
                "mean_active_sinr": float(np.nanmean([e["inrange_sinr"] for e in bundle["episodes"]])),
                "mean_active_rate": float(np.nanmean([e["inrange_rate"] for e in bundle["episodes"]])),
                "unfinished_progress": float(np.mean([e["unfinished"] for e in bundle["episodes"]])),
            }
        )
        hl_rows.append(
            {
                "model_seed": seed,
                "high_level_decision_count": float(np.mean([e["n_opt"] for e in bundle["episodes"]])),
                "assignment_change_count": float(np.mean([e["assignment_changes"] for e in bundle["episodes"]])),
                "idle_fraction": float(np.mean([e["idle_fraction"] for e in bundle["episodes"]])),
            }
        )
        # critic
        vals = np.concatenate([e["values"] for e in bundle["episodes"]])
        rets = np.concatenate([e["returns"] for e in bundle["episodes"]])
        pear = float(np.corrcoef(vals, rets)[0, 1]) if np.std(vals) > 1e-9 and np.std(rets) > 1e-9 else float("nan")
        ra = np.argsort(np.argsort(vals))
        rb = np.argsort(np.argsort(rets))
        sp = float(np.corrcoef(ra, rb)[0, 1]) if np.std(ra) > 0 and np.std(rb) > 0 else float("nan")
        crit_rows.append({"model_seed": seed, "mse": float(np.mean((vals - rets) ** 2)), "bias": float(np.mean(vals - rets)), "pearson": pear, "spearman": sp, "n": int(vals.size)})
        hid = np.concatenate([s["hid"] for s in bundle["steps"]], axis=0).reshape(-1, bundle["steps"][0]["hid"].shape[-1])
        hid = hid.reshape(-1, hid.shape[-1])
        var = float(np.mean(np.var(hid, axis=0)))
        svals = np.linalg.svd(hid - hid.mean(0), compute_uv=False)
        energy = np.cumsum(svals ** 2) / max(float(np.sum(svals ** 2)), 1e-12)
        erank = int(np.searchsorted(energy, 0.99) + 1)
        mu_w = float(torch.linalg.norm(w.actor.mu.weight.detach()).cpu())
        rng_cos = np.random.default_rng(7)
        n_h = min(hid.shape[0], 256)
        idx = rng_cos.choice(hid.shape[0], size=n_h, replace=False)
        x = hid[idx]
        xn = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)
        cos = xn @ xn.T
        iu = np.triu_indices(n_h, k=1)
        repr_rows.append(
            {
                "model_seed": seed,
                "hidden_feature_variance": var,
                "effective_rank_99": erank,
                "mean_pairwise_cosine": float(np.mean(cos[iu])),
                "output_head_weight_norm": mu_w,
                "log_std": log_std,
            }
        )
        # features on this bundle
        d_all = np.concatenate([s["d_near"] for s in bundle["steps"]])
        i_all = np.concatenate([s["interf"] for s in bundle["steps"]])
        dq = np.quantile(d_all, [1 / 3, 2 / 3])
        iq = np.quantile(i_all, [1 / 3, 2 / 3])
        acc = defaultdict(list)
        for s in bundle["steps"][::4]:
            obs = s["obs"]
            base = s["pre"]
            packed = {
                "neighbor_shuffle": {**{k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in obs.items()}},
                "interf_shuffle": {**{k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in obs.items()}},
                "sinr_shuffle": {**{k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in obs.items()}},
            }
            packed["neighbor_shuffle"]["neighbor_set"] = packed["neighbor_shuffle"]["neighbor_set"][:, rng.permutation(5)]
            packed["interf_shuffle"]["communication"][:, 2] = rng.permutation(packed["interf_shuffle"]["communication"][:, 2])
            packed["sinr_shuffle"]["communication"][:, 0] = rng.permutation(packed["sinr_shuffle"]["communication"][:, 0])
            with torch.no_grad():
                for name, alt in packed.items():
                    pre, _, _ = w.act_tensors(alt, deterministic=True)
                    diff = np.linalg.norm(pre.cpu().numpy() - base, axis=-1)
                    for u in range(6):
                        if name == "neighbor_shuffle" and float(s["d_near"][u]) <= dq[0]:
                            acc[name].append(float(diff[u]))
                        if name == "interf_shuffle" and float(s["interf"][u]) > iq[1]:
                            acc[name].append(float(diff[u]))
                        if name == "sinr_shuffle" and float(s["interf"][u]) > iq[1]:
                            acc[name].append(float(diff[u]))
        feat_rows.append({"model_seed": seed, **{k: float(np.mean(v)) if v else float("nan") for k, v in acc.items()}})
        w0 = load_worker(ckpt_update(seed, 0))
        init_rows.append(
            {
                "model_seed": seed,
                "init_actor_param_norm": param_norm(w0.actor),
                "init_critic_param_norm": param_norm(w0.critic),
                "init_log_std": float(w0.actor.log_std.detach().cpu().mean()),
            }
        )
    means = np.array([r["mean_residual_norm"] for r in residual_rows])
    for row in residual_rows:
        others = means[[i for i, s in enumerate(SEEDS) if f"seed{s}" != row["checkpoint"]]]
        row["zscore_vs_others"] = float((row["mean_residual_norm"] - float(np.mean(others))) / max(float(np.std(others)), 1e-9))
        row["rank_among_5"] = int(1 + np.sum(means < row["mean_residual_norm"]))
    # L0 nav
    arr0 = np.concatenate([np.asarray(e["arrive_steps"], dtype=np.float64) for e in l0["episodes"] if e["arrive_steps"]] or [np.array([np.nan])])
    nav_rows.append({"model_seed": "L0", "mean_arrival_steps": float(np.nanmean(arr0)), "median_arrival_steps": float(np.nanmedian(arr0)), "p90_arrival_steps": float("nan"), "never_enter_count": float(np.sum([e["never_enter"] for e in l0["episodes"]]))})
    svc_rows.append(
        {
            "model_seed": "L0",
            "time_in_range": float(np.mean([e["time_in_range"] for e in l0["episodes"]])),
            "completed_blocks": float(np.mean([e["completed_blocks"] for e in l0["episodes"]])),
            "mean_active_sinr": float(np.nanmean([e["inrange_sinr"] for e in l0["episodes"]])),
            "mean_active_rate": float(np.nanmean([e["inrange_rate"] for e in l0["episodes"]])),
            "unfinished_progress": float(np.mean([e["unfinished"] for e in l0["episodes"]])),
        }
    )
    hl_rows.append(
        {
            "model_seed": "L0",
            "high_level_decision_count": float(np.mean([e["n_opt"] for e in l0["episodes"]])),
            "assignment_change_count": float(np.mean([e["assignment_changes"] for e in l0["episodes"]])),
            "idle_fraction": float(np.mean([e["idle_fraction"] for e in l0["episodes"]])),
        }
    )
    plot_traj({"L0": l0, "seed1": bundles[1], "seed2": bundles[2]}, buoys, OUT / "figures", ids=sorted(traj_ids))
    # scenario failure post-hoc from consumed val
    scen = list(csv.DictReader((PHASE2 / "outputs" / "LLV2_PHASE2_PER_SCENARIO_METRICS.csv").open(encoding="utf-8")))
    s1 = [r for r in scen if r["model_seed"] == "1"]
    fail_rows = []
    deltas = np.array([float(r["delta"]) for r in s1])
    qneg = np.quantile(deltas[deltas < 0], 0.5) if np.any(deltas < 0) else -1e6
    for r in s1:
        d = float(r["delta"])
        if d <= min(-2e6, qneg):
            bucket = "large-negative"
        elif d < -1e-6:
            bucket = "moderate-negative"
        elif d <= 1e6:
            bucket = "near-zero"
        else:
            bucket = "positive"
        fail_rows.append({**{k: r[k] for k in r}, "bucket": bucket})
    # opt windows seed1 vs seed2
    opt_rows = []
    for seed in (1, 2):
        recs = hist1[seed]
        by_u = {int(float(r["update"])): r for r in recs}

        def wm(a, b, key):
            vals = [float(by_u[u][key]) for u in range(a, b + 1) if u in by_u]
            return float(np.mean(vals)) if vals else float("nan")

        for a, b in ((1, 20), (21, 40), (41, 60), (61, 80), (81, 100)):
            opt_rows.append(
                {
                    "model_seed": seed,
                    "window": f"{a}-{b}",
                    "kl": wm(a, b, "approx_kl"),
                    "clip": wm(a, b, "clip_fraction"),
                    "grad": wm(a, b, "gradient_norm_preclip"),
                    "entropy": wm(a, b, "entropy"),
                    "value_loss": wm(a, b, "value_loss"),
                    "policy_loss": wm(a, b, "policy_loss"),
                    "explained_variance": wm(a, b, "explained_variance"),
                }
            )
    # classify
    s1_curve = [r for r in curve_eval if int(r["model_seed"]) == 1]
    p20 = float(next(r["payload"] for r in s1_curve if int(float(r["update"])) == 20))
    p100 = float(next(r["payload"] for r in s1_curve if int(float(r["update"])) == 100))
    p0 = float(next(r["payload"] for r in s1_curve if int(float(r["update"])) == 0))
    curve_type = "B" if p20 > p0 + 5e5 and p100 < p20 - 5e5 else "A"
    s1_res = next(r for r in residual_rows if r["checkpoint"] == "seed1")
    s2_res = next(r for r in residual_rows if r["checkpoint"] == "seed2")
    s1_dir = next(r for r in dir_rows if r["model_seed"] == 1)
    s2_dir = next(r for r in dir_rows if r["model_seed"] == 2)
    s1_nav = next(r for r in nav_rows if r["model_seed"] == 1)
    s2_nav = next(r for r in nav_rows if r["model_seed"] == 2)
    l0_nav = next(r for r in nav_rows if r["model_seed"] == "L0")
    s1_lock = next(r for r in lock_rows if r["model_seed"] == 1)
    s1_crit = next(r for r in crit_rows if r["model_seed"] == 1)
    s2_crit = next(r for r in crit_rows if r["model_seed"] == 2)
    s1_feat = next(r for r in feat_rows if r["model_seed"] == 1)
    s2_feat = next(r for r in feat_rows if r["model_seed"] == 2)
    lock_ok = all(r["service_lock_violation"] == 0 and r["locked_residual_nonzero_fraction"] == 0 for r in lock_rows)
    s1_svc = next(r for r in svc_rows if r["model_seed"] == 1)
    l0_svc = next(r for r in svc_rows if r["model_seed"] == "L0")
    s1_hl = next(r for r in hl_rows if r["model_seed"] == 1)
    l0_hl = next(r for r in hl_rows if r["model_seed"] == "L0")
    nav_worse = s1_nav["mean_arrival_steps"] > l0_nav["mean_arrival_steps"] + 1.0
    reverse_high = s1_dir["reverse_fraction"] > s2_dir["reverse_fraction"] + 0.05
    idle_res_big = s1_dir["base_zero_mean_residual"] > 1.5 * s2_dir["base_zero_mean_residual"]
    res_big = s1_res["mean_residual_norm"] > 1.2 * s2_res["mean_residual_norm"]
    tir_worse = s1_svc["time_in_range"] < 0.8 * l0_svc["time_in_range"]
    idle_worse = s1_hl["idle_fraction"] > l0_hl["idle_fraction"] + 0.05
    crit_bad = (not np.isfinite(s1_crit["pearson"])) or s1_crit["pearson"] + 0.15 < s2_crit["pearson"]
    rel_unique = abs(float(s1_feat.get("neighbor_shuffle") or 0) - float(s2_feat.get("neighbor_shuffle") or 0)) > 1e-3
    w4160 = next(r for r in opt_rows if r["model_seed"] == 1 and r["window"] == "41-60")
    w2140 = next(r for r in opt_rows if r["model_seed"] == 1 and r["window"] == "21-40")
    opt_spike = float(w4160["grad"]) > 20.0 and float(w4160["grad"]) > 5.0 * max(float(w2140["grad"]), 1e-6)
    # Direction vs seed2 does NOT show extra reverse; seed1 residual is oversized, especially at base≈0.
    if not lock_ok:
        primary, secondary = "SHIELD", "ACTION"
    elif res_big or idle_res_big:
        primary = "ACTION"
        if tir_worse or nav_worse:
            secondary = "NAV"
        elif idle_worse:
            secondary = "HIGHLEVEL_FEEDBACK"
        elif opt_spike:
            secondary = "OPT"
        else:
            secondary = "NAV"
    elif nav_worse or tir_worse:
        primary, secondary = "NAV", "ACTION"
    elif crit_bad:
        primary, secondary = "CRITIC", "ACTION"
    elif opt_spike:
        primary, secondary = "OPT", "ACTION"
    else:
        primary, secondary = "NONE", "ACTION"
    if rel_unique:
        secondary = "REL"
    mechanisms = [primary, secondary]
    if idle_worse and "HIGHLEVEL_FEEDBACK" not in mechanisms:
        mechanisms.append("HIGHLEVEL_FEEDBACK")
    if opt_spike and "OPT" not in mechanisms:
        mechanisms.append("OPT")
    gate = "C"
    change = "NO_METHOD_CHANGE"
    if primary in {"ACTION", "NAV"} and (res_big or idle_res_big) and lock_ok and (not reverse_high):
        gate = "B"
        change = "uniform residual-norm cap plus freeze residual when ||a_base||≈0 (idle/base-zero), all seeds; do not add relation modules"
    elif primary == "NAV" and reverse_high:
        gate = "A"
        change = "cap residual components that oppose the gated fly-toward base (forward-trust residual), all seeds, no seed-specific training"
    elif primary == "NONE":
        gate = "C"
        change = "NO_METHOD_CHANGE"
    payload = {
        "S1": gate,
        "curve_type": curve_type,
        "primary_mechanism": primary,
        "secondary_mechanism": secondary,
        "minimal_method_change": change,
        "service_lock_ok_all_seeds": lock_ok,
        "trained": False,
        "phase2_conclusion_unchanged": {"LLV2_MS": "B", "stable_residual_advantage": False, "REL": 1},
        "seed1_train_payload_u20": p20,
        "seed1_train_payload_u100": p100,
        "labels_allowed": ["OPT", "CRITIC", "ACTION", "NAV", "REL", "SHIELD", "HIGHLEVEL_FEEDBACK", "INIT", "NONE"],
        "extra_related_mechanisms": [m for m in mechanisms if m not in {primary, secondary}],
        "evidence": {
            "reverse_high_vs_seed2": reverse_high,
            "res_big_vs_seed2": res_big,
            "idle_residual_big": idle_res_big,
            "time_in_range_worse": tir_worse,
            "idle_fraction_worse": idle_worse,
            "nav_mean_arrival_worse": nav_worse,
            "opt_mid_grad_spike": opt_spike,
            "critic_worse_than_seed2": crit_bad,
            "rel_unique": rel_unique,
        },
    }
    _csv(OUT / "LLV2_PHASE2A_LEARNING_CURVES.csv", learn_rows)
    _csv(OUT / "LLV2_PHASE2A_RESIDUAL_STATS.csv", residual_rows)
    _csv(OUT / "LLV2_PHASE2A_PROJECTOR_CONTEXT.csv", proj_ctx)
    _csv(OUT / "LLV2_PHASE2A_SEED1_SCENARIO_FAILURE.csv", fail_rows)
    _csv(OUT / "LLV2_PHASE2A_NAVIGATION.csv", nav_rows)
    _csv(OUT / "LLV2_PHASE2A_SERVICE.csv", svc_rows)
    _csv(OUT / "LLV2_PHASE2A_FEATURE_SENSITIVITY.csv", feat_rows)
    _csv(OUT / "LLV2_PHASE2A_ACTOR_REPRESENTATION.csv", repr_rows)
    _csv(OUT / "LLV2_PHASE2A_CRITIC_CALIBRATION.csv", crit_rows)
    _csv(OUT / "LLV2_PHASE2A_OPTIMIZATION.csv", opt_rows)
    _csv(OUT / "LLV2_PHASE2A_HIGHLEVEL_FEEDBACK.csv", hl_rows)
    _csv(OUT / "LLV2_PHASE2A_RESIDUAL_DIRECTION.csv", dir_rows)
    _csv(OUT / "LLV2_PHASE2A_SERVICE_LOCK.csv", lock_rows)
    _csv(OUT / "LLV2_PHASE2A_INIT.csv", init_rows)
    _dump(OUT / "LLV2_PHASE2A_SEED1_GATE.json", payload)
    _dump(
        OUT / "LLV2_PHASE2A_PROTOCOL.json",
        {
            "name": "LL-V2.2A",
            "trained": False,
            "primary_ckpts": "update100 only",
            "design_used": True,
            "validation_posthoc_only": True,
            "no_checkpoint_pick": True,
        },
    )
    for src in OUT.glob("LLV2_PHASE2A_*"):
        (PHASE2A / src.name).write_bytes(src.read_bytes())
        (ROOT / src.name).write_bytes(src.read_bytes())
    fig_dir = PHASE2A / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for png in (OUT / "figures").glob("seed1_vs_seed2_traj_*.png"):
        (fig_dir / png.name).write_bytes(png.read_bytes())
        (ROOT / png.name).write_bytes(png.read_bytes())
    return payload


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
