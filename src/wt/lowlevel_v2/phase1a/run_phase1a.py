"""LL-V2.1A frozen-checkpoint audit. No training. No hyperparameter changes."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from wt.l3b.env import L3BEnv
from wt.lowlevel_v2.contracts import env_target
from wt.lowlevel_v2.observation import communication_features
from wt.lowlevel_v2.phase1.controller import ResidualWorker, load_frozen_high, make_networks
from wt.lowlevel_v2.phase1.protocol import (
    ALPHA_RESIDUAL,
    D_SAFE_M,
    EXPECTED_HIGH_SHA,
    LAMBDA_PROJECT,
    LAMBDA_RESIDUAL,
    OUTPUT as PHASE1_OUTPUT,
    PPO,
    ROOT,
    high_sha,
    load_pools,
    refuse_untouched,
)
from wt.lowlevel_v2.service_safe_envelope import envelope_active

from .diagnose import desired_constraint_flags, intervention_label, predict_sinr_rate

PHASE1A = Path(__file__).resolve().parent
OUT = PHASE1A / "outputs"
UPDATES = (0, 20, 40, 60, 80, 100)
LARGE_CHANGE = 0.20
INTERVENE_EPS = 1.0e-6
SAT_EPS = 0.98
NEAR_FRAC = (0.0, 1.0 / 3.0)
MID_FRAC = (1.0 / 3.0, 2.0 / 3.0)
FAR_FRAC = (2.0 / 3.0, 1.0)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def inventory() -> list[dict]:
    rows = []
    for update in UPDATES:
        path = PHASE1_OUTPUT / f"ckpt_update{update}.pt"
        exists = path.is_file()
        rows.append(
            {
                "checkpoint": f"u{update}",
                "update": update,
                "path": str(path),
                "exists": exists,
                "sha256": _sha(path) if exists else None,
                "bytes": path.stat().st_size if exists else 0,
            }
        )
    _csv(OUT / "LLV2_PHASE1A_CHECKPOINT_INVENTORY.csv", rows)
    return rows


def load_ckpt(actor, critic, path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")
    actor.load_state_dict(payload["actor"])
    if "critic" in payload:
        critic.load_state_dict(payload["critic"])
    actor.eval()
    critic.eval()
    return payload


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {key: float("nan") for key in ("min", "p10", "p25", "median", "p75", "p90", "max")}
    qs = np.quantile(values, [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0])
    return {"min": float(qs[0]), "p10": float(qs[1]), "p25": float(qs[2]), "median": float(qs[3]), "p75": float(qs[4]), "p90": float(qs[5]), "max": float(qs[6])}


def _tertile_masks(values: np.ndarray) -> dict[str, np.ndarray]:
    if values.size == 0:
        empty = np.zeros(0, dtype=bool)
        return {"near": empty, "middle": empty, "far": empty}
    q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    return {"near": values <= q1, "middle": (values > q1) & (values <= q2), "far": values > q2}


def actor_pre(worker: ResidualWorker, obs: dict, *, mutate: str | None, rng: np.random.Generator) -> np.ndarray:
    packed = {key: (np.array(val, copy=True) if isinstance(val, np.ndarray) else val) for key, val in obs.items()}
    if mutate == "neighbor_zero":
        packed["neighbor_set"] = np.zeros_like(packed["neighbor_set"])
    elif mutate == "neighbor_shuffle":
        perm = rng.permutation(packed["neighbor_set"].shape[0])
        packed["neighbor_set"] = packed["neighbor_set"][perm]
    elif mutate == "interf_zero":
        packed["communication"][:, 2:4] = 0.0
    elif mutate == "interf_shuffle":
        packed["communication"][:, 2] = rng.permutation(packed["communication"][:, 2])
        packed["communication"][:, 3] = rng.permutation(packed["communication"][:, 3])
    elif mutate == "sinr_zero":
        packed["communication"][:, 0:2] = 0.0
    elif mutate == "sinr_shuffle":
        packed["communication"][:, 0] = rng.permutation(packed["communication"][:, 0])
        packed["communication"][:, 1] = rng.permutation(packed["communication"][:, 1])
    with torch.no_grad():
        pre, _ = worker.act_tensors(packed, deterministic=True)
    return pre.detach().cpu().numpy()


def rollout(worker: ResidualWorker, seeds: list[int], *, zero_residual: bool, store_traj_ids: set[int] | None = None) -> dict:
    steps: list[dict] = []
    episodes: list[dict] = []
    traj: dict[int, list] = defaultdict(list)
    for seed in seeds:
        refuse_untouched(int(seed))
        info = worker.reset(int(seed))
        payload0 = float(worker.env.payload_bits)
        drop0 = float(worker.env.dropped)
        flight = 0.0
        prev_assoc = None
        switch = 0
        serve_n = 0
        while True:
            pos0 = np.asarray(worker.env.uav_positions, dtype=np.float64)
            assoc = np.asarray(worker.option.association if worker.option is not None else worker.env.assignment)
            info, rec = worker.step(info, deterministic=True, zero_residual=zero_residual)
            assoc = np.asarray(rec.obs.get("task") is not None and worker.option.association)
            assoc = np.asarray(worker.option.association)
            flags = desired_constraint_flags(worker.env, worker.option, rec.a_des)
            # flags must use PRE-step geometry: worker.env already stepped. Use pos0.
            saved = worker.env.uav_positions.copy()
            worker.env.uav_positions = pos0
            flags = desired_constraint_flags(worker.env, worker.option, rec.a_des)
            comm = communication_features(worker.env)
            envl = envelope_active(worker.env, worker.option)
            in_service = []
            for uav in range(6):
                tgt = env_target(int(assoc[uav]), 12)
                in_service.append(bool(tgt >= 0 and worker.env.queue[tgt] > 1.0e-9 and worker.env._in_range(uav, tgt)))
            worker.env.uav_positions = saved
            intervened = float(rec.intervention_norm) > INTERVENE_EPS
            label = intervention_label(flags, intervened)
            res_norm = np.linalg.norm(rec.delta_tanh, axis=-1)
            des_minus = np.linalg.norm(rec.a_des - rec.a_base, axis=-1)
            exec_minus = np.linalg.norm(rec.a_exec - rec.a_base, axis=-1)
            sat = np.mean(np.max(np.abs(rec.delta_tanh), axis=-1) >= SAT_EPS)
            actor_mean = float(np.mean(np.linalg.norm(rec.pre_tanh, axis=-1)))
            log_std = float(worker.actor.log_std.detach().cpu().mean())
            d_near = np.asarray(rec.obs["d_nearest"], dtype=np.float64)
            if prev_assoc is not None:
                for uav in range(6):
                    if prev_assoc[uav] >= 0 and int(assoc[uav]) != int(prev_assoc[uav]):
                        switch += 1
            prev_assoc = assoc.copy()
            if store_traj_ids is not None and int(seed) in store_traj_ids:
                traj[int(seed)].append(
                    {
                        "positions": pos0.copy(),
                        "assoc": assoc.copy(),
                        "intervened": intervened,
                        "d_near": d_near.copy(),
                    }
                )
            sinr_b, rate_b = None, None
            sinr_r, rate_r = None, None
            worker.env.uav_positions, restored = pos0, saved
            worker.env.uav_positions = pos0
            try:
                sinr_b, rate_b = predict_sinr_rate(worker.env, rec.a_base)
                sinr_r, rate_r = predict_sinr_rate(worker.env, rec.a_des)
            finally:
                worker.env.uav_positions = restored
            flight += float(np.sum(np.linalg.norm(np.asarray(worker.env.uav_positions) - pos0, axis=-1)))
            c_proj = float(np.mean(np.sum((rec.a_exec - rec.a_des) ** 2, axis=-1)))
            c_res = float(np.mean(np.sum((ALPHA_RESIDUAL * rec.delta_tanh) ** 2, axis=-1)))
            r_payload = rec.payload_inc / 1.0e6
            steps.append(
                {
                    "seed": int(seed),
                    "res_norm": res_norm,
                    "des_minus": des_minus,
                    "exec_minus": exec_minus,
                    "sat": sat,
                    "actor_mean": actor_mean,
                    "log_std": log_std,
                    "pre_tanh": rec.pre_tanh.copy(),
                    "obs": rec.obs,
                    "a_base": rec.a_base,
                    "a_des": rec.a_des,
                    "a_exec": rec.a_exec,
                    "delta_tanh": rec.delta_tanh,
                    "d_near": d_near,
                    "intervened": intervened,
                    "label": label,
                    "flags": flags,
                    "intervention_norm": rec.intervention_norm,
                    "envl": envl.copy(),
                    "in_service": np.asarray(in_service, dtype=bool),
                    "base_norm": np.linalg.norm(rec.a_base, axis=-1),
                    "interf": np.asarray(comm["total_interference"], dtype=np.float64),
                    "sinr": np.asarray(comm["sinr"], dtype=np.float64),
                    "rate": np.asarray(comm["rate"], dtype=np.float64),
                    "value": rec.value,
                    "reward": rec.reward,
                    "r_payload": r_payload,
                    "c_proj": c_proj,
                    "c_res": c_res,
                    "done": rec.done,
                    "delta_sinr": float(np.mean(sinr_r - sinr_b)),
                    "delta_rate": float(np.mean(rate_r - rate_b)),
                    "positions": pos0,
                    "assoc": assoc,
                    "progress": worker.env.progress.copy(),
                }
            )
            serve_n += int(np.sum(in_service))
            if rec.done or worker.env.t >= 48:
                break
        payload = float(worker.env.payload_bits - payload0)
        episodes.append(
            {
                "seed": int(seed),
                "payload": payload,
                "completed_blocks": payload / 1.0e6,
                "flight": flight,
                "switch_loss": switch,
                "unfinished": float(np.mean(worker.env.progress)),
                "drop": float(worker.env.dropped - drop0),
                "min_sep": float(min(float(np.min(s["d_near"])) for s in steps if s["seed"] == seed) if any(s["seed"] == seed for s in steps) else np.nan),
                "mean_sinr": float(np.mean([np.mean(s["sinr"]) for s in steps if s["seed"] == seed])),
                "projector_rate": float(np.mean([s["intervened"] for s in steps if s["seed"] == seed])),
            }
        )
    return {"steps": steps, "episodes": episodes, "traj": traj}


def summarize_residual(tag: str, bundle: dict, log_std: float) -> dict:
    steps = bundle["steps"]
    res = np.concatenate([s["res_norm"] for s in steps])
    des = np.concatenate([s["des_minus"] for s in steps])
    exe = np.concatenate([s["exec_minus"] for s in steps])
    base_zero = np.concatenate([s["res_norm"][s["base_norm"] < 1.0e-6] for s in steps if np.any(s["base_norm"] < 1.0e-6)] or [np.array([0.0])])
    return {
        "checkpoint": tag,
        "mean_residual_norm": float(np.mean(res)),
        "median_residual_norm": float(np.median(res)),
        "p95_residual_norm": float(np.quantile(res, 0.95)),
        "max_residual_norm": float(np.max(res)),
        "mean_desired_minus_base_norm": float(np.mean(des)),
        "mean_executed_minus_base_norm": float(np.mean(exe)),
        "tanh_saturation_fraction": float(np.mean([s["sat"] for s in steps])),
        "actor_mean_norm": float(np.mean([s["actor_mean"] for s in steps])),
        "actor_log_std": float(log_std),
        "base_zero_state_residual_norm": float(np.mean(base_zero)),
    }


def projector_breakdown(tag: str, bundle: dict) -> dict:
    steps = bundle["steps"]
    n = max(len(steps), 1)
    labels = [s["label"] for s in steps if s["intervened"]]
    counts = defaultdict(int)
    for lab in labels:
        counts[lab] += 1
    row = {
        "checkpoint": tag,
        "projector_intervention_rate": float(np.mean([s["intervened"] for s in steps])),
        "projector_intervention_norm": float(np.mean([s["intervention_norm"] for s in steps])),
        "n_steps": len(steps),
        "n_intervened": len(labels),
    }
    for name in ("STEP_LIMIT", "PAIRWISE_COLLISION", "SERVICE_SAFE_ENVELOPE", "MULTI_CONSTRAINT", "OTHER"):
        row[name] = counts[name] / n
        row[f"{name}_among_intervened"] = counts[name] / max(len(labels), 1)
    return row


def service_behavior(tag: str, bundle: dict) -> dict:
    res_nz = []
    res_n = []
    env_int = []
    for step in bundle["steps"]:
        mask = step["in_service"] & (step["base_norm"] < 1.0e-6)
        if not np.any(mask):
            continue
        res_nz.extend((step["res_norm"][mask] > 1.0e-4).astype(float).tolist())
        res_n.extend(step["res_norm"][mask].tolist())
        env_hit = float(step["intervened"] and step["flags"]["SERVICE_SAFE_ENVELOPE"])
        env_int.extend([env_hit] * int(np.sum(mask)))
    eps = bundle["episodes"]
    return {
        "checkpoint": tag,
        "residual_nonzero_fraction": float(np.mean(res_nz) if res_nz else 0.0),
        "mean_residual_norm": float(np.mean(res_n) if res_n else 0.0),
        "envelope_intervention_rate": float(np.mean(env_int) if env_int else 0.0),
        "n_service_zero_base_steps": len(res_nz),
        "completed_blocks": float(np.mean([e["completed_blocks"] for e in eps])),
        "unfinished_progress": float(np.mean([e["unfinished"] for e in eps])),
        "target_switch_loss": float(np.mean([e["switch_loss"] for e in eps])),
    }


def utilization_rows(tag: str, worker: ResidualWorker, steps: list[dict], kind: str) -> list[dict]:
    rng = np.random.default_rng(21501)
    rows = []
    if kind == "neighbor":
        variants = [("zero", "neighbor_zero"), ("shuffle", "neighbor_shuffle")]
        return _util_simple(tag, worker, steps, variants, "distance", [s["d_near"] for s in steps], rng)
    variants = [
        ("interf_zero", "interf_zero"),
        ("interf_shuffle", "interf_shuffle"),
        ("sinr_zero", "sinr_zero"),
        ("sinr_shuffle", "sinr_shuffle"),
    ]
    return _util_simple(tag, worker, steps, variants, "interference", [s["interf"] for s in steps], rng)


def _util_simple(tag, worker, steps, variants, regime_name, per_step_scores, rng) -> list[dict]:
    all_scores = np.concatenate(per_step_scores)
    q1, q2 = np.quantile(all_scores, [1 / 3, 2 / 3])
    rows = []
    regimes = ("near", "middle", "far") if regime_name == "distance" else ("low", "middle", "high")
    buckets = {v[0]: {g: [] for g in (*regimes, "all")} for v in variants}
    rch = {v[0]: {g: [] for g in (*regimes, "all")} for v in variants}
    for step, score in zip(steps, per_step_scores):
        base_pre = step["pre_tanh"]
        for vname, mutate in variants:
            alt = actor_pre(worker, step["obs"], mutate=mutate, rng=rng)
            diff = np.linalg.norm(alt - base_pre, axis=-1)
            rn = np.linalg.norm(np.tanh(alt), axis=-1) - np.linalg.norm(np.tanh(base_pre), axis=-1)
            for uav in range(6):
                val = float(score[uav])
                if regime_name == "distance":
                    g = "near" if val <= q1 else ("middle" if val <= q2 else "far")
                else:
                    g = "low" if val <= q1 else ("middle" if val <= q2 else "high")
                buckets[vname][g].append(float(diff[uav]))
                rch[vname][g].append(float(rn[uav]))
                buckets[vname]["all"].append(float(diff[uav]))
                rch[vname]["all"].append(float(rn[uav]))
    regimes = ("near", "middle", "far") if regime_name == "distance" else ("low", "middle", "high")
    for vname, _ in variants:
        for g in (*regimes, "all"):
            arr = np.asarray(buckets[vname].get(g, []), dtype=np.float64)
            rc = np.asarray(rch[vname].get(g, []), dtype=np.float64)
            rows.append(
                {
                    "checkpoint": tag,
                    "variant": vname,
                    "regime": g,
                    "mean_action_l2": float(np.mean(arr)) if arr.size else float("nan"),
                    "p95_action_l2": float(np.quantile(arr, 0.95)) if arr.size else float("nan"),
                    "mean_residual_norm_change": float(np.mean(rc)) if rc.size else float("nan"),
                    "n": int(arr.size),
                }
            )
    return rows


def sep_response(tag: str, steps: list[dict]) -> list[dict]:
    d_all = np.concatenate([s["d_near"] for s in steps])
    q1, q2 = np.quantile(d_all, [1 / 3, 2 / 3])
    rows = []
    buckets = {"near": [], "middle": [], "far": []}
    for step in steps:
        pos = step["positions"]
        r = step["a_des"] - step["a_base"]
        for m in range(6):
            delta = pos - pos[m]
            dist = np.linalg.norm(delta, axis=-1)
            dist[m] = np.inf
            j = int(np.argmin(dist))
            v = pos[m] - pos[j]
            nrm = float(np.linalg.norm(v))
            if nrm < 1.0e-9:
                continue
            v = v / nrm
            resp = float(np.dot(r[m], v))
            d = float(step["d_near"][m])
            if d <= q1:
                buckets["near"].append(resp)
            elif d <= q2:
                buckets["middle"].append(resp)
            else:
                buckets["far"].append(resp)
    for g, arr in buckets.items():
        a = np.asarray(arr, dtype=np.float64)
        rows.append({"checkpoint": tag, "regime": g, "mean_separation_response": float(np.mean(a)) if a.size else float("nan"), "n": int(a.size)})
    return rows


def plot_trajectories(bundles: dict, buoys: np.ndarray, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    for seed in sorted(next(iter(bundles.values()))["traj"].keys()):
        fig, axes = plt.subplots(1, len(bundles), figsize=(4 * len(bundles), 4), sharex=True, sharey=True)
        if len(bundles) == 1:
            axes = [axes]
        for ax, (name, bundle) in zip(axes, bundles.items()):
            frames = bundle["traj"][seed]
            for uav in range(6):
                xy = np.array([fr["positions"][uav] for fr in frames])
                ax.plot(xy[:, 0], xy[:, 1], color=colors[uav], lw=1.2)
                ax.scatter(xy[0, 0], xy[0, 1], color=colors[uav], s=12)
            for fr in frames:
                if fr["intervened"]:
                    ax.scatter(fr["positions"][:, 0], fr["positions"][:, 1], c="k", s=4, alpha=0.2, zorder=3)
                d = fr["d_near"]
                close = d <= np.min(d) + 1e-9
                ax.scatter(fr["positions"][close, 0], fr["positions"][close, 1], marker="o", facecolors="none", edgecolors="orange", s=18, lw=0.6)
            for _k, buoy in enumerate(buoys):
                circ = plt.Circle(buoy, 16.0, fill=False, lw=0.4, color="#888")
                ax.add_patch(circ)
                ax.scatter([buoy[0]], [buoy[1]], marker="x", c="red", s=18)
            ax.set_title(f"{name} seed {seed}")
            ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(out / f"traj_seed{seed}.png", dpi=140)
        fig.savefig(out / f"traj_seed{seed}.pdf")
        plt.close(fig)


def critic_calibration(bundle: dict) -> dict:
    gamma = float(PPO.gamma)
    mses, biases, pears = [], [], []
    by_seed: dict[int, list] = defaultdict(list)
    for step in bundle["steps"]:
        by_seed[step["seed"]].append(step)
    resolved = False
    for seed, seq in by_seed.items():
        rewards = np.array([s["reward"] for s in seq], dtype=np.float64)
        values = np.array([s["value"] for s in seq], dtype=np.float64)
        dones = np.array([s["done"] for s in seq], dtype=np.float64)
        ret = np.zeros_like(rewards)
        running = 0.0
        for t in range(len(seq) - 1, -1, -1):
            running = rewards[t] + gamma * (1.0 - dones[t]) * running
            ret[t] = running
        if values.size < 3:
            continue
        resolved = True
        mses.append(float(np.mean((values - ret) ** 2)))
        biases.append(float(np.mean(values - ret)))
        if np.std(values) > 1e-9 and np.std(ret) > 1e-9:
            pears.append(float(np.corrcoef(values, ret)[0, 1]))
    return {
        "checkpoint": bundle.get("tag", ""),
        "critic_calibration_resolved": resolved,
        "mse": float(np.mean(mses)) if mses else float("nan"),
        "bias": float(np.mean(biases)) if biases else float("nan"),
        "pearson": float(np.mean(pears)) if pears else float("nan"),
    }


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    if high_sha() != EXPECTED_HIGH_SHA:
        raise RuntimeError("high-level SHA mismatch")
    train_seeds, design_seeds = load_pools()
    inv = inventory()
    present = [row for row in inv if row["exists"]]
    high, _ = load_frozen_high()
    actor, critic = make_networks()
    worker = ResidualWorker(high, actor, critic)
    env0 = L3BEnv()
    env0.reset(seed=int(design_seeds[0]))
    buoys = env0.buoy_positions.copy()
    traj_ids = set(sorted(design_seeds)[:4])
    bundles = {}
    worker_l0 = ResidualWorker(high, actor, critic)
    l0 = rollout(worker_l0, design_seeds, zero_residual=True, store_traj_ids=traj_ids)
    l0["tag"] = "L0"
    residual_rows, proj_rows, svc_rows = [], [], []
    neigh_rows, comm_rows, sep_rows, critic_rows = [], [], [], []
    high_interf_rows = []
    drift_states = [s["obs"] for s in l0["steps"]]
    drift_pres = {}
    for row in present:
        load_ckpt(actor, critic, Path(row["path"]))
        tag = row["checkpoint"]
        bundle = rollout(worker, design_seeds, zero_residual=False, store_traj_ids=traj_ids)
        bundle["tag"] = tag
        bundles[tag] = bundle
        log_std = float(actor.log_std.detach().cpu().mean())
        residual_rows.append(summarize_residual(tag, bundle, log_std))
        proj_rows.append(projector_breakdown(tag, bundle))
        svc_rows.append(service_behavior(tag, bundle))
        neigh_rows.extend(utilization_rows(tag, worker, bundle["steps"], "neighbor"))
        comm_rows.extend(utilization_rows(tag, worker, bundle["steps"], "comm"))
        sep_rows.extend(sep_response(tag, bundle["steps"]))
        cal = critic_calibration(bundle)
        cal["checkpoint"] = tag
        critic_rows.append(cal)
        high_interf_vals = []
        high_rate_vals = []
        all_interf = np.concatenate([s["interf"] for s in bundle["steps"]])
        hi_thr = float(np.quantile(all_interf, 2.0 / 3.0))
        for step in bundle["steps"]:
            mask = step["interf"] > hi_thr
            if np.any(mask):
                high_interf_vals.append(float(step["delta_sinr"]))
                high_rate_vals.append(float(step["delta_rate"]))
        high_interf_rows.append(
            {
                "checkpoint": tag,
                "mean_delta_sinr": float(np.mean(high_interf_vals)) if high_interf_vals else float("nan"),
                "mean_delta_rate": float(np.mean(high_rate_vals)) if high_rate_vals else float("nan"),
                "n_high_interf_steps": len(high_interf_vals),
                "high_interf_threshold": hi_thr,
            }
        )
        pres = []
        rng = np.random.default_rng(7)
        for obs in drift_states:
            pres.append(actor_pre(worker, obs, mutate=None, rng=rng))
        drift_pres[tag] = np.stack(pres, axis=0)
    reward_rows = []
    for row, bundle in zip(residual_rows, [bundles[r["checkpoint"]] for r in residual_rows]):
        rp = np.array([s["r_payload"] for s in bundle["steps"]])
        cp = np.array([LAMBDA_PROJECT * s["c_proj"] for s in bundle["steps"]])
        cr = np.array([LAMBDA_RESIDUAL * s["c_res"] for s in bundle["steps"]])
        eps = 1.0e-9
        pr = float(np.mean(np.abs(cp)) / (np.mean(np.abs(rp)) + eps))
        rr = float(np.mean(np.abs(cr)) / (np.mean(np.abs(rp)) + eps))
        verdict = "TOO_WEAK" if max(pr, rr) < 0.02 else ("TOO_STRONG" if max(pr, rr) > 0.5 else "BALANCED")
        reward_rows.append(
            {
                "checkpoint": row["checkpoint"],
                "source": "design_frozen_rollout_not_train_csv",
                "mean_r_payload": float(np.mean(rp)),
                "std_r_payload": float(np.std(rp)),
                "mean_0.05_C_project": float(np.mean(cp)),
                "std_0.05_C_project": float(np.std(cp)),
                "mean_0.01_C_residual": float(np.mean(cr)),
                "std_0.01_C_residual": float(np.std(cr)),
                "project_penalty_ratio": pr,
                "residual_penalty_ratio": rr,
                "verdict": verdict,
                "train_csv_has_reward_terms": False,
            }
        )
    hist = list(csv.DictReader((PHASE1_OUTPUT / "LLV2_PHASE1_TRAINING_HISTORY.csv").open(encoding="utf-8")))
    opt_rows = []
    for rec in hist:
        opt_rows.append(
            {
                "update": rec.get("update", "missing"),
                "policy_loss": rec.get("policy_loss", "missing"),
                "value_loss": rec.get("value_loss", "missing"),
                "entropy": rec.get("entropy", "missing"),
                "approx_kl": rec.get("approx_kl", "missing"),
                "clip_fraction": rec.get("clip_fraction", "missing"),
                "gradient_norm": rec.get("gradient_norm_preclip", "missing"),
                "learning_rate": PPO.learning_rate,
                "explained_variance": rec.get("explained_variance", "missing"),
                "advantage_mean": rec.get("advantage_mean", "missing"),
                "advantage_std": rec.get("advantage_std", "missing"),
                "return_mean": "missing",
                "return_std": "missing",
            }
        )
    by_u = {int(float(r["update"])): r for r in opt_rows}

    def window_mean(a, b, key):
        vals = [float(by_u[u][key]) for u in range(a, b + 1) if u in by_u]
        return float(np.mean(vals)) if vals else float("nan")

    spikes = {}
    for a, b, c, d in ((20, 40, 1, 19), (40, 60, 21, 39), (60, 80, 41, 59), (80, 100, 61, 79)):
        spikes[f"{a}-{b}"] = {
            "kl": window_mean(a, b, "approx_kl"),
            "clip": window_mean(a, b, "clip_fraction"),
            "value": window_mean(a, b, "value_loss"),
            "grad": window_mean(a, b, "gradient_norm"),
            "entropy": window_mean(a, b, "entropy"),
            "ev": window_mean(a, b, "explained_variance"),
            "kl_prev": window_mean(c, d, "approx_kl"),
        }
    drift_rows = []
    pairs = [("u20", "u40"), ("u40", "u60"), ("u60", "u80"), ("u80", "u100")]
    for left, right in pairs:
        if left not in drift_pres or right not in drift_pres:
            drift_rows.append({"pair": f"{left}->{right}", "missing": True})
            continue
        a, b = drift_pres[left], drift_pres[right]
        delta = b - a
        l2 = np.linalg.norm(delta, axis=-1)
        l2m = l2.mean(axis=-1)
        cos = []
        for i in range(a.shape[0]):
            for u in range(6):
                na, nb = np.linalg.norm(a[i, u]), np.linalg.norm(b[i, u])
                if na < 1e-9 or nb < 1e-9:
                    cos.append(1.0 if na < 1e-9 and nb < 1e-9 else 0.0)
                else:
                    cos.append(float(np.dot(a[i, u], b[i, u]) / (na * nb)))
        drift_rows.append(
            {
                "pair": f"{left}->{right}",
                "mean_l2": float(np.mean(l2m)),
                "median_l2": float(np.median(l2m)),
                "p95_l2": float(np.quantile(l2m, 0.95)),
                "mean_cosine": float(np.mean(cos)),
                "large_change_fraction": float(np.mean(l2m > LARGE_CHANGE)),
                "large_change_threshold": LARGE_CHANGE,
            }
        )
    # state signal: L0 design + train initials + L0 on train (no learning)
    dists, margins, interfs, sinrs, rates = [], [], [], [], []
    for step in l0["steps"]:
        dists.extend(step["d_near"].tolist())
        margins.extend((step["d_near"] - D_SAFE_M).tolist())
        interfs.extend(step["interf"].tolist())
        sinrs.extend(step["sinr"].tolist())
        rates.extend(step["rate"].tolist())
    for seed in train_seeds:
        refuse_untouched(int(seed))
        worker_l0.reset(int(seed))
        pos = worker_l0.env.uav_positions
        from wt.lowlevel_v2.contracts import pairwise_distances

        dmat = pairwise_distances(pos)
        for i in range(6):
            row = np.delete(dmat[i], i)
            dn = float(np.min(row))
            dists.append(dn)
            margins.append(dn - D_SAFE_M)
    d_arr = np.asarray(dists)
    near_frac = float(np.mean(d_arr < 1.5 * D_SAFE_M))
    interf_arr = np.asarray(interfs)
    high_thr = float(np.quantile(interf_arr, 0.75)) if interf_arr.size else float("nan")
    high_frac = float(np.mean(interf_arr >= high_thr)) if interf_arr.size else 0.0
    joint = 0.0
    n_joint = 0
    for step in l0["steps"]:
        for uav in range(6):
            n_joint += 1
            if step["d_near"][uav] < 1.5 * D_SAFE_M and step["interf"][uav] >= high_thr:
                joint += 1
    joint_frac = joint / max(n_joint, 1)
    signal = {
        "nearest_uav_distance": _quantiles(d_arr),
        "d_safe_margin": _quantiles(np.asarray(margins)),
        "interference": _quantiles(interf_arr),
        "sinr": _quantiles(np.asarray(sinrs)),
        "rate": _quantiles(np.asarray(rates)),
        "near_neighbor_fraction_d_lt_1.5ds": near_frac,
        "high_interference_fraction_p75": high_frac,
        "near_and_high_interf_fraction": joint_frac,
        "feature_signal_weak": bool(joint_frac < 0.02),
        "interference_median_equals_noise_floor": bool(abs(float(np.median(interf_arr)) - 0.001) < 1e-6) if interf_arr.size else True,
        "d_safe_m": D_SAFE_M,
        "includes_train_initials": True,
        "includes_design_l0_rollout": True,
        "trained": False,
    }
    ep_tables = []
    for name, bundle in [("L0", l0), ("u20", bundles.get("u20")), ("u80", bundles.get("u80")), ("u100", bundles.get("u100"))]:
        if bundle is None:
            continue
        for ep in bundle["episodes"]:
            if ep["seed"] in traj_ids:
                ep_tables.append({"method": name, **ep})
    plot_map = {"L0": l0}
    for key in ("u20", "u80", "u100"):
        if key in bundles:
            plot_map[key] = bundles[key]
    plot_trajectories(plot_map, buoys, OUT / "figures")
    # mechanism
    u100p = next(r for r in proj_rows if r["checkpoint"] == "u100")
    u100r = next(r for r in residual_rows if r["checkpoint"] == "u100")
    u20r = next(r for r in residual_rows if r["checkpoint"] == "u20")
    u80r = next((r for r in residual_rows if r["checkpoint"] == "u80"), None)
    svc100 = next(r for r in svc_rows if r["checkpoint"] == "u100")
    drift_u80 = next((r for r in drift_rows if r.get("pair") == "u80->u100"), {})
    neigh_near = [r for r in neigh_rows if r["checkpoint"] == "u100" and r["regime"] == "near" and r["variant"] == "shuffle"]
    comm_high = [r for r in comm_rows if r["checkpoint"] in {"u20", "u40", "u80", "u100"} and r["regime"] == "high"]
    sat_growth = u100r["tanh_saturation_fraction"] > 0.2 or u100r["mean_residual_norm"] > 2.0 * u20r["mean_residual_norm"]
    opt_spike = False
    if spikes.get("80-100", {}).get("kl") and spikes.get("80-100", {}).get("kl_prev"):
        opt_spike = spikes["80-100"]["kl"] > 2.0 * max(spikes["80-100"]["kl_prev"], 1e-6)
    critic_bad = any(r["checkpoint"] == "u100" and (not np.isfinite(r["pearson"]) or r["pearson"] < 0.2) for r in critic_rows)
    rel_unused = bool(neigh_near and neigh_near[0]["mean_action_l2"] < 1e-4)
    service_break = svc100["residual_nonzero_fraction"] > 0.3 and svc100["envelope_intervention_rate"] > 0.1
    envelope_share = float(u100p["SERVICE_SAFE_ENVELOPE_among_intervened"])
    primary, secondary = "NONE", "NONE"
    if service_break and envelope_share > 0.5:
        primary, secondary = "SERVICE", "ACTION" if sat_growth else "SHIELD"
    elif sat_growth:
        primary, secondary = "ACTION", "SERVICE" if service_break else "OPT"
    elif opt_spike:
        primary, secondary = "OPT", "ACTION"
    elif rel_unused and not signal["feature_signal_weak"]:
        primary, secondary = "REL", "OPT"
    elif signal["feature_signal_weak"]:
        primary, secondary = "SIGNAL", "REL"
    elif critic_bad:
        primary, secondary = "CRITIC", "OPT"
    mech = "C"
    if primary != "NONE" and secondary != "NONE":
        mech = "B"
    if primary != "NONE" and secondary in {"NONE", primary}:
        mech = "A"
    if primary == "NONE":
        mech = "C"
    change = {
        "OPT": "stabilize PPO updates only (KL/value clipping or update-age), no lr/reward/alpha/network combo",
        "ACTION": "add a residual trust-region / saturation cap on tanh residual, no other changes",
        "SERVICE": "zero residual inside active in-range service (keep L0 gate semantics for those UAVs)",
        "REL": "increase neighbor-feature gradient use only (e.g. input dropout off / residual on neighbor branch), not a new encoder",
        "SIGNAL": "rebalance train scenario geometry toward nearer UAV pairs, no network change",
        "CRITIC": "freeze actor briefly and re-fit critic only (not executed this phase)",
        "SHIELD": "do not weaken d_safe; inspect projector only",
        "REWARD": "do not change lambdas this follow-up; audit only",
        "NONE": "no justified single change; stop",
    }[primary]
    gate = {
        "MECH": mech,
        "primary_mechanism": primary,
        "secondary_mechanism": secondary,
        "minimal_change": change,
        "trained": False,
        "multiseed": False,
        "u100_projector_dominant": max(
            (
                ("PAIRWISE_COLLISION", u100p["PAIRWISE_COLLISION_among_intervened"]),
                ("SERVICE_SAFE_ENVELOPE", u100p["SERVICE_SAFE_ENVELOPE_among_intervened"]),
                ("STEP_LIMIT", u100p["STEP_LIMIT_among_intervened"]),
                ("MULTI_CONSTRAINT", u100p["MULTI_CONSTRAINT_among_intervened"]),
                ("OTHER", u100p["OTHER_among_intervened"]),
            ),
            key=lambda x: x[1],
        )[0],
        "residual_expanded": bool(sat_growth),
        "neighbor_near_unused": rel_unused,
        "feature_signal_weak": signal["feature_signal_weak"],
        "opt_kl_spike_80_100": bool(opt_spike),
        "large_change_threshold": LARGE_CHANGE,
        "u80_u100_drift": drift_u80,
        "inventory": inv,
    }
    _csv(OUT / "LLV2_PHASE1A_RESIDUAL_STATS.csv", residual_rows)
    _csv(OUT / "LLV2_PHASE1A_PROJECTOR_BREAKDOWN.csv", proj_rows)
    _csv(OUT / "LLV2_PHASE1A_SERVICE_BEHAVIOR.csv", svc_rows)
    _csv(OUT / "LLV2_PHASE1A_NEIGHBOR_UTILIZATION.csv", neigh_rows)
    _csv(OUT / "LLV2_PHASE1A_COMM_UTILIZATION.csv", comm_rows)
    _dump(OUT / "LLV2_PHASE1A_STATE_SIGNAL_AUDIT.json", signal)
    _csv(OUT / "LLV2_PHASE1A_REWARD_SCALE.csv", reward_rows)
    _csv(OUT / "LLV2_PHASE1A_OPTIMIZATION.csv", opt_rows)
    _csv(OUT / "LLV2_PHASE1A_POLICY_DRIFT.csv", drift_rows)
    _csv(OUT / "LLV2_PHASE1A_CRITIC_CALIBRATION.csv", critic_rows)
    _csv(OUT / "LLV2_PHASE1A_SEP_RESPONSE.csv", sep_rows)
    _csv(OUT / "LLV2_PHASE1A_COMM_DIRECTION.csv", high_interf_rows)
    _csv(OUT / "LLV2_PHASE1A_TRAJ_SCENARIO_METRICS.csv", ep_tables)
    _dump(OUT / "LLV2_PHASE1A_OPT_SPIKES.json", spikes)
    _dump(OUT / "LLV2_PHASE1A_MECHANISM_GATE.json", gate)
    names = [p.name for p in OUT.iterdir() if p.is_file()]
    for src in OUT.glob("LLV2_PHASE1A_*"):
        (PHASE1A / src.name).write_bytes(src.read_bytes())
        (ROOT / src.name).write_bytes(src.read_bytes())
    return {"n_ckpt": len(present), "files": names, "gate": gate}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
