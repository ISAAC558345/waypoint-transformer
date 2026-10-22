"""Design-set mechanism eval for C0/C1. Never touches Phase-3 validation."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from wt.lowlevel_v2.phase1b.controller import ResidualWorker
from wt.lowlevel_v2.phase1b.protocol import D_SAFE_M
from wt.lowlevel_v2.phase4a.protocol import refuse_phase4a


def evaluate_design_mechanics(worker: ResidualWorker, seeds: list[int], *, zero_residual: bool) -> dict:
    per = []
    n_agent = 0
    n_pair = 0
    n_service = 0
    n_multi = 0
    n_step = 0
    n_int = 0
    n_int_env = 0
    n_steps = 0
    sep_near = []
    neigh_rows = []
    rng = np.random.default_rng(22141)
    for seed in seeds:
        refuse_phase4a(int(seed))
        info = worker.reset(int(seed))
        payload0 = float(worker.env.payload_bits)
        coll = 0
        fail = 0
        viol = 0.0
        mins = []
        inter_n = 0
        time_in = 0
        step_n = 0
        lock_n = []
        env_int_n = []
        while True:
            pos0 = np.asarray(worker.env.uav_positions, dtype=np.float64)
            info, rec = worker.step(info, deterministic=True, zero_residual=zero_residual)
            n_steps += 1
            step_n += 1
            fail += int(rec.projector_failure)
            viol += rec.lock_violation
            mins.append(rec.min_sep)
            if rec.min_sep + 1.0e-6 < D_SAFE_M:
                coll += 1
            intervened_env = float(rec.intervention_norm) > 1.0e-6
            if intervened_env:
                n_int_env += 1
                inter_n += 1
            lock = np.asarray(rec.service_lock, dtype=bool)
            lock_n.extend(lock.astype(float).tolist())
            time_in += int(np.sum(lock))
            d_near = np.asarray(rec.obs["d_nearest"], dtype=np.float64)
            r_des = np.asarray(rec.a_des - rec.a_base, dtype=np.float64)
            pair = np.asarray(rec.pairwise_mask) > 0.5
            envl = np.asarray(rec.envelope_mask) > 0.5
            stepb = np.asarray(rec.step_mask) > 0.5
            inter = np.asarray(rec.intervened_mask) > 0.5
            multi = np.asarray(rec.multi_mask) > 0.5
            n_agent += 6
            n_int += int(np.sum(inter))
            n_pair += int(np.sum(inter & pair))
            n_service += int(np.sum(np.asarray(rec.service_only_mask) > 0.5))
            n_multi += int(np.sum(multi))
            n_step += int(np.sum(np.asarray(rec.step_only_mask) > 0.5))
            env_int_n.append(float(intervened_env))
            q1 = float(np.quantile(d_near, 1 / 3))
            for u in range(6):
                if d_near[u] > q1:
                    continue
                delta = pos0 - pos0[u]
                dist = np.linalg.norm(delta, axis=-1)
                dist[u] = np.inf
                j = int(np.argmin(dist))
                v = pos0[u] - pos0[j]
                nrm = float(np.linalg.norm(v))
                if nrm >= 1e-9:
                    sep_near.append(float(np.dot(r_des[u], v / nrm)))
            if (not zero_residual) and (not rec.preexisting_safety_violation) and step_n % 4 == 0:
                packed = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in rec.obs.items()}
                zeroed = dict(packed, neighbor_set=np.zeros_like(packed["neighbor_set"]))
                shuffled = dict(packed, neighbor_set=packed["neighbor_set"][:, rng.permutation(packed["neighbor_set"].shape[1])])
                with torch.no_grad():
                    pre0, _, _ = worker.act_tensors(packed, deterministic=True)
                    prez, _, _ = worker.act_tensors(zeroed, deterministic=True)
                    pres, _, _ = worker.act_tensors(shuffled, deterministic=True)
                d0 = pre0.cpu().numpy()
                for name, pre_alt in (("neighbor_zero", prez.cpu().numpy()), ("neighbor_shuffle", pres.cpu().numpy())):
                    diff = np.linalg.norm(pre_alt - d0, axis=-1)
                    for u in range(6):
                        band = "near" if d_near[u] <= q1 else ("middle" if d_near[u] <= float(np.quantile(d_near, 2 / 3)) else "far")
                        neigh_rows.append({"variant": name, "regime": band, "l2": float(diff[u])})
                        neigh_rows.append({"variant": name, "regime": "all", "l2": float(diff[u])})
            if rec.done or worker.env.t >= worker.env.config.episode_horizon:
                break
        payload = float(worker.env.payload_bits - payload0)
        per.append(
            {
                "scenario": int(seed),
                "payload": payload,
                "completed_blocks": payload / 1.0e6,
                "min_sep": float(min(mins) if mins else float("inf")),
                "collision_count": float(coll),
                "projector_failure": float(fail),
                "service_lock_violation": float(viol),
                "projector_intervention_rate": inter_n / max(step_n, 1),
                "time_in_range": float(time_in),
                "service_lock_rate": float(np.mean(lock_n) if lock_n else 0.0),
            }
        )

    def mean(key):
        return float(np.nanmean([r[key] for r in per])) if per else float("nan")

    arr = np.asarray(sep_near, dtype=np.float64)
    neigh_summary = []
    grouped = defaultdict(list)
    for row in neigh_rows:
        grouped[(row["variant"], row["regime"])].append(row["l2"])
    for (variant, regime), xs in grouped.items():
        neigh_summary.append({"variant": variant, "regime": regime, "mean_l2": float(np.mean(xs)), "n": len(xs)})
    return {
        "payload": mean("payload"),
        "completed_blocks": mean("completed_blocks"),
        "min_sep": mean("min_sep"),
        "min_sep_worst": float(min(r["min_sep"] for r in per)) if per else float("inf"),
        "collision_count": float(sum(r["collision_count"] for r in per)),
        "projector_failure_count": float(sum(r["projector_failure"] for r in per)),
        "service_lock_violation_count": float(sum(r["service_lock_violation"] for r in per)),
        "projector_intervention_rate": mean("projector_intervention_rate"),
        "time_in_range": mean("time_in_range"),
        "service_lock_rate": mean("service_lock_rate"),
        "pairwise_intervention_rate": float(n_pair / max(n_agent, 1)),
        "service_only_intervention_rate": float(n_service / max(n_agent, 1)),
        "multi_intervention_rate": float(n_multi / max(n_agent, 1)),
        "step_only_intervention_rate": float(n_step / max(n_agent, 1)),
        "agent_intervention_rate": float(n_int / max(n_agent, 1)),
        "env_projector_intervention_rate": float(n_int_env / max(n_steps, 1)),
        "n_agent": n_agent,
        "n_steps": n_steps,
        "separation": {
            "mean": float(np.mean(arr)) if arr.size else float("nan"),
            "median": float(np.median(arr)) if arr.size else float("nan"),
            "positive_fraction": float(np.mean(arr > 0)) if arr.size else float("nan"),
            "n": int(arr.size),
        },
        "neighbor": neigh_summary,
        "scenarios": per,
    }


def neighbor_l2(rows: list[dict], variant: str, regime: str) -> float:
    xs = [float(r["mean_l2"]) for r in rows if r["variant"] == variant and r["regime"] == regime]
    return float(np.mean(xs)) if xs else 0.0


__all__ = ["evaluate_design_mechanics", "neighbor_l2"]
