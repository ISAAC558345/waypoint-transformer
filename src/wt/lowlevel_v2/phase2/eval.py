"""Independent-validation episode loop. Does not change Phase-1B worker or lock."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from wt.lowlevel_v2.observation import communication_features
from wt.lowlevel_v2.phase1a.diagnose import desired_constraint_flags, intervention_label, predict_sinr_rate
from wt.lowlevel_v2.phase1b.controller import ResidualWorker
from wt.lowlevel_v2.phase1b.protocol import D_SAFE_M, refuse_untouched


def _tertile(val: float, q1: float, q2: float, names: tuple[str, str, str]) -> str:
    if val <= q1:
        return names[0]
    if val <= q2:
        return names[1]
    return names[2]


def evaluate_validation(worker: ResidualWorker, seeds: list[int], *, zero_residual: bool, diagnose: bool) -> dict:
    per_scen = []
    labels_all = defaultdict(int)
    n_steps = 0
    n_int = 0
    rel_rows = []
    sep_rows = []
    comm_rows = []
    rng = np.random.default_rng(22217)
    for seed in seeds:
        refuse_untouched(int(seed))
        info = worker.reset(int(seed))
        payload0 = float(worker.env.payload_bits)
        flight = 0.0
        inter_n = 0
        fail_n = 0
        coll = 0
        near_n = 0
        mins = []
        res = []
        smooth = []
        prev = None
        sinr_acc = []
        rate_acc = []
        viol = 0.0
        env_n = 0
        step_n = 0
        while True:
            pos0 = np.asarray(worker.env.uav_positions, dtype=np.float64)
            info, rec = worker.step(info, deterministic=True, zero_residual=zero_residual)
            saved = worker.env.uav_positions.copy()
            worker.env.uav_positions = pos0
            flags = desired_constraint_flags(worker.env, worker.option, rec.a_des)
            comm = communication_features(worker.env)
            worker.env.uav_positions = saved
            intervened = float(rec.intervention_norm) > 1.0e-6
            lab = intervention_label(flags, intervened)
            n_steps += 1
            step_n += 1
            if intervened:
                n_int += 1
                labels_all[lab] += 1
                if lab == "SERVICE_SAFE_ENVELOPE":
                    env_n += 1
            if rec.min_sep + 1.0e-6 < D_SAFE_M:
                coll += 1
            elif rec.min_sep < 1.5 * D_SAFE_M:
                near_n += 1
            mins.append(rec.min_sep)
            res.append(float(np.mean(np.linalg.norm(rec.delta_used, axis=-1))))
            viol += rec.lock_violation
            if rec.intervention_norm > 1.0e-6:
                inter_n += 1
            fail_n += int(rec.projector_failure)
            flight += float(np.sum(np.linalg.norm(np.asarray(worker.env.uav_positions) - pos0, axis=-1)))
            if prev is not None:
                smooth.append(float(np.mean(np.linalg.norm(rec.a_exec - prev, axis=-1))))
            prev = rec.a_exec
            sinr_acc.append(float(np.mean(worker.env.last_sinr)))
            rate_acc.append(float(np.mean(worker.env.last_rate)))
            if diagnose and not rec.preexisting_safety_violation and not zero_residual:
                d_near = np.asarray(rec.obs["d_nearest"], dtype=np.float64)
                interf = np.asarray(comm["total_interference"], dtype=np.float64)
                sinr = np.asarray(comm["sinr"], dtype=np.float64)
                rel_rows.append({"d_near": d_near, "interf": interf, "sinr": sinr, "pre": rec.pre_tanh.copy(), "obs": rec.obs})
                r = rec.a_des - rec.a_base
                pos = pos0
                for m in range(6):
                    delta = pos - pos[m]
                    dist = np.linalg.norm(delta, axis=-1)
                    dist[m] = np.inf
                    j = int(np.argmin(dist))
                    v = pos[m] - pos[j]
                    nrm = float(np.linalg.norm(v))
                    if nrm < 1e-9:
                        continue
                    sep_rows.append(float(np.dot(r[m], v / nrm)))
                worker.env.uav_positions = pos0
                try:
                    s0, r0 = predict_sinr_rate(worker.env, rec.a_base)
                    s1, r1 = predict_sinr_rate(worker.env, rec.a_des)
                finally:
                    worker.env.uav_positions = saved
                comm_rows.append({"interf": interf, "d_sinr": float(np.mean(s1 - s0)), "d_rate": float(np.mean(r1 - r0))})
            if rec.done or worker.env.t >= worker.env.config.episode_horizon:
                break
        payload = float(worker.env.payload_bits - payload0)
        per_scen.append(
            {
                "scenario": int(seed),
                "payload": payload,
                "completed_blocks": payload / 1.0e6,
                "residual_queue": float(np.sum(worker.env.queue)),
                "drop": float(worker.env.dropped),
                "flight_distance": flight,
                "mean_sinr": float(np.mean(sinr_acc) if sinr_acc else 0.0),
                "mean_active_rate": float(np.mean(rate_acc) if rate_acc else 0.0),
                "unfinished_progress": float(np.mean(worker.env.progress)),
                "min_sep": float(min(mins) if mins else float("inf")),
                "collision_count": float(coll),
                "near_events": float(near_n),
                "projector_intervention_rate": inter_n / max(step_n, 1),
                "projector_failure": float(fail_n),
                "service_lock_violation": float(viol),
                "mean_residual_norm": float(np.mean(res) if res else 0.0),
                "smoothness": float(np.mean(smooth) if smooth else 0.0),
                "envelope_steps": float(env_n / max(step_n, 1)),
            }
        )
    payloads = np.array([r["payload"] for r in per_scen], dtype=np.float64)

    def mean(key):
        return float(np.mean([r[key] for r in per_scen]))

    among = {k: labels_all[k] / max(n_int, 1) for k in ("STEP_LIMIT", "PAIRWISE_COLLISION", "SERVICE_SAFE_ENVELOPE", "MULTI_CONSTRAINT", "OTHER")}
    summary = {
        "payload": float(np.mean(payloads)),
        "completed_blocks": mean("completed_blocks"),
        "residual_queue": mean("residual_queue"),
        "drop": mean("drop"),
        "flight_distance": mean("flight_distance"),
        "mean_sinr": mean("mean_sinr"),
        "mean_active_rate": mean("mean_active_rate"),
        "unfinished_progress": mean("unfinished_progress"),
        "min_sep": mean("min_sep"),
        "min_sep_worst": float(min(r["min_sep"] for r in per_scen)),
        "collision_count": float(sum(r["collision_count"] for r in per_scen)),
        "near_events": float(sum(r["near_events"] for r in per_scen)),
        "projector_intervention_rate": mean("projector_intervention_rate"),
        "projector_failure_count": float(sum(r["projector_failure"] for r in per_scen)),
        "service_lock_violation_count": float(sum(r["service_lock_violation"] for r in per_scen)),
        "mean_residual_norm": mean("mean_residual_norm"),
        "smoothness": mean("smoothness"),
        "n": len(seeds),
        "n_steps": n_steps,
        "n_intervened": n_int,
        **{f"{k}_among_intervened": v for k, v in among.items()},
        "scenarios": per_scen,
    }
    if diagnose and rel_rows:
        summary["relation"] = _relation_tables(worker, rel_rows, rng)
        arr = np.asarray(sep_rows, dtype=np.float64)
        summary["separation"] = {
            "mean": float(np.mean(arr)) if arr.size else float("nan"),
            "median": float(np.median(arr)) if arr.size else float("nan"),
            "positive_fraction": float(np.mean(arr > 0)) if arr.size else float("nan"),
            "n": int(arr.size),
        }
        interfs = np.concatenate([c["interf"] for c in comm_rows])
        hi = float(np.quantile(interfs, 2 / 3))
        dsinr = [c["d_sinr"] for c in comm_rows if float(np.mean(c["interf"])) > hi]
        drate = [c["d_rate"] for c in comm_rows if float(np.mean(c["interf"])) > hi]
        summary["interference_response"] = {
            "mean_delta_sinr": float(np.mean(dsinr)) if dsinr else float("nan"),
            "mean_delta_rate": float(np.mean(drate)) if drate else float("nan"),
            "n_high": len(dsinr),
        }
    return summary


def _relation_tables(worker: ResidualWorker, rel_rows: list[dict], rng: np.random.Generator) -> list[dict]:
    d_all = np.concatenate([r["d_near"] for r in rel_rows])
    i_all = np.concatenate([r["interf"] for r in rel_rows])
    s_all = np.concatenate([r["sinr"] for r in rel_rows])
    dq = np.quantile(d_all, [1 / 3, 2 / 3])
    iq = np.quantile(i_all, [1 / 3, 2 / 3])
    sq = np.quantile(s_all, [1 / 3, 2 / 3])
    variants = [
        ("neighbor_zero", "neighbor", "d_near", dq, ("near", "middle", "far")),
        ("neighbor_shuffle", "neighbor", "d_near", dq, ("near", "middle", "far")),
        ("interf_zero", "interf", "interf", iq, ("low", "middle", "high")),
        ("interf_shuffle", "interf", "interf", iq, ("low", "middle", "high")),
        ("sinr_zero", "sinr", "sinr", sq, ("low", "middle", "high")),
        ("sinr_shuffle", "sinr", "sinr", sq, ("low", "middle", "high")),
    ]
    buckets = {(v[0], g): [] for v in variants for g in (*v[4], "all")}
    for row in rel_rows:
        obs = {k: (np.array(val, copy=True) if isinstance(val, np.ndarray) else val) for k, val in row["obs"].items()}
        base = row["pre"]
        packed = {
            "neighbor_zero": dict(obs, neighbor_set=np.zeros_like(obs["neighbor_set"])),
            "neighbor_shuffle": dict(obs, neighbor_set=obs["neighbor_set"][:, rng.permutation(obs["neighbor_set"].shape[1])]),
            "interf_zero": _zero_cols(obs, 2, 4),
            "interf_shuffle": _perm_cols(obs, (2, 3), rng),
            "sinr_zero": _zero_cols(obs, 0, 2),
            "sinr_shuffle": _perm_cols(obs, (0, 1), rng),
        }
        for vname, _kind, score_key, qs, names in variants:
            alt_obs = packed[vname]
            with torch.no_grad():
                pre, _, _ = worker.act_tensors(alt_obs, deterministic=True)
            diff = np.linalg.norm(pre.cpu().numpy() - base, axis=-1)
            scores = row[score_key]
            for uav in range(6):
                g = _tertile(float(scores[uav]), float(qs[0]), float(qs[1]), names)
                buckets[(vname, g)].append(float(diff[uav]))
                buckets[(vname, "all")].append(float(diff[uav]))
    out = []
    for (vname, g), arr in buckets.items():
        a = np.asarray(arr, dtype=np.float64)
        out.append({"variant": vname, "regime": g, "mean_l2": float(np.mean(a)) if a.size else float("nan"), "n": int(a.size)})
    return out


def _zero_cols(obs: dict, start: int, end: int) -> dict:
    packed = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
    packed["communication"][:, start:end] = 0.0
    return packed


def _perm_cols(obs: dict, cols: tuple[int, ...], rng: np.random.Generator) -> dict:
    packed = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
    for c in cols:
        packed["communication"][:, c] = rng.permutation(packed["communication"][:, c])
    return packed
