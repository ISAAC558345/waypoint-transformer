"""Phase-3 independent-validation episodes. Frozen Phase-2D worker. No method edits."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from wt.lowlevel_v2.observation import communication_features
from wt.lowlevel_v2.phase1a.diagnose import desired_constraint_flags, intervention_label, predicted_next, predict_sinr_rate
from wt.lowlevel_v2.phase1b.controller import ResidualWorker
from wt.lowlevel_v2.phase1b.protocol import D_SAFE_M
from wt.lowlevel_v2.phase3.protocol import refuse_historical


def _tertile(val: float, q1: float, q2: float, names: tuple[str, str, str]) -> str:
    if val <= q1:
        return names[0]
    if val <= q2:
        return names[1]
    return names[2]


class ValWorker(ResidualWorker):
    def reset(self, seed: int):
        refuse_historical(int(seed), allow_phase3_val=True)
        return super().reset(seed)


def evaluate_validation(worker: ResidualWorker, seeds: list[int], *, zero_residual: bool, diagnose: bool) -> dict:
    per = []
    labels = defaultdict(int)
    n_int = 0
    n_steps = 0
    rel_rows = []
    sep_rows = []
    comm_rows = []
    rng = np.random.default_rng(22317)
    for seed in seeds:
        refuse_historical(int(seed), allow_phase3_val=True)
        info = worker.reset(int(seed))
        payload0 = float(worker.env.payload_bits)
        flight = 0.0
        coll = 0
        fail = 0
        viol = 0.0
        inter_n = 0
        near_n = 0
        mins = []
        raw_n, used_n = [], []
        cap_n, freeze_n, lock_n = [], [], []
        arrive, never = [], 0
        option_open: dict[int, dict] = {}
        time_in = 0
        idle_n = 0
        n_opt = 0
        switches = 0
        assigns = []
        inrange_sinr, inrange_rate = [], []
        step_n = 0
        while True:
            pos0 = np.asarray(worker.env.uav_positions, dtype=np.float64)
            pending_before = int(worker.pending)
            info, rec = worker.step(info, deterministic=True, zero_residual=zero_residual)
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
            worker.env.uav_positions = saved
            in_svc = np.asarray(rec.service_lock, dtype=bool)
            intervened = float(rec.intervention_norm) > 1.0e-6
            lab = intervention_label(flags, intervened)
            n_steps += 1
            step_n += 1
            if intervened:
                n_int += 1
                labels[lab] += 1
                inter_n += 1
            fail += int(rec.projector_failure)
            viol += rec.lock_violation
            mins.append(rec.min_sep)
            if rec.min_sep + 1.0e-6 < D_SAFE_M:
                coll += 1
            elif rec.min_sep < 1.5 * D_SAFE_M:
                near_n += 1
            flight += float(np.sum(np.linalg.norm(np.asarray(worker.env.uav_positions) - pos0, axis=-1)))
            raw = rec.r_raw if rec.r_raw is not None else rec.delta_used
            raw_n.extend(np.linalg.norm(np.asarray(raw), axis=-1).tolist())
            used_n.extend(np.linalg.norm(rec.delta_used, axis=-1).tolist())
            cap_n.extend(np.asarray(rec.cap_active, dtype=float).tolist())
            freeze_n.extend(np.asarray(rec.base_zero_lock, dtype=float).tolist())
            lock_n.extend(np.asarray(rec.service_lock, dtype=float).tolist())
            for u in range(6):
                tgt_valid = int(assoc[u]) >= 0
                in_u = bool(in_svc[u])
                if tgt_valid and u not in option_open and not in_u:
                    option_open[u] = {"t0": int(worker.env.t), "entered": False}
                if u in option_open and in_u and not option_open[u]["entered"]:
                    option_open[u]["entered"] = True
                    arrive.append(max(int(worker.env.t) - option_open[u]["t0"], 0))
                if pending_before <= 0 and u in option_open and not option_open[u]["entered"]:
                    never += 1
                    option_open.pop(u, None)
                if in_u:
                    time_in += 1
                    inrange_sinr.append(float(comm["sinr"][u]))
                    inrange_rate.append(float(comm["rate"][u]))
            if diagnose and (not zero_residual) and (not rec.preexisting_safety_violation) and step_n % 4 == 0:
                d_near = np.asarray(rec.obs["d_nearest"], dtype=np.float64)
                interf = np.asarray(comm["total_interference"], dtype=np.float64)
                sinr = np.asarray(comm["sinr"], dtype=np.float64)
                rel_rows.append({"d_near": d_near, "interf": interf, "sinr": sinr, "pre": rec.pre_tanh.copy(), "obs": rec.obs})
                r = rec.a_des - rec.a_base
                for m in range(6):
                    delta = pos0 - pos0[m]
                    dist = np.linalg.norm(delta, axis=-1)
                    dist[m] = np.inf
                    j = int(np.argmin(dist))
                    v = pos0[m] - pos0[j]
                    nrm = float(np.linalg.norm(v))
                    if nrm >= 1e-9:
                        sep_rows.append({"dot": float(np.dot(r[m], v / nrm)), "near": bool(d_near[m] <= float(np.quantile(d_near, 1 / 3)))})
                worker.env.uav_positions = pos0
                try:
                    s0, r0 = predict_sinr_rate(worker.env, rec.a_base)
                    s1, r1 = predict_sinr_rate(worker.env, rec.a_des)
                    nxt0 = predicted_next(worker.env, rec.a_base)
                    nxt1 = predicted_next(worker.env, rec.a_des)
                    saved2 = worker.env.uav_positions.copy()
                    worker.env.uav_positions = nxt0
                    i0 = float(np.mean(communication_features(worker.env)["total_interference"]))
                    worker.env.uav_positions = nxt1
                    i1 = float(np.mean(communication_features(worker.env)["total_interference"]))
                    worker.env.uav_positions = saved2
                finally:
                    worker.env.uav_positions = saved
                comm_rows.append(
                    {
                        "interf": interf,
                        "d_sinr": float(np.mean(s1 - s0)),
                        "d_rate": float(np.mean(r1 - r0)),
                        "d_interf": float(i1 - i0),
                    }
                )
            if rec.done or worker.env.t >= worker.env.config.episode_horizon:
                break
        payload = float(worker.env.payload_bits - payload0)
        arr = np.asarray(arrive, dtype=np.float64) if arrive else np.array([np.nan])
        per.append(
            {
                "scenario": int(seed),
                "payload": payload,
                "completed_blocks": payload / 1.0e6,
                "residual_queue": float(np.sum(worker.env.queue)),
                "drop": float(worker.env.dropped),
                "flight_distance": flight,
                "unfinished_progress": float(np.mean(worker.env.progress)),
                "min_sep": float(min(mins) if mins else float("inf")),
                "collision_count": float(coll),
                "near_events": float(near_n),
                "projector_intervention_rate": inter_n / max(step_n, 1),
                "projector_failure": float(fail),
                "service_lock_violation": float(viol),
                "mean_raw_residual": float(np.mean(raw_n) if raw_n else 0.0),
                "mean_used_residual": float(np.mean(used_n) if used_n else 0.0),
                "cap_activation_rate": float(np.mean(cap_n) if cap_n else 0.0),
                "base_zero_freeze_rate": float(np.mean(freeze_n) if freeze_n else 0.0),
                "service_lock_rate": float(np.mean(lock_n) if lock_n else 0.0),
                "mean_arrival_steps": float(np.nanmean(arr)),
                "never_reach_fraction": float(never / max(never + len(arrive), 1)),
                "time_in_range": float(time_in),
                "idle_fraction": idle_n / max(6 * max(step_n, 1), 1),
                "assignment_change_count": float(switches),
                "target_switch_count": 0.0,
                "mean_active_sinr": float(np.mean(inrange_sinr) if inrange_sinr else float("nan")),
                "mean_active_rate": float(np.mean(inrange_rate) if inrange_rate else float("nan")),
            }
        )

    def mean(key):
        return float(np.nanmean([r[key] for r in per])) if per else float("nan")

    among = {k: labels[k] / max(n_int, 1) for k in ("STEP_LIMIT", "PAIRWISE_COLLISION", "SERVICE_SAFE_ENVELOPE", "MULTI_CONSTRAINT", "OTHER")}
    summary = {
        "payload": mean("payload"),
        "completed_blocks": mean("completed_blocks"),
        "residual_queue": mean("residual_queue"),
        "drop": mean("drop"),
        "flight_distance": mean("flight_distance"),
        "unfinished_progress": mean("unfinished_progress"),
        "min_sep": mean("min_sep"),
        "min_sep_worst": float(min(r["min_sep"] for r in per)) if per else float("inf"),
        "collision_count": float(sum(r["collision_count"] for r in per)),
        "near_events": float(sum(r["near_events"] for r in per)),
        "projector_intervention_rate": mean("projector_intervention_rate"),
        "projector_failure_count": float(sum(r["projector_failure"] for r in per)),
        "service_lock_violation_count": float(sum(r["service_lock_violation"] for r in per)),
        "mean_raw_residual": mean("mean_raw_residual"),
        "mean_used_residual": mean("mean_used_residual"),
        "cap_activation_rate": mean("cap_activation_rate"),
        "base_zero_freeze_rate": mean("base_zero_freeze_rate"),
        "service_lock_rate": mean("service_lock_rate"),
        "mean_arrival_steps": mean("mean_arrival_steps"),
        "never_reach_fraction": mean("never_reach_fraction"),
        "time_in_range": mean("time_in_range"),
        "idle_fraction": mean("idle_fraction"),
        "assignment_change_count": mean("assignment_change_count"),
        "target_switch_count": mean("target_switch_count"),
        "mean_active_sinr": mean("mean_active_sinr"),
        "mean_active_rate": mean("mean_active_rate"),
        "n": len(seeds),
        "n_steps": n_steps,
        **{f"{k}_among_intervened": v for k, v in among.items()},
        "scenarios": per,
    }
    if diagnose and rel_rows:
        summary["relation"] = _relation_tables(worker, rel_rows, rng)
        near_dots = [r["dot"] for r in sep_rows if r["near"]]
        all_dots = [r["dot"] for r in sep_rows]
        use = near_dots if near_dots else all_dots
        arr = np.asarray(use, dtype=np.float64)
        summary["separation"] = {
            "mean": float(np.mean(arr)) if arr.size else float("nan"),
            "median": float(np.median(arr)) if arr.size else float("nan"),
            "positive_fraction": float(np.mean(arr > 0)) if arr.size else float("nan"),
            "n": int(arr.size),
        }
        if comm_rows:
            interfs = np.concatenate([c["interf"] for c in comm_rows])
            hi = float(np.quantile(interfs, 2 / 3))
            high = [c for c in comm_rows if float(np.mean(c["interf"])) > hi]
            summary["interference_response"] = {
                "mean_delta_sinr": float(np.mean([c["d_sinr"] for c in high])) if high else float("nan"),
                "mean_delta_rate": float(np.mean([c["d_rate"] for c in high])) if high else float("nan"),
                "mean_delta_interference": float(np.mean([c["d_interf"] for c in high])) if high else float("nan"),
                "n_high": len(high),
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
        ("neighbor_zero", "d_near", dq, ("near", "middle", "far")),
        ("neighbor_shuffle", "d_near", dq, ("near", "middle", "far")),
        ("interf_zero", "interf", iq, ("low", "middle", "high")),
        ("interf_shuffle", "interf", iq, ("low", "middle", "high")),
        ("sinr_zero", "sinr", sq, ("low", "middle", "high")),
        ("sinr_shuffle", "sinr", sq, ("low", "middle", "high")),
    ]
    buckets = {(v[0], g): [] for v in variants for g in (*v[3], "all")}
    for row in rel_rows:
        obs = {k: (np.array(val, copy=True) if isinstance(val, np.ndarray) else val) for k, val in row["obs"].items()}
        packed = {
            "neighbor_zero": dict(obs, neighbor_set=np.zeros_like(obs["neighbor_set"])),
            "neighbor_shuffle": dict(obs, neighbor_set=obs["neighbor_set"][:, rng.permutation(obs["neighbor_set"].shape[1])]),
            "interf_zero": _zero_cols(obs, 2, 4),
            "interf_shuffle": _perm_cols(obs, (2, 3), rng),
            "sinr_zero": _zero_cols(obs, 0, 2),
            "sinr_shuffle": _perm_cols(obs, (0, 1), rng),
        }
        base = row["pre"]
        for vname, score_key, qs, names in variants:
            with torch.no_grad():
                pre, _, _ = worker.act_tensors(packed[vname], deterministic=True)
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


__all__ = ["ValWorker", "evaluate_validation"]
