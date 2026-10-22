"""Design-set diagnostics for Phase-2B. Never loads Phase-2 validation seeds."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from wt.lowlevel_v2.observation import communication_features
from wt.lowlevel_v2.phase1a.diagnose import desired_constraint_flags, intervention_label
from wt.lowlevel_v2.phase1b.controller import ResidualWorker
from wt.lowlevel_v2.phase1b.protocol import D_SAFE_M
from wt.lowlevel_v2.phase2b.protocol import refuse_phase2_validation


def evaluate_design(worker: ResidualWorker, seeds: list[int], *, zero_residual: bool) -> dict:
    per = []
    labels = defaultdict(int)
    n_int = 0
    n_steps = 0
    feat = defaultdict(list)
    rng = np.random.default_rng(22101)
    for seed in seeds:
        refuse_phase2_validation(int(seed))
        info = worker.reset(int(seed))
        payload0 = float(worker.env.payload_bits)
        flight = 0.0
        coll = 0
        fail = 0
        viol = 0.0
        inter_n = 0
        mins = []
        raw_n, used_n = [], []
        cap_n, freeze_n, base_n = [], [], []
        fwd, lat, rev = [], [], []
        arrive, never = [], 0
        option_open: dict[int, dict] = {}
        time_in = 0
        idle_n = 0
        n_opt = 0
        switches = 0
        assigns = []
        tgt_sw = 0
        inrange_sinr, inrange_rate, unfinished = [], [], []
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
            flight += float(np.sum(np.linalg.norm(np.asarray(worker.env.uav_positions) - pos0, axis=-1)))
            raw = rec.r_raw if rec.r_raw is not None else rec.delta_used
            raw_n.extend(np.linalg.norm(np.asarray(raw), axis=-1).tolist())
            used_n.extend(np.linalg.norm(rec.delta_used, axis=-1).tolist())
            cap_n.extend(np.asarray(rec.cap_active, dtype=float).tolist())
            freeze_n.extend(np.asarray(rec.base_zero_lock, dtype=float).tolist())
            base_n.extend(np.asarray(rec.base_norm, dtype=float).tolist())
            for u in range(6):
                bn = float(np.linalg.norm(rec.a_base[u]))
                r = rec.delta_used[u]
                if bn >= 1e-6:
                    unit = rec.a_base[u] / bn
                    f = float(np.dot(r, unit))
                    fwd.append(f)
                    lat.append(float(np.linalg.norm(r - f * unit)))
                    rev.append(float(f < -1e-6))
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
            if not zero_residual and not rec.preexisting_safety_violation and step_n % 4 == 0:
                d_near = np.asarray(rec.obs["d_nearest"])
                interf = np.asarray(comm["total_interference"])
                packed = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in rec.obs.items()}
                neigh = packed.copy()
                neigh["neighbor_set"] = neigh["neighbor_set"][:, rng.permutation(5)]
                inters = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in rec.obs.items()}
                inters["communication"] = np.array(inters["communication"], copy=True)
                inters["communication"][:, 2] = rng.permutation(inters["communication"][:, 2])
                sinr_s = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in rec.obs.items()}
                sinr_s["communication"] = np.array(sinr_s["communication"], copy=True)
                sinr_s["communication"][:, 0] = rng.permutation(sinr_s["communication"][:, 0])
                dq = np.quantile(d_near, 1 / 3)
                iq = np.quantile(interf, 2 / 3)
                with torch.no_grad():
                    for name, alt, cond in (
                        ("neighbor_shuffle", neigh, d_near <= dq),
                        ("interf_shuffle", inters, interf > iq),
                        ("sinr_shuffle", sinr_s, interf > iq),
                    ):
                        pre_s, _, _ = worker.act_tensors(alt, deterministic=True)
                        diff = np.linalg.norm(pre_s.cpu().numpy() - rec.pre_tanh, axis=-1)
                        feat[name].extend(diff[cond].tolist())
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
                "projector_intervention_rate": inter_n / max(step_n, 1),
                "projector_failure": float(fail),
                "service_lock_violation": float(viol),
                "mean_raw_residual": float(np.mean(raw_n) if raw_n else 0.0),
                "mean_used_residual": float(np.mean(used_n) if used_n else 0.0),
                "median_raw_residual": float(np.median(raw_n) if raw_n else 0.0),
                "median_used_residual": float(np.median(used_n) if used_n else 0.0),
                "p95_raw_residual": float(np.quantile(raw_n, 0.95) if raw_n else 0.0),
                "p95_used_residual": float(np.quantile(used_n, 0.95) if used_n else 0.0),
                "cap_activation_rate": float(np.mean(cap_n) if cap_n else 0.0),
                "base_zero_freeze_rate": float(np.mean(freeze_n) if freeze_n else 0.0),
                "mean_base_norm": float(np.mean(base_n) if base_n else 0.0),
                "mean_forward": float(np.mean(fwd) if fwd else 0.0),
                "mean_lateral": float(np.mean(lat) if lat else 0.0),
                "reverse_fraction": float(np.mean(rev) if rev else 0.0),
                "mean_arrival_steps": float(np.nanmean(arr)),
                "median_arrival_steps": float(np.nanmedian(arr)),
                "p90_arrival_steps": float(np.nanquantile(arr, 0.9)) if np.isfinite(arr).any() else float("nan"),
                "never_reach_count": float(never),
                "n_option_windows": float(never + len(arrive)),
                "never_reach_fraction": float(never / max(never + len(arrive), 1)),
                "time_in_range": float(time_in),
                "idle_fraction": idle_n / max(6 * max(step_n, 1), 1),
                "high_level_decision_count": float(n_opt),
                "assignment_change_count": float(switches),
                "target_switch_count": float(tgt_sw),
                "mean_active_sinr": float(np.mean(inrange_sinr) if inrange_sinr else float("nan")),
                "mean_active_rate": float(np.mean(inrange_rate) if inrange_rate else float("nan")),
            }
        )

    def mean(key):
        vals = [r[key] for r in per]
        return float(np.nanmean(vals)) if vals else float("nan")

    among = {k: labels[k] / max(n_int, 1) for k in ("STEP_LIMIT", "PAIRWISE_COLLISION", "SERVICE_SAFE_ENVELOPE", "MULTI_CONSTRAINT", "OTHER")}
    return {
        "payload": mean("payload"),
        "completed_blocks": mean("completed_blocks"),
        "residual_queue": mean("residual_queue"),
        "drop": mean("drop"),
        "flight_distance": mean("flight_distance"),
        "unfinished_progress": mean("unfinished_progress"),
        "min_sep": mean("min_sep"),
        "min_sep_worst": float(min(r["min_sep"] for r in per)) if per else float("inf"),
        "collision_count": float(sum(r["collision_count"] for r in per)),
        "projector_intervention_rate": mean("projector_intervention_rate"),
        "projector_failure_count": float(sum(r["projector_failure"] for r in per)),
        "service_lock_violation_count": float(sum(r["service_lock_violation"] for r in per)),
        "mean_raw_residual": mean("mean_raw_residual"),
        "mean_used_residual": mean("mean_used_residual"),
        "median_raw_residual": mean("median_raw_residual"),
        "median_used_residual": mean("median_used_residual"),
        "p95_raw_residual": mean("p95_raw_residual"),
        "p95_used_residual": mean("p95_used_residual"),
        "cap_activation_rate": mean("cap_activation_rate"),
        "base_zero_freeze_rate": mean("base_zero_freeze_rate"),
        "mean_base_norm": mean("mean_base_norm"),
        "mean_forward": mean("mean_forward"),
        "mean_lateral": mean("mean_lateral"),
        "reverse_fraction": mean("reverse_fraction"),
        "mean_arrival_steps": mean("mean_arrival_steps"),
        "median_arrival_steps": mean("median_arrival_steps"),
        "p90_arrival_steps": mean("p90_arrival_steps"),
        "never_reach_count": mean("never_reach_count"),
        "never_reach_fraction": mean("never_reach_fraction"),
        "time_in_range": mean("time_in_range"),
        "idle_fraction": mean("idle_fraction"),
        "high_level_decision_count": mean("high_level_decision_count"),
        "assignment_change_count": mean("assignment_change_count"),
        "target_switch_count": mean("target_switch_count"),
        "mean_active_sinr": mean("mean_active_sinr"),
        "mean_active_rate": mean("mean_active_rate"),
        "n": len(seeds),
        "n_steps": n_steps,
        "n_intervened": n_int,
        **{f"{k}_among_intervened": v for k, v in among.items()},
        "neighbor_shuffle": float(np.mean(feat["neighbor_shuffle"])) if feat["neighbor_shuffle"] else float("nan"),
        "interf_shuffle": float(np.mean(feat["interf_shuffle"])) if feat["interf_shuffle"] else float("nan"),
        "sinr_shuffle": float(np.mean(feat["sinr_shuffle"])) if feat["sinr_shuffle"] else float("nan"),
        "scenarios": per,
    }


__all__ = ["evaluate_design"]
