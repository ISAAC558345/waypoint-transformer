"""Offline Phase-3 failure diagnosis. Reads frozen artifacts only. No training."""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
import torch

from wt.htw_phase943.order_free_decoder import OrderFreeAssignmentModel
from wt.htw_phase943.set_context import make_snap
from wt.htw_phase943.training import _encode_one
from wt.htw_phase945.utility_loss import candidate_logprobs
from wt.htw_phase945.utility_targets import soft_targets
from wt.lowlevel_v2.integration_phase1.stats import kendall_tau, spearman

from .dataset import load_exec_split
from .exec_loss import Q_EPS, executor_targets
from .protocol import MODEL_SEEDS, OUTPUT, PHASE2B_OUTPUT, PHASE2D_OUTPUT, TRAIN_NAME, VAL_NAME, exec_ckpt, prefix_ckpt


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _dump(path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n", encoding="utf-8")


def _summ(xs) -> dict:
    a = np.asarray(list(xs), dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {"n": int(a.size), "mean": float(np.mean(a)), "median": float(np.median(a)), "p90": float(np.quantile(a, 0.90))}


def entropy(p) -> float:
    q = np.asarray(p, dtype=np.float64)
    q = q[q > 0]
    return float(-np.sum(q * np.log(q)))


def ndcg(u, score) -> float:
    u = np.asarray(u, dtype=np.float64)
    lo, hi = float(np.min(u)), float(np.max(u))
    rel = np.zeros_like(u) if hi <= lo else (u - lo) / (hi - lo)
    order = np.argsort(-np.asarray(score, dtype=np.float64))
    denom = np.log2(np.arange(2, rel.size + 2))
    gains = rel[order] / denom
    ideal = np.sort(rel)[::-1] / denom
    total = float(np.sum(ideal))
    if total <= 0:
        return float("nan")
    return float(np.sum(gains) / total)


def q_stats(utilities: np.ndarray) -> dict:
    raw = np.asarray(utilities, dtype=np.float64)
    q = executor_targets(raw)
    q_noshift = raw / max(float(np.sum(np.clip(raw, 0.0, None))), 1.0e-12)
    q_soft = soft_targets(raw, 1.0)
    shifted = bool(np.any(raw <= 0.0))
    rank_raw = np.argsort(np.argsort(-raw))
    rank_q = np.argsort(np.argsort(-q))
    rank_soft = np.argsort(np.argsort(-q_soft))
    return {
        "shifted": shifted,
        "q_entropy": entropy(q),
        "n_eff": float(np.exp(entropy(q))),
        "q_max": float(np.max(q)),
        "q_support_gt_0.05": int(np.sum(q > 0.05)),
        "q_support_gt_0.01": int(np.sum(q > 0.01)),
        "rank_unchanged": bool(np.array_equal(rank_raw, rank_q)),
        "soft_rank_unchanged": bool(np.array_equal(rank_raw, rank_soft)),
        "q_noshift_entropy": entropy(q_noshift),
        "q_soft_entropy": entropy(q_soft),
        "n_eff_soft": float(np.exp(entropy(q_soft))),
        "q_soft_max": float(np.max(q_soft)),
        "max_q_vs_noshift": float(np.max(np.abs(q - q_noshift))),
        "max_q_vs_soft": float(np.max(np.abs(q - q_soft))),
        "l1_q_vs_soft": float(np.sum(np.abs(q - q_soft))),
    }


def utility_sparsity(u: np.ndarray) -> dict:
    flat = np.asarray(u, dtype=np.float64).reshape(-1)
    pos = flat[flat > 0]
    total = float(np.sum(np.clip(flat, 0, None)))
    top = np.sort(flat)[::-1]
    k = max(1, int(np.ceil(0.05 * flat.size)))
    ranges = []
    cvs = []
    for row in np.asarray(u, dtype=np.float64):
        ranges.append(float(np.max(row) - np.min(row)))
        m = float(np.mean(row))
        cvs.append(float(np.std(row) / m) if abs(m) > 1e-12 else float("nan"))
    return {
        "n": int(flat.size),
        "zero_frac": float(np.mean(flat == 0.0)),
        "nonpositive_frac": float(np.mean(flat <= 0.0)),
        "near_zero_frac": float(np.mean(np.abs(flat) < 1.0)),
        "top5_share": float(np.sum(top[:k]) / total) if total > 0 else float("nan"),
        "range_mean": float(np.mean(ranges)),
        "cv_mean": float(np.nanmean(cvs)),
    }


def load_high(path):
    model = OrderFreeAssignmentModel("deepsets")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def align_one(model, data, index) -> dict:
    encoded = _encode_one(model, data, index, "cpu")
    snap = make_snap(data["uav_positions"][index], data["buoy_positions"][index])
    u = np.asarray(data["utility"][index], dtype=np.float64)
    with torch.no_grad():
        lp = candidate_logprobs(model, encoded, snap, data["candidates"][index], data["buoy_mask"][index]).detach().cpu().numpy()
    n = u.size
    iu = int(np.argmax(u))
    ip = int(np.argmax(lp))
    top_u = set(np.argsort(-u)[:3].tolist())
    top_p = set(np.argsort(-lp)[:3].tolist())
    k25 = max(1, int(np.ceil(0.25 * n)))
    set_u = set(np.argsort(-u)[:k25].tolist())
    set_p = set(np.argsort(-lp)[:k25].tolist())
    best = float(np.max(u))
    regret = (best - float(u[ip])) / max(abs(best), 1.0)
    return {
        "top1_agree": float(iu == ip),
        "top3_overlap": float(len(top_u & top_p) / 3.0),
        "top25_overlap": float(len(set_u & set_p) / float(k25)),
        "spearman": spearman(u, lp),
        "kendall": kendall_tau(u, lp),
        "ndcg": ndcg(u, lp),
        "regret": regret,
        "U_argmax": float(u[iu]),
        "U_pmax": float(u[ip]),
    }


def _truthy(value) -> bool:
    return str(value) in {"True", "true", "1"}


def decode_audit(path, family: str | None = None) -> dict:
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    by = defaultdict(list)
    for r in rows:
        fam = r.get("family") or r.get("model")
        if family is not None and fam != family:
            continue
        by[f"{fam}:{r.get('high_seed')}"].append(r)
    out = {}
    for key, items in by.items():
        nrs = [float(x["DecodeNR"]) for x in items]
        inp = np.array([_truthy(x.get("in_pool", False)) for x in items], dtype=bool)
        mass = [float(x["PoolMass"]) for x in items if x.get("PoolMass") not in {None, ""}]
        udec = np.array([float(x["U_decode"]) for x in items], dtype=np.float64)
        ubest = np.array([float(x["U_best_pool"]) for x in items], dtype=np.float64)
        out[key] = {
            "DecodeNR": _summ(nrs),
            "in_pool_rate": float(np.mean(inp)),
            "PoolMass": _summ(mass),
            "U_decode": _summ(udec),
            "U_best_pool": _summ(ubest),
            "DecodeNR_in_pool": _summ(np.array(nrs)[inp]) if np.any(inp) else _summ([]),
            "DecodeNR_ood": _summ(np.array(nrs)[~inp]) if np.any(~inp) else _summ([]),
            "n": len(items),
            "n_negative_DecodeNR": int(np.sum(np.array(nrs) < 0)),
        }
    return out


def closed_loop_audit(path) -> dict:
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    by = defaultdict(list)
    for r in rows:
        by[r.get("family") or r.get("high")].append(r)
    out = {}
    for fam, items in by.items():
        out[str(fam)] = {
            "payload": _summ([float(x["payload"]) for x in items]),
            "decisions": _summ([float(x["high_level_decision_count"]) for x in items]),
            "assignment_changes": _summ([float(x["assignment_change_count"]) for x in items]),
            "idle_fraction": _summ([float(x["idle_fraction"]) for x in items]),
            "n_steps": _summ([float(x["n_steps"]) for x in items]),
            "n": len(items),
        }
    return out


def hash_variance(data) -> dict:
    buckets = defaultdict(list)
    for i, hashes in enumerate(data["hashes"]):
        u = np.asarray(data["utility"][i], dtype=np.float64)
        meta = data["meta"][i]
        for h, val in zip(hashes, u):
            buckets[str(h)].append((float(val), int(meta["scenario"]), int(meta["t"])))
    reused = {h: v for h, v in buckets.items() if len(v) >= 2}
    within = []
    within_same_scenario = []
    for vals in reused.values():
        arr = np.array([x[0] for x in vals], dtype=np.float64)
        within.append(float(np.var(arr, ddof=1)) if arr.size > 1 else 0.0)
        by_sc = defaultdict(list)
        for val, sc, _t in vals:
            by_sc[sc].append(val)
        for group in by_sc.values():
            if len(group) >= 2:
                within_same_scenario.append(float(np.var(np.asarray(group, dtype=np.float64), ddof=1)))
    all_u = np.asarray(data["utility"], dtype=np.float64)
    state_means = np.mean(all_u, axis=1)
    state_vars = np.var(all_u, axis=1, ddof=1)
    total = float(np.var(all_u, ddof=1))
    between_state = float(np.var(state_means, ddof=1))
    within_state = float(np.mean(state_vars))
    return {
        "unique_hashes": len(buckets),
        "reused_hashes": len(reused),
        "reuse_frac": float(len(reused) / max(len(buckets), 1)),
        "n_reused_observations": int(sum(len(v) for v in reused.values())),
        "within_reused_var_mean": float(np.mean(within)) if within else float("nan"),
        "within_reused_var_median": float(np.median(within)) if within else float("nan"),
        "within_same_scenario_var_mean": float(np.mean(within_same_scenario)) if within_same_scenario else float("nan"),
        "between_state_meanU_var": between_state,
        "within_state_var_mean": within_state,
        "global_var": total,
        "between_state_share": float(between_state / total) if total > 0 else float("nan"),
        "within_state_share": float(within_state / total) if total > 0 else float("nan"),
        "note": "same assignment hash across different (scenario,t) snapshots; candidate==full 6-UAV assignment",
    }


def run() -> dict:
    train = load_exec_split(TRAIN_NAME)
    val = load_exec_split(VAL_NAME)
    u_all = np.concatenate([train["utility"], val["utility"]], axis=0)
    q_rows = [q_stats(row) for row in u_all]
    align = {}
    for seed in MODEL_SEEDS:
        model = load_high(exec_ckpt(int(seed)))
        rows = [align_one(model, val, i) for i in range(int(val["assignment"].shape[0]))]
        align[f"EXEC:{seed}"] = {
            "top1_agree": _summ([r["top1_agree"] for r in rows]),
            "top3_overlap": _summ([r["top3_overlap"] for r in rows]),
            "top25_overlap": _summ([r["top25_overlap"] for r in rows]),
            "spearman": _summ([r["spearman"] for r in rows]),
            "kendall": _summ([r["kendall"] for r in rows]),
            "ndcg": _summ([r["ndcg"] for r in rows]),
            "regret": _summ([r["regret"] for r in rows]),
        }
        del model
    for seed in (0, 1):
        model = load_high(prefix_ckpt(int(seed)))
        rows = [align_one(model, val, i) for i in range(int(val["assignment"].shape[0]))]
        align[f"PREFIX:{seed}"] = {
            "top1_agree": _summ([r["top1_agree"] for r in rows]),
            "top3_overlap": _summ([r["top3_overlap"] for r in rows]),
            "top25_overlap": _summ([r["top25_overlap"] for r in rows]),
            "spearman": _summ([r["spearman"] for r in rows]),
            "kendall": _summ([r["kendall"] for r in rows]),
            "ndcg": _summ([r["ndcg"] for r in rows]),
            "regret": _summ([r["regret"] for r in rows]),
        }
        del model
    hist_rows = list(csv.DictReader((OUTPUT / "HL_EXEC_TRAINING_HISTORY.csv").open(encoding="utf-8", newline="")))
    val_hist = [r for r in hist_rows if r.get("val_spearman")]
    payload = {
        "sparsity": utility_sparsity(u_all),
        "q": {
            "shift_rate": float(np.mean([r["shifted"] for r in q_rows])),
            "entropy": _summ([r["q_entropy"] for r in q_rows]),
            "entropy_noshift": _summ([r["q_noshift_entropy"] for r in q_rows]),
            "entropy_soft": _summ([r["q_soft_entropy"] for r in q_rows]),
            "n_eff": _summ([r["n_eff"] for r in q_rows]),
            "n_eff_soft": _summ([r["n_eff_soft"] for r in q_rows]),
            "q_max": _summ([r["q_max"] for r in q_rows]),
            "q_soft_max": _summ([r["q_soft_max"] for r in q_rows]),
            "support_gt_0.05": _summ([r["q_support_gt_0.05"] for r in q_rows]),
            "rank_unchanged_rate": float(np.mean([r["rank_unchanged"] for r in q_rows])),
            "soft_rank_unchanged_rate": float(np.mean([r["soft_rank_unchanged"] for r in q_rows])),
            "l1_q_vs_soft": _summ([r["l1_q_vs_soft"] for r in q_rows]),
        },
        "hash_variance_train": hash_variance(train),
        "hash_variance_val": hash_variance(val),
        "within_pool_align": align,
        "decode_exec": decode_audit(OUTPUT / "HL_EXEC_DECODE_UTILITY.csv"),
        "decode_prefix_file": decode_audit(PHASE2D_OUTPUT / "HL_PREFIX_DECODE_UTILITY.csv"),
        "closed_loop": closed_loop_audit(OUTPUT / "HL_EXEC_CLOSED_LOOP_DESIGN.csv"),
        "train_seed4": {
            "L_exec_ep1": float(hist_rows[0]["L_exec"]),
            "L_exec_ep80": float(hist_rows[-1]["L_exec"]),
            "val_spearman_ep10": float(next(r["val_spearman"] for r in val_hist if int(r["epoch"]) == 10)),
            "val_spearman_ep70": float(next(r["val_spearman"] for r in val_hist if int(r["epoch"]) == 70)),
            "val_spearman_ep80": float(next(r["val_spearman"] for r in val_hist if int(r["epoch"]) == 80)),
            "val_regret_ep10": float(next(r["val_top1_regret"] for r in val_hist if int(r["epoch"]) == 10)),
            "val_regret_ep70": float(next(r["val_top1_regret"] for r in val_hist if int(r["epoch"]) == 70)),
            "val_nom5_ep10": float(next(r["val_nom5"] for r in val_hist if int(r["epoch"]) == 10)),
            "val_nom5_ep70": float(next(r["val_nom5"] for r in val_hist if int(r["epoch"]) == 70)),
        },
        "code_evidence": {
            "U_LL": "shadow_h15 payload_increment: env.payload_bits after HOLD_STEPS with frozen assignment",
            "closed_loop": "evaluate_design: full episode payload_bits from reset to horizon; CompatWorker frozen_assignment is None so decide() re-decodes every hold",
            "L_exec": "-sum q_i log P_set(Gamma_i|s) absolute set logP, not log_softmax over the 24-pool",
            "official_decode": "joint greedy pick_candidate / greedy_trace, not argmax over labeled pool",
        },
    }
    _dump(OUTPUT / "HL_EXEC_FAILURE_DIAGNOSIS.json", payload)
    return payload


if __name__ == "__main__":
    out = run()
    print(json.dumps({
        "top1": {k: v["top1_agree"]["mean"] for k, v in out["within_pool_align"].items()},
        "ndcg": {k: v["ndcg"]["mean"] for k, v in out["within_pool_align"].items()},
        "in_pool": {k: v["in_pool_rate"] for k, v in out["decode_exec"].items()},
        "q_shift_rate": out["q"]["shift_rate"],
        "q_n_eff": out["q"]["n_eff"],
    }, indent=2), flush=True)
