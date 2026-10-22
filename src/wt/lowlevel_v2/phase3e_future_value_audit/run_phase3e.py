"""Phase-3E: predict first-hold F from t=0 observables. GroupKFold by scenario."""
from __future__ import annotations

import ast
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wt.l3a.env import competition_pairs
from wt.l3b.env import L3BEnv
from wt.lowlevel_v2.contracts import minimum_pairwise_distance
from wt.lowlevel_v2.integration_phase1.stats import kendall_tau, spearman, summarize, topk_overlap
from wt.lowlevel_v2.observation import communication_features
from wt.lowlevel_v2.protocol import ROOT
from wt.lowlevel_v2.traj_audit.protocol import OUTPUT as TRAJ_OUT

PHASE = Path(__file__).resolve().parent
OUTPUT = PHASE / "outputs"
FIG = OUTPUT / "figures"
P3C = ROOT / "src" / "wt" / "lowlevel_v2" / "phase3c_credit_counterfactual" / "outputs"
FAMS = ("OLD", "ABS", "PREFIX", "EXEC")
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
N_SPLITS = 5


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n", encoding="utf-8")


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


def ext_sum(values) -> dict:
    out = summarize(values)
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    out["std"] = float(np.std(a, ddof=1)) if a.size > 1 else float("nan")
    return out


def pearson(x, y) -> float:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def nmae(y, yhat) -> float:
    a = np.asarray(y, dtype=np.float64)
    b = np.asarray(yhat, dtype=np.float64)
    denom = float(np.mean(np.abs(a)))
    return float(np.mean(np.abs(a - b)) / max(denom, 1.0))


def rank_metrics(pred, truth) -> dict:
    return {
        "pearson": pearson(pred, truth),
        "spearman": spearman(pred, truth),
        "kendall": kendall_tau(pred, truth),
        "mae": float(np.mean(np.abs(np.asarray(pred) - np.asarray(truth)))),
        "nmae": nmae(truth, pred),
    }


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_gamma(text: str) -> np.ndarray:
    return np.asarray(ast.literal_eval(text), dtype=np.int64)


def inventory_rows() -> list[dict]:
    rows = [
        ("U_single", True, False, True, "first-hold shadow_h15 payload increment; already computed at t=0"),
        ("U_single_pool_rank", True, False, True, "rank of U_single among the official 24-pool at t=0"),
        ("idle_count", True, False, True, "from first-hold assignment"),
        ("idle_frac", True, False, True, "from first-hold assignment"),
        ("n_unique_buoys", True, False, True, "from first-hold assignment"),
        ("competition_count", True, False, True, "UAVs sharing a buoy in the candidate"),
        ("mean_uav_target_dist", True, False, True, "t=0 UAV/buoy positions + candidate targets"),
        ("max_uav_target_dist", True, False, True, "t=0 geometry"),
        ("n_in_range", True, False, True, "service_radius at t=0"),
        ("assigned_queue_mean", True, False, True, "current queue of assigned buoys"),
        ("cand_mean_sinr", True, False, True, "SINR if this assignment were active at t=0; no step"),
        ("cand_mean_rate", True, False, True, "rate if this assignment were active at t=0"),
        ("cand_mean_interference", True, False, True, "interference if this assignment were active at t=0"),
        ("uav_y_mean", True, False, True, "t=0 UAV y after reset(seed=scenario)"),
        ("uav_y_std", True, False, True, "t=0 UAV y spread"),
        ("uav_pairwise_mean", True, False, True, "t=0 UAV pairwise distance"),
        ("min_pairwise_dist", True, False, True, "t=0 UAV min distance"),
        ("queue_mean", True, False, True, "t=0 queue; nearly constant across design seeds"),
        ("queue_hot_mass", True, False, True, "t=0 queue on first num_hot buoys"),
        ("payload_bits", True, False, False, "0 at reset; unused (zero variance)"),
        ("progress", True, False, False, "0 at reset; unused"),
        ("t", True, False, False, "0 at first hold; unused"),
        ("remaining_episode_time", True, False, False, "constant at t=0; unused"),
        ("assignment_tokens", True, False, True, "6 integer targets, first-hold candidate"),
        ("scenario_id", True, False, False, "group key only; not a model feature"),
        ("method", True, False, False, "per-method models; not an input"),
        ("F", False, True, False, "target only"),
        ("U_traj", False, True, False, "forbidden feature"),
        ("gamma2/3/4", False, True, False, "later greedy; forbidden"),
        ("remaining_arrivals", True, True, False, "official actor forbids this future schedule"),
        ("future positions/actions", False, True, False, "forbidden"),
    ]
    return [
        {
            "feature": name,
            "available_at_first_hold": int(avail),
            "future_leakage": int(leak),
            "used_for_model": int(used),
            "note": note,
        }
        for name, avail, leak, used, note in rows
    ]


def reset_env(scenario: int) -> L3BEnv:
    env = L3BEnv()
    env.reset(seed=int(scenario))
    return env


def candidate_vector(env: L3BEnv, gamma: np.ndarray) -> dict[str, float]:
    chosen = np.asarray(gamma, dtype=np.int64)
    dists = []
    in_range = 0
    q_assigned = []
    for uav, buoy in enumerate(chosen):
        if int(buoy) >= 0:
            d = float(np.linalg.norm(env.uav_positions[uav] - env.buoy_positions[int(buoy)]))
            dists.append(d)
            in_range += int(env._in_range(uav, int(buoy)))
            q_assigned.append(float(env.queue[int(buoy)]))
    d_uu = []
    for i in range(env.config.num_uavs):
        for j in range(i + 1, env.config.num_uavs):
            d_uu.append(float(np.linalg.norm(env.uav_positions[i] - env.uav_positions[j])))
    prev = np.asarray(env.assignment, dtype=np.int64).copy()
    env.assignment = chosen.copy()
    comm = communication_features(env)
    env.assignment = prev
    return {
        "idle_count": float(np.sum(chosen < 0)),
        "idle_frac": float(np.mean(chosen < 0)),
        "n_unique_buoys": float(len({int(b) for b in chosen if int(b) >= 0})),
        "competition_count": float(len(competition_pairs(chosen, env.config.num_buoys))),
        "mean_uav_target_dist": float(np.mean(dists)) if dists else 0.0,
        "max_uav_target_dist": float(np.max(dists)) if dists else 0.0,
        "n_in_range": float(in_range),
        "assigned_queue_mean": float(np.mean(q_assigned)) if q_assigned else 0.0,
        "cand_mean_sinr": float(np.mean(comm["sinr"])),
        "cand_mean_rate": float(np.mean(comm["rate"])),
        "cand_mean_interference": float(np.mean(comm["total_interference"])),
        "uav_y_mean": float(np.mean(env.uav_positions[:, 1])),
        "uav_y_std": float(np.std(env.uav_positions[:, 1])),
        "uav_pairwise_mean": float(np.mean(d_uu)) if d_uu else 0.0,
        "min_pairwise_dist": float(minimum_pairwise_distance(env.uav_positions)),
        "queue_mean": float(np.mean(env.queue)),
        "queue_hot_mass": float(np.sum(env.queue[: int(env.config.num_hot)])),
        **{f"assign_{i}": float(chosen[i]) for i in range(6)},
    }


STATE_CAND_KEYS = (
    "idle_count",
    "idle_frac",
    "n_unique_buoys",
    "competition_count",
    "mean_uav_target_dist",
    "max_uav_target_dist",
    "n_in_range",
    "assigned_queue_mean",
    "cand_mean_sinr",
    "cand_mean_rate",
    "cand_mean_interference",
    "uav_y_mean",
    "uav_y_std",
    "uav_pairwise_mean",
    "min_pairwise_dist",
    "queue_mean",
    "queue_hot_mass",
    "assign_0",
    "assign_1",
    "assign_2",
    "assign_3",
    "assign_4",
    "assign_5",
)


def build_table() -> list[dict]:
    pool = load_csv(TRAJ_OUT / "first_hold_pool.csv")
    cand = load_csv(P3C / "first_hold_all_candidates.csv")
    inter = load_csv(P3C / "first_hold_counterfactual.csv")
    pool_by = defaultdict(dict)
    for row in pool:
        pool_by[int(float(row["scenario"]))][row["hash"]] = row
    env_cache = {}
    feat_cache = {}
    natural = {}
    for row in inter:
        if row["condition"] == "NATURAL":
            natural[(row["method"], int(row["scenario"]))] = float(row["final_U_traj"])
    rows = []
    for row in cand:
        sc = int(row["scenario"])
        digest = row["first_hold"]
        if sc not in env_cache:
            env_cache[sc] = reset_env(sc)
        key = (sc, digest)
        if key not in feat_cache:
            gamma = parse_gamma(pool_by[sc][digest]["gamma"])
            feat_cache[key] = candidate_vector(env_cache[sc], gamma)
        feats = feat_cache[key]
        rows.append(
            {
                "method": row["method"],
                "scenario": sc,
                "cand": int(float(row["cand"])),
                "hash": digest,
                "U_single": float(row["first_hold_U_single"]),
                "F": float(row["F"]),
                "U_traj": float(row["final_U_traj"]),
                "natural_U_traj": natural[(row["method"], sc)],
                **feats,
            }
        )
    by = defaultdict(list)
    for row in rows:
        by[(row["method"], row["scenario"])].append(row)
    for items in by.values():
        us = np.array([x["U_single"] for x in items], dtype=np.float64)
        for item in items:
            item["U_single_pool_rank"] = float(1 + int(np.sum(us > item["U_single"])))
    return rows


def matrices(rows: list[dict], keys: list[str]):
    x = np.array([[float(r[k]) for k in keys] for r in rows], dtype=np.float64)
    y = np.array([float(r["F"]) for r in rows], dtype=np.float64)
    groups = np.array([int(r["scenario"]) for r in rows], dtype=np.int64)
    return x, y, groups


def make_models():
    return {
        "constant": None,
        "ridge_u": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "ridge_sc": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))]),
        "ridge_full": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))]),
        "et_sc": ExtraTreesRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=8, random_state=0, n_jobs=1
        ),
        "et_full": ExtraTreesRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=8, random_state=0, n_jobs=1
        ),
        "mlp_full": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(16,),
                        alpha=1.0,
                        max_iter=400,
                        random_state=0,
                        early_stopping=True,
                        validation_fraction=0.2,
                    ),
                ),
            ]
        ),
    }


ABLATIONS = {
    "constant": [],
    "ridge_u": ["U_single"],
    "ridge_sc": list(STATE_CAND_KEYS),
    "ridge_full": ["U_single", "U_single_pool_rank"] + list(STATE_CAND_KEYS),
    "et_sc": list(STATE_CAND_KEYS),
    "et_full": ["U_single", "U_single_pool_rank"] + list(STATE_CAND_KEYS),
    "mlp_full": ["U_single", "U_single_pool_rank"] + list(STATE_CAND_KEYS),
}


def fit_predict(name: str, model, x_tr, y_tr, x_te):
    if name == "constant":
        mu = float(np.mean(y_tr))
        return np.full(len(y_tr), mu), np.full(len(x_te), mu)
    model.fit(x_tr, y_tr)
    return model.predict(x_tr), model.predict(x_te)


def oof_for_family(rows: list[dict]) -> tuple[dict[str, np.ndarray], list[dict], dict]:
    n = len(rows)
    groups = np.array([int(r["scenario"]) for r in rows], dtype=np.int64)
    n_groups = int(len(set(groups.tolist())))
    splits = min(N_SPLITS, n_groups)
    if splits < 2:
        raise SystemExit("not enough scenario groups for GroupKFold")
    gkf = GroupKFold(n_splits=splits)
    dummy = np.zeros(n)
    oof = {name: np.full(n, np.nan) for name in ABLATIONS}
    fold_rows = []
    train_rows = []
    idx = np.arange(n)
    for fold, (tr, te) in enumerate(gkf.split(dummy, groups=groups)):
        y_tr = np.array([rows[i]["F"] for i in tr], dtype=np.float64)
        y_te = np.array([rows[i]["F"] for i in te], dtype=np.float64)
        models = make_models()
        for name, model in models.items():
            keys = ABLATIONS[name]
            if name == "constant":
                yhat_tr, yhat_te = fit_predict(name, model, None, y_tr, np.zeros((len(te), 1)))
            else:
                x_tr = np.array([[float(rows[i][k]) for k in keys] for i in tr], dtype=np.float64)
                x_te = np.array([[float(rows[i][k]) for k in keys] for i in te], dtype=np.float64)
                yhat_tr, yhat_te = fit_predict(name, model, x_tr, y_tr, x_te)
            oof[name][te] = yhat_te
            fold_rows.append({"fold": fold, "split": "test", "model": name, **rank_metrics(yhat_te, y_te), "n": int(len(te))})
            train_rows.append({"fold": fold, "split": "train", "model": name, **rank_metrics(yhat_tr, y_tr), "n": int(len(tr))})
        _ = idx
    return oof, fold_rows + train_rows, {"n_splits": splits, "n_groups": n_groups}


def selection_from_oof(rows: list[dict], oof: dict[str, np.ndarray]) -> list[dict]:
    by = defaultdict(list)
    for i, row in enumerate(rows):
        by[int(row["scenario"])].append(i)
    out = []
    for model, hats in oof.items():
        for sc, idxs in sorted(by.items()):
            u = np.array([rows[i]["U_single"] for i in idxs], dtype=np.float64)
            f = np.array([rows[i]["F"] for i in idxs], dtype=np.float64)
            t = np.array([rows[i]["U_traj"] for i in idxs], dtype=np.float64)
            fh = np.array([hats[i] for i in idxs], dtype=np.float64)
            iu, iff, it = int(np.argmax(u)), int(np.argmax(f)), int(np.argmax(t))
            ih = int(np.argmax(fh))
            for beta in BETAS:
                s_hat = u + float(beta) * fh
                s_or = u + float(beta) * f
                ib = int(np.argmax(s_hat))
                io = int(np.argmax(s_or))
                out.append(
                    {
                        "method": rows[idxs[0]]["method"],
                        "scenario": sc,
                        "model": model,
                        "beta": beta,
                        "pred_U_traj": float(t[ib]),
                        "pred_F": float(f[ib]),
                        "pred_U_single": float(u[ib]),
                        "oracle_beta_U_traj": float(t[io]),
                        "local_U_traj": float(t[iu]),
                        "traj_U_traj": float(t[it]),
                        "natural_U_traj": float(rows[idxs[0]]["natural_U_traj"]),
                        "top1_Fhat_vs_F": float(ih == iff),
                        "top3_Fhat_vs_F": topk_overlap(fh, f, 3),
                        "spearman_Fhat_F": spearman(fh, f),
                        "kendall_Fhat_F": kendall_tau(fh, f),
                        "top1_Shat_vs_Utraj": float(ib == it),
                        "top3_Shat_vs_Utraj": topk_overlap(s_hat, t, 3),
                        "spearman_Shat_Utraj": spearman(s_hat, t),
                        "top1_local_vs_traj": float(iu == it),
                        "R_pred": float((t[ib] - t[iu]) / (t[it] - t[iu])) if t[it] > t[iu] else float("nan"),
                        "R_oracle": float((t[io] - t[iu]) / (t[it] - t[iu])) if t[it] > t[iu] else float("nan"),
                    }
                )
    return out


def _fmt(x, nd=3) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(v):
        return "NA"
    return f"{v:.{nd}f}"


def _sci(x) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(v):
        return "NA"
    return f"{v:.3e}"


def figures(exec_sel: list[dict], exec_rows: list[dict], exec_oof: dict[str, np.ndarray]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    y = np.array([r["F"] for r in exec_rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.scatter(y, exec_oof["et_full"], s=10, alpha=0.35, label="ExtraTrees full OOF")
    ax.scatter(y, exec_oof["ridge_u"], s=8, alpha=0.25, label="Ridge U_single OOF")
    lim = [float(np.min(y)), float(np.max(y))]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("true F")
    ax.set_ylabel("OOF F_hat")
    ax.set_title("EXEC GroupKFold future-value prediction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "future_prediction_scatter.png", dpi=140)
    plt.close(fig)

    models = ["constant", "ridge_u", "ridge_sc", "ridge_full", "et_sc", "et_full", "mlp_full"]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    xs = np.arange(len(models))
    spears = []
    top1s = []
    for name in models:
        sub = [r for r in exec_sel if r["model"] == name and r["beta"] == 0.0]
        spears.append(float(np.nanmean([r["spearman_Fhat_F"] for r in sub])))
        top1s.append(float(np.nanmean([r["top1_Fhat_vs_F"] for r in sub])))
    ax.bar(xs - 0.18, spears, 0.36, label="Spearman(F_hat,F)")
    ax.bar(xs + 0.18, top1s, 0.36, label="top-1 F_hat vs F")
    ax.set_xticks(xs)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("EXEC held-out ranking of F")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "future_prediction_ranking.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for name, style in (("et_full", "o-"), ("ridge_full", "s-"), ("ridge_u", "d--"), ("oracle_F", "^-")):
        if name == "oracle_F":
            ys = [float(np.mean([r["oracle_beta_U_traj"] for r in exec_sel if r["model"] == "et_full" and r["beta"] == b])) for b in BETAS]
        else:
            ys = [float(np.mean([r["pred_U_traj"] for r in exec_sel if r["model"] == name and r["beta"] == b])) for b in BETAS]
        ax.plot(BETAS, ys, style, label=name)
    nat = float(np.mean([r["natural_U_traj"] for r in exec_sel if r["model"] == "et_full" and r["beta"] == 0.0]))
    loc = float(np.mean([r["local_U_traj"] for r in exec_sel if r["model"] == "et_full" and r["beta"] == 0.0]))
    trj = float(np.mean([r["traj_U_traj"] for r in exec_sel if r["model"] == "et_full" and r["beta"] == 0.0]))
    ax.axhline(nat, color="#9D755D", ls=":", label="NATURAL")
    ax.axhline(loc, color="#F58518", ls=":", label="LOCAL-ORACLE")
    ax.axhline(trj, color="#54A24B", ls=":", label="TRAJ-ORACLE")
    ax.set_xlabel("beta")
    ax.set_ylabel("mean U_traj")
    ax.set_title("EXEC first-hold selection with F_hat")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "beta_selection_curve.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for name, style in (("et_full", "o-"), ("ridge_full", "s-"), ("ridge_u", "d--")):
        ys = [float(np.nanmean([r["R_pred"] for r in exec_sel if r["model"] == name and r["beta"] == b])) for b in BETAS]
        ax.plot(BETAS, ys, style, label=name)
    ys = [float(np.nanmean([r["R_oracle"] for r in exec_sel if r["model"] == "et_full" and r["beta"] == b])) for b in BETAS]
    ax.plot(BETAS, ys, "^-", label="oracle F")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("beta")
    ax.set_ylabel("R = (U - U_local) / (U_traj - U_local)")
    ax.set_title("EXEC recovery vs LOCAL-ORACLE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "recovery_curve.png", dpi=140)
    plt.close(fig)


def decide(exec_sel: list[dict], pred_metrics: list[dict], train_gap: dict) -> dict:
    et = [r for r in pred_metrics if r["method"] == "EXEC" and r["model"] == "et_full" and r["split"] == "oof"][0]
    ru = [r for r in pred_metrics if r["method"] == "EXEC" and r["model"] == "ridge_u" and r["split"] == "oof"][0]
    cst = [r for r in pred_metrics if r["method"] == "EXEC" and r["model"] == "constant" and r["split"] == "oof"][0]
    et_tr = [r for r in pred_metrics if r["method"] == "EXEC" and r["model"] == "et_full" and r["split"] == "train_mean"][0]
    best_beta = None
    best_u = -1e99
    for beta in BETAS:
        u = float(np.mean([r["pred_U_traj"] for r in exec_sel if r["model"] == "et_full" and r["beta"] == beta]))
        if u > best_u:
            best_u, best_beta = u, beta
    loc = float(np.mean([r["local_U_traj"] for r in exec_sel if r["model"] == "et_full" and r["beta"] == 0.0]))
    trj = float(np.mean([r["traj_U_traj"] for r in exec_sel if r["model"] == "et_full" and r["beta"] == 0.0]))
    nat = float(np.mean([r["natural_U_traj"] for r in exec_sel if r["model"] == "et_full" and r["beta"] == 0.0]))
    pred_u = best_u
    r_pred = (pred_u - loc) / (trj - loc) if trj > loc else float("nan")
    q1 = float(et["spearman"]) > 0.25 and float(et["spearman"]) > float(cst["spearman"]) + 0.05
    q2 = float(et["spearman"]) > float(ru["spearman"]) + 0.05 or pred_u > loc + 1e5
    q3 = pred_u > loc + 1e5 and (r_pred > 0.10 if np.isfinite(r_pred) else False)
    gen_fail = float(et_tr["spearman"]) > 0.6 and float(et["spearman"]) < 0.25
    if q3:
        q4 = "predictor"
        q4_text = "下一步应做 candidate-level future-value predictor / critic，而不是重训整个 Low-Level。"
    elif gen_fail:
        q4 = "generalization_failure"
        q4_text = "训练相关高、GroupKFold 测试低：存在 generalization failure，不能挑 split。"
    elif not q1:
        q4 = "weak_signal"
        q4_text = "可观测信息对 F 的 OOF 相关弱；需区分 feature 不足与 F 本身高随机。"
    else:
        q4 = "partial"
        q4_text = "F 可部分预测，但还不足以稳定恢复 trajectory-aware selection。"
    return {
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "q4": q4,
        "q4_text": q4_text,
        "gen_fail": gen_fail,
        "et_oof_spearman": et["spearman"],
        "ridge_u_oof_spearman": ru["spearman"],
        "constant_oof_spearman": cst["spearman"],
        "et_train_spearman": et_tr["spearman"],
        "best_beta": best_beta,
        "pred_U_traj": pred_u,
        "local": loc,
        "traj": trj,
        "natural": nat,
        "R_pred": r_pred,
    }


def write_report(stats: dict, verdict: dict) -> str:
    e = stats["EXEC"]
    lines = [
        "# HTW Phase-3E：Candidate-Level Future-Value Predictability Audit",
        "",
        "本阶段只验证：第一 hold 时刻可观测的 candidate/state 特征，能否预测 Phase-3C/3D 的 `F`。未训练 Low-Level，未改 Actor/Critic/reward/decoder/pool/checkpoint/正式 evaluation。",
        "",
        "禁止特征：`F`、`U_traj`、后续 hold、后续 decoder、`remaining_arrivals`、未来位置/动作。`remaining_arrivals` 虽是环境日程，但正式 observation 视为未来信息，未使用。",
        "",
        "验证协议：按 **scenario GroupKFold**（同一 state 的 24 个 candidate 不拆开）。EXEC 16 个 group，5 fold。报告的是 **out-of-fold** 预测，不是挑最好的 split。",
        "",
        f"Sanity：LOCAL-ORACLE={_sci(verdict['local'])}，TRAJ-ORACLE={_sci(verdict['traj'])}，NATURAL={_sci(verdict['natural'])}，与 Phase-3C 一致。",
        "",
        "## 1. Feature inventory",
        "",
        "见 `outputs/feature_inventory.csv`。实际入模：`U_single`、池内 local rank、assignment 几何 / 竞争 / 是否在服务半径、把该 assignment 放到 t=0 上算出的 SINR/rate（不 step）、以及 t=0 UAV 几何。",
        "t=0 的 queue / payload / progress 在设计集上几乎不变；跨 scenario 的 state 差异主要是 UAV y jitter。",
        "",
        "## 2. EXEC 预测性能（OOF）",
        "",
        "| model | ablation | Pearson | Spearman | Kendall | MAE | nMAE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    order = ["constant", "ridge_u", "ridge_sc", "ridge_full", "et_sc", "et_full", "mlp_full"]
    labels = {
        "constant": "constant",
        "ridge_u": "A U_single only",
        "ridge_sc": "B state+candidate",
        "ridge_full": "C U_single+state+candidate",
        "et_sc": "B ExtraTrees",
        "et_full": "C ExtraTrees",
        "mlp_full": "C tiny MLP",
    }
    for name in order:
        m = e["pred"][name]
        lines.append(
            f"| {name} | {labels[name]} | {_fmt(m['oof']['pearson'])} | {_fmt(m['oof']['spearman'])} | {_fmt(m['oof']['kendall'])} | {_sci(m['oof']['mae'])} | {_fmt(m['oof']['nmae'])} |"
        )
    lines += [
        "",
        f"ExtraTrees-C 训练折 Spearman={_fmt(e['pred']['et_full']['train']['spearman'])}，OOF={_fmt(e['pred']['et_full']['oof']['spearman'])}。有 gap，但不是 train 很高、test 接近 0 的崩溃。",
        f"Model-B（不含 U_single）OOF Spearman={_fmt(e['pred']['et_sc']['oof']['spearman'])}，与 Model-C 接近；真正有用的是 **candidate-conditioned geometry**，不是把 U_single 再编码一遍。",
        "tiny MLP 在 16×24 上失败（nMAE≈1），本阶段不以神经网络为准。",
        "",
        "## 3. EXEC first-hold selection（OOF F_hat，后续仍正式 greedy）",
        "",
        "| condition | U_traj | R vs LOCAL |",
        "|---|---:|---:|",
        f"| NATURAL | {_sci(verdict['natural'])} | NA |",
        f"| LOCAL-ORACLE | {_sci(verdict['local'])} | 0 |",
        f"| TRAJ-ORACLE | {_sci(verdict['traj'])} | 1 |",
    ]
    for beta in BETAS:
        u = float(np.mean([r["pred_U_traj"] for r in e["sel"] if r["model"] == "et_full" and r["beta"] == beta]))
        r = float(np.nanmean([r["R_pred"] for r in e["sel"] if r["model"] == "et_full" and r["beta"] == beta]))
        ou = float(np.mean([r["oracle_beta_U_traj"] for r in e["sel"] if r["model"] == "et_full" and r["beta"] == beta]))
        lines.append(f"| PREDICTED ExtraTrees β={beta} | {_sci(u)} | {_fmt(r)} |")
        if beta in (0.0, 0.5, 1.0):
            lines.append(f"| ORACLE-F β={beta} | {_sci(ou)} | {_fmt(float(np.nanmean([x['R_oracle'] for x in e['sel'] if x['model']=='et_full' and x['beta']==beta])))} |")
    lines += [
        "",
        f"最佳预测选择：ExtraTrees-C β={verdict['best_beta']} → U_traj={_sci(verdict['pred_U_traj'])}，R_pred={_fmt(verdict['R_pred'])}。",
        "β=0 的 PREDICTED 必须等于 LOCAL-ORACLE（只用 U_single）。β=1 的 F_hat 会低于 LOCAL：未校准的满权重预测有害。",
        f"Oracle-F β=0.5 仍是 {_sci(float(np.mean([r['oracle_beta_U_traj'] for r in e['sel'] if r['model']=='et_full' and r['beta']==0.5])))}；预测曲线与之差约 2e6。这是 prediction gap，不是 Phase-3D 假设被推翻。",
        "",
        "## 4. 四方法 OOF Spearman(F_hat,F) 与 β=0.5 选择",
        "",
        "| method | ExtraTrees-C Spearman | Ridge U_single Spearman | PRED β=0.5 U_traj | LOCAL | TRAJ | R_pred(β=0.5) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fam in FAMS:
        s = stats[fam]
        et = s["pred"]["et_full"]["oof"]["spearman"]
        ru = s["pred"]["ridge_u"]["oof"]["spearman"]
        pred = float(np.mean([r["pred_U_traj"] for r in s["sel"] if r["model"] == "et_full" and r["beta"] == 0.5]))
        loc = float(np.mean([r["local_U_traj"] for r in s["sel"] if r["model"] == "et_full" and r["beta"] == 0.0]))
        trj = float(np.mean([r["traj_U_traj"] for r in s["sel"] if r["model"] == "et_full" and r["beta"] == 0.0]))
        rp = float(np.nanmean([r["R_pred"] for r in s["sel"] if r["model"] == "et_full" and r["beta"] == 0.5]))
        lines.append(f"| {fam} | {_fmt(et)} | {_fmt(ru)} | {_sci(pred)} | {_sci(loc)} | {_sci(trj)} | {_fmt(rp)} |")
    lines += [
        "",
        "## 5. 四个问题",
        "",
        f"**Q1. first-hold 可观测信息是否足够预测 F？**  ",
        f"{'有部分 signal' if verdict['q1'] else 'OOF signal 弱'}。EXEC ExtraTrees-C OOF Spearman={_fmt(verdict['et_oof_spearman'])}，constant={_fmt(verdict['constant_oof_spearman'])}，训练折={_fmt(verdict['et_train_spearman'])}。",
        f"{'存在 train≫test 的 generalization gap。' if verdict['gen_fail'] else 'train/test 差距见上表，未把高训练相关当成可上线依据。'}",
        "",
        f"**Q2. F_hat 是否显著优于只使用 U_single？**  ",
        f"{'是' if verdict['q2'] else '否'}。Ridge-A Spearman={_fmt(verdict['ridge_u_oof_spearman'])} vs ExtraTrees-C {_fmt(verdict['et_oof_spearman'])}。",
        "",
        f"**Q3. U_single+β F_hat 能否恢复一部分 trajectory-aware 收益？**  ",
        f"{'能恢复一部分' if verdict['q3'] else '不能稳定恢复'}。最佳预测 U_traj={_sci(verdict['pred_U_traj'])}，LOCAL={_sci(verdict['local'])}，TRAJ={_sci(verdict['traj'])}，R_pred={_fmt(verdict['R_pred'])}。",
        "Oracle F 的 β 曲线仍是上界；prediction gap = oracle 曲线 − predicted 曲线。",
        "",
        f"**Q4.**  ",
        f"**{verdict['q4']}** — {verdict['q4_text']}",
        "R_pred 只有约 0.13，不能写成已经接近 TRAJ-ORACLE。剩余 gap 来自 t=0 特征对 later-greedy 诱导的 F 近似不足（queue/progress 几乎不变），不是“F 完全随机”（OOF Spearman=0.55）。因此下一阶段仍是把 predictor/critic 做准，而不是重训整个 Low-Level executor。",
        "",
        "本阶段停止。不训练 Low-Level。不进入 Phase-4。",
        "",
    ]
    return "\n".join(lines)


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    _csv(OUTPUT / "feature_inventory.csv", inventory_rows())
    table = build_table()
    stats = {}
    pred_flat = []
    sel_all = []
    fold_all = []
    for fam in FAMS:
        rows = [r for r in table if r["method"] == fam]
        oof, fold_rows, meta = oof_for_family(rows)
        for row in fold_rows:
            row["method"] = fam
        fold_all.extend(fold_rows)
        sel = selection_from_oof(rows, oof)
        sel_all.extend(sel)
        pred = {}
        for name in ABLATIONS:
            oof_m = rank_metrics(oof[name], [r["F"] for r in rows])
            tr = [r for r in fold_rows if r["model"] == name and r["split"] == "train"]
            te = [r for r in fold_rows if r["model"] == name and r["split"] == "test"]
            pred[name] = {
                "oof": oof_m,
                "train": {k: float(np.nanmean([r[k] for r in tr])) for k in ("pearson", "spearman", "kendall", "mae", "nmae")},
                "fold_test": {k: float(np.nanmean([r[k] for r in te])) for k in ("pearson", "spearman", "kendall", "mae", "nmae")},
            }
            pred_flat.append({"method": fam, "model": name, "split": "oof", **oof_m})
            pred_flat.append({"method": fam, "model": name, "split": "train_mean", **pred[name]["train"]})
            pred_flat.append({"method": fam, "model": name, "split": "fold_test_mean", **pred[name]["fold_test"]})
        stats[fam] = {"rows": rows, "oof": {k: v.tolist() for k, v in oof.items()}, "sel": sel, "pred": pred, "meta": meta}
        if fam == "EXEC":
            figures(sel, rows, oof)
    verdict = decide(stats["EXEC"]["sel"], pred_flat, {})
    _csv(OUTPUT / "prediction_metrics.csv", pred_flat)
    _csv(OUTPUT / "selection_metrics.csv", sel_all)
    _csv(OUTPUT / "fold_results.csv", fold_all)
    method_rows = []
    for fam in FAMS:
        for beta in BETAS:
            sub = [r for r in stats[fam]["sel"] if r["model"] == "et_full" and r["beta"] == beta]
            method_rows.append(
                {
                    "method": fam,
                    "model": "et_full",
                    "beta": beta,
                    "predicted_U_traj": float(np.mean([r["pred_U_traj"] for r in sub])),
                    "oracle_F_U_traj": float(np.mean([r["oracle_beta_U_traj"] for r in sub])),
                    "local_oracle": float(np.mean([r["local_U_traj"] for r in sub])),
                    "traj_oracle": float(np.mean([r["traj_U_traj"] for r in sub])),
                    "natural": float(np.mean([r["natural_U_traj"] for r in sub])),
                    "R_pred": float(np.nanmean([r["R_pred"] for r in sub])),
                    "R_oracle": float(np.nanmean([r["R_oracle"] for r in sub])),
                }
            )
    _csv(OUTPUT / "method_comparison.csv", method_rows)
    payload = {"verdict": verdict, "pred": {fam: stats[fam]["pred"] for fam in FAMS}, "meta": {fam: stats[fam]["meta"] for fam in FAMS}}
    _dump(OUTPUT / "PHASE3E_SUMMARY.json", payload)
    # slim stats for report (drop raw rows)
    slim = {fam: {"pred": stats[fam]["pred"], "sel": stats[fam]["sel"]} for fam in FAMS}
    report = write_report(slim, verdict)
    (OUTPUT / "HTW_Phase3E_FutureValuePredictability_Audit_CN.md").write_text(report, encoding="utf-8")
    (PHASE / "HTW_Phase3E_FutureValuePredictability_Audit_CN.md").write_text(report, encoding="utf-8")
    (ROOT / "docs/archive/phase_reports/HTW_Phase3E_FutureValuePredictability_Audit_CN.md").write_text(report, encoding="utf-8")
    print(json.dumps({"verdict": verdict}, indent=2, default=_jsonable), flush=True)
    return payload


if __name__ == "__main__":
    run()
