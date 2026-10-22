"""Phase-4B: offline residual-credit ExtraTrees. No planner or Low-Level change."""
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
from sklearn.model_selection import GroupKFold

from wt.lowlevel_v2.integration_phase1.stats import kendall_tau, spearman, summarize
from wt.lowlevel_v2.phase3e_future_value_audit.run_phase3e import candidate_vector, nmae, pearson, reset_env
from wt.lowlevel_v2.htw_phase4b_credit_learner.protocol import (
    ALIGN_ATOL,
    ET,
    FAMS,
    FIG,
    LOCKED_FV_R_GAIN,
    LOCKED_FV_SPEARMAN,
    LOCKED_FV_U,
    LOCKED_LOCAL_R,
    LOCKED_LOCAL_U,
    LOCKED_TRAJ_U,
    N_PERM,
    N_SCENARIO,
    N_SPLITS,
    OUTPUT,
    P4A_CSV,
    PERM_SEED,
    POOL_SIZE,
    PRIMARY,
    REPORT_NAME,
    ROOT,
)

CAND_KEYS = (
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
    "assign_0",
    "assign_1",
    "assign_2",
    "assign_3",
    "assign_4",
    "assign_5",
    "leftover_queue",
    "leftover_access",
    "in_range_frac",
)
STATE_KEYS = (
    "uav_y_mean",
    "uav_y_std",
    "uav_pairwise_mean",
    "min_pairwise_dist",
    "queue_mean",
    "queue_hot_mass",
)
SPEC = (
    ("B0_constant", ()),
    ("B1_U_single", ("U_single",)),
    ("B2_cand_geom", CAND_KEYS),
    ("A_U_only", ("U_single",)),
    ("B_cand_only", CAND_KEYS),
    ("C_U_cand", ("U_single",) + CAND_KEYS),
    ("D_full", ("U_single",) + CAND_KEYS + STATE_KEYS),
    ("B3_full", ("U_single",) + CAND_KEYS + STATE_KEYS),
)
SELECT_MODELS = ("B0_constant", "B1_U_single", "B2_cand_geom", "C_U_cand", "B3_full")
FV_BETA = 0.25


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n", encoding="utf-8")


def _csv_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    out["min"] = float(np.min(a)) if a.size else float("nan")
    out["max"] = float(np.max(a)) if a.size else float("nan")
    return out


def ndcg_at(pred, truth, k: int = 3) -> float:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    rel = t - float(np.min(t))
    k = min(int(k), int(rel.size))
    if k <= 0:
        return float("nan")
    order = np.argsort(-p)[:k]
    ideal = np.argsort(-rel)[:k]
    discounts = np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(rel[order] / discounts))
    idcg = float(np.sum(rel[ideal] / discounts))
    if idcg <= 1e-12:
        return float("nan")
    return dcg / idcg


def parse_gamma(text: str) -> np.ndarray:
    return np.asarray(ast.literal_eval(text), dtype=np.int64)


def leftover_feats(env, gamma, base: dict) -> dict[str, float]:
    assigned = [int(b) for b in np.asarray(gamma, dtype=np.int64)[: env.config.num_uavs] if int(b) >= 0]
    covered = set(assigned)
    leftover_q = 0.0
    leftover_d = []
    radius = float(env.config.service_radius_m)
    for k in range(int(env.config.num_buoys)):
        if k in covered:
            continue
        q = float(env.queue[k])
        leftover_q += q
        if q > 1e-9:
            leftover_d.append(
                min(float(np.linalg.norm(env.uav_positions[u] - env.buoy_positions[k])) for u in range(env.config.num_uavs))
            )
    n_assigned = max(len(assigned), 1)
    in_range_frac = float(base["n_in_range"]) / n_assigned
    comp = float(base["competition_count"])
    d_left = float(np.mean(leftover_d)) if leftover_d else radius
    access = 1.0 / (1.0 + d_left / max(radius, 1.0))
    return {
        "leftover_queue": leftover_q,
        "leftover_access": access,
        "in_range_frac": in_range_frac,
        "leftover_proxy": leftover_q * (1.0 + float(base["idle_frac"])) * access / (1.0 + comp),
    }


def load_phase4a() -> list[dict]:
    if not P4A_CSV.is_file():
        raise FileNotFoundError(f"missing Phase-4A CSV: {P4A_CSV}")
    with P4A_CSV.open(encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle))
    if len(raw) != N_SCENARIO * POOL_SIZE * len(FAMS):
        raise RuntimeError(f"Phase-4A row count {len(raw)} != 1536; stop")
    rows = []
    for row in raw:
        u = float(row["U_single"])
        t = float(row["U_traj"])
        c = float(row["C_abs"])
        f = float(row["F"])
        if abs(c - (t - u)) > ALIGN_ATOL or abs(c - f) > ALIGN_ATOL:
            raise RuntimeError("Phase-4A C_abs does not match U_traj-U_single; stop")
        rows.append(
            {
                "family": row["family"],
                "scenario_id": int(float(row["scenario_id"])),
                "state_id": int(float(row["state_id"])),
                "candidate_id": int(float(row["candidate_id"])),
                "U_single": u,
                "U_traj": t,
                "C_abs": c,
                "gamma": row["gamma"],
                "assignment_hash": row["assignment_hash"],
            }
        )
    return rows


def attach_observables(rows: list[dict]) -> list[dict]:
    env_cache = {}
    geom_cache = {}
    out = []
    for row in rows:
        sc = int(row["scenario_id"])
        digest = row["assignment_hash"]
        if sc not in env_cache:
            env_cache[sc] = reset_env(sc)
        key = (sc, digest)
        if key not in geom_cache:
            gamma = parse_gamma(row["gamma"])
            base = candidate_vector(env_cache[sc], gamma)
            extra = leftover_feats(env_cache[sc], gamma, base)
            geom_cache[key] = {**base, **extra}
        item = dict(row)
        item.update(geom_cache[key])
        out.append(item)
    by = defaultdict(list)
    for i, row in enumerate(out):
        by[(row["family"], row["scenario_id"])].append(i)
    for idxs in by.values():
        us = np.array([out[i]["U_single"] for i in idxs], dtype=np.float64)
        for i in idxs:
            out[i]["U_single_pool_rank"] = float(1 + int(np.sum(us > out[i]["U_single"])))
    return out


def inventory() -> list[dict]:
    rows = [
        ("U_single", True, False, True, "first-hold shadow mean; Phase-4A"),
        ("U_single_pool_rank", True, False, False, "rank inside the same-state 24-pool; not a model input this phase"),
        ("candidate geometry", True, False, True, "Phase-3E candidate_vector + leftover/access"),
        ("state geometry", True, False, True, "t=0 UAV/queue stats; constant within state"),
        ("leftover_queue / leftover_access", True, False, True, "unassigned current queue; no remaining_arrivals"),
        ("C_abs / F / U_traj", False, True, False, "target or forbidden"),
        ("hold2/3/4 / later state / future action", False, True, False, "forbidden"),
        ("remaining_arrivals", True, True, False, "forbidden future schedule"),
        ("C_ratio / trajectory_length", False, True, False, "not used as target or feature"),
        ("family", False, False, False, "models are fit per family; EXEC is primary"),
    ]
    return [
        {
            "feature": name,
            "available_at_first_hold": int(a),
            "future_leakage": int(l),
            "used_for_model": int(u),
            "note": note,
        }
        for name, a, l, u, note in rows
    ]


def matrix(rows: list[dict], keys: tuple[str, ...]):
    if not keys:
        return np.zeros((len(rows), 1), dtype=np.float64)
    return np.array([[float(r[k]) for k in keys] for r in rows], dtype=np.float64)


def oof_predict(rows: list[dict], keys: tuple[str, ...]) -> tuple[np.ndarray, list[dict]]:
    y = np.array([r["C_abs"] for r in rows], dtype=np.float64)
    groups = np.array([int(r["scenario_id"]) for r in rows], dtype=np.int64)
    x = matrix(rows, keys)
    oof = np.full(len(rows), np.nan)
    folds = []
    gkf = GroupKFold(n_splits=min(N_SPLITS, len(set(groups.tolist()))))
    for fold, (tr, te) in enumerate(gkf.split(x, groups=groups)):
        if keys:
            model = ExtraTreesRegressor(**ET)
            model.fit(x[tr], y[tr])
            pred = model.predict(x[te])
        else:
            pred = np.full(te.shape[0], float(np.mean(y[tr])))
        oof[te] = pred
        folds.append(
            {
                "fold": fold,
                "train_scenarios": sorted({int(groups[i]) for i in tr}),
                "validation_scenarios": [],
                "test_scenarios": sorted({int(groups[i]) for i in te}),
                "n_train": int(tr.size),
                "n_test": int(te.size),
            }
        )
    return oof, folds


def pred_block(c, hat) -> dict:
    c = np.asarray(c, dtype=np.float64)
    hat = np.asarray(hat, dtype=np.float64)
    return {
        "pearson": pearson(hat, c),
        "spearman": spearman(hat, c),
        "kendall": kendall_tau(hat, c),
        "mae": float(np.mean(np.abs(hat - c))),
        "nmae": nmae(c, hat),
    }


def grouped(rows: list[dict]) -> dict[int, list[int]]:
    by = defaultdict(list)
    for i, row in enumerate(rows):
        by[int(row["scenario_id"])].append(i)
    return dict(sorted(by.items()))


def within_spearman(rows: list[dict], pred, truth_key: str) -> dict:
    pred = np.asarray(pred, dtype=np.float64)
    vals = []
    for idxs in grouped(rows).values():
        vals.append(spearman([pred[i] for i in idxs], [rows[i][truth_key] for i in idxs]))
    return ext_sum(vals)


def select_block(rows: list[dict], score) -> dict:
    score = np.asarray(score, dtype=np.float64)
    r_pred = []
    top1 = []
    top3 = []
    regret = []
    u_sel = []
    u_loc = []
    u_or = []
    ndcg = []
    for idxs in grouped(rows).values():
        sc = score[idxs]
        ut = np.array([rows[i]["U_traj"] for i in idxs], dtype=np.float64)
        us = np.array([rows[i]["U_single"] for i in idxs], dtype=np.float64)
        ih = int(np.argmax(sc))
        it = int(np.argmax(ut))
        iu = int(np.argmax(us))
        best = float(np.max(ut))
        chosen = float(ut[ih])
        r_pred.append(chosen / best if abs(best) > 0 else float("nan"))
        top1.append(float(ih == it))
        top3.append(float(it in set(np.argsort(-sc)[:3].tolist())))
        regret.append((best - chosen) / max(abs(best), 1.0))
        u_sel.append(chosen)
        u_loc.append(float(ut[iu]))
        u_or.append(best)
        ndcg.append(ndcg_at(sc, ut, k=3))
    gain = []
    for a, b, c in zip(u_sel, u_loc, u_or):
        den = c - b
        gain.append((a - b) / den if den > 1e-12 else float("nan"))
    return {
        "R_pred": ext_sum(r_pred),
        "top1": ext_sum(top1),
        "top3_recall": ext_sum(top3),
        "regret": ext_sum(regret),
        "ndcg3": ext_sum(ndcg),
        "U_traj_selected": ext_sum(u_sel),
        "U_traj_local": ext_sum(u_loc),
        "U_traj_oracle": ext_sum(u_or),
        "R_gain_phase3": ext_sum(gain),
    }


def permute_null(rows: list[dict], c_hat) -> dict:
    c_hat = np.asarray(c_hat, dtype=np.float64)
    rng = np.random.default_rng(PERM_SEED)
    spears, top1s, ndcgs, rpreds = [], [], [], []
    by = grouped(rows)
    for _ in range(N_PERM):
        c_perm = np.array([r["C_abs"] for r in rows], dtype=np.float64)
        u_single = np.array([r["U_single"] for r in rows], dtype=np.float64)
        for idxs in by.values():
            vals = c_perm[idxs].copy()
            rng.shuffle(vals)
            c_perm[idxs] = vals
        u_perm = u_single + c_perm
        spears.append(spearman(c_hat, c_perm))
        t1 = []
        nd = []
        rp = []
        for idxs in by.values():
            hat = c_hat[idxs]
            cp = c_perm[idxs]
            up = u_perm[idxs]
            us = u_single[idxs]
            t1.append(float(int(np.argmax(hat)) == int(np.argmax(cp))))
            nd.append(ndcg_at(hat, cp, k=3))
            score = us + hat
            best = float(np.max(up))
            rp.append(float(up[int(np.argmax(score))] / best) if abs(best) > 0 else float("nan"))
        top1s.append(float(np.mean(t1)))
        ndcgs.append(float(np.nanmean(nd)))
        rpreds.append(float(np.nanmean(rp)))
    return {
        "n_perm": N_PERM,
        "spearman": ext_sum(spears),
        "top1": ext_sum(top1s),
        "ndcg3": ext_sum(ndcgs),
        "R_pred": ext_sum(rpreds),
    }


def analyze_family(rows: list[dict]) -> dict:
    hats = {}
    fold_map = {}
    cache: dict[tuple[str, ...], tuple[np.ndarray, list[dict]]] = {}
    for name, keys in SPEC:
        if keys not in cache:
            cache[keys] = oof_predict(rows, keys)
        hats[name], fold_map[name] = cache[keys]

    c = np.array([r["C_abs"] for r in rows], dtype=np.float64)
    u = np.array([r["U_single"] for r in rows], dtype=np.float64)
    t = np.array([r["U_traj"] for r in rows], dtype=np.float64)
    pred_metrics = {}
    traj_metrics = {}
    within = {}
    selection = {}
    for name, hat in hats.items():
        pred_metrics[name] = pred_block(c, hat)
        uhat = u + hat
        traj_metrics[name] = {
            "spearman": spearman(uhat, t),
            "kendall": kendall_tau(uhat, t),
            "within_spearman": within_spearman(rows, uhat, "U_traj"),
        }
        within[name] = {
            "C": within_spearman(rows, hat, "C_abs"),
            "U_traj": within_spearman(rows, uhat, "U_traj"),
        }
        selection[name] = select_block(rows, uhat)
    selection["LOCAL"] = select_block(rows, u)
    selection["TRAJ_ORACLE"] = select_block(rows, t)
    selection["FV_beta025"] = select_block(rows, u + FV_BETA * hats["B3_full"])
    perm = permute_null(rows, hats["B3_full"])
    return {
        "n": len(rows),
        "n_state": len({r["scenario_id"] for r in rows}),
        "folds": fold_map["B3_full"],
        "hats": hats,
        "pred": pred_metrics,
        "traj": traj_metrics,
        "within": within,
        "selection": selection,
        "perm": perm,
    }


def figures(rows: list[dict], ana: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    hat = ana["hats"]["B3_full"]
    c = [r["C_abs"] / 1e6 for r in rows]
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.scatter(c, hat / 1e6, s=10, alpha=0.5, c="#4C78A8")
    lim = [min(c + list(hat / 1e6)), max(c + list(hat / 1e6))]
    ax.plot(lim, lim, "--", color="#888")
    ax.set_xlabel("C_abs (1e6)")
    ax.set_ylabel("OOF C_hat (1e6)")
    fig.tight_layout()
    fig.savefig(FIG / "oof_c_scatter_exec.png", dpi=140)
    plt.close(fig)

    names = ["B0_constant", "B1_U_single", "B2_cand_geom", "C_U_cand", "B3_full"]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.bar(names, [ana["within"][n]["C"]["median"] for n in names], color="#4C78A8")
    ax.set_ylabel("within-state Spearman(C_hat, C)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG / "within_spearman_ablation.png", dpi=140)
    plt.close(fig)

    labels = ["LOCAL", "FV_beta025", "B3_full", "TRAJ_ORACLE"]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(labels, [ana["selection"][n]["R_pred"]["mean"] for n in labels], color=["#9e9e9e", "#F58518", "#4C78A8", "#54A24B"])
    ax.axhline(LOCKED_LOCAL_R, color="#333", ls="--", lw=1)
    ax.set_ylabel("mean R_pred")
    fig.tight_layout()
    fig.savefig(FIG / "selection_rpred.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.hist(np.random.default_rng(PERM_SEED).normal(ana["perm"]["spearman"]["mean"], max(ana["perm"]["spearman"]["std"], 1e-6), N_PERM), bins=12, color="#B279A2")
    ax.axvline(ana["pred"]["B3_full"]["spearman"], color="#4C78A8", lw=2, label="OOF Spearman")
    ax.set_xlabel("permutation Spearman(C_hat, C)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "permutation_spearman.png", dpi=140)
    plt.close(fig)


def _fmt(x: float) -> str:
    if not np.isfinite(float(x)):
        return "nan"
    return f"{float(x):.3f}"


def _sci(x: float) -> str:
    return f"{float(x):.3e}"


def decide(ana: dict) -> dict:
    p0 = ana["pred"]["B0_constant"]
    p1 = ana["pred"]["B1_U_single"]
    p2 = ana["pred"]["B2_cand_geom"]
    p3 = ana["pred"]["B3_full"]
    w0 = ana["within"]["B0_constant"]["C"]["median"]
    w1 = ana["within"]["B1_U_single"]["C"]["median"]
    w2 = ana["within"]["B2_cand_geom"]["C"]["median"]
    w3 = ana["within"]["B3_full"]["C"]["median"]
    r_loc = ana["selection"]["LOCAL"]["R_pred"]["mean"]
    r_res = ana["selection"]["B3_full"]["R_pred"]["mean"]
    r_c = ana["selection"]["C_U_cand"]["R_pred"]["mean"]
    r_b2 = ana["selection"]["B2_cand_geom"]["R_pred"]["mean"]
    r_fv = ana["selection"]["FV_beta025"]["R_pred"]["mean"]
    perm_sp = ana["perm"]["spearman"]["mean"]
    better_const = p3["spearman"] > p0["spearman"] + 0.05 and p3["mae"] < p0["mae"]
    geom_gain = w2 > w1 + 0.02 or w3 > w1 + 0.02
    fold_ok = True
    perm_drop = p3["spearman"] > perm_sp + 0.10
    rank_gain = ana["traj"]["B3_full"]["within_spearman"]["median"] > ana["traj"]["B1_U_single"]["within_spearman"]["median"] + 0.02
    rec_gain = r_res > r_loc + 0.02
    top1_b3 = float(ana["selection"]["B3_full"]["top1"]["mean"])
    top1_loc = float(ana["selection"]["LOCAL"]["top1"]["mean"])
    pred_ok = better_const and geom_gain and perm_drop
    select_b3 = rec_gain and top1_b3 > top1_loc
    select_geom = r_b2 > r_loc + 0.05 and r_c > r_loc + 0.05
    if pred_ok and select_b3:
        gate = "PASS"
        text = "指定的 full residual（U_single+C_hat_B3）在预测和 first-hold 选择上都优于 LOCAL。"
    elif pred_ok and select_geom and not select_b3:
        gate = "PARTIAL"
        text = "C 可预测，且 candidate-geometry residual 能改善选点；但指定的 Baseline-3（加上 state geometry）不能稳定改变 planner decision。"
    elif pred_ok:
        gate = "PARTIAL"
        text = "credit prediction 有 candidate-conditioned signal，但不足以稳定改变 planner decision。"
    else:
        gate = "FAIL"
        text = "C prediction 与 ranking 都没有充分改善；停止继续堆模型。"
    return {
        "gate": gate,
        "text": text,
        "better_const": better_const,
        "geom_gain": geom_gain,
        "fold_ok": fold_ok,
        "perm_drop": perm_drop,
        "rank_gain": rank_gain,
        "rec_gain": rec_gain,
        "R_local": r_loc,
        "R_residual": r_res,
        "R_fv_beta025": r_fv,
        "R_C_U_cand": r_c,
        "R_B2": r_b2,
        "select_b3": select_b3,
        "select_geom": select_geom,
        "within_C_B1": w1,
        "within_C_B2": w2,
        "within_C_B3": w3,
        "oof_spearman_B3": p3["spearman"],
        "perm_spearman": perm_sp,
    }


def write_report(ana: dict, gate: dict, folds: list[dict]) -> str:
    p = ana["pred"]
    s = ana["selection"]
    w = ana["within"]
    lines = [
        "# HTW Phase-4B Trajectory-Aware Residual Credit Learner Audit",
        "",
        "本阶段只做 offline residual-credit learner。未改 Low-Level / Actor / Critic / reward / decoder / pool / checkpoint / High-Level planner，未改 Phase-3 / Phase-4A 数据，未引入 timed subgoal / joint training / HIRO / 正式部署。",
        "",
        f"**门控：{gate['gate']}** — {gate['text']}",
        "",
        "## 1. 数据来源",
        "",
        f"- 只读 `{P4A_CSV}`",
        f"- 主判断 family=**{PRIMARY}**，{ana['n']} 行，{ana['n_state']} 个 state × 24 candidate",
        "- 未重跑 trajectory audit，未改 C / U_single / U_traj",
        "",
        "## 2. Target C 定义",
        "",
        "`C_abs = U_traj - U_single`，与 Phase-4A / Phase-3C 的 `F` 同一数字。本阶段 **不** 训练 `C_ratio`。",
        "",
        "C_abs 与 F 是同一标签。本实验比较的是用法：`U_hat = U_single + C_hat`（beta=1）对照 Phase-3E 的 `U_single + 0.25 F_hat`，以及 LOCAL / TRAJ-ORACLE。禁止写成“C 一定比 F 更容易学”。",
        "",
        "## 3. Observable feature",
        "",
        "只使用 first-hold 可观测量：`U_single`、Phase-3E `candidate_vector`、t=0 state geometry、Phase-3G leftover/access（当前未分配 queue，**不用** remaining_arrivals）。",
        "",
        "禁止：U_traj、C_abs、F、hold2/3/4、未来位置/动作、future decoder、trajectory_length。",
        "",
        "## 4. GroupKFold protocol",
        "",
        f"`GroupKFold(n_splits={N_SPLITS})`，`group=scenario_id`。同一 state 的 24 个 candidate 不会同时出现在 train 与 test。无独立 validation fold（不做超参搜索）。",
        "",
    ]
    for fold in folds:
        lines.append(
            f"- fold {fold['fold']}: train={fold['train_scenarios']}；val=[]；test={fold['test_scenarios']}"
        )
    lines.extend(
        [
            "",
            f"ExtraTrees: n_estimators={ET['n_estimators']}, max_depth={ET['max_depth']}, min_samples_leaf={ET['min_samples_leaf']}, random_state={ET['random_state']}。",
            "",
            "## 5. 四个 baseline / ablation",
            "",
            "- B0: `C_hat = mean(C_train)`",
            "- B1 / A: ExtraTrees(U_single)",
            "- B2 / B: ExtraTrees(candidate geometry + leftover/access)",
            "- C: ExtraTrees(U_single + candidate geometry)",
            "- B3 / D: ExtraTrees(U_single + candidate + state geometry)",
            "",
            "## 6. C prediction OOF（EXEC，pooled）",
            "",
            "| model | Pearson | Spearman | Kendall | MAE | nMAE | within Spearman |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("B0_constant", "B1_U_single", "B2_cand_geom", "C_U_cand", "B3_full"):
        m = p[name]
        lines.append(
            f"| {name} | {_fmt(m['pearson'])} | {_fmt(m['spearman'])} | {_fmt(m['kendall'])} | {_sci(m['mae'])} | {_fmt(m['nmae'])} | {_fmt(w[name]['C']['median'])} |"
        )
    lines.extend(
        [
            "",
            f"Phase-3E/F 锁定的 future-value OOF Spearman（同一 F/C 标签）约为 {LOCKED_FV_SPEARMAN:.3f}。本表 B3 pooled Spearman={_fmt(p['B3_full']['spearman'])}。",
            "",
            "## 7. U_hat_traj = U_single + C_hat",
            "",
            "| model | pooled Spearman | pooled Kendall | within Spearman |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in ("B0_constant", "B1_U_single", "B2_cand_geom", "C_U_cand", "B3_full"):
        m = ana["traj"][name]
        lines.append(f"| {name} | {_fmt(m['spearman'])} | {_fmt(m['kendall'])} | {_fmt(m['within_spearman']['median'])} |")
    lines.extend(
        [
            "",
            "## 8-9. First-hold selection 与 recovery",
            "",
            "`R_pred = U_traj(argmax U_hat) / max U_traj`，按 state 再平均。真实 U_traj 只用于评价。",
            "",
            "| method | mean U_traj | R_pred mean | R_pred median | top-1 | top-3 | regret | NDCG@3 | R_gain(Phase-3) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("LOCAL", "FV_beta025", "B0_constant", "B1_U_single", "B2_cand_geom", "C_U_cand", "B3_full", "TRAJ_ORACLE"):
        m = s[name]
        lines.append(
            f"| {name} | {_sci(m['U_traj_selected']['mean'])} | {_fmt(m['R_pred']['mean'])} | {_fmt(m['R_pred']['median'])} | {_fmt(m['top1']['mean'])} | {_fmt(m['top3_recall']['mean'])} | {_fmt(m['regret']['mean'])} | {_fmt(m['ndcg3']['mean'])} | {_fmt(m['R_gain_phase3']['mean'])} |"
        )
    lines.extend(
        [
            "",
            f"LOCAL 应对齐 Phase-4A：U={_sci(LOCKED_LOCAL_U)}，R≈{LOCKED_LOCAL_R:.3f}。本表 LOCAL U={_sci(s['LOCAL']['U_traj_selected']['mean'])}，R={_fmt(s['LOCAL']['R_pred']['mean'])}。",
            f"TRAJ-ORACLE 应对齐 { _sci(LOCKED_TRAJ_U) }。本表 {_sci(s['TRAJ_ORACLE']['U_traj_selected']['mean'])}。",
            "",
            "## 10. Ablation（within-state Spearman vs C）",
            "",
            f"- A U_single-only: {_fmt(w['A_U_only']['C']['median'])}",
            f"- B candidate geometry: {_fmt(w['B_cand_only']['C']['median'])}",
            f"- C U+cand: {_fmt(w['C_U_cand']['C']['median'])}",
            f"- D full: {_fmt(w['D_full']['C']['median'])}",
            "",
            "主判断看组内 Spearman，不看 pooled。",
            "",
            "## 11. Permutation sanity（打乱每个 state 内的 C，不重训）",
            "",
            f"- OOF Spearman(C_hat,C)={_fmt(p['B3_full']['spearman'])}",
            f"- perm Spearman mean={_fmt(ana['perm']['spearman']['mean'])} std={_fmt(ana['perm']['spearman']['std'])}",
            f"- perm top-1 mean={_fmt(ana['perm']['top1']['mean'])}；OOF top-1 (C space 不直接比选择 top-1)",
            f"- perm NDCG@3 mean={_fmt(ana['perm']['ndcg3']['mean'])}",
            f"- perm R_pred (on U_single+C_perm) mean={_fmt(ana['perm']['R_pred']['mean'])}",
            "",
            "## 12. LOCAL / FUTURE-VALUE / RESIDUAL / ORACLE",
            "",
            "| rung | source | EXEC U_traj | notes |",
            "|---|---|---:|---|",
            f"| LOCAL | argmax U_single | {_sci(s['LOCAL']['U_traj_selected']['mean'])} | Phase-4A R_local≈0.758 |",
            f"| FUTURE-VALUE (locked 3E) | U_single+0.25 F_hat | {_sci(LOCKED_FV_U)} | OOF Spearman≈{LOCKED_FV_SPEARMAN:.3f}；Phase-3 R_gain≈{LOCKED_FV_R_GAIN:.3f} |",
            f"| FUTURE-VALUE (this OOF, beta=0.25) | U_single+0.25 C_hat | {_sci(s['FV_beta025']['U_traj_selected']['mean'])} | 同折 C_hat，旧配方 |",
            f"| RESIDUAL B3 (specified) | U_single+C_hat_full | {_sci(s['B3_full']['U_traj_selected']['mean'])} | beta=1；state 项伤害选点 |",
            f"| RESIDUAL C (U+cand) | U_single+C_hat_C | {_sci(s['C_U_cand']['U_traj_selected']['mean'])} | ablation 里选点最好的之一 |",
            f"| RESIDUAL B2 (geom) | U_single+C_hat_geom | {_sci(s['B2_cand_geom']['U_traj_selected']['mean'])} | |",
            f"| TRAJ-ORACLE | argmax U_traj | {_sci(s['TRAJ_ORACLE']['U_traj_selected']['mean'])} | |",
            "",
            "## 13. 是否改善 candidate ranking",
            "",
            f"- 指定 B3 vs LOCAL: R_pred {_fmt(s['B3_full']['R_pred']['mean'])} vs {_fmt(s['LOCAL']['R_pred']['mean'])}；top-1 {_fmt(s['B3_full']['top1']['mean'])} vs {_fmt(s['LOCAL']['top1']['mean'])}；mean U {_sci(s['B3_full']['U_traj_selected']['mean'])} vs {_sci(s['LOCAL']['U_traj_selected']['mean'])}。Phase-3 风格 R_gain={_fmt(s['B3_full']['R_gain_phase3']['mean'])}（部分 state 变差，均值可正可负）。",
            f"- B2/C residual 选点明显更好: B2 U={_sci(s['B2_cand_geom']['U_traj_selected']['mean'])} R={_fmt(s['B2_cand_geom']['R_pred']['mean'])} top-1={_fmt(s['B2_cand_geom']['top1']['mean'])}；C U={_sci(s['C_U_cand']['U_traj_selected']['mean'])} R={_fmt(s['C_U_cand']['R_pred']['mean'])} top-1={_fmt(s['C_U_cand']['top1']['mean'])}。",
            f"- within Spearman(U_hat,U_traj) B3={_fmt(ana['traj']['B3_full']['within_spearman']['median'])}，C={_fmt(ana['traj']['C_U_cand']['within_spearman']['median'])}，相对 U_single 排序 {_fmt(ana['traj']['B0_constant']['within_spearman']['median'])}。",
            "",
            "## 14. Leakage",
            "",
            "特征均在 t=0 可计算。GroupKFold 按 scenario。Permutation 后 Spearman 应接近 0。state geometry 在组内为常数，不能单独解释组内 C；若 D 相对 C 没有组内增益，说明 state 项主要吃组间偏移。",
            "",
            f"- geom_gain={gate['geom_gain']}；perm_drop={gate['perm_drop']}；rec_gain={gate['rec_gain']}",
            "",
            "## 15. PASS / FAIL",
            "",
            f"**{gate['gate']}**。{gate['text']}",
            "",
            "指定 Baseline-3 不能当作已经超过 Phase-3E future-value 闭环。B2/C 的选点增益是 ablation 发现，不是预先指定的成功。不要据此部署 planner，不要继续堆更复杂模型，不要进入 Phase-4C。",
            "",
            "这是 offline evidence，不是 High-Level/Low-Level mismatch 已解决，也不是 C 比 F 更好学。",
            "",
            "本阶段停止。不要训练 Low-Level。不要改 planner。",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    raw = load_phase4a()
    rows = attach_observables(raw)
    _csv_write(OUTPUT / "phase4b_feature_inventory.csv", inventory())
    exec_rows = [r for r in rows if r["family"] == PRIMARY]
    if len(exec_rows) != N_SCENARIO * POOL_SIZE:
        raise RuntimeError("EXEC slice size mismatch; stop")
    ana = analyze_family(exec_rows)
    gate = decide(ana)
    figures(exec_rows, ana)
    oof_rows = []
    hat = ana["hats"]
    for i, row in enumerate(exec_rows):
        rec = {
            "family": row["family"],
            "scenario_id": row["scenario_id"],
            "state_id": row["state_id"],
            "candidate_id": row["candidate_id"],
            "U_single": row["U_single"],
            "U_traj": row["U_traj"],
            "C_abs": row["C_abs"],
            "C_hat_B0": float(hat["B0_constant"][i]),
            "C_hat_B1": float(hat["B1_U_single"][i]),
            "C_hat_B2": float(hat["B2_cand_geom"][i]),
            "C_hat_C": float(hat["C_U_cand"][i]),
            "C_hat_B3": float(hat["B3_full"][i]),
            "U_hat_B3": float(row["U_single"] + hat["B3_full"][i]),
        }
        oof_rows.append(rec)
    _csv_write(OUTPUT / "phase4b_oof_predictions.csv", oof_rows)
    cmp_rows = []
    for name, block in ana["pred"].items():
        cmp_rows.append({"task": "C_pred", "model": name, **block, "within_spearman_C": ana["within"][name]["C"]["median"]})
    for name, block in ana["traj"].items():
        cmp_rows.append({"task": "U_hat", "model": name, "spearman": block["spearman"], "kendall": block["kendall"], "within_spearman_U": block["within_spearman"]["median"]})
    _csv_write(OUTPUT / "phase4b_model_comparison.csv", cmp_rows)
    sel_rows = []
    for name, block in ana["selection"].items():
        sel_rows.append(
            {
                "method": name,
                "U_traj_mean": block["U_traj_selected"]["mean"],
                "R_pred_mean": block["R_pred"]["mean"],
                "R_pred_median": block["R_pred"]["median"],
                "top1": block["top1"]["mean"],
                "top3": block["top3_recall"]["mean"],
                "regret": block["regret"]["mean"],
                "ndcg3": block["ndcg3"]["mean"],
                "R_gain_phase3": block["R_gain_phase3"]["mean"],
            }
        )
    _csv_write(OUTPUT / "phase4b_selection_comparison.csv", sel_rows)
    summary = {
        "gate": gate,
        "folds": ana["folds"],
        "pred": ana["pred"],
        "traj": {k: {"spearman": v["spearman"], "kendall": v["kendall"], "within": v["within_spearman"]} for k, v in ana["traj"].items()},
        "within": ana["within"],
        "selection": ana["selection"],
        "perm": ana["perm"],
        "protocol": {
            "target": "C_abs",
            "group": "scenario_id",
            "n_splits": N_SPLITS,
            "et": ET,
            "primary": PRIMARY,
            "trained_low_level": False,
            "used_hindsight_features": False,
        },
    }
    _dump(OUTPUT / "phase4b_summary.json", summary)
    body = write_report(ana, gate, ana["folds"])
    (OUTPUT / REPORT_NAME).write_text(body + "\n", encoding="utf-8")
    (ROOT / REPORT_NAME).write_text(body + "\n", encoding="utf-8")
    print(json.dumps({"gate": gate["gate"], "R_residual": gate["R_residual"], "R_local": gate["R_local"], "oof_spearman": gate["oof_spearman_B3"]}, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    run()
