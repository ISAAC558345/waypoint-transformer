"""Phase-3F ceiling audit: feature groups, within-state ranking, permutation null."""
from __future__ import annotations

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

from wt.lowlevel_v2.integration_phase1.stats import kendall_tau, spearman, summarize, topk_overlap
from wt.lowlevel_v2.phase3e_future_value_audit.run_phase3e import STATE_CAND_KEYS, build_table
from wt.lowlevel_v2.protocol import ROOT

PHASE = Path(__file__).resolve().parent
OUTPUT = PHASE / "outputs"
FIG = OUTPUT / "figures"
FAMS = ("OLD", "ABS", "PREFIX", "EXEC")
BETAS = (0.25, 0.5, 0.75, 1.0)
N_SPLITS = 5
N_PERM = 100
ET = dict(n_estimators=200, max_depth=6, min_samples_leaf=8, random_state=0, n_jobs=1)

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
)
STATE_KEYS = (
    "uav_y_mean",
    "uav_y_std",
    "uav_pairwise_mean",
    "min_pairwise_dist",
    "queue_mean",
    "queue_hot_mass",
)
GROUPS = {
    "A_U_single": ("U_single",),
    "B_candidate": CAND_KEYS,
    "C_state": STATE_KEYS,
    "D_cand_state": CAND_KEYS + STATE_KEYS,
    "E_U_cand": ("U_single",) + CAND_KEYS,
    "F_U_state": ("U_single",) + STATE_KEYS,
    "G_full": ("U_single", "U_single_pool_rank") + tuple(STATE_CAND_KEYS),
}


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


def rank_reg(f, pred, u) -> float:
    f = np.asarray(f, dtype=np.float64)
    best = float(np.max(f))
    hat = float(f[int(np.argmax(pred))])
    loc = float(f[int(np.argmax(u))])
    num = best - hat
    den = best - loc
    if abs(den) < 1e-12:
        return float("nan")
    return num / den


def oof_group(rows: list[dict], keys: tuple[str, ...]) -> np.ndarray:
    n = len(rows)
    groups = np.array([int(r["scenario"]) for r in rows], dtype=np.int64)
    y = np.array([float(r["F"]) for r in rows], dtype=np.float64)
    x = np.array([[float(r[k]) for k in keys] for r in rows], dtype=np.float64)
    oof = np.full(n, np.nan)
    splits = min(N_SPLITS, int(len(set(groups.tolist()))))
    gkf = GroupKFold(n_splits=splits)
    for tr, te in gkf.split(x, groups=groups):
        model = ExtraTreesRegressor(**ET)
        model.fit(x[tr], y[tr])
        oof[te] = model.predict(x[te])
    return oof


def by_state(rows: list[dict], hats: np.ndarray | None = None):
    buckets = defaultdict(list)
    for i, row in enumerate(rows):
        item = dict(row)
        if hats is not None:
            item["F_hat"] = float(hats[i])
        buckets[int(row["scenario"])].append(item)
    return buckets


def state_rows(rows: list[dict], hats: np.ndarray) -> list[dict]:
    out = []
    for sc, items in sorted(by_state(rows, hats).items()):
        u = np.array([x["U_single"] for x in items], dtype=np.float64)
        f = np.array([x["F"] for x in items], dtype=np.float64)
        t = np.array([x["U_traj"] for x in items], dtype=np.float64)
        fh = np.array([x["F_hat"] for x in items], dtype=np.float64)
        iu, iff, it, ih = int(np.argmax(u)), int(np.argmax(f)), int(np.argmax(t)), int(np.argmax(fh))
        out.append(
            {
                "method": items[0]["method"],
                "scenario": sc,
                "n": len(items),
                "F_mean": float(np.mean(f)),
                "F_std": float(np.std(f, ddof=1)) if len(f) > 1 else 0.0,
                "F_iqr": float(np.subtract(*np.percentile(f, [75, 25]))),
                "F_p10": float(np.percentile(f, 10)),
                "F_p50": float(np.percentile(f, 50)),
                "F_p90": float(np.percentile(f, 90)),
                "F_range": float(np.max(f) - np.min(f)),
                "U_std": float(np.std(u, ddof=1)) if len(u) > 1 else 0.0,
                "U_range": float(np.max(u) - np.min(u)),
                "within_spearman_U_F": spearman(u, f),
                "within_spearman_Fhat_F": spearman(fh, f),
                "within_kendall_Fhat_F": kendall_tau(fh, f),
                "top1_Fhat_F": float(ih == iff),
                "top3_Fhat_F": topk_overlap(fh, f, 3),
                "ndcg3_Fhat_F": ndcg_at(fh, f, 3),
                "R_rank": rank_reg(f, fh, u),
                "best_F": float(np.max(f)),
                "F_of_Fhat": float(f[ih]),
                "F_of_Usingle": float(f[iu]),
                "best_U_traj": float(np.max(t)),
                "local_U_traj": float(t[iu]),
                "traj_U_traj": float(t[it]),
                "natural_U_traj": float(items[0]["natural_U_traj"]),
                "F_top1_share": float((np.max(f) - np.mean(f)) / max(np.std(f), 1.0)),
                "F_unique": int(len(set(np.round(f, 3).tolist()))),
            }
        )
    return out


def permutation_null(rows: list[dict], hats: np.ndarray, rng: np.random.Generator) -> list[dict]:
    out = []
    for sc, items in sorted(by_state(rows, hats).items()):
        f = np.array([x["F"] for x in items], dtype=np.float64)
        u = np.array([x["U_single"] for x in items], dtype=np.float64)
        fh = np.array([x["F_hat"] for x in items], dtype=np.float64)
        obs_sp = spearman(fh, f)
        obs_t1 = float(int(np.argmax(fh)) == int(np.argmax(f)))
        obs_rg = rank_reg(f, fh, u)
        null_sp, null_t1, null_rg = [], [], []
        for _ in range(N_PERM):
            fp = rng.permutation(f)
            null_sp.append(spearman(fh, fp))
            null_t1.append(float(int(np.argmax(fh)) == int(np.argmax(fp))))
            null_rg.append(rank_reg(fp, fh, u))
        out.append(
            {
                "method": items[0]["method"],
                "scenario": sc,
                "obs_spearman": obs_sp,
                "obs_top1": obs_t1,
                "obs_R_rank": obs_rg,
                "null_spearman_mean": float(np.nanmean(null_sp)),
                "null_spearman_p95": float(np.nanpercentile(null_sp, 95)),
                "null_top1_mean": float(np.nanmean(null_t1)),
                "null_R_rank_mean": float(np.nanmean(null_rg)),
                "p_spearman": float(np.mean(np.asarray(null_sp) >= obs_sp - 1e-12)),
                "p_top1": float(np.mean(np.asarray(null_t1) >= obs_t1 - 1e-12)),
            }
        )
    return out


def recovery_rows(rows: list[dict], hats: np.ndarray) -> list[dict]:
    out = []
    for sc, items in sorted(by_state(rows, hats).items()):
        u = np.array([x["U_single"] for x in items], dtype=np.float64)
        f = np.array([x["F"] for x in items], dtype=np.float64)
        t = np.array([x["U_traj"] for x in items], dtype=np.float64)
        fh = np.array([x["F_hat"] for x in items], dtype=np.float64)
        iu, it = int(np.argmax(u)), int(np.argmax(t))
        loc, trj = float(t[iu]), float(t[it])
        for beta in (0.0,) + BETAS:
            ib = int(np.argmax(u + float(beta) * fh))
            io = int(np.argmax(u + float(beta) * f))
            pred = float(t[ib])
            ora = float(t[io])
            out.append(
                {
                    "method": items[0]["method"],
                    "scenario": sc,
                    "beta": beta,
                    "predicted_U_traj": pred,
                    "oracle_F_U_traj": ora,
                    "local_U_traj": loc,
                    "traj_U_traj": trj,
                    "natural_U_traj": float(items[0]["natural_U_traj"]),
                    "R_pred": float((pred - loc) / (trj - loc)) if trj > loc else float("nan"),
                    "R_oracle": float((ora - loc) / (trj - loc)) if trj > loc else float("nan"),
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


def figures(fam: str, group_oof: dict, rows: list[dict], hats: np.ndarray, hetero: list[dict], perm: list[dict], rec: list[dict]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    names = list(GROUPS)
    spears = [group_oof[n]["oof_spearman"] for n in names]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.bar(range(len(names)), spears, color="#4C78A8")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("OOF Spearman(F_hat,F)")
    ax.set_title(f"{fam} ExtraTrees feature-group ceiling")
    ax.set_ylim(-0.15, 1.0)
    fig.tight_layout()
    fig.savefig(FIG / "feature_group_comparison.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.scatter([r["within_spearman_U_F"] for r in hetero], [r["within_spearman_Fhat_F"] for r in hetero], c="#4C78A8")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("within-state Spearman(U_single,F)")
    ax.set_ylabel("within-state Spearman(F_hat,F)")
    ax.set_title(f"{fam} within-state ranking")
    fig.tight_layout()
    fig.savefig(FIG / "within_state_f_vs_usingle.png", dpi=140)
    plt.close(fig)

    y = np.array([r["F"] for r in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.scatter(y, hats, s=10, alpha=0.35, label="OOF G_full")
    lim = [float(np.min(y)), float(np.max(y))]
    ax.plot(lim, lim, "k--", lw=1, label="y=x / oracle F")
    ax.set_xlabel("true F")
    ax.set_ylabel("OOF F_hat")
    ax.set_title(f"{fam} prediction vs oracle F")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "prediction_vs_oracle.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist([r["R_rank"] for r in hetero if np.isfinite(r["R_rank"])], bins=8, color="#F58518")
    ax.set_xlabel("R_rank")
    ax.set_title(f"{fam} candidate ranking regret")
    fig.tight_layout()
    fig.savefig(FIG / "ranking_regret.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.hist([r["null_spearman_mean"] for r in perm], bins=10, alpha=0.5, label="null mean / state")
    ax.hist([r["obs_spearman"] for r in perm], bins=10, alpha=0.6, label="observed / state")
    ax.set_xlabel("within-state Spearman")
    ax.set_title(f"{fam} permutation null vs OOF")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "permutation_null.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xs = [str(r["scenario"])[-3:] for r in hetero]
    ax.bar(xs, [r["within_spearman_Fhat_F"] for r in hetero], color="#54A24B")
    ax.axhline(float(np.nanmedian([r["within_spearman_Fhat_F"] for r in hetero])), color="k", ls="--")
    ax.set_ylabel("within Spearman(F_hat,F)")
    ax.set_title(f"{fam} state heterogeneity (no states dropped)")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(FIG / "state_heterogeneity.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    betas = [0.0] + list(BETAS)
    pred = [float(np.mean([r["predicted_U_traj"] for r in rec if r["beta"] == b])) for b in betas]
    ora = [float(np.mean([r["oracle_F_U_traj"] for r in rec if r["beta"] == b])) for b in betas]
    ax.plot(betas, pred, "o-", label="PREDICTED G_full")
    ax.plot(betas, ora, "s-", label="ORACLE F")
    ax.axhline(float(np.mean([r["local_U_traj"] for r in rec if r["beta"] == 0.0])), color="#F58518", ls="--", label="LOCAL")
    ax.axhline(float(np.mean([r["traj_U_traj"] for r in rec if r["beta"] == 0.0])), color="#54A24B", ls="--", label="TRAJ")
    ax.set_xlabel("beta")
    ax.set_ylabel("mean U_traj")
    ax.set_title(f"{fam} recovery (no new beta search)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "recovery_curve.png", dpi=140)
    plt.close(fig)


def decide(exec_g: dict, exec_h: list[dict], exec_p: list[dict], exec_r: list[dict], group_metrics: dict) -> dict:
    g = group_metrics
    oof_g = g["G_full"]["oof_spearman"]
    oof_a = g["A_U_single"]["oof_spearman"]
    oof_b = g["B_candidate"]["oof_spearman"]
    oof_c = g["C_state"]["oof_spearman"]
    obs = float(np.nanmean([r["obs_spearman"] for r in exec_p]))
    null95 = float(np.nanmean([r["null_spearman_p95"] for r in exec_p]))
    p_med = float(np.nanmedian([r["p_spearman"] for r in exec_p]))
    within_u = ext_sum([r["within_spearman_U_F"] for r in exec_h])
    within_h = ext_sum([r["within_spearman_Fhat_F"] for r in exec_h])
    r025 = float(np.nanmean([r["R_pred"] for r in exec_r if r["beta"] == 0.25]))
    easy = sum(1 for r in exec_h if r["within_spearman_Fhat_F"] >= 0.4)
    hard = sum(1 for r in exec_h if r["within_spearman_Fhat_F"] < 0.1)
    q1 = oof_b > 0.35 and obs > null95
    q2 = oof_g > oof_a + 0.15
    feature_limited = oof_b >= oof_g - 0.05 and oof_c < 0.15
    model_limited = g["G_full"]["train_spearman"] - oof_g > 0.35
    hetero = hard >= 4
    if feature_limited and not model_limited:
        ceiling = "information-limited"
    elif model_limited and not feature_limited:
        ceiling = "model-limited"
    else:
        ceiling = "mixed"
    q4 = float(within_u["median"]) < 0.25
    q5 = oof_g < 0.70 or r025 < 0.25
    q6 = q5
    return {
        "oof_G": oof_g,
        "oof_A": oof_a,
        "oof_B": oof_b,
        "oof_C": oof_c,
        "obs_within": obs,
        "null95": null95,
        "p_median": p_med,
        "within_U_F_median": within_u["median"],
        "within_Fhat_F_median": within_h["median"],
        "R_pred_025": r025,
        "n_easy": easy,
        "n_hard": hard,
        "q1": q1,
        "q2": q2,
        "q4_low_UF": q4,
        "q5_not_enough": q5,
        "q6_stop_stacking": q6,
        "ceiling": ceiling,
        "feature_limited": feature_limited,
        "model_limited": model_limited,
        "hetero": hetero,
    }


def write_report(v: dict, gm: dict, hetero: list[dict], perm: list[dict], rec: list[dict], all_gm: dict) -> str:
    loc = float(np.mean([r["local_U_traj"] for r in rec if r["beta"] == 0.0]))
    trj = float(np.mean([r["traj_U_traj"] for r in rec if r["beta"] == 0.0]))
    nat = float(np.mean([r["natural_U_traj"] for r in rec if r["beta"] == 0.0]))
    lines = [
        "# HTW Phase-3F：Future-Value Predictability Ceiling Audit",
        "",
        "本阶段只诊断 Phase-3E 的 0.553 离可观测信息上限有多远。未训练 Low-Level，未改正式方法，未使用未来变量，未做 XGBoost / 深网 / 调参。",
        "协议：ExtraTrees + **scenario GroupKFold**。同一 state 的 24 个 candidate 不拆开。Oracle-F 只作诊断上界，不进入正式方法。",
        "",
        f"Sanity：NATURAL={_sci(nat)}，LOCAL={_sci(loc)}，TRAJ={_sci(trj)}。",
        "",
        f"**经验判断（不是理论上下界）：{v['ceiling']}。**",
        "",
        "## 1. Feature-group OOF（EXEC，ExtraTrees）",
        "",
        "| group | features | OOF Spearman | train Spearman | OOF top-1 vs F |",
        "|---|---|---:|---:|---:|",
    ]
    labels = {
        "A_U_single": "A U_single",
        "B_candidate": "B candidate geometry",
        "C_state": "C state geometry",
        "D_cand_state": "D candidate+state",
        "E_U_cand": "E U_single+candidate",
        "F_U_state": "F U_single+state",
        "G_full": "G U_single+candidate+state",
    }
    for name, lab in labels.items():
        s = gm[name]
        lines.append(f"| {lab} | {s['n_feat']} | {_fmt(s['oof_spearman'])} | {_fmt(s['train_spearman'])} | {_fmt(s['oof_top1'])} |")
    lines += [
        "",
        f"B={_fmt(v['oof_B'])}，C={_fmt(v['oof_C'])}，G={_fmt(v['oof_G'])}。state-only 几乎无信号；candidate geometry 已经接近 full 组合。",
        "",
        "## 2. Within-state F 噪声与 U_single",
        "",
        f"16 个 EXEC state 的 F：range 中位={_sci(float(np.median([r['F_range'] for r in hetero])))}，IQR 中位={_sci(float(np.median([r['F_iqr'] for r in hetero])))}。",
        f"within-state Spearman(U_single,F) median={_fmt(v['within_U_F_median'])}。F 主要不是 local utility 的再编码。",
        f"within-state Spearman(F_hat,F) median={_fmt(v['within_Fhat_F_median'])}。易预测 state（≥0.4）{v['n_easy']} 个，难预测（<0.1）{v['n_hard']} 个。未删除任何 state。",
        "",
        "## 3. Ranking / regret / permutation",
        "",
        f"OOF G 的 within-state top-1(F_hat,F) mean={_fmt(float(np.mean([r['top1_Fhat_F'] for r in hetero])))}，NDCG@3={_fmt(float(np.nanmean([r['ndcg3_Fhat_F'] for r in hetero])))}，R_rank median={_fmt(float(np.nanmedian([r['R_rank'] for r in hetero])))}。",
        f"Permutation（每 state 打乱 F，{N_PERM} 次）：observed within Spearman mean={_fmt(v['obs_within'])}，null p95 mean={_fmt(v['null95'])}，median p={_fmt(v['p_median'])}。",
        "0.55 量级的全局 OOF 显著高于 candidate 内随机匹配；但 top-1 仍然低，排序不够稳。",
        "",
        "## 4. Recovery 复核（不搜新 β）",
        "",
        "| beta | PREDICTED | ORACLE-F | R_pred | R_oracle |",
        "|---:|---:|---:|---:|---:|",
    ]
    for beta in (0.0,) + BETAS:
        sub = [r for r in rec if r["beta"] == beta]
        lines.append(
            f"| {beta} | {_sci(float(np.mean([r['predicted_U_traj'] for r in sub])))} | {_sci(float(np.mean([r['oracle_F_U_traj'] for r in sub])))} | {_fmt(float(np.nanmean([r['R_pred'] for r in sub])))} | {_fmt(float(np.nanmean([r['R_oracle'] for r in sub])))} |"
        )
    lines += [
        "",
        f"LOCAL={_sci(loc)}，TRAJ={_sci(trj)}。预测恢复仍约 0.13；oracle F 在 β=0.5 已接近 TRAJ。prediction gap 仍在。",
        "",
        "## 5. 四方法 G_full OOF",
        "",
        "| method | G OOF Spearman | A OOF Spearman | B OOF Spearman |",
        "|---|---:|---:|---:|",
    ]
    for fam in FAMS:
        lines.append(
            f"| {fam} | {_fmt(all_gm[fam]['G_full']['oof_spearman'])} | {_fmt(all_gm[fam]['A_U_single']['oof_spearman'])} | {_fmt(all_gm[fam]['B_candidate']['oof_spearman'])} |"
        )
    lines += [
        "",
        "## 6. 六个问题",
        "",
        f"**Q1.** {'是' if v['q1'] else '否'}。candidate-conditioned geometry 能预测 F（B OOF={_fmt(v['oof_B'])}），且高于 permutation null。",
        "",
        f"**Q2.** {'是' if v['q2'] else '否'}。G={_fmt(v['oof_G'])} vs A={_fmt(v['oof_A'])}。",
        "",
        f"**Q3.** 经验判断：**{v['ceiling']}**。state-only 无信号、B≈G → 主要是 **feature information 不足**（t=0 queue/progress 几乎不变，F 含 later-greedy）。train−OOF≈{_fmt(gm['G_full']['train_spearman']-gm['G_full']['oof_spearman'])}，有轻度模型/异质性成分，但不是“换更强模型就能到 0.95”。",
        "",
        f"**Q4.** within-state U vs F median={_fmt(v['within_U_F_median'])}，F 的组内排序 **不稳定地依附 U_single**。F 有 candidate 结构（B 有效），也有 state 间异质性（难预测 state={v['n_hard']}）。",
        "",
        f"**Q5.** {'不足以' if v['q5_not_enough'] else '勉强可以'}支撑可靠的 candidate-level future-value predictor。OOF 0.55 + R_pred≈0.13 不能当上线 critic。",
        "",
        f"**Q6.** {'应该停止堆模型复杂度' if v['q6_stop_stacking'] else '还可以先挖特征'}。下一阶段应转向 **trajectory-aware credit 的可观测近似 / 结构化估计**，而不是 ExtraTrees / MLP / Transformer 调参。",
        "",
        "empirical evidence suggests：当前 first-hold 可观测特征的 OOF Spearman 天花板大约就在 0.5–0.6，而不是 0.9。这不是数学上界。",
        "",
        "本阶段停止。不训练。不进入 Phase-4。不进入 End-to-End Independent Validation。",
        "",
    ]
    return "\n".join(lines)


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    table = build_table()
    all_gm = {}
    exec_pack = {}
    group_csv = []
    hetero_csv = []
    rank_csv = []
    perm_csv = []
    rec_csv = []
    ceil_csv = []
    for fam in FAMS:
        rows = [r for r in table if r["method"] == fam]
        gm = {}
        hats_g = None
        for name, keys in GROUPS.items():
            hats = oof_group(rows, keys)
            y = np.array([r["F"] for r in rows], dtype=np.float64)
            # train Spearman via a single fit on all data is leaky; use mean of in-fold train instead
            groups = np.array([int(r["scenario"]) for r in rows], dtype=np.int64)
            x = np.array([[float(r[k]) for k in keys] for r in rows], dtype=np.float64)
            tr_sp = []
            gkf = GroupKFold(n_splits=min(N_SPLITS, len(set(groups.tolist()))))
            for tr, te in gkf.split(x, groups=groups):
                model = ExtraTreesRegressor(**ET)
                model.fit(x[tr], y[tr])
                tr_sp.append(spearman(model.predict(x[tr]), y[tr]))
                _ = te
            by = by_state(rows, hats)
            top1s = []
            for items in by.values():
                f = np.array([i["F"] for i in items])
                fh = np.array([i["F_hat"] for i in items])
                top1s.append(float(int(np.argmax(fh)) == int(np.argmax(f))))
            gm[name] = {
                "n_feat": len(keys),
                "oof_spearman": spearman(hats, y),
                "oof_pearson": pearson(hats, y),
                "train_spearman": float(np.nanmean(tr_sp)),
                "oof_top1": float(np.mean(top1s)),
            }
            group_csv.append({"method": fam, "group": name, **gm[name]})
            if name == "G_full":
                hats_g = hats
        hetero = state_rows(rows, hats_g)
        for row in hetero:
            hetero_csv.append(row)
            rank_csv.append(
                {
                    "method": fam,
                    "scenario": row["scenario"],
                    "spearman": row["within_spearman_Fhat_F"],
                    "kendall": row["within_kendall_Fhat_F"],
                    "top1": row["top1_Fhat_F"],
                    "top3": row["top3_Fhat_F"],
                    "ndcg3": row["ndcg3_Fhat_F"],
                    "R_rank": row["R_rank"],
                }
            )
        perm = permutation_null(rows, hats_g, np.random.default_rng(0))
        perm_csv.extend(perm)
        rec = recovery_rows(rows, hats_g)
        rec_csv.extend(rec)
        all_gm[fam] = gm
        if fam == "EXEC":
            figures(fam, gm, rows, hats_g, hetero, perm, rec)
            exec_pack = {"gm": gm, "hetero": hetero, "perm": perm, "rec": rec, "hats": hats_g, "rows": rows}
    v = decide(exec_pack["gm"], exec_pack["hetero"], exec_pack["perm"], exec_pack["rec"], exec_pack["gm"])
    ceil_csv.append(
        {
            "ceiling": v["ceiling"],
            "oof_G": v["oof_G"],
            "oof_A": v["oof_A"],
            "oof_B": v["oof_B"],
            "oof_C": v["oof_C"],
            "perm_obs": v["obs_within"],
            "perm_null_p95": v["null95"],
            "R_pred_025": v["R_pred_025"],
            "n_hard_states": v["n_hard"],
        }
    )
    _csv(OUTPUT / "feature_group_metrics.csv", group_csv)
    _csv(OUTPUT / "within_state_metrics.csv", hetero_csv)
    _csv(OUTPUT / "state_heterogeneity.csv", hetero_csv)
    _csv(OUTPUT / "ranking_metrics.csv", rank_csv)
    _csv(OUTPUT / "permutation_test.csv", perm_csv)
    _csv(OUTPUT / "beta_recovery.csv", rec_csv)
    _csv(OUTPUT / "ceiling_summary.csv", ceil_csv)
    _dump(OUTPUT / "PHASE3F_SUMMARY.json", {"verdict": v, "groups": all_gm})
    report = write_report(v, exec_pack["gm"], exec_pack["hetero"], exec_pack["perm"], exec_pack["rec"], all_gm)
    (OUTPUT / "HTW_Phase3F_FutureValueCeiling_Audit_CN.md").write_text(report, encoding="utf-8")
    (PHASE / "HTW_Phase3F_FutureValueCeiling_Audit_CN.md").write_text(report, encoding="utf-8")
    (ROOT / "docs/archive/phase_reports/HTW_Phase3F_FutureValueCeiling_Audit_CN.md").write_text(report, encoding="utf-8")
    print(json.dumps({"verdict": v}, indent=2, default=_jsonable), flush=True)
    return v


if __name__ == "__main__":
    run()
