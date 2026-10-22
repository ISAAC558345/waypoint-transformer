"""Phase-3H: candidate-conditioned discrete decision-credit audit. No official training."""
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

from wt.l3a.env import competition_pairs
from wt.lowlevel_v2.integration_phase1.stats import kendall_tau, spearman, topk_overlap
from wt.lowlevel_v2.phase3e_future_value_audit.run_phase3e import STATE_CAND_KEYS, build_table, reset_env
from wt.lowlevel_v2.phase3f_future_value_ceiling_audit.run_phase3f import ndcg_at
from wt.lowlevel_v2.phase3g_trajectory_credit_structural_audit.run_phase3g import enrich, zscore
from wt.lowlevel_v2.protocol import ROOT

PHASE = Path(__file__).resolve().parent
OUTPUT = PHASE / "outputs"
FIG = OUTPUT / "figures"
FAMS = ("OLD", "ABS", "PREFIX", "EXEC")
ALPHAS = (0.25, 0.5, 0.75)
DISC_ALPHA = 0.5
FHAT_BETA = 0.25
STRUCT_OC = (0.5, 0.25)
N_SPLITS = 5
ET = dict(n_estimators=200, max_depth=6, min_samples_leaf=8, random_state=0, n_jobs=1)


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


def recovery(u_m, u_loc, u_trj) -> float:
    den = float(u_trj) - float(u_loc)
    if abs(den) < 1e-12:
        return float("nan")
    return (float(u_m) - float(u_loc)) / den


def gamma_of(row: dict) -> np.ndarray:
    return np.asarray([int(row[f"assign_{i}"]) for i in range(6)], dtype=np.int64)


def discrete_of(env, gamma: np.ndarray) -> dict[str, float]:
    chosen = np.asarray(gamma, dtype=np.int64)
    n_uav = int(env.config.num_uavs)
    n_buoy = int(env.config.num_buoys)
    occupied = {int(b) for b in chosen if int(b) >= 0}
    in_range = np.zeros((n_uav, n_buoy), dtype=bool)
    for uav in range(n_uav):
        for buoy in range(n_buoy):
            in_range[uav, buoy] = bool(env._in_range(uav, buoy))
    leftover = [b for b in range(n_buoy) if b not in occupied]
    degree = 0
    weighted = 0.0
    for uav in range(n_uav):
        for buoy in leftover:
            if in_range[uav, buoy]:
                degree += 1
                weighted += float(env.queue[buoy])
    leftover_coverable = float(sum(1 for b in leftover if bool(in_range[:, b].any())))
    flex = 0
    for uav in range(n_uav):
        cur = int(chosen[uav])
        if cur >= 0:
            flex += 1
        for buoy in range(n_buoy):
            if buoy == cur:
                continue
            if buoy in occupied:
                continue
            if in_range[uav, buoy]:
                flex += 1
    return {
        "degree": float(degree),
        "weighted_degree": float(weighted),
        "Flex": float(flex),
        "leftover_coverable": leftover_coverable,
        "conflict": float(len(competition_pairs(chosen, n_buoy))),
        "n_occupied": float(len(occupied)),
    }


def attach_discrete(rows: list[dict]) -> list[dict]:
    envs = {}
    out = []
    for row in rows:
        sc = int(row["scenario"])
        if sc not in envs:
            envs[sc] = reset_env(sc)
        item = dict(row)
        item.update(discrete_of(envs[sc], gamma_of(row)))
        out.append(item)
    by = defaultdict(list)
    for row in out:
        by[(row["method"], int(row["scenario"]))].append(row)
    for items in by.values():
        gams = [gamma_of(r) for r in items]
        for i, row in enumerate(items):
            pool_deg = 0
            wdeg = 0.0
            for j, other in enumerate(items):
                if i == j:
                    continue
                ham = int(np.sum(gams[i] != gams[j]))
                if ham == 1:
                    pool_deg += 1
                    wdeg += float(other["U_single"])
            row["pool_degree"] = float(pool_deg)
            row["pool_weighted_degree"] = float(wdeg)
    return out


def fhat_oof(rows: list[dict]) -> np.ndarray:
    keys = ("U_single", "U_single_pool_rank") + tuple(STATE_CAND_KEYS)
    x = np.array([[float(r[k]) for k in keys] for r in rows], dtype=np.float64)
    y = np.array([float(r["F"]) for r in rows], dtype=np.float64)
    groups = np.array([int(r["scenario"]) for r in rows], dtype=np.int64)
    oof = np.full(len(rows), np.nan)
    gkf = GroupKFold(n_splits=min(N_SPLITS, len(set(groups.tolist()))))
    for tr, te in gkf.split(x, groups=groups):
        model = ExtraTreesRegressor(**ET)
        model.fit(x[tr], y[tr])
        oof[te] = model.predict(x[te])
    return oof


def attach_scores(rows: list[dict], hats: np.ndarray) -> list[dict]:
    by = defaultdict(list)
    for i, row in enumerate(rows):
        row = dict(row)
        row["F_hat"] = float(hats[i])
        by[int(row["scenario"])].append(row)
    out = []
    for items in by.values():
        u = np.array([r["U_single"] for r in items], dtype=np.float64)
        o = zscore([r["O_traj"] for r in items])
        c = zscore([r["C_transition"] for r in items])
        zu = zscore(u)
        zf = zscore([r["Flex"] for r in items])
        su = float(np.std(u))
        if su < 1e-12:
            su = 1.0
        for i, row in enumerate(items):
            row["S_fhat"] = float(row["U_single"] + FHAT_BETA * row["F_hat"])
            row["S_OC"] = float(row["U_single"] + STRUCT_OC[0] * su * o[i] + STRUCT_OC[1] * su * c[i])
            for a in ALPHAS:
                row[f"S_disc_{a}"] = float(zu[i] + a * zf[i])
            row["S_disc"] = float(zu[i] + DISC_ALPHA * zf[i])
            out.append(row)
    return out


def within_block(items: list[dict], score_key: str, target_key: str) -> dict:
    s = np.array([r[score_key] for r in items], dtype=np.float64)
    t = np.array([r[target_key] for r in items], dtype=np.float64)
    return {
        "spearman": spearman(s, t),
        "kendall": kendall_tau(s, t),
        "top1": float(int(np.argmax(s)) == int(np.argmax(t))),
        "top3": topk_overlap(s, t, 3),
        "ndcg3": ndcg_at(s, t, 3),
    }


def select_row(items: list[dict], key: str) -> dict:
    return items[int(np.argmax([r[key] for r in items]))]


def family_eval(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for row in rows:
        by[int(row["scenario"])].append(row)
    score_keys = {
        "LOCAL": "U_single",
        "STRUCT_OC": "S_OC",
        "DEGREE": "degree",
        "WDEGREE": "weighted_degree",
        "FLEX": "Flex",
        "SDISC": "S_disc",
        "FHAT": "S_fhat",
        "TRAJ": "U_traj",
    }
    for a in ALPHAS:
        score_keys[f"SDISC_a{a}"] = f"S_disc_{a}"
    within_rows = []
    sel_rows = []
    repl_rows = []
    rank_names = (
        ("LOCAL", "U_single"),
        ("degree", "degree"),
        ("weighted_degree", "weighted_degree"),
        ("Flex", "Flex"),
        ("STRUCT_OC", "S_OC"),
        ("SDISC", "S_disc"),
        ("FHAT", "S_fhat"),
        ("pool_degree", "pool_degree"),
    )
    for sc, items in sorted(by.items()):
        u = np.array([r["U_single"] for r in items], dtype=np.float64)
        f = np.array([r["F"] for r in items], dtype=np.float64)
        loc = select_row(items, "U_single")
        trj = select_row(items, "U_traj")
        topf = select_row(items, "F")
        w = {"method": items[0]["method"], "scenario": sc}
        for name, key in rank_names:
            blk = within_block(items, key, "F")
            w.update({f"{name}_{k}": v for k, v in blk.items()})
            blk_t = within_block(items, key, "U_traj")
            w.update({f"{name}_vsUtraj_{k}": v for k, v in blk_t.items()})
        within_rows.append(w)
        for name, key in score_keys.items():
            chosen = select_row(items, key)
            sel_rows.append(
                {
                    "method": items[0]["method"],
                    "scenario": sc,
                    "score": name,
                    "U_traj": chosen["U_traj"],
                    "F": chosen["F"],
                    "U_single": chosen["U_single"],
                    "payload": chosen["U_traj"],
                    "F_percentile": float(np.mean(f <= chosen["F"])),
                    "R": recovery(chosen["U_traj"], loc["U_traj"], trj["U_traj"]),
                    "top1_vs_traj": float(chosen["hash"] == trj["hash"]),
                    "degree": chosen["degree"],
                    "Flex": chosen["Flex"],
                    "conflict": chosen["conflict"],
                }
            )
        f_q75 = float(np.quantile(f, 0.75))
        f_q25 = float(np.quantile(f, 0.25))
        high = [r for r in items if r["F"] >= f_q75]
        low = [r for r in items if r["F"] <= f_q25]
        repl_rows.append(
            {
                "method": items[0]["method"],
                "scenario": sc,
                "local_degree": loc["degree"],
                "local_wdegree": loc["weighted_degree"],
                "local_flex": loc["Flex"],
                "local_conflict": loc["conflict"],
                "topF_degree": topf["degree"],
                "topF_wdegree": topf["weighted_degree"],
                "topF_flex": topf["Flex"],
                "topF_conflict": topf["conflict"],
                "d_degree": topf["degree"] - loc["degree"],
                "d_wdegree": topf["weighted_degree"] - loc["weighted_degree"],
                "d_flex": topf["Flex"] - loc["Flex"],
                "d_conflict": topf["conflict"] - loc["conflict"],
                "topF_higher_degree": float(topf["degree"] > loc["degree"]),
                "topF_higher_flex": float(topf["Flex"] > loc["Flex"]),
                "topF_lower_conflict": float(topf["conflict"] < loc["conflict"]),
                "highF_mean_degree": float(np.mean([r["degree"] for r in high])) if high else float("nan"),
                "lowF_mean_degree": float(np.mean([r["degree"] for r in low])) if low else float("nan"),
                "highF_mean_flex": float(np.mean([r["Flex"] for r in high])) if high else float("nan"),
                "lowF_mean_flex": float(np.mean([r["Flex"] for r in low])) if low else float("nan"),
                "n_unique_degree": int(len({r["degree"] for r in items})),
                "n_unique_flex": int(len({r["Flex"] for r in items})),
                "std_U": float(np.std(u)),
            }
        )
    return {"within": within_rows, "sel": sel_rows, "repl": repl_rows}


def raw_dist(rows: list[dict]) -> dict:
    keys = ("degree", "weighted_degree", "Flex", "leftover_coverable", "conflict", "pool_degree")
    out = {}
    for key in keys:
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        out[key] = {
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
            "n_unique": int(len(set(vals.tolist()))),
        }
    return out


def figures(fam: str, within: list[dict], sel: list[dict], rows: list[dict]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.scatter([r["degree"] for r in rows], [r["F"] for r in rows], s=8, alpha=0.35)
    ax.set_xlabel("degree")
    ax.set_ylabel("F")
    ax.set_title(f"{fam} F vs degree  Spearman={spearman([r['degree'] for r in rows], [r['F'] for r in rows]):.2f}")
    fig.tight_layout()
    fig.savefig(FIG / "candidate_f_vs_degree.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.scatter([r["Flex"] for r in rows], [r["F"] for r in rows], s=8, alpha=0.35)
    ax.set_xlabel("Flex")
    ax.set_ylabel("F")
    ax.set_title(f"{fam} F vs Flex  Spearman={spearman([r['Flex'] for r in rows], [r['F'] for r in rows]):.2f}")
    fig.tight_layout()
    fig.savefig(FIG / "candidate_f_vs_flex.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = ["LOCAL", "degree", "Flex", "SDISC", "STRUCT_OC", "FHAT"]
    meds = [float(np.nanmedian([r[f"{n}_spearman"] for r in within])) for n in labels]
    ax.bar(labels, meds, color=["#4C78A8", "#F58518", "#54A24B", "#72B7B2", "#E45756", "#B279A2"])
    ax.set_ylabel("median within-state Spearman vs F")
    ax.set_title(f"{fam} within-state ranking")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG / "within_state_ranking.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    names = ["LOCAL", "STRUCT_OC", "DEGREE", "FLEX", "SDISC", "FHAT", "TRAJ"]
    means = [float(np.mean([r["U_traj"] for r in sel if r["score"] == n])) for n in names]
    ax.bar(names, means, color="#4C78A8")
    ax.set_ylabel("mean U_traj")
    ax.set_title(f"{fam} first-hold selection")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG / "first_hold_selection.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    rnames = ["LOCAL", "STRUCT_OC", "SDISC", "FHAT"]
    rs = [float(np.nanmean([r["R"] for r in sel if r["score"] == n])) for n in rnames]
    ax.bar(rnames, rs, color="#54A24B")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_ylabel("mean R vs LOCAL")
    ax.set_title(f"{fam} recovery comparison")
    fig.tight_layout()
    fig.savefig(FIG / "recovery_comparison.png", dpi=140)
    plt.close(fig)


def decide(within: list[dict], sel: list[dict], repl: list[dict]) -> dict:
    def mean_u(name):
        return float(np.mean([r["U_traj"] for r in sel if r["score"] == name]))

    def mean_r(name):
        return float(np.nanmean([r["R"] for r in sel if r["score"] == name]))

    def mean_fp(name):
        return float(np.mean([r["F_percentile"] for r in sel if r["score"] == name]))

    def med(name, metric="spearman"):
        return float(np.nanmedian([r[f"{name}_{metric}"] for r in within]))

    loc, struct, disc, fhat, trj = mean_u("LOCAL"), mean_u("STRUCT_OC"), mean_u("SDISC"), mean_u("FHAT"), mean_u("TRAJ")
    med_u = med("LOCAL")
    med_deg = med("degree")
    med_wd = med("weighted_degree")
    med_flex = med("Flex")
    med_disc = med("SDISC")
    med_fhat = med("FHAT")
    med_struct = med("STRUCT_OC")
    best_disc = max([("degree", med_deg), ("weighted_degree", med_wd), ("Flex", med_flex)], key=lambda x: x[1] if np.isfinite(x[1]) else -9)
    q1 = bool(best_disc[1] >= 0.15)
    q3 = bool(med_disc > med_u + 0.05 and disc > loc + 1e5)
    q4 = bool(disc > loc + 1e5)
    q5 = bool(disc >= fhat - 1e5 and med_disc >= med_fhat - 0.02)
    if q1 and best_disc[1] >= 0.30:
        q6 = "B"
    elif q1:
        q6 = "B_weak"
    elif best_disc[1] < 0.10 and med_fhat < 0.40:
        q6 = "C"
    else:
        q6 = "C"
    return {
        "local": loc,
        "struct": struct,
        "disc": disc,
        "fhat": fhat,
        "traj": trj,
        "R_struct": mean_r("STRUCT_OC"),
        "R_disc": mean_r("SDISC"),
        "R_fhat": mean_r("FHAT"),
        "Fpct_local": mean_fp("LOCAL"),
        "Fpct_disc": mean_fp("SDISC"),
        "Fpct_fhat": mean_fp("FHAT"),
        "med_U_F": med_u,
        "med_deg_F": med_deg,
        "med_wd_F": med_wd,
        "med_flex_F": med_flex,
        "med_disc_F": med_disc,
        "med_fhat_F": med_fhat,
        "med_struct_F": med_struct,
        "best_disc": best_disc[0],
        "best_disc_val": best_disc[1],
        "frac_topF_higher_flex": float(np.mean([r["topF_higher_flex"] for r in repl])),
        "frac_topF_higher_degree": float(np.mean([r["topF_higher_degree"] for r in repl])),
        "frac_topF_lower_conflict": float(np.mean([r["topF_lower_conflict"] for r in repl])),
        "mean_d_flex": float(np.mean([r["d_flex"] for r in repl])),
        "mean_d_degree": float(np.mean([r["d_degree"] for r in repl])),
        "q1": q1,
        "q3": q3,
        "q4": q4,
        "q5": q5,
        "q6": q6,
    }


def write_report(v: dict, within: list[dict], sel: list[dict], repl: list[dict], all_v: dict, dist: dict) -> str:
    def row_rank(name):
        return (
            f"| {name} | {_fmt(float(np.nanmedian([r[f'{name}_spearman'] for r in within])))} "
            f"| {_fmt(float(np.nanpercentile([r[f'{name}_spearman'] for r in within], 25)))} "
            f"| {_fmt(float(np.nanpercentile([r[f'{name}_spearman'] for r in within], 75)))} "
            f"| {_fmt(float(np.nanmedian([r[f'{name}_kendall'] for r in within])))} "
            f"| {_fmt(float(np.mean([r[f'{name}_top1'] for r in within])))} "
            f"| {_fmt(float(np.mean([r[f'{name}_top3'] for r in within])))} "
            f"| {_fmt(float(np.nanmean([r[f'{name}_ndcg3'] for r in within])))} |"
        )

    lines = [
        "# HTW Phase-3H：Candidate-Conditioned Decision Credit Audit",
        "",
        "本阶段只检验：Phase-3F 的 candidate-geometry signal 来自连续距离，还是 first-hold 可观测的 **离散决策 / feasible-set 结构**。未训练，未改正式方法，未读取未来 trajectory / F / U_traj 作为输入。不重复 Phase-3G 的连续 leftover×access 公式作为新发现。",
        "",
        "指定离散 proxy（先看原始计数，再组内 z-score）：",
        "",
        "- `degree` = 选择 Γ 后，t=0 leftover buoy 上仍 in-range 的 (UAV, buoy) 边数。",
        "- `weighted_degree` = 同上，边权为该 leftover buoy 的当前 queue。",
        "- `Flex` = exclusive occupation 下，每个 UAV 的合法单边改写数（改 idle 或改到未占用且 in-range 的 buoy）。",
        "",
        r"`S_disc = z(U_single) + α z(Flex)`，指定 α=0.50；另报 0.25 / 0.75 敏感性。STRUCT_OC 与 FHAT 沿用 Phase-3G/3E 定义。",
        "",
        f"Sanity：LOCAL={_sci(v['local'])}，TRAJ={_sci(v['traj'])}。",
        "",
        "## 1. 原始分布（EXEC，未归一化）",
        "",
        "| proxy | min | max | mean | std | n_unique |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("degree", "weighted_degree", "Flex", "leftover_coverable", "conflict", "pool_degree"):
        d = dist[key]
        lines.append(f"| {key} | {_fmt(d['min'])} | {_fmt(d['max'])} | {_fmt(d['mean'])} | {_fmt(d['std'])} | {d['n_unique']} |")
    lines += [
        "",
        "## 2. Within-state ranking vs F（EXEC）",
        "",
        "| score | Spearman med | p25 | p75 | Kendall med | top-1 | top-3 | NDCG@3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        row_rank("LOCAL"),
        row_rank("degree"),
        row_rank("weighted_degree"),
        row_rank("Flex"),
        row_rank("STRUCT_OC"),
        row_rank("SDISC"),
        row_rank("FHAT"),
        row_rank("pool_degree"),
        "",
        "## 3. First-hold 闭环（后续仍正式 greedy）",
        "",
        "| method | U_traj | F percentile | R | top-1 vs TRAJ |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("LOCAL", "STRUCT_OC", "DEGREE", "WDEGREE", "FLEX", "SDISC", "FHAT", "TRAJ"):
        sub = [r for r in sel if r["score"] == name]
        lines.append(
            f"| {name} | {_sci(float(np.mean([r['U_traj'] for r in sub])))} | {_fmt(float(np.mean([r['F_percentile'] for r in sub])))} | {_fmt(float(np.nanmean([r['R'] for r in sub])))} | {_fmt(float(np.mean([r['top1_vs_traj'] for r in sub])))} |"
        )
    lines += [
        "",
        "α 敏感性（指定 α=0.50，不取最优）：",
        "",
        "| alpha | S_disc U_traj | R |",
        "|---:|---:|---:|",
    ]
    for a in ALPHAS:
        sub = [r for r in sel if r["score"] == f"SDISC_a{a}"]
        lines.append(f"| {a} | {_sci(float(np.mean([r['U_traj'] for r in sub])))} | {_fmt(float(np.nanmean([r['R'] for r in sub])))} |")
    lines += [
        "",
        "## 4. Replacement / structure consistency（不是因果）",
        "",
        f"top-F 比 LOCAL 有更高 Flex 的 state 比例={_fmt(v['frac_topF_higher_flex'])}；更高 degree={_fmt(v['frac_topF_higher_degree'])}；更低 conflict={_fmt(v['frac_topF_lower_conflict'])}。",
        f"mean ΔFlex(topF−LOCAL)={_fmt(v['mean_d_flex'])}，mean Δdegree={_fmt(v['mean_d_degree'])}。",
        "这只说明高 F candidate 在 first-hold 可观测结构上是否已经看起来更灵活，不能写成 Flex 导致 F。",
        "",
        "## 5. 四方法指定 S_disc",
        "",
        "| method | LOCAL | STRUCT_OC | S_disc | FHAT | TRAJ | R_disc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fam in FAMS:
        d = all_v[fam]
        lines.append(f"| {fam} | {_sci(d['local'])} | {_sci(d['struct'])} | {_sci(d['disc'])} | {_sci(d['fhat'])} | {_sci(d['traj'])} | {_fmt(d['R_disc'])} |")
    q1_txt = "能部分解释" if v["q1"] else "不能解释"
    q6_map = {
        "A": "A. future value 主要是连续 geometry",
        "B": "B. future value 主要是 candidate-conditioned discrete structure",
        "B_weak": "B 的弱形式：离散结构有弱相关，但不足以成为主要解释",
        "C": "C. 两者都不足，observable information 本身不足",
    }
    lines += [
        "",
        "## 6. 六个问题",
        "",
        f"**Q1.** candidate geometry 的 signal 能否由当前 feasible-set structure 解释？**{q1_txt}**。最强离散项是 {v['best_disc']}（within-state median Spearman={_fmt(v['best_disc_val'])}），对照 Phase-3G `O_traj`=0.002、Phase-3F geometry OOF=0.546。",
        "",
        f"**Q2.** degree={_fmt(v['med_deg_F'])}，weighted_degree={_fmt(v['med_wd_F'])}，Flex={_fmt(v['med_flex_F'])}（均为 vs F 的 within-state median Spearman）。",
        "",
        f"**Q3.** S_disc 是否比 U_single 有稳定改善？**{'是' if v['q3'] else '否'}**。within Spearman S_disc={_fmt(v['med_disc_F'])} vs U={_fmt(v['med_U_F'])}；闭环 {_sci(v['disc'])} vs {_sci(v['local'])}。",
        "",
        f"**Q4.** S_disc 能否提高实际 first-hold recovery？**{'能抬一点' if v['q4'] else '不能稳定提高'}**。S_disc={_sci(v['disc'])}，LOCAL={_sci(v['local'])}，R={_fmt(v['R_disc'])}。选中 F percentile：LOCAL={_fmt(v['Fpct_local'])}，S_disc={_fmt(v['Fpct_disc'])}。",
        "",
        f"**Q5.** S_disc 是否超过 Phase-3E FHAT？**{'是' if v['q5'] else '否'}**。S_disc={_sci(v['disc'])} vs FHAT={_sci(v['fhat'])}；within Spearman {_fmt(v['med_disc_F'])} vs {_fmt(v['med_fhat_F'])}。",
        "",
        f"**Q6.** **{q6_map[v['q6']]}**。相关 ≠ 因果。本阶段不把任一 α 写成最终参数。",
        "",
        "本阶段停止。不训练。不进入 Phase-4。",
        "",
    ]
    return "\n".join(lines)


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    table = attach_discrete(enrich(build_table()))
    all_v = {}
    within_all = []
    sel_all = []
    repl_all = []
    struct_all = []
    exec_pack = {}
    exec_dist = {}
    for fam in FAMS:
        rows = [r for r in table if r["method"] == fam]
        hats = fhat_oof(rows)
        scored = attach_scores(rows, hats)
        ev = family_eval(scored)
        v = decide(ev["within"], ev["sel"], ev["repl"])
        all_v[fam] = v
        within_all.extend(ev["within"])
        sel_all.extend(ev["sel"])
        repl_all.extend(ev["repl"])
        struct_all.extend(
            [
                {
                    "method": r["method"],
                    "scenario": r["scenario"],
                    "hash": r["hash"],
                    "U_single": r["U_single"],
                    "F": r["F"],
                    "U_traj": r["U_traj"],
                    "degree": r["degree"],
                    "weighted_degree": r["weighted_degree"],
                    "Flex": r["Flex"],
                    "leftover_coverable": r["leftover_coverable"],
                    "conflict": r["conflict"],
                    "pool_degree": r["pool_degree"],
                    "n_occupied": r["n_occupied"],
                }
                for r in scored
            ]
        )
        if fam == "EXEC":
            exec_dist = raw_dist(scored)
            figures(fam, ev["within"], ev["sel"], scored)
            exec_pack = ev
            exec_pack["verdict"] = v
    _csv(OUTPUT / "candidate_structure.csv", struct_all)
    _csv(OUTPUT / "within_state_metrics.csv", within_all)
    _csv(OUTPUT / "replacement_audit.csv", repl_all)
    _csv(OUTPUT / "selection_metrics.csv", sel_all)
    _csv(OUTPUT / "recovery_metrics.csv", sel_all)
    cmp_rows = []
    for fam in FAMS:
        for score in ("LOCAL", "STRUCT_OC", "DEGREE", "WDEGREE", "FLEX", "SDISC", "FHAT", "TRAJ"):
            sub = [r for r in sel_all if r["method"] == fam and r["score"] == score]
            wsub = [r for r in within_all if r["method"] == fam]
            key = {"LOCAL": "LOCAL", "STRUCT_OC": "STRUCT_OC", "DEGREE": "degree", "WDEGREE": "weighted_degree", "FLEX": "Flex", "SDISC": "SDISC", "FHAT": "FHAT"}.get(score)
            cmp_rows.append(
                {
                    "method": fam,
                    "score": score,
                    "U_traj": float(np.mean([r["U_traj"] for r in sub])),
                    "F_percentile": float(np.mean([r["F_percentile"] for r in sub])),
                    "R": float(np.nanmean([r["R"] for r in sub])),
                    "top1_vs_traj": float(np.mean([r["top1_vs_traj"] for r in sub])),
                    "within_spearman": float(np.nanmedian([r[f"{key}_spearman"] for r in wsub])) if key else float("nan"),
                    "within_kendall": float(np.nanmedian([r[f"{key}_kendall"] for r in wsub])) if key else float("nan"),
                    "within_top1": float(np.mean([r[f"{key}_top1"] for r in wsub])) if key else float("nan"),
                    "within_top3": float(np.mean([r[f"{key}_top3"] for r in wsub])) if key else float("nan"),
                    "within_ndcg3": float(np.nanmean([r[f"{key}_ndcg3"] for r in wsub])) if key else float("nan"),
                }
            )
    _csv(OUTPUT / "method_comparison.csv", cmp_rows)
    _dump(OUTPUT / "PHASE3H_SUMMARY.json", {"verdict": exec_pack["verdict"], "all": all_v, "exec_dist": exec_dist})
    report = write_report(exec_pack["verdict"], exec_pack["within"], exec_pack["sel"], exec_pack["repl"], all_v, exec_dist)
    (OUTPUT / "PHASE3H_AUTO_SUMMARY.md").write_text(report, encoding="utf-8")
    print(json.dumps({"EXEC": exec_pack["verdict"], "dist": exec_dist}, indent=2, default=_jsonable), flush=True)
    return exec_pack["verdict"]


if __name__ == "__main__":
    run()
