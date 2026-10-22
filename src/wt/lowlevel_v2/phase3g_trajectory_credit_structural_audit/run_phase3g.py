"""Phase-3G: structured first-hold credit proxies. No official training."""
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
from wt.lowlevel_v2.phase3e_future_value_audit.run_phase3e import STATE_CAND_KEYS, build_table, reset_env
from wt.lowlevel_v2.phase3f_future_value_ceiling_audit.run_phase3f import ndcg_at
from wt.lowlevel_v2.protocol import ROOT

PHASE = Path(__file__).resolve().parent
OUTPUT = PHASE / "outputs"
FIG = OUTPUT / "figures"
FAMS = ("OLD", "ABS", "PREFIX", "EXEC")
ALPHAS = (0.25, 0.5, 0.75, 1.0)
DELTAS = (0.25, 0.5)
N_SPLITS = 5
ET = dict(n_estimators=200, max_depth=6, min_samples_leaf=8, random_state=0, n_jobs=1)
# Designated structural scores. Not a search winner.
STRUCT_O = (0.5, 0.0)
STRUCT_C = (0.0, 0.25)
STRUCT_OC = (0.5, 0.25)
FHAT_BETA = 0.25


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


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    s = float(np.std(x))
    if s < 1e-12:
        return np.zeros_like(x)
    return (x - float(np.mean(x))) / s


def rank_reg(f, score, u) -> float:
    f = np.asarray(f, dtype=np.float64)
    best = float(np.max(f))
    hat = float(f[int(np.argmax(score))])
    loc = float(f[int(np.argmax(u))])
    den = best - loc
    if abs(den) < 1e-12:
        return float("nan")
    return (best - hat) / den


def recovery(u_m, u_loc, u_trj) -> float:
    den = float(u_trj) - float(u_loc)
    if abs(den) < 1e-12:
        return float("nan")
    return (float(u_m) - float(u_loc)) / den


def inventory() -> list[dict]:
    rows = [
        ("U_single", "A local value", True, True, "first-hold shadow increment"),
        ("mean_uav_target_dist", "B cand-state / E access", True, True, "t=0 UAV to assigned buoy"),
        ("max_uav_target_dist", "B/C UAV geometry", True, True, "t=0"),
        ("n_in_range", "E accessibility", True, True, "service_radius at t=0"),
        ("assigned_queue_mean", "D/H task queue", True, True, "queue of assigned buoys"),
        ("n_unique_buoys", "D other-task geometry", True, True, "assignment coverage"),
        ("leftover_queue", "F/H remaining-task", True, True, "queue mass not assigned now"),
        ("leftover_hot", "F remaining hot", True, True, "unassigned hot-buoy queue"),
        ("leftover_min_dist", "E leftover access", True, True, "min UAV distance to leftover queued buoys"),
        ("idle_frac", "F flexibility", True, True, "idle UAVs at first hold"),
        ("competition_count", "G competition", True, True, "shared-buoy conflicts"),
        ("cand_mean_sinr", "B comm geometry", True, True, "assignment on t=0, no step"),
        ("queue_hot_mass", "H state queue", True, True, "t=0 hot queue; nearly constant"),
        ("O_traj", "opportunity proxy", True, True, "leftover_queue*(1+idle)/(1+comp)/(1+d/R)"),
        ("C_transition", "transition proxy", True, True, "in_range_frac/(1+competition)"),
        ("F", "target", False, False, "forbidden as feature"),
        ("U_traj", "target", False, False, "forbidden as feature"),
        ("future trajectory", "forbidden", False, False, "not used"),
    ]
    return [
        {
            "proxy": n,
            "component": c,
            "available_at_first_hold": int(a),
            "used": int(u),
            "note": note,
        }
        for n, c, a, u, note in rows
    ]


def enrich(rows: list[dict]) -> list[dict]:
    envs = {}
    out = []
    for row in rows:
        sc = int(row["scenario"])
        if sc not in envs:
            envs[sc] = reset_env(sc)
        env = envs[sc]
        assigned = [int(row[f"assign_{i}"]) for i in range(6) if int(row[f"assign_{i}"]) >= 0]
        covered = set(assigned)
        leftover_q = 0.0
        leftover_hot = 0.0
        leftover_d = []
        radius = float(env.config.service_radius_m)
        n_hot = int(env.config.num_hot)
        for k in range(int(env.config.num_buoys)):
            if k in covered:
                continue
            q = float(env.queue[k])
            leftover_q += q
            if k < n_hot:
                leftover_hot += q
            if q > 1e-9:
                dmin = min(float(np.linalg.norm(env.uav_positions[u] - env.buoy_positions[k])) for u in range(env.config.num_uavs))
                leftover_d.append(dmin)
        n_assigned = max(len(assigned), 1)
        in_range_frac = float(row["n_in_range"]) / n_assigned
        comp = float(row["competition_count"])
        idle = float(row["idle_frac"])
        d_left = float(np.mean(leftover_d)) if leftover_d else radius
        access = 1.0 / (1.0 + d_left / max(radius, 1.0))
        o_traj = leftover_q * (1.0 + idle) * access / (1.0 + comp)
        c_tr = in_range_frac / (1.0 + comp)
        item = dict(row)
        item.update(
            {
                "leftover_queue": leftover_q,
                "leftover_hot": leftover_hot,
                "leftover_min_dist": d_left,
                "leftover_access": access,
                "in_range_frac": in_range_frac,
                "O_traj": o_traj,
                "C_transition": c_tr,
            }
        )
        out.append(item)
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
        su = float(np.std(u))
        if su < 1e-12:
            su = 1.0
        for i, row in enumerate(items):
            row["O_traj_z"] = float(o[i])
            row["C_transition_z"] = float(c[i])
            row["S_fhat"] = float(row["U_single"] + FHAT_BETA * row["F_hat"])
            for a in ALPHAS:
                row[f"S_O_{a}"] = float(row["U_single"] + a * su * o[i])
            for d in DELTAS:
                row[f"S_C_{d}"] = float(row["U_single"] + d * su * c[i])
            row["S_OC"] = float(row["U_single"] + STRUCT_OC[0] * su * o[i] + STRUCT_OC[1] * su * c[i])
            row["S_O"] = float(row["U_single"] + STRUCT_O[0] * su * o[i])
            row["S_C"] = float(row["U_single"] + STRUCT_C[1] * su * c[i])
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
        "FHAT": "S_fhat",
        "STRUCT_O": "S_O",
        "STRUCT_C": "S_C",
        "STRUCT_OC": "S_OC",
        "TRAJ": "U_traj",
        "ORACLE_F": "F",
    }
    for a in ALPHAS:
        score_keys[f"O_a{a}"] = f"S_O_{a}"
    within_rows = []
    sel_rows = []
    for sc, items in sorted(by.items()):
        u = np.array([r["U_single"] for r in items], dtype=np.float64)
        f = np.array([r["F"] for r in items], dtype=np.float64)
        t = np.array([r["U_traj"] for r in items], dtype=np.float64)
        loc = select_row(items, "U_single")
        trj = select_row(items, "U_traj")
        best_u = float(np.max(u))
        w = {
            "method": items[0]["method"],
            "scenario": sc,
            "spearman_U_F": spearman(u, f),
            "spearman_O_F": spearman([r["O_traj"] for r in items], f),
            "spearman_C_F": spearman([r["C_transition"] for r in items], f),
            "spearman_leftoverQ_F": spearman([r["leftover_queue"] for r in items], f),
            "spearman_access_F": spearman([r["leftover_access"] for r in items], f),
            "spearman_comp_F": spearman([-r["competition_count"] for r in items], f),
            "spearman_inrange_F": spearman([r["n_in_range"] for r in items], f),
            "spearman_dist_F": spearman([-r["mean_uav_target_dist"] for r in items], f),
        }
        for name, key in (("LOCAL", "U_single"), ("O_traj", "O_traj"), ("C_tr", "C_transition"), ("STRUCT_OC", "S_OC"), ("FHAT", "S_fhat")):
            blk = within_block(items, key, "F")
            w.update({f"{name}_{k}": v for k, v in blk.items()})
            blk_t = within_block(items, key, "U_traj")
            w.update({f"{name}_vsUtraj_{k}": v for k, v in blk_t.items()})
        within_rows.append(w)
        for name, key in score_keys.items():
            chosen = select_row(items, key)
            f_pct = float(np.mean(f <= chosen["F"]))
            sel_rows.append(
                {
                    "method": items[0]["method"],
                    "scenario": sc,
                    "score": name,
                    "U_traj": chosen["U_traj"],
                    "F": chosen["F"],
                    "U_single": chosen["U_single"],
                    "payload": chosen["U_traj"],
                    "DecodeNR": (best_u - chosen["U_single"]) / (abs(best_u) + 1.0),
                    "F_percentile": f_pct,
                    "R": recovery(chosen["U_traj"], loc["U_traj"], trj["U_traj"]),
                    "R_rank": rank_reg(f, [r[key] for r in items], u),
                    "top1_vs_traj": float(chosen["hash"] == trj["hash"]),
                    "local_U_traj": loc["U_traj"],
                    "traj_U_traj": trj["U_traj"],
                    "natural_U_traj": items[0]["natural_U_traj"],
                }
            )
    return {"within": within_rows, "sel": sel_rows}


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


def figures(fam: str, within: list[dict], sel: list[dict], rows: list[dict]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    comps = [
        ("U_single", [r["U_single"] for r in rows], [r["F"] for r in rows]),
        ("O_traj", [r["O_traj"] for r in rows], [r["F"] for r in rows]),
        ("C_transition", [r["C_transition"] for r in rows], [r["F"] for r in rows]),
        ("leftover_queue", [r["leftover_queue"] for r in rows], [r["F"] for r in rows]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.2))
    for ax, (name, x, y) in zip(axes.ravel(), comps):
        ax.scatter(x, y, s=8, alpha=0.35)
        ax.set_xlabel(name)
        ax.set_ylabel("F")
        ax.set_title(f"{name} vs F  Spearman={spearman(x, y):.2f}")
    fig.tight_layout()
    fig.savefig(FIG / "component_vs_F.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    labels = ["LOCAL", "O_traj", "STRUCT_OC", "FHAT"]
    meds = [float(np.nanmedian([r[f"{n}_spearman"] for r in within])) for n in labels]
    ax.bar(labels, meds, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    ax.set_ylabel("median within-state Spearman vs F")
    ax.set_title(f"{fam} within-state ranking")
    fig.tight_layout()
    fig.savefig(FIG / "within_state_ranking.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    names = ["LOCAL", "FHAT", "STRUCT_O", "STRUCT_C", "STRUCT_OC", "TRAJ"]
    means = [float(np.mean([r["U_traj"] for r in sel if r["score"] == n])) for n in names]
    ax.bar(names, means, color="#4C78A8")
    ax.set_ylabel("mean U_traj")
    ax.set_title(f"{fam} first-hold selection")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG / "first_hold_selection.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    rnames = ["LOCAL", "FHAT", "STRUCT_O", "STRUCT_OC"]
    rs = [float(np.nanmean([r["R"] for r in sel if r["score"] == n])) for n in rnames]
    ax.bar(rnames, rs, color="#54A24B")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_ylabel("mean R vs LOCAL")
    ax.set_title(f"{fam} recovery comparison")
    fig.tight_layout()
    fig.savefig(FIG / "recovery_comparison.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist([r["R_rank"] for r in sel if r["score"] == "STRUCT_OC" and np.isfinite(r["R_rank"])], bins=8, color="#F58518")
    ax.set_xlabel("STRUCT_OC ranking regret vs F")
    ax.set_title(f"{fam} ranking regret")
    fig.tight_layout()
    fig.savefig(FIG / "ranking_regret.png", dpi=140)
    plt.close(fig)


def decide(within: list[dict], sel: list[dict]) -> dict:
    def mean_u(name):
        return float(np.mean([r["U_traj"] for r in sel if r["score"] == name]))

    def mean_r(name):
        return float(np.nanmean([r["R"] for r in sel if r["score"] == name]))

    def mean_fp(name):
        return float(np.mean([r["F_percentile"] for r in sel if r["score"] == name]))

    loc, fhat, so, soc, trj = mean_u("LOCAL"), mean_u("FHAT"), mean_u("STRUCT_O"), mean_u("STRUCT_OC"), mean_u("TRAJ")
    med_u = float(np.nanmedian([r["spearman_U_F"] for r in within]))
    med_o = float(np.nanmedian([r["spearman_O_F"] for r in within]))
    med_soc = float(np.nanmedian([r["STRUCT_OC_spearman"] for r in within]))
    med_fhat = float(np.nanmedian([r["FHAT_spearman"] for r in within]))
    q1_best = max(
        [
            ("leftover_queue", float(np.nanmedian([r["spearman_leftoverQ_F"] for r in within]))),
            ("O_traj", med_o),
            ("C_transition", float(np.nanmedian([r["spearman_C_F"] for r in within]))),
            ("-distance", float(np.nanmedian([r["spearman_dist_F"] for r in within]))),
            ("n_in_range", float(np.nanmedian([r["spearman_inrange_F"] for r in within]))),
            ("-competition", float(np.nanmedian([r["spearman_comp_F"] for r in within]))),
            ("U_single", med_u),
        ],
        key=lambda x: x[1] if np.isfinite(x[1]) else -9,
    )
    q2 = med_o > med_u + 0.05
    q3 = med_soc >= med_fhat - 0.02 and mean_r("STRUCT_OC") >= mean_r("FHAT") - 1e-6
    q4 = soc > loc + 1e5
    q5 = q4 and (soc >= fhat - 1e5) and q2
    return {
        "local": loc,
        "fhat": fhat,
        "struct_o": so,
        "struct_oc": soc,
        "traj": trj,
        "R_fhat": mean_r("FHAT"),
        "R_o": mean_r("STRUCT_O"),
        "R_oc": mean_r("STRUCT_OC"),
        "Fpct_local": mean_fp("LOCAL"),
        "Fpct_fhat": mean_fp("FHAT"),
        "Fpct_oc": mean_fp("STRUCT_OC"),
        "med_U_F": med_u,
        "med_O_F": med_o,
        "med_SOC_F": med_soc,
        "med_FHAT_F": med_fhat,
        "q1_best": q1_best[0],
        "q1_best_val": q1_best[1],
        "q2": q2,
        "q3": q3,
        "q4": q4,
        "q5": q5,
    }


def write_report(v: dict, within: list[dict], sel: list[dict], all_v: dict) -> str:
    lines = [
        "# HTW Phase-3G：Trajectory-Aware Credit Structural Approximation Audit",
        "",
        "本阶段只检验：能否用 **first-hold 可观测结构** 近似 trajectory-aware credit，而不是再学 x→F。未训练 Low-Level / High-Level，未改 decoder / pool / greedy / checkpoint。",
        "",
        "指定结构（不是搜参赢家）：",
        "",
        r"`O_traj = leftover_queue · (1+idle_frac) · leftover_access / (1+competition)`",
        "",
        r"`C_transition = in_range_frac / (1+competition)`",
        "",
        "组内 z-score 后：`S = U_single + α·std(U)·z(O) + δ·std(U)·z(C)`。指定 `STRUCT_O=(α=0.5,δ=0)`，`STRUCT_OC=(0.5,0.25)`。Phase-3E 对照：`U_single + 0.25 F_hat`（OOF ExtraTrees-C）。",
        "",
        f"Sanity：LOCAL={_sci(v['local'])}，TRAJ={_sci(v['traj'])}。",
        "",
        "## 1. Component 与 F（EXEC within-state median）",
        "",
        "| component | Spearman vs F median | p25 | p75 |",
        "|---|---:|---:|---:|",
    ]
    pairs = [
        ("U_single", "spearman_U_F"),
        ("O_traj", "spearman_O_F"),
        ("C_transition", "spearman_C_F"),
        ("leftover_queue", "spearman_leftoverQ_F"),
        ("leftover_access", "spearman_access_F"),
        ("-mean_dist", "spearman_dist_F"),
        ("n_in_range", "spearman_inrange_F"),
        ("-competition", "spearman_comp_F"),
    ]
    for name, key in pairs:
        vals = [r[key] for r in within]
        lines.append(f"| {name} | {_fmt(float(np.nanmedian(vals)))} | {_fmt(float(np.nanpercentile(vals, 25)))} | {_fmt(float(np.nanpercentile(vals, 75)))} |")
    lines += [
        "",
        f"与 F 最相关的指定 component：**{v['q1_best']}**（median={_fmt(v['q1_best_val'])}）。",
        "",
        "## 2. Within-state ranking vs F",
        "",
        "| score | Spearman med | top-1 | top-3 | NDCG@3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("LOCAL", "O_traj", "STRUCT_OC", "FHAT"):
        lines.append(
            f"| {name} | {_fmt(float(np.nanmedian([r[f'{name}_spearman'] for r in within])))} | {_fmt(float(np.mean([r[f'{name}_top1'] for r in within])))} | {_fmt(float(np.mean([r[f'{name}_top3'] for r in within])))} | {_fmt(float(np.nanmean([r[f'{name}_ndcg3'] for r in within])))} |"
        )
    lines += [
        "",
        "## 3. First-hold 实际闭环（后续仍正式 greedy）",
        "",
        "| method | U_traj | F percentile | R vs LOCAL | DecodeNR | top-1 vs TRAJ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("LOCAL", "FHAT", "STRUCT_O", "STRUCT_C", "STRUCT_OC", "TRAJ"):
        sub = [r for r in sel if r["score"] == name]
        lines.append(
            f"| {name} | {_sci(float(np.mean([r['U_traj'] for r in sub])))} | {_fmt(float(np.mean([r['F_percentile'] for r in sub])))} | {_fmt(float(np.nanmean([r['R'] for r in sub])))} | {_fmt(float(np.mean([r['DecodeNR'] for r in sub])))} | {_fmt(float(np.mean([r['top1_vs_traj'] for r in sub])))} |"
        )
    lines += [
        "",
        "α 网格（只作敏感性，不取最优当结论）：",
        "",
        "| alpha | STRUCT_O U_traj | R |",
        "|---:|---:|---:|",
    ]
    for a in ALPHAS:
        sub = [r for r in sel if r["score"] == f"O_a{a}"]
        lines.append(f"| {a} | {_sci(float(np.mean([r['U_traj'] for r in sub])))} | {_fmt(float(np.nanmean([r['R'] for r in sub])))} |")
    lines += [
        "",
        "## 4. 四方法指定 STRUCT_OC",
        "",
        "| method | LOCAL | FHAT | STRUCT_OC | TRAJ | R_OC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for fam in FAMS:
        d = all_v[fam]
        lines.append(f"| {fam} | {_sci(d['local'])} | {_sci(d['fhat'])} | {_sci(d['struct_oc'])} | {_sci(d['traj'])} | {_fmt(d['R_oc'])} |")
    lines += [
        "",
        "## 5. 五个问题",
        "",
        f"**Q1.** 与 F 最相关的可观测结构是 **{v['q1_best']}**（within-state median Spearman={_fmt(v['q1_best_val'])}）。U_single 只有 {_fmt(v['med_U_F'])}。",
        "",
        f"**Q2.** O_traj 是否比 U_single 更能解释 F？**{'是' if v['q2'] else '否'}**（O={_fmt(v['med_O_F'])} vs U={_fmt(v['med_U_F'])}）。",
        "",
        f"**Q3.** 指定 STRUCT 是否比 Phase-3E F_hat 更稳定？**{'是' if v['q3'] else '否'}**。within Spearman STRUCT_OC={_fmt(v['med_SOC_F'])} vs FHAT={_fmt(v['med_FHAT_F'])}；闭环 R_OC={_fmt(v['R_oc'])} vs R_FHAT={_fmt(v['R_fhat'])}。",
        "",
        f"**Q4.** 结构化 score 能否提高实际 first-hold recovery？**{'能抬一点' if v['q4'] else '不能稳定提高'}**。STRUCT_OC={_sci(v['struct_oc'])}，LOCAL={_sci(v['local'])}，FHAT={_sci(v['fhat'])}，TRAJ={_sci(v['traj'])}。选中 candidate 的真实 F percentile：LOCAL={_fmt(v['Fpct_local'])}，STRUCT_OC={_fmt(v['Fpct_oc'])}，FHAT={_fmt(v['Fpct_fhat'])}。",
        "",
        f"**Q5.** 是否足以支持“应采用可观测结构化近似，而不是直接预测完整 F”？**{'证据不足，不能这么写' if not v['q5'] else '有条件支持：结构可解释，但尚未压过 / 稳定优于 F_hat 的闭环'}**。",
        "相关变高 ≠ 算法有效。本阶段不把任一 α 写成最终参数。",
        "",
        "本阶段停止。不训练。不进入 Phase-4。",
        "",
    ]
    return "\n".join(lines)


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    _csv(OUTPUT / "observable_proxy_inventory.csv", inventory())
    table = enrich(build_table())
    all_v = {}
    within_all = []
    sel_all = []
    exec_pack = {}
    for fam in FAMS:
        rows = [r for r in table if r["method"] == fam]
        hats = fhat_oof(rows)
        scored = attach_scores(rows, hats)
        ev = family_eval(scored)
        v = decide(ev["within"], ev["sel"])
        all_v[fam] = v
        within_all.extend(ev["within"])
        sel_all.extend(ev["sel"])
        if fam == "EXEC":
            figures(fam, ev["within"], ev["sel"], scored)
            exec_pack = ev
            exec_pack["verdict"] = v
    _csv(OUTPUT / "within_state_metrics.csv", within_all)
    _csv(OUTPUT / "selection_metrics.csv", sel_all)
    _csv(OUTPUT / "ranking_metrics.csv", within_all)
    _csv(OUTPUT / "recovery_metrics.csv", sel_all)
    comp = []
    for fam in FAMS:
        sub = [r for r in within_all if r["method"] == fam]
        for name, key in (
            ("U_single", "spearman_U_F"),
            ("O_traj", "spearman_O_F"),
            ("C_transition", "spearman_C_F"),
            ("leftover_queue", "spearman_leftoverQ_F"),
            ("leftover_access", "spearman_access_F"),
            ("n_in_range", "spearman_inrange_F"),
            ("-distance", "spearman_dist_F"),
            ("-competition", "spearman_comp_F"),
        ):
            vals = [r[key] for r in sub]
            comp.append({"method": fam, "component": name, "median": float(np.nanmedian(vals)), "p25": float(np.nanpercentile(vals, 25)), "p75": float(np.nanpercentile(vals, 75))})
    _csv(OUTPUT / "component_metrics.csv", comp)
    abl = []
    for fam in FAMS:
        for score in ("LOCAL", "FHAT", "STRUCT_O", "STRUCT_C", "STRUCT_OC", "TRAJ"):
            sub = [r for r in sel_all if r["method"] == fam and r["score"] == score]
            abl.append(
                {
                    "method": fam,
                    "score": score,
                    "U_traj": float(np.mean([r["U_traj"] for r in sub])),
                    "F_percentile": float(np.mean([r["F_percentile"] for r in sub])),
                    "R": float(np.nanmean([r["R"] for r in sub])),
                    "DecodeNR": float(np.mean([r["DecodeNR"] for r in sub])),
                    "top1_vs_traj": float(np.mean([r["top1_vs_traj"] for r in sub])),
                }
            )
    _csv(OUTPUT / "ablation_metrics.csv", abl)
    _dump(OUTPUT / "PHASE3G_SUMMARY.json", {"verdict": exec_pack["verdict"], "all": all_v})
    # Formal CN report is maintained as a locked narrative; do not clobber it
    # with the short auto template after the audit numbers are frozen.
    report = write_report(exec_pack["verdict"], exec_pack["within"], exec_pack["sel"], all_v)
    (OUTPUT / "PHASE3G_AUTO_SUMMARY.md").write_text(report, encoding="utf-8")
    print(json.dumps({"EXEC": exec_pack["verdict"]}, indent=2, default=_jsonable), flush=True)
    return exec_pack["verdict"]


if __name__ == "__main__":
    run()
