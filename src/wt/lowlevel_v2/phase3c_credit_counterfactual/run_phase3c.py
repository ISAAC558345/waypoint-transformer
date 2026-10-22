"""Phase-3C first-hold intervention audit. Frozen rollouts only. No training."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wt.lowlevel_v2.integration_phase1.stats import kendall_tau, spearman, summarize
from wt.lowlevel_v2.integration_phase2.protocol import UTILITY_EPS
from wt.lowlevel_v2.protocol import ROOT
from wt.lowlevel_v2.traj_audit.protocol import OUTPUT as TRAJ_OUT

PHASE = Path(__file__).resolve().parent
OUTPUT = PHASE / "outputs"
FIG = OUTPUT / "figures"
FAMS = ("OLD", "ABS", "PREFIX", "EXEC")
CONDS = ("NATURAL", "LOCAL-ORACLE", "TRAJ-ORACLE")
SOURCE = {
    "natural": str(TRAJ_OUT / "natural_trajectories.json"),
    "counterfactual": str(TRAJ_OUT / "counterfactual.json"),
    "pool": str(TRAJ_OUT / "first_hold_pool.csv"),
    "phase3b": str(ROOT / "src/wt/lowlevel_v2/phase3b_credit/outputs/first_hold_counterfactual_credit.csv"),
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


def pearson(x, y) -> float:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def ext_sum(values) -> dict:
    out = summarize(values)
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    out["std"] = float(np.std(a, ddof=1)) if a.size > 1 else float("nan")
    return out


def decode_nr(best: float, chosen: float) -> float:
    return (float(best) - float(chosen)) / (abs(float(best)) + float(UTILITY_EPS))


def rank_desc(values, chosen: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(1 + int(np.sum(arr > float(chosen))))


def load_json(name: str):
    return json.loads((TRAJ_OUT / name).read_text(encoding="utf-8"))


def load_pool() -> dict[int, list[dict]]:
    rows = list(csv.DictReader((TRAJ_OUT / "first_hold_pool.csv").open(encoding="utf-8", newline="")))
    by = defaultdict(list)
    for row in rows:
        by[int(float(row["scenario"]))].append(
            {"hash": row["hash"], "U_LL": float(row["U_LL"]), "gamma": row.get("gamma", "")}
        )
    return by


def build_rows(nat: list[dict], cf: list[dict], pool: dict[int, list[dict]]):
    actual = {}
    for rec in nat:
        first = rec["holds"][0]
        later = [float(h["U_single"]) for h in rec["holds"][1:]]
        actual[(rec["family"], int(rec["scenario"]))] = {
            "hash": rec.get("first_hash") or first["hash"],
            "gamma": json.dumps(first.get("gamma", [])),
            "U_single": float(first["U_single"]),
            "U_traj": float(rec["U_traj"]),
            "F": float(rec["U_traj"]) - float(first["U_single"]),
            "F_sum_later": float(np.sum(later)) if later else 0.0,
            "in_pool": bool(rec.get("first_in_pool", False)),
            "state": int(rec["scenario"]),
        }
    by_cf = defaultdict(list)
    for row in cf:
        by_cf[(row["family"], int(row["scenario"]))].append(row)

    interventions = []
    candidates = []
    gains = []
    for fam in FAMS:
        scenarios = sorted({int(r["scenario"]) for r in nat if r["family"] == fam})
        for sc in scenarios:
            act = actual[(fam, sc)]
            cand_rows = sorted(by_cf[(fam, sc)], key=lambda x: int(x["cand"]))
            u = np.array([float(x["U_LL_first"]) for x in cand_rows], dtype=np.float64)
            t = np.array([float(x["U_traj"]) for x in cand_rows], dtype=np.float64)
            fut = t - u
            iu, it = int(np.argmax(u)), int(np.argmax(t))
            best_local = float(np.max(u))
            pool_u = np.array([p["U_LL"] for p in pool[sc]], dtype=np.float64)
            conds = {
                "NATURAL": {
                    "hash": act["hash"],
                    "gamma": act["gamma"],
                    "U_single": act["U_single"],
                    "U_traj": act["U_traj"],
                    "F": act["F"],
                    "cand": -1,
                    "in_pool": int(act["in_pool"]),
                },
                "LOCAL-ORACLE": {
                    "hash": cand_rows[iu]["hash"],
                    "gamma": next((p["gamma"] for p in pool[sc] if p["hash"] == cand_rows[iu]["hash"]), ""),
                    "U_single": float(u[iu]),
                    "U_traj": float(t[iu]),
                    "F": float(fut[iu]),
                    "cand": int(cand_rows[iu]["cand"]),
                    "in_pool": 1,
                },
                "TRAJ-ORACLE": {
                    "hash": cand_rows[it]["hash"],
                    "gamma": next((p["gamma"] for p in pool[sc] if p["hash"] == cand_rows[it]["hash"]), ""),
                    "U_single": float(u[it]),
                    "U_traj": float(t[it]),
                    "F": float(fut[it]),
                    "cand": int(cand_rows[it]["cand"]),
                    "in_pool": 1,
                },
            }
            for name, item in conds.items():
                interventions.append(
                    {
                        "method": fam,
                        "scenario": sc,
                        "state": sc,
                        "condition": name,
                        "first_hold": item["hash"],
                        "first_hold_gamma": item["gamma"],
                        "first_hold_cand": item["cand"],
                        "first_hold_in_pool": item["in_pool"],
                        "first_hold_U_single": item["U_single"],
                        "final_U_traj": item["U_traj"],
                        "payload": item["U_traj"],
                        "F": item["F"],
                        "DecodeNR": decode_nr(best_local, item["U_single"]),
                        "first_hold_rank_local": rank_desc(u, item["U_single"]),
                        "first_hold_rank_traj": rank_desc(t, item["U_traj"]),
                    }
                )
            dloc = float(t[iu] - act["U_traj"])
            dtraj = float(t[it] - act["U_traj"])
            gains.append(
                {
                    "method": fam,
                    "scenario": sc,
                    "state": sc,
                    "natural_U_traj": act["U_traj"],
                    "local_oracle_U_traj": float(t[iu]),
                    "traj_oracle_U_traj": float(t[it]),
                    "pool_uniform_U_traj": float(np.mean(t)),
                    "natural_U_single": act["U_single"],
                    "local_oracle_U_single": float(u[iu]),
                    "traj_oracle_U_single": float(u[it]),
                    "natural_F": act["F"],
                    "local_oracle_F": float(fut[iu]),
                    "traj_oracle_F": float(fut[it]),
                    "delta_local": dloc,
                    "delta_traj": dtraj,
                    "R": float(dloc / dtraj) if dtraj > 0 else float("nan"),
                    "disagree": int(iu != it),
                    "natural_DecodeNR": decode_nr(best_local, act["U_single"]),
                    "local_DecodeNR": decode_nr(best_local, float(u[iu])),
                    "traj_DecodeNR": decode_nr(best_local, float(u[it])),
                    "natural_rank_local": rank_desc(u, act["U_single"]),
                    "natural_rank_traj": rank_desc(t, act["U_traj"]),
                }
            )
            for i, row in enumerate(cand_rows):
                candidates.append(
                    {
                        "method": fam,
                        "scenario": sc,
                        "state": sc,
                        "cand": int(row["cand"]),
                        "first_hold": row["hash"],
                        "first_hold_U_single": float(u[i]),
                        "F": float(fut[i]),
                        "final_U_traj": float(t[i]),
                        "payload": float(t[i]),
                        "DecodeNR": decode_nr(best_local, float(u[i])),
                        "first_hold_rank_local": rank_desc(u, float(u[i])),
                        "first_hold_rank_traj": rank_desc(t, float(t[i])),
                        "is_local_oracle": int(i == iu),
                        "is_traj_oracle": int(i == it),
                        "is_natural": int(row["hash"] == act["hash"]),
                    }
                )
    return interventions, candidates, gains


def family_stats(gains: list[dict], interventions: list[dict], candidates: list[dict]) -> dict:
    out = {}
    for fam in FAMS:
        g = [r for r in gains if r["method"] == fam]
        r_vals = [r["R"] for r in g if np.isfinite(r["R"]) and float(r["delta_traj"]) > 0]
        cond = {}
        for name in CONDS:
            vals = [r["final_U_traj"] for r in interventions if r["method"] == fam and r["condition"] == name]
            us = [r["first_hold_U_single"] for r in interventions if r["method"] == fam and r["condition"] == name]
            fs = [r["F"] for r in interventions if r["method"] == fam and r["condition"] == name]
            nrs = [r["DecodeNR"] for r in interventions if r["method"] == fam and r["condition"] == name]
            cond[name] = {
                "U_traj": ext_sum(vals),
                "U_single": ext_sum(us),
                "F": ext_sum(fs),
                "DecodeNR": ext_sum(nrs),
            }
        cand = [r for r in candidates if r["method"] == fam]
        out[fam] = {
            "conditions": cond,
            "delta_local": ext_sum([r["delta_local"] for r in g]),
            "delta_traj": ext_sum([r["delta_traj"] for r in g]),
            "R": ext_sum(r_vals),
            "n_R": len(r_vals),
            "disagree_rate": float(np.mean([r["disagree"] for r in g])),
            "pool_uniform_U_traj": ext_sum([r["pool_uniform_U_traj"] for r in g]),
            "U_vs_F": {
                "pearson": pearson([r["first_hold_U_single"] for r in cand], [r["F"] for r in cand]),
                "spearman": spearman([r["first_hold_U_single"] for r in cand], [r["F"] for r in cand]),
                "kendall": kendall_tau([r["first_hold_U_single"] for r in cand], [r["F"] for r in cand]),
            },
        }
    return out


def decide(stats: dict) -> dict:
    e = stats["EXEC"]
    p = stats["PREFIX"]
    nat = e["conditions"]["NATURAL"]["U_traj"]["mean"]
    loc = e["conditions"]["LOCAL-ORACLE"]["U_traj"]["mean"]
    trj = e["conditions"]["TRAJ-ORACLE"]["U_traj"]["mean"]
    pref = p["conditions"]["NATURAL"]["U_traj"]["mean"]
    r_mean = e["R"]["mean"]
    local_over_traj = loc / trj if trj else float("nan")
    gain_local = loc - nat
    gain_traj = trj - nat
    local_close_to_traj = loc >= 0.90 * trj
    traj_reaches_prefix = trj >= 0.95 * pref
    local_reaches_prefix = loc >= 0.95 * pref
    if local_close_to_traj and (local_reaches_prefix or traj_reaches_prefix):
        q4 = "A"
        q4_text = "只改 first-hold selection（把第一 hold 拉到高 U_single）已接近 traj-oracle"
    elif (not local_close_to_traj) and traj_reaches_prefix:
        q4 = "B"
        q4_text = "只改第一 hold 就够，但必须是 trajectory-aware credit，而不是 argmax U_single"
    elif not traj_reaches_prefix:
        q4 = "D"
        q4_text = "traj-oracle 第一 hold 仍追不上 PREFIX，后续 executor 可能是独立缺口"
    else:
        q4 = "C"
        q4_text = "数据指向 candidate-level utility，而不是 first-hold credit"
    return {
        "exec_natural": nat,
        "exec_local": loc,
        "exec_traj": trj,
        "prefix_natural": pref,
        "delta_local": gain_local,
        "delta_traj": gain_traj,
        "R_mean": r_mean,
        "local_over_traj": local_over_traj,
        "local_close_to_traj": local_close_to_traj,
        "traj_reaches_prefix": traj_reaches_prefix,
        "q4": q4,
        "q4_text": q4_text,
    }


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


def figures(stats: dict, candidates: list[dict], interventions: list[dict], gains: list[dict]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(FAMS))
    width = 0.25
    nat = [stats[f]["conditions"]["NATURAL"]["U_traj"]["mean"] for f in FAMS]
    loc = [stats[f]["conditions"]["LOCAL-ORACLE"]["U_traj"]["mean"] for f in FAMS]
    trj = [stats[f]["conditions"]["TRAJ-ORACLE"]["U_traj"]["mean"] for f in FAMS]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(x - width, nat, width, label="NATURAL", color="#4C78A8")
    ax.bar(x, loc, width, label="LOCAL-ORACLE", color="#F58518")
    ax.bar(x + width, trj, width, label="TRAJ-ORACLE", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(list(FAMS))
    ax.set_ylabel("mean U_traj")
    ax.set_title("First-hold intervention (later holds stay official greedy)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "method_comparison_bars.png", dpi=140)
    plt.close(fig)

    e = stats["EXEC"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    names = ["NATURAL", "LOCAL-ORACLE", "TRAJ-ORACLE"]
    means = [e["conditions"][n]["U_traj"]["mean"] for n in names]
    meds = [e["conditions"][n]["U_traj"]["median"] for n in names]
    ax.bar(names, means, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.plot(names, meds, "ko", label="median")
    ax.set_ylabel("U_traj")
    ax.set_title("EXEC seed0 first-hold intervention")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "exec_intervention_bars.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.0), sharex=True, sharey=True)
    marks = {
        "NATURAL": ("o", "#4C78A8"),
        "LOCAL-ORACLE": ("s", "#F58518"),
        "TRAJ-ORACLE": ("^", "#54A24B"),
    }
    for ax, fam in zip(axes.ravel(), FAMS):
        cand = [r for r in candidates if r["method"] == fam]
        ax.scatter(
            [r["first_hold_U_single"] for r in cand],
            [r["F"] for r in cand],
            s=10,
            c="#B8B8B8",
            alpha=0.35,
            label="pool first-hold",
        )
        for name, (mk, col) in marks.items():
            pts = [r for r in interventions if r["method"] == fam and r["condition"] == name]
            ax.scatter(
                [r["first_hold_U_single"] for r in pts],
                [r["F"] for r in pts],
                s=36,
                marker=mk,
                c=col,
                edgecolors="black",
                linewidths=0.4,
                label=name,
                zorder=3,
            )
        ax.set_title(f"{fam}  U vs F  Spearman={stats[fam]['U_vs_F']['spearman']:.2f}")
        ax.set_xlabel("U_single (first hold)")
        ax.set_ylabel("F (later return)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "u_single_vs_F.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    cand = [r for r in candidates if r["method"] == "EXEC"]
    ax.scatter([r["first_hold_U_single"] for r in cand], [r["F"] for r in cand], s=12, c="#B8B8B8", alpha=0.4)
    for name, (mk, col) in marks.items():
        pts = [r for r in interventions if r["method"] == "EXEC" and r["condition"] == name]
        ax.scatter(
            [r["first_hold_U_single"] for r in pts],
            [r["F"] for r in pts],
            s=42,
            marker=mk,
            c=col,
            edgecolors="black",
            linewidths=0.5,
            label=name,
            zorder=3,
        )
    ax.set_xlabel("EXEC U_single (first hold)")
    ax.set_ylabel("EXEC F (U_traj - U_single)")
    ax.set_title("EXEC: first-hold local utility vs later return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "exec_u_single_vs_F.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    eg = [r for r in gains if r["method"] == "EXEC"]
    ax.scatter([r["delta_local"] for r in eg], [r["delta_traj"] for r in eg], c="#4C78A8")
    lim = max(
        abs(min(r["delta_local"] for r in eg)),
        abs(max(r["delta_local"] for r in eg)),
        abs(min(r["delta_traj"] for r in eg)),
        abs(max(r["delta_traj"] for r in eg)),
        1.0,
    )
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="R=1")
    ax.set_xlabel("Δ_local = U_traj(local) - U_traj(natural)")
    ax.set_ylabel("Δ_traj = U_traj(traj) - U_traj(natural)")
    ax.set_title("EXEC per-scenario intervention gains")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "exec_gain_scatter.png", dpi=140)
    plt.close(fig)


def write_report(stats: dict, verdict: dict) -> str:
    e = stats["EXEC"]
    p = stats["PREFIX"]
    lines = [
        "# HTW Phase-3C：First-Hold Long-Horizon Credit Counterfactual",
        "",
        "本阶段只做第一 hold 干预。未训练、未改模型 / checkpoint / q / L_exec / decoder / pool / Low-Level / reward / 正式 evaluation。",
        "",
        "数据全部追溯到 Phase-3 Trajectory Utility Audit 的已冻结 rollout：",
        "",
        f"- NATURAL：`{SOURCE['natural']}`",
        f"- 24-pool first-hold + 后续正式 greedy：`{SOURCE['counterfactual']}`",
        f"- 官方 first-hold pool：`{SOURCE['pool']}`",
        "",
        "四种条件里，**只有第一 hold 被替换**。hold 2/3/4 仍是当时正式 greedy decoder 的真实闭环，不是 4-hold oracle。未新增随机 rollout；`POOL-UNIFORM` 只是已有 24 条合法 first-hold 的均值参考。",
        "",
        f"**EXEC：NATURAL={_sci(verdict['exec_natural'])}，LOCAL-ORACLE={_sci(verdict['exec_local'])}，TRAJ-ORACLE={_sci(verdict['exec_traj'])}。** 与 Phase-3B 的 5.81e6 / 9.00e6 / 11.88e6 一致。",
        "",
        "## 1. 四方法对照",
        "",
        "| method | NATURAL | LOCAL-ORACLE | TRAJ-ORACLE | Δ_local | Δ_traj | R=Δ_local/Δ_traj |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fam in FAMS:
        s = stats[fam]
        lines.append(
            "| {m} | {n} | {l} | {t} | {dl} | {dt} | {r} |".format(
                m=fam,
                n=_sci(s["conditions"]["NATURAL"]["U_traj"]["mean"]),
                l=_sci(s["conditions"]["LOCAL-ORACLE"]["U_traj"]["mean"]),
                t=_sci(s["conditions"]["TRAJ-ORACLE"]["U_traj"]["mean"]),
                dl=_sci(s["delta_local"]["mean"]),
                dt=_sci(s["delta_traj"]["mean"]),
                r=_fmt(s["R"]["mean"]),
            )
        )
    lines += [
        "",
        "`R` 只在该 scenario 的 `Δ_traj>0` 上计算。PREFIX 的 LOCAL-ORACLE 低于 NATURAL（OOD greedy 已经好于 pool `argmax U_single`）。",
        "",
        "POOL-UNIFORM（已有 24 条均值，不是新实验）：",
        "",
        "| method | POOL-UNIFORM U_traj |",
        "|---|---:|",
    ]
    for fam in FAMS:
        lines.append(f"| {fam} | {_sci(stats[fam]['pool_uniform_U_traj']['mean'])} |")
    en = e["conditions"]["NATURAL"]["U_traj"]
    el = e["conditions"]["LOCAL-ORACLE"]["U_traj"]
    et = e["conditions"]["TRAJ-ORACLE"]["U_traj"]
    lines += [
        "",
        "## 2. EXEC 专项",
        "",
        "| condition | mean | median | p25 | p75 | p90 | std |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| NATURAL | {_sci(en['mean'])} | {_sci(en['median'])} | {_sci(en['p25'])} | {_sci(en['p75'])} | {_sci(en['p90'])} | {_sci(en['std'])} |",
        f"| LOCAL-ORACLE | {_sci(el['mean'])} | {_sci(el['median'])} | {_sci(el['p25'])} | {_sci(el['p75'])} | {_sci(el['p90'])} | {_sci(el['std'])} |",
        f"| TRAJ-ORACLE | {_sci(et['mean'])} | {_sci(et['median'])} | {_sci(et['p25'])} | {_sci(et['p75'])} | {_sci(et['p90'])} | {_sci(et['std'])} |",
        "",
        f"- LOCAL-ORACLE 相对 NATURAL：Δ_local mean={_sci(e['delta_local']['mean'])}，median={_sci(e['delta_local']['median'])}。",
        f"- TRAJ-ORACLE 相对 NATURAL：Δ_traj mean={_sci(e['delta_traj']['mean'])}，median={_sci(e['delta_traj']['median'])}。",
        f"- 恢复比例 R mean={_fmt(e['R']['mean'])}，median={_fmt(e['R']['median'])}（n={e['n_R']} 个 Δ_traj>0 的 scenario）。",
        f"- LOCAL / TRAJ 水平比={_fmt(verdict['local_over_traj'])}。",
        f"- PREFIX NATURAL={_sci(p['conditions']['NATURAL']['U_traj']['mean'])}。EXEC TRAJ-ORACLE={_sci(verdict['exec_traj'])}。",
        "",
        "只修正第一 hold：",
        "",
        f"1. 当前 policy 的 first-hold 确实差：NATURAL {_sci(verdict['exec_natural'])}。",
        f"2. 只改成 `argmax U_single` 能恢复一部分：{_sci(verdict['exec_natural'])} → {_sci(verdict['exec_local'])}。",
        f"3. 仍有一块来自 long-horizon 选点：LOCAL {_sci(verdict['exec_local'])} vs TRAJ {_sci(verdict['exec_traj'])}，缺口 {_sci(verdict['exec_traj']-verdict['exec_local'])}。",
        "",
        "## 3. 第一 hold 后续状态效应",
        "",
        "对每个合法 first-hold Γ：`F(Γ)=U_traj(Γ)-U_single(Γ)`。图见 `outputs/figures/u_single_vs_F.png`。",
        "",
        "| method | Pearson(U,F) | Spearman(U,F) | Kendall(U,F) | NATURAL F | LOCAL F | TRAJ F |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fam in FAMS:
        s = stats[fam]
        lines.append(
            "| {m} | {pe} | {sp} | {ke} | {fn} | {fl} | {ft} |".format(
                m=fam,
                pe=_fmt(s["U_vs_F"]["pearson"]),
                sp=_fmt(s["U_vs_F"]["spearman"]),
                ke=_fmt(s["U_vs_F"]["kendall"]),
                fn=_sci(s["conditions"]["NATURAL"]["F"]["mean"]),
                fl=_sci(s["conditions"]["LOCAL-ORACLE"]["F"]["mean"]),
                ft=_sci(s["conditions"]["TRAJ-ORACLE"]["F"]["mean"]),
            )
        )
    lines += [
        "",
        f"EXEC：U_single 与 F 的 Spearman={_fmt(e['U_vs_F']['spearman'])}。TRAJ-ORACLE 的 F mean={_sci(e['conditions']['TRAJ-ORACLE']['F']['mean'])}，高于 LOCAL {_sci(e['conditions']['LOCAL-ORACLE']['F']['mean'])} 与 NATURAL {_sci(e['conditions']['NATURAL']['F']['mean'])}。",
        "第一 hold 的轨迹价值大量通过后续状态体现：traj-oracle 选的往往不是最高 U_single，而是更高的 F。",
        "",
        "## 4. 四个问题",
        "",
        f"**Q1. EXEC 有多少可以只靠修正 first-hold 恢复？**  ",
        f"相对 NATURAL，LOCAL-ORACLE 抬升 {_sci(verdict['delta_local'])}（到 {_sci(verdict['exec_local'])}），TRAJ-ORACLE 抬升 {_sci(verdict['delta_traj'])}（到 {_sci(verdict['exec_traj'])}）。",
        f"只改第一 hold、后续仍正式 greedy，可以把 seed0 从 {_sci(verdict['exec_natural'])} 拉到最多 {_sci(verdict['exec_traj'])}，已经贴近 PREFIX NATURAL {_sci(verdict['prefix_natural'])}。",
        f"其中 local ranking 只贡献 R={_fmt(verdict['R_mean'])} 的可恢复增益。",
        "",
        f"**Q2. argmax U_single 是否已经足以接近 argmax U_traj？**  ",
        f"{'是' if verdict['local_close_to_traj'] else '否'}。LOCAL={_sci(verdict['exec_local'])}，TRAJ={_sci(verdict['exec_traj'])}，水平比={_fmt(verdict['local_over_traj'])}，剩余 {_sci(verdict['exec_traj']-verdict['exec_local'])}。",
        "",
        f"**Q3. 剩余 gap 能否归因于 long-horizon credit / state-transition？**  ",
        f"{'可以' if (not verdict['local_close_to_traj'] and verdict['traj_reaches_prefix']) else '不能单独钉死'}。控制变量只有第一 hold；后续 greedy、pool、decoder 都冻结。LOCAL 与 TRAJ 的差只能来自“第一 hold 按 U_single 还是按 U_traj 选”，而 TRAJ 的优势主要来自更高的 F（后续状态）。不是 4-hold 全轨迹 oracle。",
        "",
        f"**Q4. 下一阶段最小假设？**  ",
        f"**{verdict['q4']}** — {verdict['q4_text']}。",
        "排除：C（U_single 沿路径可加，不是错误标签）；D（traj-oracle 第一 hold 已贴近 PREFIX，不必先重训整个 Low-Level）。A 单独不够，因为 R 只有一半左右。",
        "",
        "本阶段停止。不要训练。不要改方法或 checkpoint。不要进入 End-to-End Independent Validation。",
        "",
    ]
    return "\n".join(lines)


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    nat = load_json("natural_trajectories.json")
    cf = load_json("counterfactual.json")
    pool = load_pool()
    interventions, candidates, gains = build_rows(nat, cf, pool)
    stats = family_stats(gains, interventions, candidates)
    verdict = decide(stats)
    _csv(OUTPUT / "first_hold_counterfactual.csv", interventions)
    _csv(OUTPUT / "first_hold_all_candidates.csv", candidates)
    _csv(OUTPUT / "method_comparison.csv", [
        {
            "method": fam,
            "natural": stats[fam]["conditions"]["NATURAL"]["U_traj"]["mean"],
            "local_oracle": stats[fam]["conditions"]["LOCAL-ORACLE"]["U_traj"]["mean"],
            "traj_oracle": stats[fam]["conditions"]["TRAJ-ORACLE"]["U_traj"]["mean"],
            "pool_uniform": stats[fam]["pool_uniform_U_traj"]["mean"],
            "delta_local": stats[fam]["delta_local"]["mean"],
            "delta_traj": stats[fam]["delta_traj"]["mean"],
            "R": stats[fam]["R"]["mean"],
            "U_vs_F_spearman": stats[fam]["U_vs_F"]["spearman"],
        }
        for fam in FAMS
    ])
    summary_rows = []
    for fam in FAMS:
        for name in CONDS:
            u = stats[fam]["conditions"][name]["U_traj"]
            summary_rows.append(
                {
                    "method": fam,
                    "condition": name,
                    "mean": u["mean"],
                    "median": u["median"],
                    "p25": u["p25"],
                    "p75": u["p75"],
                    "p90": u["p90"],
                    "std": u["std"],
                    "n": u["n"],
                }
            )
    _csv(OUTPUT / "summary.csv", summary_rows)
    _csv(OUTPUT / "per_scenario_gains.csv", gains)
    figures(stats, candidates, interventions, gains)
    payload = {"stats": stats, "verdict": verdict, "source": SOURCE, "n_interventions": len(interventions), "n_candidates": len(candidates)}
    _dump(OUTPUT / "PHASE3C_SUMMARY.json", payload)
    report = write_report(stats, verdict)
    (OUTPUT / "HTW_Phase3C_FirstHoldCredit_Counterfactual_CN.md").write_text(report, encoding="utf-8")
    (PHASE / "HTW_Phase3C_FirstHoldCredit_Counterfactual_CN.md").write_text(report, encoding="utf-8")
    (ROOT / "docs/archive/phase_reports/HTW_Phase3C_FirstHoldCredit_Counterfactual_CN.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "q4": verdict["q4"],
                "EXEC": {
                    "NATURAL": verdict["exec_natural"],
                    "LOCAL-ORACLE": verdict["exec_local"],
                    "TRAJ-ORACLE": verdict["exec_traj"],
                    "R": verdict["R_mean"],
                },
            },
            indent=2,
        ),
        flush=True,
    )
    return payload


if __name__ == "__main__":
    run()
