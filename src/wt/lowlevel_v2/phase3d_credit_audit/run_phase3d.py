"""Phase-3D trajectory-aware first-hold score audit. Frozen rollouts only."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wt.lowlevel_v2.integration_phase1.stats import kendall_tau, spearman, summarize, topk_overlap
from wt.lowlevel_v2.protocol import ROOT

PHASE = Path(__file__).resolve().parent
OUTPUT = PHASE / "outputs"
FIG = OUTPUT / "figures"
P3C = ROOT / "src" / "wt" / "lowlevel_v2" / "phase3c_credit_counterfactual" / "outputs"
FAMS = ("OLD", "ABS", "PREFIX", "EXEC")
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
SANITY_TOL = 1.0
PHASE3C_EXPECT = {
    "EXEC": {"NATURAL": 5812500.0, "LOCAL-ORACLE": 9000000.0, "TRAJ-ORACLE": 11875000.0},
    "PREFIX": {"NATURAL": 11562500.0, "LOCAL-ORACLE": 10312500.0, "TRAJ-ORACLE": 14375000.0},
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


def load_csv(name: str) -> list[dict]:
    rows = []
    with (P3C / name).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows


def _f(row: dict, key: str) -> float:
    return float(row[key])


def build():
    cand = load_csv("first_hold_all_candidates.csv")
    inter = load_csv("first_hold_counterfactual.csv")
    add_err = []
    by = defaultdict(list)
    for row in cand:
        u = _f(row, "first_hold_U_single")
        f = _f(row, "F")
        t = _f(row, "final_U_traj")
        add_err.append(abs((u + f) - t))
        by[(row["method"], int(row["scenario"]))].append(
            {
                "method": row["method"],
                "scenario": int(row["scenario"]),
                "cand": int(float(row["cand"])),
                "hash": row["first_hold"],
                "U_single": u,
                "F": f,
                "U_traj": t,
                "DecodeNR": _f(row, "DecodeNR"),
            }
        )
    if max(add_err) > SANITY_TOL:
        raise SystemExit(f"STOP: U_single+F != U_traj, max abs error={max(add_err)}")

    natural = {}
    local = {}
    traj = {}
    for row in inter:
        key = (row["method"], int(row["scenario"]))
        if row["condition"] == "NATURAL":
            natural[key] = row
        elif row["condition"] == "LOCAL-ORACLE":
            local[key] = row
        elif row["condition"] == "TRAJ-ORACLE":
            traj[key] = row

    score_rows = []
    pick_rows = []
    rank_rows = []
    for (fam, sc), items in sorted(by.items()):
        items = sorted(items, key=lambda x: x["cand"])
        u = np.array([x["U_single"] for x in items], dtype=np.float64)
        f = np.array([x["F"] for x in items], dtype=np.float64)
        t = np.array([x["U_traj"] for x in items], dtype=np.float64)
        it = int(np.argmax(t))
        iu = int(np.argmax(u))
        for beta in BETAS:
            s = u + float(beta) * f
            ib = int(np.argmax(s))
            chosen = items[ib]
            for i, item in enumerate(items):
                score_rows.append(
                    {
                        "method": fam,
                        "scenario": sc,
                        "beta": beta,
                        "cand": item["cand"],
                        "first_hold": item["hash"],
                        "U_single": item["U_single"],
                        "F": item["F"],
                        "S_beta": float(s[i]),
                        "U_traj": item["U_traj"],
                        "is_selected": int(i == ib),
                    }
                )
            pick_rows.append(
                {
                    "method": fam,
                    "scenario": sc,
                    "beta": beta,
                    "first_hold": chosen["hash"],
                    "first_hold_cand": chosen["cand"],
                    "first_hold_U_single": chosen["U_single"],
                    "future_return": chosen["F"],
                    "final_U_traj": chosen["U_traj"],
                    "payload": chosen["U_traj"],
                    "DecodeNR": chosen["DecodeNR"],
                    "S_beta": float(s[ib]),
                    "matches_local_oracle": int(ib == iu),
                    "matches_traj_oracle": int(ib == it),
                    "natural_U_traj": _f(natural[(fam, sc)], "final_U_traj"),
                    "local_oracle_U_traj": _f(local[(fam, sc)], "final_U_traj"),
                    "traj_oracle_U_traj": _f(traj[(fam, sc)], "final_U_traj"),
                }
            )
            rank_rows.append(
                {
                    "method": fam,
                    "scenario": sc,
                    "beta": beta,
                    "spearman": spearman(s, t),
                    "kendall": kendall_tau(s, t),
                    "top1": float(ib == it),
                    "top3": topk_overlap(s, t, 3),
                }
            )
    return score_rows, pick_rows, rank_rows, natural, local, traj


def family_tables(pick_rows, rank_rows):
    stats = {}
    for fam in FAMS:
        stats[fam] = {"beta": {}}
        for beta in BETAS:
            picks = [r for r in pick_rows if r["method"] == fam and r["beta"] == beta]
            ranks = [r for r in rank_rows if r["method"] == fam and r["beta"] == beta]
            stats[fam]["beta"][str(beta)] = {
                "U_traj": ext_sum([r["final_U_traj"] for r in picks]),
                "U_single": ext_sum([r["first_hold_U_single"] for r in picks]),
                "F": ext_sum([r["future_return"] for r in picks]),
                "DecodeNR": ext_sum([r["DecodeNR"] for r in picks]),
                "spearman": ext_sum([r["spearman"] for r in ranks]),
                "kendall": ext_sum([r["kendall"] for r in ranks]),
                "top1": ext_sum([r["top1"] for r in ranks]),
                "top3": ext_sum([r["top3"] for r in ranks]),
                "match_local": float(np.mean([r["matches_local_oracle"] for r in picks])),
                "match_traj": float(np.mean([r["matches_traj_oracle"] for r in picks])),
            }
        nat = [r["natural_U_traj"] for r in pick_rows if r["method"] == fam and r["beta"] == 0.0]
        loc = [r["local_oracle_U_traj"] for r in pick_rows if r["method"] == fam and r["beta"] == 0.0]
        trj = [r["traj_oracle_U_traj"] for r in pick_rows if r["method"] == fam and r["beta"] == 0.0]
        stats[fam]["NATURAL"] = ext_sum(nat)
        stats[fam]["LOCAL-ORACLE"] = ext_sum(loc)
        stats[fam]["TRAJ-ORACLE"] = ext_sum(trj)
    return stats


def sanity(stats: dict) -> dict:
    checks = []
    for fam in FAMS:
        b0 = stats[fam]["beta"]["0.0"]["U_traj"]["mean"]
        b1 = stats[fam]["beta"]["1.0"]["U_traj"]["mean"]
        loc = stats[fam]["LOCAL-ORACLE"]["mean"]
        trj = stats[fam]["TRAJ-ORACLE"]["mean"]
        checks.append(
            {
                "method": fam,
                "beta0": b0,
                "local_oracle": loc,
                "abs_beta0_local": abs(b0 - loc),
                "beta1": b1,
                "traj_oracle": trj,
                "abs_beta1_traj": abs(b1 - trj),
                "ok_beta0": abs(b0 - loc) <= SANITY_TOL,
                "ok_beta1": abs(b1 - trj) <= SANITY_TOL,
            }
        )
    ok = all(c["ok_beta0"] and c["ok_beta1"] for c in checks)
    exec_trj = stats["EXEC"]["TRAJ-ORACLE"]["mean"]
    exec_expect = PHASE3C_EXPECT["EXEC"]["TRAJ-ORACLE"]
    if abs(exec_trj - exec_expect) > SANITY_TOL:
        ok = False
    if not ok:
        raise SystemExit("STOP: beta sanity failed vs Phase-3C LOCAL/TRAJ-ORACLE. " + json.dumps(checks, default=_jsonable))
    return {"ok": True, "checks": checks}


def decide(stats: dict) -> dict:
    e = stats["EXEC"]
    means = [e["beta"][str(b)]["U_traj"]["mean"] for b in BETAS]
    spears = [e["beta"][str(b)]["spearman"]["mean"] for b in BETAS]
    top1s = [e["beta"][str(b)]["top1"]["mean"] for b in BETAS]
    nat, loc, trj = e["NATURAL"]["mean"], e["LOCAL-ORACLE"]["mean"], e["TRAJ-ORACLE"]["mean"]
    diffs = np.diff(np.asarray(means, dtype=np.float64))
    monotonic = bool(np.all(diffs >= -SANITY_TOL))
    q1 = bool(means[-1] >= means[0] - SANITY_TOL and (means[2] > means[0] + SANITY_TOL or means[-1] > means[0] + SANITY_TOL))
    # sweet spot: smallest beta whose U_traj is within 5% of beta=1 / TRAJ
    sweet = None
    for b, m in zip(BETAS, means):
        if m >= 0.95 * trj:
            sweet = b
            break
    must_near_one = sweet is None or sweet >= 0.75
    mid = e["beta"]["0.5"]["U_traj"]["mean"]
    mid_frac = (mid - loc) / (trj - loc) if trj > loc else float("nan")
    # The tested object is candidate-level F, not later-hold executor credit.
    # A discounted F (β≈0.5) is a sufficient *label*, so do not jump to multi-hold retraining.
    if q1:
        q4 = "C"
        q4_text = (
            "有效信号是每个第一 hold 候选的未来回报 F；"
            f"β=0.5 已回收 LOCAL→TRAJ 剩余增益的 {_fmt(mid_frac)}，"
            "因此下一阶段应预测候选级 F（或折扣 F），而不是重训后续 hold / Low-Level。"
            "A 只是把同一分数做成 first-hold critic 的实现方式；B 会被误读成 multi-hold executor credit，与 Phase-3C 不符。"
        )
    else:
        q4 = "C"
        q4_text = "β>0 没有稳定抬升，当前线性 S_beta 不足以作为下一阶段主假设"
    return {
        "q1": q1,
        "monotonic": monotonic,
        "sweet_beta": sweet,
        "must_near_one": must_near_one,
        "exec_means": {str(b): m for b, m in zip(BETAS, means)},
        "exec_spearman": {str(b): s for b, s in zip(BETAS, spears)},
        "exec_top1": {str(b): t for b, t in zip(BETAS, top1s)},
        "mid_frac_of_remaining": mid_frac,
        "natural": nat,
        "local": loc,
        "traj": trj,
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


def figures(stats: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    e = stats["EXEC"]
    xs = list(BETAS)
    ys = [e["beta"][str(b)]["U_traj"]["mean"] for b in BETAS]
    ax.plot(xs, ys, "o-", color="#4C78A8", label="argmax S_beta")
    ax.axhline(e["NATURAL"]["mean"], color="#9D755D", ls="--", label="NATURAL")
    ax.axhline(e["LOCAL-ORACLE"]["mean"], color="#F58518", ls="--", label="LOCAL-ORACLE / β=0")
    ax.axhline(e["TRAJ-ORACLE"]["mean"], color="#54A24B", ls="--", label="TRAJ-ORACLE / β=1")
    ax.set_xlabel("beta")
    ax.set_ylabel("EXEC mean U_traj")
    ax.set_title("EXEC first-hold S_beta intervention")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "exec_beta_sweep.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for fam, col in zip(FAMS, ("#4C78A8", "#F58518", "#54A24B", "#E45756")):
        ax.plot(xs, [stats[fam]["beta"][str(b)]["U_traj"]["mean"] for b in BETAS], "o-", color=col, label=fam)
    ax.set_xlabel("beta")
    ax.set_ylabel("mean U_traj")
    ax.set_title("S_beta first-hold intervention")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "method_beta_sweep.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(xs, [e["beta"][str(b)]["spearman"]["mean"] for b in BETAS], "o-", label="Spearman")
    ax.plot(xs, [e["beta"][str(b)]["kendall"]["mean"] for b in BETAS], "s-", label="Kendall")
    ax.plot(xs, [e["beta"][str(b)]["top1"]["mean"] for b in BETAS], "^-", label="top-1")
    ax.plot(xs, [e["beta"][str(b)]["top3"]["mean"] for b in BETAS], "d-", label="top-3")
    ax.set_xlabel("beta")
    ax.set_ylabel("ranking vs U_traj")
    ax.set_title("EXEC S_beta ranking quality")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "exec_ranking.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for fam, col in zip(FAMS, ("#4C78A8", "#F58518", "#54A24B", "#E45756")):
        axes[0].plot(xs, [stats[fam]["beta"][str(b)]["spearman"]["mean"] for b in BETAS], "o-", color=col, label=fam)
        axes[1].plot(xs, [stats[fam]["beta"][str(b)]["top1"]["mean"] for b in BETAS], "o-", color=col, label=fam)
    axes[0].set_title("Spearman(S_beta, U_traj)")
    axes[1].set_title("top-1 agreement")
    for ax in axes:
        ax.set_xlabel("beta")
        ax.set_ylim(-0.05, 1.05)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG / "method_ranking.png", dpi=140)
    plt.close(fig)


def write_report(stats: dict, verdict: dict, sanity_info: dict) -> str:
    e = stats["EXEC"]
    p = stats["PREFIX"]
    lines = [
        "# HTW Phase-3D：Trajectory-Aware First-Hold Credit Minimal Audit",
        "",
        "本阶段只做离线假设检验。未训练、未改模型 / checkpoint / Low-Level / decoder / pool / U_single / 正式 evaluation。",
        "",
        "`F` 来自 Phase-3C 已冻结的 first-hold counterfactual remaining return，**不是**在线可观测特征。因此本报告不声称已经解决 Low-Level。",
        "",
        "分数（只用于离线选第一 hold）：",
        "",
        "`S_β(Γ) = U_single(Γ) + β F(Γ)`，`β ∈ {0, 0.25, 0.5, 0.75, 1}`。",
        "",
        "选 `Γ* = argmax S_β` 后，hold 2/3/4 仍是当时正式 greedy。β=0 必须等于 LOCAL-ORACLE；β=1 必须等于 TRAJ-ORACLE。",
        "",
        f"Sanity：**通过**。β=0 与 LOCAL-ORACLE、β=1 与 TRAJ-ORACLE 的均值差均 ≤ {SANITY_TOL}。EXEC TRAJ-ORACLE={_sci(e['TRAJ-ORACLE']['mean'])}，与 Phase-3C 11.875e6 一致。",
        "",
        f"**EXEC：NATURAL={_sci(e['NATURAL']['mean'])}，β=0 / LOCAL={_sci(e['beta']['0.0']['U_traj']['mean'])}，β=1 / TRAJ={_sci(e['beta']['1.0']['U_traj']['mean'])}。**",
        "",
        "## 1. EXEC β 扫描",
        "",
        "| beta | final U_traj | F | U_single | DecodeNR | Spearman | Kendall | top-1 | top-3 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for beta in BETAS:
        s = e["beta"][str(beta)]
        lines.append(
            "| {b} | {u} | {f} | {us} | {nr} | {sp} | {ke} | {t1} | {t3} |".format(
                b=beta,
                u=_sci(s["U_traj"]["mean"]),
                f=_sci(s["F"]["mean"]),
                us=_sci(s["U_single"]["mean"]),
                nr=_fmt(s["DecodeNR"]["mean"]),
                sp=_fmt(s["spearman"]["mean"]),
                ke=_fmt(s["kendall"]["mean"]),
                t1=_fmt(s["top1"]["mean"]),
                t3=_fmt(s["top3"]["mean"]),
            )
        )
    lines += [
        "",
        f"对照：NATURAL U_traj={_sci(e['NATURAL']['mean'])}，LOCAL-ORACLE={_sci(e['LOCAL-ORACLE']['mean'])}，TRAJ-ORACLE={_sci(e['TRAJ-ORACLE']['mean'])}。",
        f"EXEC 均值序列是否随 β 非降：{verdict['monotonic']}。β=0.5 相对 LOCAL→TRAJ 剩余增益的回收比例={_fmt(verdict['mid_frac_of_remaining'])}。",
        "",
        "## 2. 四方法 β audit",
        "",
        "| method | beta | final U_traj | top-1 | Spearman | Kendall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for fam in FAMS:
        for beta in BETAS:
            s = stats[fam]["beta"][str(beta)]
            lines.append(
                f"| {fam} | {beta} | {_sci(s['U_traj']['mean'])} | {_fmt(s['top1']['mean'])} | {_fmt(s['spearman']['mean'])} | {_fmt(s['kendall']['mean'])} |"
            )
    lines += [
        "",
        "PREFIX / OLD / ABS 同样扫描，用来看 trajectory-aware score 是否系统性改善 first-hold 排序，而不是比较谁最高。",
        "",
        f"PREFIX：NATURAL={_sci(p['NATURAL']['mean'])}，β=0={_sci(p['beta']['0.0']['U_traj']['mean'])}，β=1={_sci(p['beta']['1.0']['U_traj']['mean'])}。",
        "",
        "## 3. 四个问题",
        "",
        f"**Q1. β>0 是否比 β=0 更接近 TRAJ-ORACLE？**  ",
        f"{'是' if verdict['q1'] else '否'}。EXEC β=0 → {_sci(e['beta']['0.0']['U_traj']['mean'])}，β=0.5 → {_sci(e['beta']['0.5']['U_traj']['mean'])}，β=1 → {_sci(e['beta']['1.0']['U_traj']['mean'])}。",
        f"排序：Spearman β=0={_fmt(e['beta']['0.0']['spearman']['mean'])}，β=0.5={_fmt(e['beta']['0.5']['spearman']['mean'])}，β=1={_fmt(e['beta']['1.0']['spearman']['mean'])}；top-1 {_fmt(e['beta']['0.0']['top1']['mean'])} → {_fmt(e['beta']['0.5']['top1']['mean'])} → {_fmt(e['beta']['1.0']['top1']['mean'])}。",
        "",
        f"**Q2. 是否存在明显 sweet spot，还是必须接近 β=1？**  ",
    ]
    if verdict["sweet_beta"] is None:
        lines.append("在本网格上，没有任何 β<1 达到 TRAJ-ORACLE 的 95%。有效性随 β 增大，接近 β=1 才贴上 oracle。")
    elif verdict["sweet_beta"] == 0.0:
        lines.append("β=0 已达到 TRAJ 的 95%，不需要 future term。")
    else:
        lines.append(
            f"对 **闭环 U_traj**，最早达到 TRAJ 95% 的网格点是 β={verdict['sweet_beta']}（EXEC β=0.5 为 {_sci(e['beta']['0.5']['U_traj']['mean'])} vs TRAJ {_sci(e['TRAJ-ORACLE']['mean'])}）。"
            f"对 **精确 top-1**，仍要到 β=1 才到 1.0（β=0.5 只有 {_fmt(e['beta']['0.5']['top1']['mean'])}）。"
            "因此：不必精确 β=1 才能恢复 payload；但若目标是命中 argmax U_traj，则要完整 F。"
        )
    lines += [
        "",
        f"**Q3. EXEC 的 first-hold mismatch 能否靠 future-return 明显修正？**  ",
        f"{'能' if verdict['q1'] else '不能单靠这个线性项'}。加入 F 后，选点从 LOCAL {_sci(e['LOCAL-ORACLE']['mean'])} 移向 TRAJ {_sci(e['TRAJ-ORACLE']['mean'])}。",
        "这只证明 **存在** 可用的 trajectory-aware credit signal。F 本身由完整 4-hold counterfactual 得到，不能直接喂给在线执行器。",
        "",
        f"**Q4. 下一阶段最符合数据的最小假设？**  ",
        f"**{verdict['q4']}** — {verdict['q4_text']}",
        "不是最终算法。不要改 Low-Level，不要把 β 或 F 当成在线输入。",
        "",
        "本阶段停止。不训练。不改方法或 checkpoint。不进入 End-to-End Independent Validation。",
        "",
    ]
    return "\n".join(lines)


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    score_rows, pick_rows, rank_rows, _natural, _local, _traj = build()
    stats = family_tables(pick_rows, rank_rows)
    sanity_info = sanity(stats)
    verdict = decide(stats)
    _csv(OUTPUT / "first_hold_scores.csv", score_rows)
    _csv(OUTPUT / "beta_sweep.csv", pick_rows)
    _csv(OUTPUT / "ranking.csv", rank_rows)
    method_rows = []
    summary_rows = []
    for fam in FAMS:
        for beta in BETAS:
            s = stats[fam]["beta"][str(beta)]
            method_rows.append(
                {
                    "method": fam,
                    "beta": beta,
                    "final_U_traj": s["U_traj"]["mean"],
                    "top1_agreement": s["top1"]["mean"],
                    "spearman": s["spearman"]["mean"],
                    "kendall": s["kendall"]["mean"],
                    "top3": s["top3"]["mean"],
                    "F": s["F"]["mean"],
                    "U_single": s["U_single"]["mean"],
                    "DecodeNR": s["DecodeNR"]["mean"],
                }
            )
            summary_rows.append(
                {
                    "method": fam,
                    "beta": beta,
                    "mean": s["U_traj"]["mean"],
                    "median": s["U_traj"]["median"],
                    "p25": s["U_traj"]["p25"],
                    "p75": s["U_traj"]["p75"],
                    "p90": s["U_traj"]["p90"],
                    "std": s["U_traj"]["std"],
                    "spearman": s["spearman"]["mean"],
                    "kendall": s["kendall"]["mean"],
                    "top1": s["top1"]["mean"],
                    "top3": s["top3"]["mean"],
                }
            )
    _csv(OUTPUT / "method_comparison.csv", method_rows)
    _csv(OUTPUT / "summary.csv", summary_rows)
    figures(stats)
    payload = {
        "stats": stats,
        "verdict": verdict,
        "sanity": sanity_info,
        "source": {
            "candidates": str(P3C / "first_hold_all_candidates.csv"),
            "interventions": str(P3C / "first_hold_counterfactual.csv"),
        },
    }
    _dump(OUTPUT / "PHASE3D_SUMMARY.json", payload)
    report = write_report(stats, verdict, sanity_info)
    (OUTPUT / "HTW_Phase3D_TrajectoryAwareFirstHold_Audit_CN.md").write_text(report, encoding="utf-8")
    (PHASE / "HTW_Phase3D_TrajectoryAwareFirstHold_Audit_CN.md").write_text(report, encoding="utf-8")
    (ROOT / "docs/archive/phase_reports/HTW_Phase3D_TrajectoryAwareFirstHold_Audit_CN.md").write_text(report, encoding="utf-8")
    print(json.dumps({"sanity": True, "q4": verdict["q4"], "exec_means": verdict["exec_means"], "sweet": verdict["sweet_beta"]}, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    run()
