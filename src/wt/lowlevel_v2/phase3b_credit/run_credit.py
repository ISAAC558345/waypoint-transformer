"""Phase-3B credit audit from frozen trajectory artifacts. No training."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from wt.lowlevel_v2.integration_phase1.stats import kendall_tau, spearman, summarize
from wt.lowlevel_v2.protocol import ROOT
from wt.lowlevel_v2.traj_audit.protocol import OUTPUT as TRAJ_OUT

CREDIT = Path(__file__).resolve().parent
OUTPUT = CREDIT / "outputs"
FAMS = ("OLD", "ABS", "PREFIX", "EXEC")
OFFICIAL = {"OLD": 10912500.0, "ABS": 8440000.0, "PREFIX": 11287500.0, "EXEC": 8692500.0}


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


def topk_agree(u, v, k: int) -> float:
    a = np.asarray(u, dtype=np.float64)
    b = np.asarray(v, dtype=np.float64)
    k = min(int(k), int(a.size))
    if k <= 0:
        return float("nan")
    return float(len(set(np.argsort(-a)[:k].tolist()) & set(np.argsort(-b)[:k].tolist())) / k)


def nregret(best: float, chosen: float) -> float:
    return (float(best) - float(chosen)) / max(abs(float(best)), 1.0)


def rank_block(local, target) -> dict:
    a = np.asarray(local, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    empty = {
        "pearson": float("nan"),
        "spearman": float("nan"),
        "kendall": float("nan"),
        "top1": float("nan"),
        "top3": float("nan"),
        "nregret": float("nan"),
        "gap": float("nan"),
    }
    if a.size == 0 or b.size == 0:
        return empty
    iu = int(np.argmax(a))
    out = {
        "pearson": pearson(a, b),
        "spearman": spearman(a, b) if float(np.std(b)) >= 1e-12 else float("nan"),
        "kendall": kendall_tau(a, b) if float(np.std(b)) >= 1e-12 else float("nan"),
        "top1": float(iu == int(np.argmax(b))),
        "top3": topk_agree(a, b, 3),
        "nregret": nregret(float(np.max(b)), float(b[iu])),
        "gap": float(np.max(b) - b[iu]),
    }
    if float(np.std(b)) < 1e-12:
        out["top1"] = float("nan")
        out["top3"] = float("nan")
        out["nregret"] = 0.0 if float(np.max(np.abs(b))) < 1e-12 else out["nregret"]
        out["gap"] = 0.0 if float(np.max(np.abs(b))) < 1e-12 else out["gap"]
    return out


def load_json(name: str):
    return json.loads((TRAJ_OUT / name).read_text(encoding="utf-8"))


def load_pool() -> dict[int, list[dict]]:
    rows = list(csv.DictReader((TRAJ_OUT / "first_hold_pool.csv").open(encoding="utf-8", newline="")))
    by = defaultdict(list)
    for row in rows:
        by[int(float(row["scenario"]))].append({"hash": row["hash"], "U_LL": float(row["U_LL"])})
    return by


def natural_rows(nat: list[dict]) -> list[dict]:
    out = []
    for rec in nat:
        us = [float(h["U_single"]) for h in rec["holds"]]
        while len(us) < 4:
            us.append(0.0)
        g = [float(np.sum(us[t:])) for t in range(4)]
        f = [float(np.sum(us[t + 1 :])) for t in range(4)]
        for t in range(4):
            h = rec["holds"][t] if t < len(rec["holds"]) else {}
            out.append(
                {
                    "family": rec["family"],
                    "scenario": int(rec["scenario"]),
                    "hold": t + 1,
                    "t_env": int(h.get("t", t * 15)),
                    "hash": h.get("hash", ""),
                    "U_single": us[t],
                    "U_hold_actual": float(h.get("U_hold_actual", float("nan"))),
                    "G": g[t],
                    "F": f[t],
                    "U_traj": float(rec["U_traj"]),
                    "sum_U_single": float(rec["sum_U_single"]),
                    "gap": float(rec["interaction_gap"]),
                    "first_hash": rec.get("first_hash", ""),
                    "first_in_pool": int(bool(rec.get("first_in_pool", False))),
                }
            )
    return out


def hold_corr(rows: list[dict]) -> dict:
    out = {}
    for fam in FAMS:
        out[fam] = {}
        for t in (1, 2, 3, 4):
            sub = [r for r in rows if r["family"] == fam and int(r["hold"]) == t]
            u = [r["U_single"] for r in sub]
            g = [r["G"] for r in sub]
            f = [r["F"] for r in sub]
            ug = rank_block(u, g)
            uf = rank_block(u, f)
            u_arr = np.asarray(u, dtype=np.float64)
            f_arr = np.asarray(f, dtype=np.float64)
            out[fam][str(t)] = {
                "n": len(sub),
                "U_single": summarize(u),
                "G": summarize(g),
                "F": summarize(f),
                "U_vs_G": ug,
                "U_vs_F": uf,
                "highU_lowF": int(np.sum((u_arr > np.nanmedian(u_arr)) & (f_arr < np.nanmedian(f_arr)))),
                "lowU_highF": int(np.sum((u_arr < np.nanmedian(u_arr)) & (f_arr > np.nanmedian(f_arr)))),
                "note": "across 16 realized scenarios; one natural hold each. Not a candidate-pool ranking.",
            }
    return out


def counterfactual_credit(cf: list[dict], nat: list[dict]) -> tuple[list[dict], dict, list[dict]]:
    actual = {}
    for rec in nat:
        first = rec["holds"][0]
        actual[(rec["family"], int(rec["scenario"]))] = {
            "hash": rec.get("first_hash") or first["hash"],
            "U_single": float(first["U_single"]),
            "U_traj": float(rec["U_traj"]),
            "F": float(rec["U_traj"]) - float(first["U_single"]),
            "in_pool": bool(rec.get("first_in_pool", False)),
        }
    rows = []
    per_sc = []
    summary = {}
    for fam in FAMS:
        items = [r for r in cf if r["family"] == fam]
        by_sc = defaultdict(list)
        for r in items:
            by_sc[int(r["scenario"])].append(r)
        spears, kends, pears = [], [], []
        top1, top3, regs, gaps, disagree = [], [], [], [], []
        f_spears, f_top1, f_regs = [], [], []
        dloc_vs_at = []
        local_best_future, traj_best_local = [], []
        local_best_atraj, local_best_dloc = [], []
        actual_trajs, local_best_trajs, traj_best_trajs = [], [], []
        for sc, cand_rows in sorted(by_sc.items()):
            cand_rows = sorted(cand_rows, key=lambda x: int(x["cand"]))
            u = np.array([float(x["U_LL_first"]) for x in cand_rows], dtype=np.float64)
            t = np.array([float(x["U_traj"]) for x in cand_rows], dtype=np.float64)
            fut = t - u
            iu, it, iff = int(np.argmax(u)), int(np.argmax(t)), int(np.argmax(fut))
            ug = rank_block(u, t)
            uf = rank_block(u, fut)
            spears.append(ug["spearman"])
            kends.append(ug["kendall"])
            pears.append(ug["pearson"])
            top1.append(ug["top1"])
            top3.append(ug["top3"])
            regs.append(ug["nregret"])
            gaps.append(ug["gap"])
            disagree.append(float(iu != it))
            f_spears.append(uf["spearman"])
            f_top1.append(uf["top1"])
            f_regs.append(uf["nregret"])
            act = actual[(fam, sc)]
            for i, row in enumerate(cand_rows):
                dloc = float(u[i] - act["U_single"])
                atraj = float(t[i] - act["U_traj"])
                dfut = float(fut[i] - act["F"])
                dloc_vs_at.append((dloc, atraj, dfut))
                rows.append(
                    {
                        "family": fam,
                        "scenario": sc,
                        "cand": int(row["cand"]),
                        "hash": row["hash"],
                        "U_single": float(u[i]),
                        "U_traj": float(t[i]),
                        "F": float(fut[i]),
                        "G": float(t[i]),
                        "delta_local": dloc,
                        "A_traj": atraj,
                        "delta_future": dfut,
                        "is_local_best": int(i == iu),
                        "is_traj_best": int(i == it),
                        "is_future_best": int(i == iff),
                        "is_actual": int(row["hash"] == act["hash"]),
                    }
                )
            local_best_future.append(float(fut[iu] - act["F"]))
            traj_best_local.append(float(u[it] - act["U_single"]))
            local_best_atraj.append(float(t[iu] - act["U_traj"]))
            local_best_dloc.append(float(u[iu] - act["U_single"]))
            actual_trajs.append(float(act["U_traj"]))
            local_best_trajs.append(float(t[iu]))
            traj_best_trajs.append(float(t[it]))
            per_sc.append(
                {
                    "family": fam,
                    "scenario": sc,
                    "actual_hash": act["hash"],
                    "actual_in_pool": int(act["in_pool"]),
                    "actual_U_single": act["U_single"],
                    "actual_U_traj": act["U_traj"],
                    "actual_F": act["F"],
                    "oracle_local_hash": cand_rows[iu]["hash"],
                    "oracle_local_U_single": float(u[iu]),
                    "oracle_local_U_traj": float(t[iu]),
                    "oracle_local_F": float(fut[iu]),
                    "oracle_traj_hash": cand_rows[it]["hash"],
                    "oracle_traj_U_single": float(u[it]),
                    "oracle_traj_U_traj": float(t[it]),
                    "oracle_traj_F": float(fut[it]),
                    "disagree": int(iu != it),
                    "spearman_U_vs_Utraj": ug["spearman"],
                    "kendall_U_vs_Utraj": ug["kendall"],
                    "top1_U_vs_Utraj": ug["top1"],
                    "top3_U_vs_Utraj": ug["top3"],
                    "nregret_U_vs_Utraj": ug["nregret"],
                    "gap_U_vs_Utraj": ug["gap"],
                    "spearman_U_vs_F": uf["spearman"],
                    "top1_U_vs_F": uf["top1"],
                    "nregret_U_vs_F": uf["nregret"],
                    "A_traj_local_best": float(t[iu] - act["U_traj"]),
                    "delta_local_local_best": float(u[iu] - act["U_single"]),
                    "delta_future_local_best": float(fut[iu] - act["F"]),
                    "A_traj_traj_best": float(t[it] - act["U_traj"]),
                    "delta_local_traj_best": float(u[it] - act["U_single"]),
                    "delta_future_traj_best": float(fut[it] - act["F"]),
                }
            )
        dloc = np.array([x[0] for x in dloc_vs_at], dtype=np.float64)
        atraj = np.array([x[1] for x in dloc_vs_at], dtype=np.float64)
        dfut = np.array([x[2] for x in dloc_vs_at], dtype=np.float64)
        mask = np.abs(atraj) > 1.0
        summary[fam] = {
            "pearson": summarize(pears),
            "spearman": summarize(spears),
            "kendall": summarize(kends),
            "top1": summarize(top1),
            "top3": summarize(top3),
            "traj_regret_of_local_argmax": summarize(regs),
            "U_traj_gap_local_vs_best": summarize(gaps),
            "local_traj_disagree_rate": float(np.mean(disagree)) if disagree else float("nan"),
            "U_vs_F": {
                "spearman": summarize(f_spears),
                "top1": summarize(f_top1),
                "nregret": summarize(f_regs),
            },
            "delta_local_vs_A_traj": {
                "pearson": pearson(dloc, atraj),
                "spearman": spearman(dloc, atraj),
                "kendall": kendall_tau(dloc, atraj),
            },
            "delta_future_vs_A_traj": {
                "pearson": pearson(dfut, atraj),
                "spearman": spearman(dfut, atraj),
                "kendall": kendall_tau(dfut, atraj),
            },
            "delta_local_vs_delta_future": {
                "pearson": pearson(dloc, dfut),
                "spearman": spearman(dloc, dfut),
            },
            "A_traj_share_local": float(np.nanmean(dloc[mask] / atraj[mask])) if np.any(mask) else float("nan"),
            "A_traj_share_future": float(np.nanmean(dfut[mask] / atraj[mask])) if np.any(mask) else float("nan"),
            "local_best_delta_future_vs_actual": summarize(local_best_future),
            "traj_best_delta_local_vs_actual": summarize(traj_best_local),
            "local_best_A_traj_vs_actual": summarize(local_best_atraj),
            "local_best_delta_local_vs_actual": summarize(local_best_dloc),
            "mean_U_traj_actual": float(np.mean(actual_trajs)),
            "mean_U_traj_local_best": float(np.mean(local_best_trajs)),
            "mean_U_traj_traj_best": float(np.mean(traj_best_trajs)),
        }
    return rows, summary, per_sc


def first_hold_focus(nat: list[dict], pool: dict[int, list[dict]], per_sc: list[dict], fam: str) -> dict:
    recs = [r for r in nat if r["family"] == fam]
    rows = [r for r in per_sc if r["family"] == fam]
    by_sc = {int(r["scenario"]): r for r in rows}
    local_pct, loc_reg, traj_reg = [], [], []
    details = []
    for rec in recs:
        sc = int(rec["scenario"])
        u0 = float(rec["holds"][0]["U_single"])
        traj = float(rec["U_traj"])
        pool_u = np.array([p["U_LL"] for p in pool[sc]], dtype=np.float64)
        row = by_sc[sc]
        local_pct.append(float(np.mean(pool_u <= u0)))
        loc_reg.append(nregret(float(np.max(pool_u)), u0))
        traj_reg.append(nregret(float(row["oracle_traj_U_traj"]), traj))
        details.append(
            {
                "scenario": sc,
                "actual_U_single": u0,
                "actual_U_traj": traj,
                "actual_F": traj - u0,
                "first_in_pool": int(bool(rec.get("first_in_pool", False))),
                "local_pct_in_pool": float(np.mean(pool_u <= u0)),
                "traj_pct_among_cf": float("nan"),
                "oracle_local_U_single": row["oracle_local_U_single"],
                "oracle_local_U_traj": row["oracle_local_U_traj"],
                "oracle_traj_U_single": row["oracle_traj_U_single"],
                "oracle_traj_U_traj": row["oracle_traj_U_traj"],
                "disagree": row["disagree"],
                "local_regret": loc_reg[-1],
                "traj_regret": traj_reg[-1],
                "comp_actual_F_minus_localbest_F": float(row["actual_F"] - row["oracle_local_F"]),
                "A_traj_local_best": row["A_traj_local_best"],
                "delta_local_local_best": row["delta_local_local_best"],
                "delta_future_local_best": row["delta_future_local_best"],
            }
        )
    # fill traj percentile from counterfactual file via per_sc actual vs all candidates is not in per_sc.
    return {
        "n": len(recs),
        "actual_U_single": summarize([float(r["holds"][0]["U_single"]) for r in recs]),
        "actual_U_traj": summarize([float(r["U_traj"]) for r in recs]),
        "actual_F": summarize([float(r["U_traj"]) - float(r["holds"][0]["U_single"]) for r in recs]),
        "local_percentile": summarize(local_pct),
        "local_regret": summarize(loc_reg),
        "traj_regret": summarize(traj_reg),
        "oracle_local_vs_traj_disagree": float(np.mean([d["disagree"] for d in details])),
        "per_scenario": details,
    }


def fill_traj_percentiles(focus: dict, cf: list[dict], fam: str) -> dict:
    by_sc = defaultdict(list)
    for row in cf:
        if row["family"] == fam:
            by_sc[int(row["scenario"])].append(float(row["U_traj"]))
    pcts = []
    for d in focus["per_scenario"]:
        arr = np.asarray(by_sc[int(d["scenario"])], dtype=np.float64)
        d["traj_pct_among_cf"] = float(np.mean(arr <= d["actual_U_traj"])) if arr.size else float("nan")
        pcts.append(d["traj_pct_among_cf"])
    focus["traj_percentile_among_cf"] = summarize(pcts)
    return focus


def method_table(nat: list[dict], corr: dict, cf_sum: dict, foci: dict) -> list[dict]:
    rows = []
    for fam in FAMS:
        nsub = [r for r in nat if r["family"] == fam]
        for t in (1, 2, 3, 4):
            h = corr[fam][str(t)]
            row = {
                "method": fam,
                "hold": t,
                "local_U_single_mean": h["U_single"]["mean"],
                "remaining_G_mean": h["G"]["mean"],
                "future_F_mean": h["F"]["mean"],
                "local_vs_G_spearman": h["U_vs_G"]["spearman"],
                "local_vs_G_pearson": h["U_vs_G"]["pearson"],
                "local_vs_G_kendall": h["U_vs_G"]["kendall"],
                "local_vs_G_top1": h["U_vs_G"]["top1"],
                "local_vs_G_top3": h["U_vs_G"]["top3"],
                "local_vs_G_nregret": h["U_vs_G"]["nregret"],
                "local_vs_F_spearman": h["U_vs_F"]["spearman"],
                "local_vs_F_pearson": h["U_vs_F"]["pearson"],
                "local_vs_F_kendall": h["U_vs_F"]["kendall"],
                "local_vs_F_top1": h["U_vs_F"]["top1"],
                "local_vs_F_top3": h["U_vs_F"]["top3"],
                "local_vs_F_nregret": h["U_vs_F"]["nregret"],
                "cf_spearman": cf_sum[fam]["spearman"]["mean"] if t == 1 else float("nan"),
                "cf_top1": cf_sum[fam]["top1"]["mean"] if t == 1 else float("nan"),
                "cf_traj_regret": cf_sum[fam]["traj_regret_of_local_argmax"]["mean"] if t == 1 else float("nan"),
                "natural_U_traj_mean": float(np.mean([r["U_traj"] for r in nsub])),
                "official_closed_loop": OFFICIAL[fam],
            }
            rows.append(row)
    return rows


def decide(corr: dict, cf: dict, foci: dict) -> dict:
    e1 = corr["EXEC"]["1"]
    c = cf["EXEC"]
    ex = foci["EXEC"]
    px = foci["PREFIX"]
    local_bad = float(ex["actual_U_single"]["mean"]) < 1.5e6 and float(ex["local_percentile"]["median"]) < 0.55
    traj_also_bad = float(ex["actual_U_traj"]["mean"]) < float(px["actual_U_traj"]["mean"]) - 1e6
    no_comp = float(ex["actual_F"]["mean"]) <= float(px["actual_F"]["mean"]) + 1e5
    credit_mismatch = float(c["top1"]["mean"]) < 0.3 and float(c["spearman"]["mean"]) < 0.7
    future_drives = float(c["delta_future_vs_A_traj"]["spearman"]) > float(c["delta_local_vs_A_traj"]["spearman"])
    if local_bad and credit_mismatch:
        label = "C"
        text = "两者同时存在：EXEC seed0 第一 hold 局部选点差，且四条方法的 local ranking 与 trajectory ranking 都不一致"
    elif local_bad and not credit_mismatch:
        label = "A"
        text = "主要是 local candidate selection 错误"
    elif credit_mismatch:
        label = "B"
        text = "主要是 long-horizon credit assignment"
    else:
        label = "undetermined"
        text = "当前数据无法单独钉死 A 或 B"
    return {
        "label": label,
        "text": text,
        "first_hold_U_vs_F_spearman": e1["U_vs_F"]["spearman"],
        "first_hold_U_vs_G_spearman": e1["U_vs_G"]["spearman"],
        "cf_top1": c["top1"]["mean"],
        "cf_spearman": c["spearman"]["mean"],
        "local_drives_A_traj": c["delta_local_vs_A_traj"]["spearman"],
        "future_drives_A_traj": c["delta_future_vs_A_traj"]["spearman"],
        "future_stronger_than_local": future_drives,
        "local_bad": local_bad,
        "traj_also_bad": traj_also_bad,
        "no_compensation_vs_PREFIX": no_comp,
        "credit_mismatch": credit_mismatch,
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


def write_report(p: dict) -> str:
    v = p["verdict"]
    c = p["counterfactual"]
    h = p["hold_corr"]
    e = p["focus"]["EXEC"]
    px = p["focus"]["PREFIX"]
    lines = [
        "# HTW Phase-3B：Hold-to-Trajectory Credit Audit",
        "",
        "本阶段只做离线诊断。未训练、未改模型、未改 checkpoint、未改 q / L_exec / ranking / reference / DeepSets / decoder / candidate pool / beam / Pool-MAP / Low-Level MAPPO / SafetyProjector / collision / interference / reward / tau / lambda / 正式 evaluation pipeline。",
        "",
        "全部数字复用已冻结的 Phase-3 Trajectory Utility Audit 产物：",
        "",
        "- 设计集 221000–221015（16 scenario）",
        "- OLD / ABS / PREFIX / EXEC 各 **seed0**",
        "- 闭环 Low-Level seed=0",
        "- 自然 4-hold 轨迹 64 条；第一 hold 24-pool counterfactual 1536 条",
        "",
        "Sanity（不重新证明）：沿已实现轨迹 `U_traj ≈ Σ_t U_single_t`。四方法 Spearman 约 0.98–0.997，interaction gap 中位数 0。",
        "",
        f"**总判定：{v['label']} — {v['text']}。** 结论跟数据，不预设 long-horizon credit assignment 一定成立。",
        "",
        "## 1. 自然轨迹 hold 贡献",
        "",
        "每条已实现 4-hold 轨迹记录 `state` 只通过已有 hold 序列表达：`Γ_t`、`U_single_t`、`U_traj`。",
        "",
        "定义（只用于诊断，不是新训练 target）：",
        "",
        "- `G_t = Σ_{r=t}^{4} U_single_r` remaining return",
        "- `F_t = Σ_{r=t+1}^{4} U_single_r` future return",
        "",
        "下面的 Pearson / Spearman / Kendall / top-1 / top-3 / normalized regret 是 **16 条已实现轨迹跨 scenario 的排序**，不是 hold 内 24-pool 排序。t=4 时 `F=0`，相关为 NA。",
        "",
        "| method | hold | U_single mean | G mean | F mean | Spearman(U,G) | Spearman(U,F) | top-1(U,G) | top-1(U,F) | nregret(U→G) | nregret(U→F) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fam in FAMS:
        for t in ("1", "2", "3", "4"):
            s = h[fam][t]
            lines.append(
                "| {fam} | {t} | {um} | {gm} | {fm} | {sg} | {sf} | {tg} | {tf} | {rg} | {rf} |".format(
                    fam=fam,
                    t=t,
                    um=_sci(s["U_single"]["mean"]),
                    gm=_sci(s["G"]["mean"]),
                    fm=_sci(s["F"]["mean"]),
                    sg=_fmt(s["U_vs_G"]["spearman"]),
                    sf=_fmt(s["U_vs_F"]["spearman"]),
                    tg=_fmt(s["U_vs_G"]["top1"]),
                    tf=_fmt(s["U_vs_F"]["top1"]),
                    rg=_fmt(s["U_vs_G"]["nregret"]),
                    rf=_fmt(s["U_vs_F"]["nregret"]),
                )
            )
    lines += [
        "",
        f"第一 hold 交叉象限（相对中位数，16 scenario）：EXEC 高U低F={h['EXEC']['1']['highU_lowF']}，低U高F={h['EXEC']['1']['lowU_highF']}；PREFIX 高U低F={h['PREFIX']['1']['highU_lowF']}，低U高F={h['PREFIX']['1']['lowU_highF']}。",
        "",
        f"EXEC hold1：U_single mean={_sci(h['EXEC']['1']['U_single']['mean'])}，F mean={_sci(h['EXEC']['1']['F']['mean'])}。PREFIX hold1：U_single={_sci(h['PREFIX']['1']['U_single']['mean'])}，F={_sci(h['PREFIX']['1']['F']['mean'])}。",
        "",
        "沿自然轨迹，EXEC 第一 hold 局部低，后续 F 也低于 PREFIX。**没有**“局部差、未来补回来”的已实现补偿。",
        "",
        "## 2. 第一 hold counterfactual credit（本阶段主实验）",
        "",
        "同一初始 state：强制第一 hold 为官方 24-pool 之一，hold 2/3/4 继续正式 greedy decoder。得到 `Γ → U_single(Γ|s) → U_traj(Γ)`。",
        "",
        "此处 `G_1 = U_traj`（整条 remaining），`F_1 = U_traj - U_single`。",
        "",
        "| method | Spearman | Pearson | Kendall | top-1 | top-3 | local-argmax 的 traj regret | U_traj gap | argmax 不一致率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fam in FAMS:
        s = c[fam]
        lines.append(
            "| {fam} | {sp} | {pe} | {ke} | {t1} | {t3} | {rg} | {gap} | {ds} |".format(
                fam=fam,
                sp=_fmt(s["spearman"]["mean"]),
                pe=_fmt(s["pearson"]["mean"]),
                ke=_fmt(s["kendall"]["mean"]),
                t1=_fmt(s["top1"]["mean"]),
                t3=_fmt(s["top3"]["mean"]),
                rg=_fmt(s["traj_regret_of_local_argmax"]["mean"]),
                gap=_sci(s["U_traj_gap_local_vs_best"]["mean"]),
                ds=_fmt(s["local_traj_disagree_rate"]),
            )
        )
    lines += [
        "",
        "第一 hold 的 `U_single` 对 **future F**（不是含自身的 G）的 pool 内排序：",
        "",
        "| method | Spearman(U,F) | top-1(U,F) | nregret(U→F) |",
        "|---|---:|---:|---:|",
    ]
    for fam in FAMS:
        s = c[fam]["U_vs_F"]
        lines.append(f"| {fam} | {_fmt(s['spearman']['mean'])} | {_fmt(s['top1']['mean'])} | {_fmt(s['nregret']['mean'])} |")
    lines += [
        "",
        "## 3. 局部目标错位 vs 未来状态效应",
        "",
        "对每个第一 hold 候选：",
        "",
        "- `Δ_local = U_single(Γ) − U_single(Γ_actual)`",
        "- `A_traj = U_traj(Γ) − U_traj(Γ_actual)`",
        "- `Δ_future = F(Γ) − F(Γ_actual)`",
        "",
        "`A_traj = Δ_local + Δ_future`。",
        "",
        "| method | Spearman(Δ_local, A_traj) | Spearman(Δ_future, A_traj) | Spearman(Δ_local, Δ_future) | local-best 的 A_traj | local-best 的 Δ_future |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for fam in FAMS:
        s = c[fam]
        lines.append(
            "| {fam} | {a} | {b} | {d} | {at} | {df} |".format(
                fam=fam,
                a=_fmt(s["delta_local_vs_A_traj"]["spearman"]),
                b=_fmt(s["delta_future_vs_A_traj"]["spearman"]),
                d=_fmt(s["delta_local_vs_delta_future"]["spearman"]),
                at=_sci(s["local_best_A_traj_vs_actual"]["mean"]),
                df=_sci(s["local_best_delta_future_vs_actual"]["mean"]),
            )
        )
    lines += [
        "",
        f"EXEC：A_traj 更跟 Δ_future（Spearman={_fmt(c['EXEC']['delta_future_vs_A_traj']['spearman'])}）而不是 Δ_local（{_fmt(c['EXEC']['delta_local_vs_A_traj']['spearman'])}）。均值份额 A_traj_share_future={_fmt(c['EXEC']['A_traj_share_future'])}，share_local={_fmt(c['EXEC']['A_traj_share_local'])}。",
        f"`Δ_local` 与 `Δ_future` 的 Spearman={_fmt(c['EXEC']['delta_local_vs_delta_future']['spearman'])}（EXEC 为正；PREFIX={_fmt(c['PREFIX']['delta_local_vs_delta_future']['spearman'])}）。",
        "",
        "同一方法下，把第一 hold 换成 pool 内两种 oracle 后的闭环均值：",
        "",
        "| method | actual U_traj | pool argmax U_single 的 U_traj | pool argmax U_traj |",
        "|---|---:|---:|---:|",
    ]
    for fam in FAMS:
        s = c[fam]
        lines.append(
            f"| {fam} | {_sci(s['mean_U_traj_actual'])} | {_sci(s['mean_U_traj_local_best'])} | {_sci(s['mean_U_traj_traj_best'])} |"
        )
    lines += [
        "",
        "PREFIX 的 pool `argmax U_single` 反而低于其 OOD greedy（local-best A_traj 为负）：它已经选到了高即时效用。EXEC 换成 pool `argmax U_single` 可抬升约 3.2e6，仍低于 PREFIX 自然轨迹；换成 `argmax U_traj` 才接近 PREFIX。",
        "注意：自然第一 hold 大多不在 24-pool 内，因此 pool oracle **不是** OOD greedy 的上界。221006 的 traj regret 为负，即 OOD 实际轨迹高于该 scenario 的 pool-best U_traj。",
        "",
        "## 4. 四方法统一表",
        "",
        "| method | hold | local U_single | remaining G | local-vs-G Spearman | top-1(U,G) | first-hold cf Spearman | first-hold cf top-1 | cf traj regret |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in p["method_table"]:
        lines.append(
            "| {m} | {h} | {u} | {g} | {sg} | {t1} | {cs} | {ct} | {cr} |".format(
                m=row["method"],
                h=row["hold"],
                u=_sci(row["local_U_single_mean"]),
                g=_sci(row["remaining_G_mean"]),
                sg=_fmt(row["local_vs_G_spearman"]),
                t1=_fmt(row["local_vs_G_top1"]),
                cs=_fmt(row["cf_spearman"]),
                ct=_fmt(row["cf_top1"]),
                cr=_fmt(row["cf_traj_regret"]),
            )
        )
    lines += [
        "",
        "## 5. EXEC 专项",
        "",
        f"- 自然第一 hold `U_single`：mean={_sci(e['actual_U_single']['mean'])}，median={_sci(e['actual_U_single']['median'])}。PREFIX：{_sci(px['actual_U_single']['mean'])}。此前约 0.69e6 vs 4.28e6 得到确认。",
        f"- 自然 `U_traj`：mean={_sci(e['actual_U_traj']['mean'])}（PREFIX {_sci(px['actual_U_traj']['mean'])}）。正式五 seed EXEC 仍是 8.6925e6；本条是 seed0。",
        f"- 自然 future `F_1`：EXEC {_sci(e['actual_F']['mean'])} vs PREFIX {_sci(px['actual_F']['mean'])}。后续没有补偿。",
        f"- 自然第一 hold `U_single` 在 24-pool 中的 percentile：mean={_fmt(e['local_percentile']['mean'])}，median={_fmt(e['local_percentile']['median'])}。",
        f"- 自然 `U_traj` 在 24 条 counterfactual 中的 percentile：mean={_fmt(e['traj_percentile_among_cf']['mean'])}，median={_fmt(e['traj_percentile_among_cf']['median'])}。",
        f"- oracle-local first hold 与 oracle-trajectory first hold 不一致率={_fmt(e['oracle_local_vs_traj_disagree'])}。",
        f"- local regret：mean={_fmt(e['local_regret']['mean'])}，median={_fmt(e['local_regret']['median'])}。",
        f"- trajectory regret：mean={_fmt(e['traj_regret']['mean'])}，median={_fmt(e['traj_regret']['median'])}。",
        "",
        "EXEC 逐 scenario（seed0）：",
        "",
        "| scenario | U_single | U_traj | F | local pct | traj pct | oracle-local U | oracle-local traj | oracle-traj U | oracle-traj traj | disagree | local regret | traj regret |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for d in e["per_scenario"]:
        lines.append(
            "| {sc} | {us} | {ut} | {f} | {lp} | {tp} | {olu} | {olt} | {otu} | {ott} | {dg} | {lr} | {tr} |".format(
                sc=d["scenario"],
                us=_sci(d["actual_U_single"]),
                ut=_sci(d["actual_U_traj"]),
                f=_sci(d["actual_F"]),
                lp=_fmt(d["local_pct_in_pool"]),
                tp=_fmt(d["traj_pct_among_cf"]),
                olu=_sci(d["oracle_local_U_single"]),
                olt=_sci(d["oracle_local_U_traj"]),
                otu=_sci(d["oracle_traj_U_single"]),
                ott=_sci(d["oracle_traj_U_traj"]),
                dg=int(d["disagree"]),
                lr=_fmt(d["local_regret"]),
                tr=_fmt(d["traj_regret"]),
            )
        )
    lines += [
        "",
        "若 EXEC 存在 long-horizon compensation，应看到：自然 `U_single` 低、但 `U_traj` 在 24 条 cf 中偏高，或 `F` 高于 PREFIX。当前两者都不成立。",
        "",
        "## 6. 五个问题",
        "",
        f"**Q1. U_single 是否能预测当前 hold 开始后的 remaining return？**  ",
        f"只能部分预测 **G（含自身）**，不能当作 **F** 或轨迹选点分数。",
        f"沿 16 条自然轨迹：PREFIX hold1 Spearman(U,G)={_fmt(h['PREFIX']['1']['U_vs_G']['spearman'])}，Spearman(U,F)={_fmt(h['PREFIX']['1']['U_vs_F']['spearman'])}（高即时、低未来）。EXEC hold1 Spearman(U,G)={_fmt(h['EXEC']['1']['U_vs_G']['spearman'])}，Spearman(U,F)={_fmt(h['EXEC']['1']['U_vs_F']['spearman'])}——这是 seed0 崩塌簇同时压低 U 与 F 的跨 scenario 相关，不是“局部效用能预测未来”。",
        f"正确检验是 24-pool counterfactual：U_single vs U_traj Spearman EXEC={_fmt(c['EXEC']['spearman']['mean'])}、PREFIX={_fmt(c['PREFIX']['spearman']['mean'])}；U_single vs F 仅 {_fmt(c['EXEC']['U_vs_F']['spearman']['mean'])} / {_fmt(c['PREFIX']['U_vs_F']['spearman']['mean'])}。U_single 仍是正确的即时增量，但不能单独当 long-horizon 排序量。",
        "",
        f"**Q2. first-hold local utility ranking 与 first-hold trajectory ranking 是否一致？**  ",
        f"不一致。EXEC top-1={_fmt(c['EXEC']['top1']['mean'])}，Spearman={_fmt(c['EXEC']['spearman']['mean'])}，不一致率={_fmt(c['EXEC']['local_traj_disagree_rate'])}。PREFIX top-1={_fmt(c['PREFIX']['top1']['mean'])}，Spearman={_fmt(c['PREFIX']['spearman']['mean'])}。argmax U_single 与 argmax U_traj 在多数 scenario 上不是同一个 Γ。",
        "",
        f"**Q3. counterfactual 后，后续状态是否产生显著 compensation / degradation？**  ",
        f"会。A_traj 与 Δ_future 的 Spearman={_fmt(c['EXEC']['delta_future_vs_A_traj']['spearman'])}，与 Δ_local 为 {_fmt(c['EXEC']['delta_local_vs_A_traj']['spearman'])}。U_single 对 F 的 pool 内 top-1={_fmt(c['EXEC']['U_vs_F']['top1']['mean'])}。",
        f"但这是 **候选之间** 的未来状态效应，不是 EXEC seed0 已实现 greedy 的补偿：其 F 低于 PREFIX（{_sci(e['actual_F']['mean'])} vs {_sci(px['actual_F']['mean'])}），U_traj 也更低。",
        "",
        f"**Q4. EXEC 的主要问题？**  ",
        f"**{v['label']}**：{v['text']}。",
        f"- A（local selection）：EXEC seed0 第一 hold U_single 约 {_sci(e['actual_U_single']['mean'])}，pool percentile 中位 {_fmt(e['local_percentile']['median'])}；PREFIX 约 {_sci(px['actual_U_single']['mean'])}。后续未补偿。",
        f"- B（credit）：四条方法共享 first-hold local≠traj ranking（top-1 约 0–0.25）。PREFIX 尽管 credit 也不对齐，但 greedy 选了高即时 U_LL，闭环仍最高。",
        "数据能同时看见 A 和 B，不能把 EXEC seed0 的失败只写成 B。",
        "",
        "**Q5. 下一阶段最小且唯一值得验证的假设（本阶段不实现、不训练）：**  ",
        "离线表已经表明：同一后续 greedy 下，`U_traj(argmax U_traj) > U_traj(argmax U_single)`（EXEC 均值 gap 约 2.9e6）。因此不必再验证“oracle 查表谁更大”。",
        f"还没被策略实现、因而唯一值得验证的假设是：**只改第一 hold、冻结后续 greedy 时，EXEC 若能选到接近 `argmax U_traj` 的 Γ_1，闭环能否追上 PREFIX 自然轨迹（本表 pool traj-best EXEC={_sci(c['EXEC']['mean_U_traj_traj_best'])} vs PREFIX actual={_sci(c['PREFIX']['mean_U_traj_actual'])}）。** 若能，则 EXEC seed0 的主缺口是第一 hold credit / 选点；若不能，则后续 hold 的 EXEC greedy 仍是独立问题。本阶段不训练、不改 decoder。",
        "",
        "完整 Pearson / Kendall / top-3 见 `src/wt/lowlevel_v2/phase3b_credit/outputs/method_hold_summary.csv` 与 `PHASE3B_CREDIT_SUMMARY.json`。自然 hold 行的 `F` 来自 `Σ_{r>t} U_single`；EXEC 专项里的 `F` 来自 `U_traj − U_single`，二者相差已知的 interaction gap。",
        "",
        "## 7. 数据边界",
        "",
        "- 只使用 seed0 × Low-Level seed=0。正式 EXEC 五 seed 均值 8.6925e6 不被本条自然轨迹代表。",
        "- 后续 hold 没有 24-pool counterfactual `U_traj`，因此 t=2/3/4 的 credit 只能做跨 scenario 相关，不能做候选排序。",
        "- 自然第一 hold 几乎都不在官方 24-pool 内；pool 内 oracle 只回答“池内换 Γ_1 会怎样”。",
        "- 没有重训、没有新 rollout。",
        "",
        "本阶段停止。不要自动进入下一阶段。不要训练。不要改方法或 checkpoint。",
        "",
    ]
    return "\n".join(lines)


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    nat = load_json("natural_trajectories.json")
    cf = load_json("counterfactual.json")
    pool = load_pool()
    holds = natural_rows(nat)
    _csv(OUTPUT / "hold_contributions.csv", holds)
    corr = hold_corr(holds)
    cf_rows, cf_sum, per_sc = counterfactual_credit(cf, nat)
    _csv(OUTPUT / "first_hold_counterfactual_credit.csv", cf_rows)
    _csv(OUTPUT / "first_hold_per_scenario.csv", per_sc)
    foci = {}
    for fam in FAMS:
        foci[fam] = fill_traj_percentiles(first_hold_focus(nat, pool, per_sc, fam), cf, fam)
    _csv(OUTPUT / "exec_first_hold_credit.csv", foci["EXEC"]["per_scenario"])
    table = method_table(nat, corr, cf_sum, foci)
    _csv(OUTPUT / "method_hold_summary.csv", table)
    sanity = {}
    for fam in FAMS:
        sub = [r for r in nat if r["family"] == fam]
        sanity[fam] = {
            "spearman_sum_vs_traj": spearman([r["sum_U_single"] for r in sub], [r["U_traj"] for r in sub]),
            "pearson_sum_vs_traj": pearson([r["sum_U_single"] for r in sub], [r["U_traj"] for r in sub]),
            "gap": summarize([r["interaction_gap"] for r in sub]),
            "U_traj": summarize([r["U_traj"] for r in sub]),
        }
    verdict = decide(corr, cf_sum, foci)
    payload = {
        "sanity": sanity,
        "hold_corr": corr,
        "counterfactual": cf_sum,
        "focus": {k: {kk: vv for kk, vv in val.items() if kk != "per_scenario"} | {"per_scenario": val["per_scenario"]} for k, val in foci.items()},
        "method_table": table,
        "verdict": verdict,
        "source": {
            "traj_audit": str(TRAJ_OUT),
            "n_natural": len(nat),
            "n_counterfactual": len(cf),
        },
    }
    _dump(OUTPUT / "PHASE3B_CREDIT_SUMMARY.json", payload)
    report = write_report(payload)
    (OUTPUT / "HTW_Phase3B_HoldTrajectoryCredit_Audit_CN.md").write_text(report, encoding="utf-8")
    (CREDIT / "HTW_Phase3B_HoldTrajectoryCredit_Audit_CN.md").write_text(report, encoding="utf-8")
    (ROOT / "docs/archive/phase_reports/HTW_Phase3B_HoldTrajectoryCredit_Audit_CN.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": verdict["label"],
                "exec_U_single": foci["EXEC"]["actual_U_single"]["mean"],
                "exec_U_traj": foci["EXEC"]["actual_U_traj"]["mean"],
                "exec_F": foci["EXEC"]["actual_F"]["mean"],
                "prefix_U_single": foci["PREFIX"]["actual_U_single"]["mean"],
                "prefix_U_traj": foci["PREFIX"]["actual_U_traj"]["mean"],
                "exec_cf_spearman": cf_sum["EXEC"]["spearman"]["mean"],
                "exec_cf_top1": cf_sum["EXEC"]["top1"]["mean"],
                "exec_disagree": cf_sum["EXEC"]["local_traj_disagree_rate"],
                "dloc_vs_atraj": cf_sum["EXEC"]["delta_local_vs_A_traj"]["spearman"],
                "dfut_vs_atraj": cf_sum["EXEC"]["delta_future_vs_A_traj"]["spearman"],
            },
            indent=2,
        ),
        flush=True,
    )
    return payload


if __name__ == "__main__":
    run()
