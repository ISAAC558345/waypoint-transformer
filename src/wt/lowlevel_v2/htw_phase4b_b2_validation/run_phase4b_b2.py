"""Phase-4B B2 leave-scenario-out verification. No retune. No planner change."""
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

from wt.lowlevel_v2.htw_phase4b_b2_validation.protocol import (
    ET,
    FIG,
    N_PERM,
    N_SPLITS,
    OUTPUT,
    PERM_SEED,
    PRIMARY,
    REPORT_NAME,
    ROOT,
)
from wt.lowlevel_v2.htw_phase4a_hindsight_credit.protocol import SOURCE as P4A_SOURCE
from wt.lowlevel_v2.htw_phase4b_credit_learner.protocol import LOCKED_FV_U, LOCKED_TRAJ_U, P4A_CSV
from wt.lowlevel_v2.htw_phase4b_credit_learner.run_phase4b import (
    CAND_KEYS,
    STATE_KEYS,
    attach_observables,
    ext_sum,
    load_phase4a,
)
from wt.lowlevel_v2.integration_phase1.stats import spearman
from wt.lowlevel_v2.phase1.protocol import load_pools

# User-specified B2: ExtraTrees(U_single + Phase-3E candidate geometry).
P3E_CAND = tuple(k for k in CAND_KEYS if k not in {"leftover_queue", "leftover_access", "in_range_frac"})
B2_KEYS = ("U_single",) + P3E_CAND
# Phase-4B table B2 (geometry only, leftover included) — locked R=0.891, not retuned.
B2_P4B_KEYS = CAND_KEYS
B3_KEYS = ("U_single",) + CAND_KEYS + STATE_KEYS


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


def _fmt(x: float) -> str:
    if not np.isfinite(float(x)):
        return "nan"
    return f"{float(x):.3f}"


def _sci(x: float) -> str:
    return f"{float(x):.3e}"


def probe_independence(rows: list[dict]) -> dict:
    train, design = load_pools()
    have = sorted({int(r["scenario_id"]) for r in rows})
    cf_path = Path(str(P4A_SOURCE.get("counterfactual", "")))
    extra = [s for s in have if s not in set(design)]
    return {
        "independent_available": False,
        "reason": "frozen first-hold U_traj/C exists only on LLV2_design (221000-221015); all 16 states already entered Phase-4B GroupKFold. LLV2_train has no Phase-3 counterfactual labels. Other families are not independent EXEC scenarios (hold 2/3/4 greedy differs). Validation seeds are refused and unlabeled.",
        "design": design,
        "labeled_scenarios": have,
        "labeled_minus_design": extra,
        "n_train_registry": len(train),
        "phase4a_csv": str(P4A_CSV),
        "counterfactual_exists": cf_path.is_file() if cf_path.name else False,
        "do_not_call_this_independent_validation": True,
    }


def matrix(rows: list[dict], keys: tuple[str, ...]) -> np.ndarray:
    return np.array([[float(r[k]) for k in keys] for r in rows], dtype=np.float64)


def grouped(rows: list[dict]) -> dict[int, list[int]]:
    by = defaultdict(list)
    for i, row in enumerate(rows):
        by[int(row["scenario_id"])].append(i)
    return dict(sorted(by.items()))


def state_select(u_traj, score) -> dict:
    ut = np.asarray(u_traj, dtype=np.float64)
    sc = np.asarray(score, dtype=np.float64)
    ih = int(np.argmax(sc))
    it = int(np.argmax(ut))
    best = float(np.max(ut))
    chosen = float(ut[ih])
    return {
        "idx": ih,
        "oracle_idx": it,
        "U_traj": chosen,
        "R_pred": chosen / best if abs(best) > 0 else float("nan"),
        "top1": float(ih == it),
        "top3": float(it in set(np.argsort(-sc)[:3].tolist())),
        "best": best,
    }


def fit_oof(rows: list[dict], keys: tuple[str, ...]):
    x = matrix(rows, keys)
    y = np.array([r["C_abs"] for r in rows], dtype=np.float64)
    groups = np.array([int(r["scenario_id"]) for r in rows], dtype=np.int64)
    oof = np.full(len(rows), np.nan)
    folds = []
    models = []
    gkf = GroupKFold(n_splits=min(N_SPLITS, len(set(groups.tolist()))))
    for fold, (tr, te) in enumerate(gkf.split(x, groups=groups)):
        model = ExtraTreesRegressor(**ET)
        model.fit(x[tr], y[tr])
        pred = model.predict(x[te])
        oof[te] = pred
        models.append((model, te, sorted({int(groups[i]) for i in te}), sorted({int(groups[i]) for i in tr})))
        folds.append(
            {
                "fold": fold,
                "train_scenarios": sorted({int(groups[i]) for i in tr}),
                "test_scenarios": sorted({int(groups[i]) for i in te}),
                "n_train": int(tr.size),
                "n_test": int(te.size),
            }
        )
    return oof, folds, models, x, y, groups


def fold_metrics(rows: list[dict], oof: np.ndarray, folds: list[dict], models) -> list[dict]:
    groups = grouped(rows)
    out = []
    for fold, (model, te, test_sc, train_sc) in enumerate(models):
        te = np.asarray(te)
        c = np.array([rows[i]["C_abs"] for i in te], dtype=np.float64)
        hat = oof[te]
        u = np.array([rows[i]["U_single"] for i in te], dtype=np.float64)
        t = np.array([rows[i]["U_traj"] for i in te], dtype=np.float64)
        within = []
        r_b2, r_loc, top1, top3, u_b2 = [], [], [], [], []
        for sc in test_sc:
            idxs = groups[sc]
            loc = state_select([rows[i]["U_traj"] for i in idxs], [rows[i]["U_single"] for i in idxs])
            b2 = state_select([rows[i]["U_traj"] for i in idxs], [rows[i]["U_single"] + oof[i] for i in idxs])
            within.append(spearman([oof[i] for i in idxs], [rows[i]["C_abs"] for i in idxs]))
            r_b2.append(b2["R_pred"])
            r_loc.append(loc["R_pred"])
            top1.append(b2["top1"])
            top3.append(b2["top3"])
            u_b2.append(b2["U_traj"])
        out.append(
            {
                "fold": fold,
                "train_scenarios": train_sc,
                "test_scenarios": test_sc,
                "spearman_C": spearman(hat, c),
                "within_spearman_C_mean": float(np.nanmean(within)),
                "R_pred_B2": float(np.nanmean(r_b2)),
                "R_pred_LOCAL": float(np.nanmean(r_loc)),
                "top1_B2": float(np.mean(top1)),
                "top3_B2": float(np.mean(top3)),
                "mean_U_traj_B2": float(np.mean(u_b2)),
                "median_U_traj_B2": float(np.median(u_b2)),
            }
        )
    return out


def per_scenario(rows: list[dict], hats: dict[str, np.ndarray]) -> list[dict]:
    out = []
    for sc, idxs in grouped(rows).items():
        ut = [rows[i]["U_traj"] for i in idxs]
        us = [rows[i]["U_single"] for i in idxs]
        loc = state_select(ut, us)
        rec = {
            "scenario_id": sc,
            "LOCAL_R": loc["R_pred"],
            "LOCAL_U": loc["U_traj"],
            "LOCAL_cand": int(rows[idxs[loc["idx"]]]["candidate_id"]),
            "ORACLE_cand": int(rows[idxs[loc["oracle_idx"]]]["candidate_id"]),
            "ORACLE_U": loc["best"],
        }
        for name, hat in hats.items():
            sel = state_select(ut, [rows[i]["U_single"] + hat[i] for i in idxs])
            rec[f"{name}_R"] = sel["R_pred"]
            rec[f"{name}_U"] = sel["U_traj"]
            rec[f"{name}_cand"] = int(rows[idxs[sel["idx"]]]["candidate_id"])
            rec[f"{name}_top1"] = sel["top1"]
            rec[f"{name}_top3"] = sel["top3"]
            rec[f"{name}_dR"] = float(sel["R_pred"] - loc["R_pred"])
            rec[f"{name}_changed"] = int(sel["idx"] != loc["idx"])
            rec[f"{name}_oracle_overlap"] = int(sel["idx"] == loc["oracle_idx"])
            rec[f"{name}_regret_vs_local"] = float(sel["U_traj"] - loc["U_traj"])
            rec[f"{name}_gap_vs_oracle"] = float(loc["best"] - sel["U_traj"])
            rec[f"{name}_spearman_C"] = spearman([hat[i] for i in idxs], [rows[i]["C_abs"] for i in idxs])
        out.append(rec)
    return out


def ranking_rows(rows: list[dict], hat: np.ndarray) -> list[dict]:
    out = []
    for sc, idxs in grouped(rows).items():
        ut = np.array([rows[i]["U_traj"] for i in idxs], dtype=np.float64)
        us = np.array([rows[i]["U_single"] for i in idxs], dtype=np.float64)
        uh = us + hat[idxs]
        il, ib, it = int(np.argmax(us)), int(np.argmax(uh)), int(np.argmax(ut))
        out.append(
            {
                "scenario_id": sc,
                "cand_LOCAL": int(rows[idxs[il]]["candidate_id"]),
                "cand_B2": int(rows[idxs[ib]]["candidate_id"]),
                "cand_ORACLE": int(rows[idxs[it]]["candidate_id"]),
                "hash_LOCAL": rows[idxs[il]]["assignment_hash"],
                "hash_B2": rows[idxs[ib]]["assignment_hash"],
                "hash_ORACLE": rows[idxs[it]]["assignment_hash"],
                "B2_changed_first_hold": int(il != ib),
                "B2_oracle_overlap": int(ib == it),
                "LOCAL_oracle_overlap": int(il == it),
                "U_LOCAL": float(ut[il]),
                "U_B2": float(ut[ib]),
                "U_ORACLE": float(ut[it]),
                "regret_vs_LOCAL": float(ut[ib] - ut[il]),
                "gap_vs_ORACLE": float(ut[it] - ut[ib]),
            }
        )
    return out


def permute_geometry(rows: list[dict], keys: tuple[str, ...], models, x: np.ndarray) -> dict:
    rng = np.random.default_rng(PERM_SEED)
    geom = [k for k in keys if k != "U_single"]
    cols = [keys.index(k) for k in geom]
    r_real = []
    r_perm = []
    for _ in range(N_PERM):
        fold_r = []
        fold_p = []
        for model, te, test_sc, _train in models:
            x_te = x[te].copy()
            x_perm = x_te.copy()
            local = {int(rows[i]["scenario_id"]): [] for i in te}
            for j, i in enumerate(te):
                local[int(rows[i]["scenario_id"])].append(j)
            for idxs in local.values():
                order = rng.permutation(len(idxs))
                sl = np.asarray(idxs, dtype=np.int64)
                block = x_perm[sl][:, cols]
                tmp = x_perm[sl]
                tmp[:, cols] = block[order]
                x_perm[sl] = tmp
            hat = model.predict(x_te)
            hat_p = model.predict(x_perm)
            for sc, idxs in local.items():
                glob = [int(te[j]) for j in idxs]
                ut = [rows[i]["U_traj"] for i in glob]
                us = [rows[i]["U_single"] for i in glob]
                fold_r.append(state_select(ut, [us[k] + hat[idxs[k]] for k in range(len(idxs))])["R_pred"])
                fold_p.append(state_select(ut, [us[k] + hat_p[idxs[k]] for k in range(len(idxs))])["R_pred"])
        r_real.append(float(np.nanmean(fold_r)))
        r_perm.append(float(np.nanmean(fold_p)))
    return {"n_perm": N_PERM, "R_pred_real_repeat": ext_sum(r_real), "R_pred_perm": ext_sum(r_perm)}


def decide(indep: dict, per_sc: list[dict], fold_b2: list[dict], perm: dict) -> dict:
    dR = np.array([r["B2_dR"] for r in per_sc], dtype=np.float64)
    n_pos = int(np.sum(dR > 1e-12))
    n_neg = int(np.sum(dR < -1e-12))
    n_zero = int(len(dR) - n_pos - n_neg)
    mean_r = float(np.mean([r["B2_R"] for r in per_sc]))
    loc_r = float(np.mean([r["LOCAL_R"] for r in per_sc]))
    p4b_r = float(np.mean([r["B2P4B_R"] for r in per_sc]))
    b3_r = float(np.mean([r["B3_R"] for r in per_sc]))
    worst = min(per_sc, key=lambda r: r["B2_dR"])
    best = max(per_sc, key=lambda r: r["B2_dR"])
    fold_std = float(np.std([f["R_pred_B2"] for f in fold_b2], ddof=1)) if len(fold_b2) > 1 else float("nan")
    dominated = abs(best["B2_dR"]) > 3.0 * max(float(np.median(np.abs(dR))), 1e-6) and n_pos <= 4
    perm_drop = perm["R_pred_real_repeat"]["mean"] - perm["R_pred_perm"]["mean"]
    stable = n_pos >= n_neg and mean_r > loc_r + 0.02 and fold_std < 0.15
    b2_beats_b3 = p4b_r > b3_r + 0.03 or mean_r > b3_r + 0.03
    if indep["independent_available"]:
        gate = "PASS" if stable else "FAIL"
        text = "存在独立 scenario，并按该集合判定。"
    else:
        if stable and (not dominated) and perm_drop > 0.03:
            gate = "PARTIAL"
            text = "无法真正独立验证。leave-scenario-out 上 B2 相对 LOCAL 有可重复增益，但不能当作 held-out deployment 证据。"
        elif mean_r > loc_r:
            gate = "PARTIAL"
            text = "无法真正独立验证。总体均值有增益，但跨 scenario 不够稳，或可能由少数 state 拉动。"
        else:
            gate = "FAIL"
            text = "无法真正独立验证，且 LSO 未显示相对 LOCAL 的稳定改善。"
    return {
        "gate": gate,
        "text": text,
        "independent_available": False,
        "n_improve": n_pos,
        "n_worse": n_neg,
        "n_tie": n_zero,
        "mean_R_B2": mean_r,
        "mean_R_LOCAL": loc_r,
        "mean_R_B2P4B": p4b_r,
        "mean_R_B3": b3_r,
        "mean_dR": float(np.mean(dR)),
        "std_dR": float(np.std(dR, ddof=1)),
        "fold_R_std": fold_std,
        "dominated_by_few": bool(dominated),
        "perm_drop": float(perm_drop),
        "b2_beats_b3": bool(b2_beats_b3),
        "planner_ready": False,
        "worst_scenario": worst["scenario_id"],
        "worst_dR": worst["B2_dR"],
        "best_scenario": best["scenario_id"],
        "best_dR": best["B2_dR"],
    }


def figures(per_sc: list[dict], fold_b2: list[dict], perm: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    xs = [str(r["scenario_id"])[-3:] for r in per_sc]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.bar(np.arange(len(xs)) - 0.18, [r["LOCAL_R"] for r in per_sc], 0.36, label="LOCAL", color="#9e9e9e")
    ax.bar(np.arange(len(xs)) + 0.18, [r["B2_R"] for r in per_sc], 0.36, label="B2 spec", color="#4C78A8")
    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels(xs, rotation=60)
    ax.set_ylabel("R_pred")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "per_scenario_rpred.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    colors = ["#54A24B" if r["B2_dR"] >= 0 else "#E45756" for r in per_sc]
    ax.bar(xs, [r["B2_dR"] for r in per_sc], color=colors)
    ax.axhline(0.0, color="#333")
    ax.set_ylabel("dR = R_B2 - R_LOCAL")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(FIG / "per_scenario_delta_r.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.bar(["fold R_B2"], [0])
    ax.clear()
    ax.bar([f"f{f['fold']}" for f in fold_b2], [f["R_pred_B2"] for f in fold_b2], color="#4C78A8")
    ax.plot([f["R_pred_LOCAL"] for f in fold_b2], "o--", color="#9e9e9e", label="LOCAL")
    ax.set_ylabel("fold mean R_pred")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fold_rpred.png", dpi=140)
    plt.close(fig)


def write_report(indep: dict, fold_b2: list[dict], per_sc: list[dict], rank: list[dict], perm: dict, gate: dict, folds: list[dict]) -> str:
    r_b2 = ext_sum([r["B2_R"] for r in per_sc])
    r_loc = ext_sum([r["LOCAL_R"] for r in per_sc])
    r_p4b = ext_sum([r["B2P4B_R"] for r in per_sc])
    r_b3 = ext_sum([r["B3_R"] for r in per_sc])
    lines = [
        "# HTW Phase-4B B2 Leave-Scenario-Out Verification",
        "",
        "本阶段只验证固定 B2 residual，不是方法开发。未训练 Low-Level，未改 planner / reward / decoder / pool / checkpoint，未重跑 Phase-3 rollout，未把结果写入正式 checkpoint。",
        "",
        f"**门控：{gate['gate']}** — {gate['text']}",
        "",
        "## A. 是否存在真正独立验证数据",
        "",
        "**不存在。无法进行真正独立验证。** 不伪造 independent validation。",
        "",
        f"- 已标注 first-hold `U_traj`/`C` 的 scenario 只有 `LLV2_design`：{indep['labeled_scenarios']}",
        "- 这 16 个 state 已全部进入 Phase-4B GroupKFold，没有留下未拟合过的 EXEC 标签",
        f"- registry 中另有 {indep['n_train_registry']} 个 `LLV2_train` seed，但没有 Phase-3 counterfactual，因此没有 `C_abs`",
        "- PREFIX/OLD/ABS 不是独立 EXEC scenario：后续 hold greedy 不同，`C` 不可直接当 EXEC 测试",
        "- Phase-3 validation seed 被 `refuse_compat` 拒绝，且无本协议标签",
        "",
        "因此本报告只做 **leave-scenario-out stability audit**。",
        "",
        "## 固定方法",
        "",
        "本审计称 **B2** = 用户指定：`C_hat = ExtraTrees(U_single + Phase-3E candidate geometry)`，`U_hat = U_single + C_hat`。",
        "这对应 Phase-4B 的 ablation **C_U_cand**（U+cand），不是 Phase-4B 表里的 geometry-only `B2_cand_geom`。",
        "",
        "为回答“0.891 是否少数 scenario 驱动”，另报固定的 Phase-4B `B2_cand_geom`（只含 candidate geometry / leftover，不含 U_single）的 LSO 结果。不调参。",
        "",
        f"ExtraTrees 固定：{ET}。`GroupKFold({N_SPLITS})`，group=`scenario_id`。",
        "",
        "## C. Fold 稳定性（指定 B2 = U+cand）",
        "",
        "| fold | test scenarios | Spearman(C) | within C | R_B2 | R_LOCAL | top-1 | top-3 | mean U_B2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for f in fold_b2:
        lines.append(
            f"| {f['fold']} | {f['test_scenarios']} | {_fmt(f['spearman_C'])} | {_fmt(f['within_spearman_C_mean'])} | {_fmt(f['R_pred_B2'])} | {_fmt(f['R_pred_LOCAL'])} | {_fmt(f['top1_B2'])} | {_fmt(f['top3_B2'])} | {_sci(f['mean_U_traj_B2'])} |"
        )
    lines.extend(
        [
            "",
            f"- fold R_B2 mean±std = {_fmt(float(np.mean([f['R_pred_B2'] for f in fold_b2])))} ± {_fmt(gate['fold_R_std'])}",
            f"- fold R_LOCAL mean = {_fmt(float(np.mean([f['R_pred_LOCAL'] for f in fold_b2])))}",
            "",
            "## F. 每 scenario 的 recovery（必须保留恶化）",
            "",
            "| scenario | R_LOCAL | R_B2 | dR | R_B2P4B(0.891源) | R_B3 | B2 changed | vs ORACLE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in per_sc:
        lines.append(
            f"| {r['scenario_id']} | {_fmt(r['LOCAL_R'])} | {_fmt(r['B2_R'])} | {_fmt(r['B2_dR'])} | {_fmt(r['B2P4B_R'])} | {_fmt(r['B3_R'])} | {r['B2_changed']} | {_fmt(1.0 - r['B2_R'])} |"
        )
    lines.extend(
        [
            "",
            f"指定 B2：mean R={_fmt(r_b2['mean'])} median={_fmt(r_b2['median'])}；LOCAL mean={_fmt(r_loc['mean'])}。改善 {gate['n_improve']} / 恶化 {gate['n_worse']} / 平 {gate['n_tie']}。",
            f"Phase-4B geometry-only B2P4B：mean R={_fmt(r_p4b['mean'])}（对应此前 0.891）。B3 full：mean R={_fmt(r_b3['mean'])}。",
            f"最大增益 scenario={gate['best_scenario']} dR={_fmt(gate['best_dR'])}；最大恶化 scenario={gate['worst_scenario']} dR={_fmt(gate['worst_dR'])}。dominated_by_few={gate['dominated_by_few']}。",
            "",
            "## D. Permutation（打乱 geometry 与 candidate 对应，U_single 不动，不重训）",
            "",
            f"- 真实折上重复均值 R={_fmt(perm['R_pred_real_repeat']['mean'])}",
            f"- 打乱 geometry 后 R={_fmt(perm['R_pred_perm']['mean'])} ± {_fmt(perm['R_pred_perm']['std'])}",
            f"- drop={_fmt(gate['perm_drop'])}",
            "",
            "## E. Ranking consistency",
            "",
            f"- B2 改变 first-hold：{sum(r['B2_changed_first_hold'] for r in rank)}/16",
            f"- B2 与 TRAJ-ORACLE overlap：{sum(r['B2_oracle_overlap'] for r in rank)}/16",
            f"- LOCAL 与 ORACLE overlap：{sum(r['LOCAL_oracle_overlap'] for r in rank)}/16",
            f"- 相对 LOCAL 的平均 payload 差：{_sci(float(np.mean([r['regret_vs_LOCAL'] for r in rank])))}",
            f"- 相对 ORACLE 的剩余 gap：{_sci(float(np.mean([r['gap_vs_ORACLE'] for r in rank])))}",
            "",
            "## 对照",
            "",
            f"| method | mean R_pred | mean U |",
            f"|---|---:|---:|",
            f"| LOCAL | {_fmt(r_loc['mean'])} | {_sci(float(np.mean([r['LOCAL_U'] for r in per_sc])))} |",
            f"| Phase-3E locked FV | (cite) | {_sci(LOCKED_FV_U)} |",
            f"| B2 specified (U+cand) | {_fmt(r_b2['mean'])} | {_sci(float(np.mean([r['B2_U'] for r in per_sc])))} |",
            f"| B2 Phase-4B geom-only | {_fmt(r_p4b['mean'])} | {_sci(float(np.mean([r['B2P4B_U'] for r in per_sc])))} |",
            f"| B3 +state | {_fmt(r_b3['mean'])} | {_sci(float(np.mean([r['B3_U'] for r in per_sc])))} |",
            f"| TRAJ-ORACLE | 1.000 | {_sci(LOCKED_TRAJ_U)} |",
            "",
            "## 五个问题",
            "",
            f"1. 跨 scenario 稳定性？**有限。** fold std={_fmt(gate['fold_R_std'])}，{gate['n_improve']} 个改善、{gate['n_worse']} 个恶化。这是 LSO，不是独立验证。",
            f"2. 0.891 是否少数 scenario 驱动？Phase-4B geom-only 的 LSO 均值={_fmt(r_p4b['mean'])}。见每行 dR，不能只看 0.891。dominated_by_few={gate['dominated_by_few']}。",
            f"3. 相对 LOCAL 是否稳定改善？指定 B2 mean dR={_fmt(gate['mean_dR'])}，恶化数={gate['n_worse']}。",
            f"4. 是否明显优于 B3？**{'是' if gate['b2_beats_b3'] else '否'}**。B3 mean R={_fmt(r_b3['mean'])}。",
            "5. 是否足够进入正式 planner integration？**否。** 没有独立标签集，未改 planner，也不能把 LSO 当部署证据。",
            "",
            "## PASS / PARTIAL / FAIL",
            "",
            f"**{gate['gate']}**。{gate['text']}",
            "",
            "本阶段停止。不修改 B2，不修改 planner，不进入 Phase-4C。",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    raw = [r for r in attach_observables(load_phase4a()) if r["family"] == PRIMARY]
    indep = probe_independence(raw)
    oof_b2, folds, models, x, y, groups = fit_oof(raw, B2_KEYS)
    oof_p4b, _, _, _, _, _ = fit_oof(raw, B2_P4B_KEYS)
    oof_b3, _, _, _, _, _ = fit_oof(raw, B3_KEYS)
    fold_b2 = fold_metrics(raw, oof_b2, folds, models)
    hats = {"B2": oof_b2, "B2P4B": oof_p4b, "B3": oof_b3}
    per_sc = per_scenario(raw, hats)
    rank = ranking_rows(raw, oof_b2)
    perm = permute_geometry(raw, B2_KEYS, models, x)
    gate = decide(indep, per_sc, fold_b2, perm)
    figures(per_sc, fold_b2, perm)
    _csv_write(OUTPUT / "phase4b_b2_fold_metrics.csv", fold_b2)
    _csv_write(OUTPUT / "phase4b_b2_per_scenario.csv", per_sc)
    _csv_write(OUTPUT / "phase4b_b2_ranking.csv", rank)
    _dump(
        OUTPUT / "phase4b_b2_summary.json",
        {
            "independence": indep,
            "gate": gate,
            "folds": folds,
            "fold_metrics": fold_b2,
            "perm": perm,
            "B2_keys": list(B2_KEYS),
            "B2P4B_keys": list(B2_P4B_KEYS),
            "B3_keys": list(B3_KEYS),
            "et": ET,
        },
    )
    body = write_report(indep, fold_b2, per_sc, rank, perm, gate, folds)
    (OUTPUT / REPORT_NAME).write_text(body + "\n", encoding="utf-8")
    (ROOT / REPORT_NAME).write_text(body + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "gate": gate["gate"],
                "independent": False,
                "R_B2": gate["mean_R_B2"],
                "R_LOCAL": gate["mean_R_LOCAL"],
                "R_B2P4B": gate["mean_R_B2P4B"],
                "n_worse": gate["n_worse"],
            },
            indent=2,
        ),
        flush=True,
    )
    return gate


if __name__ == "__main__":
    run()
