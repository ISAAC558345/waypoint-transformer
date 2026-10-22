"""Phase-4C: frozen B2 on truly unseen scenarios. No retune. No planner change."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from wt.lowlevel_v2.htw_phase4b_b2_shadow.shadow_scorer import B2ShadowScorer, assert_no_future_keys, feature_matrix
from wt.lowlevel_v2.htw_phase4b_credit_learner.run_phase4b import attach_observables, ext_sum, load_phase4a
from wt.lowlevel_v2.integration_phase1.protocol import refuse_compat
from wt.lowlevel_v2.integration_phase1.stats import spearman
from wt.lowlevel_v2.phase1.protocol import load_pools
from wt.lowlevel_v2.phase4c_independent_validation.labels import generate_exec_labels, unseen_train_scenarios
from wt.lowlevel_v2.phase4c_independent_validation.protocol import (
    ALIGN_ATOL,
    B2_FEATURE_KEYS,
    ET,
    FIG,
    N_PERM,
    OUTPUT,
    P3E_CAND_KEYS,
    P4B_DESIGN,
    P4B_FOLDS,
    PERM_DROP_EXISTING,
    PERM_SEED,
    PRIMARY,
    REPORT_NAME,
    ROOT,
)


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


def grouped(rows: list[dict]) -> dict[int, list[int]]:
    by = defaultdict(list)
    for i, row in enumerate(rows):
        by[int(row["scenario_id"])].append(i)
    return dict(sorted(by.items()))


def p4b_membership(scenario: int) -> dict:
    sc = int(scenario)
    train_folds = [f["fold"] for f in P4B_FOLDS if sc in f["train"]]
    val_folds = [f["fold"] for f in P4B_FOLDS if sc in f["validation"]]
    test_folds = [f["fold"] for f in P4B_FOLDS if sc in f["test"]]
    return {
        "scenario_id": sc,
        "in_phase4b_design": sc in P4B_DESIGN,
        "in_any_train_fold": bool(train_folds),
        "in_any_validation_fold": bool(val_folds),
        "in_any_test_fold": bool(test_folds),
        "train_folds": train_folds,
        "validation_folds": val_folds,
        "test_folds": test_folds,
        "used_for_model_selection": False,
        "used_for_feature_selection": False,
        "used_for_threshold_selection": False,
        "used_for_hyperparameter_selection": False,
    }


def audit_independence() -> dict:
    train, design = load_pools()
    labeled_design = list(P4B_DESIGN)
    unseen = unseen_train_scenarios()
    refused = []
    for sc in list(range(223000, 223064)) + list(range(222000, 222064)) + list(range(106000, 106064)):
        try:
            refuse_compat(int(sc))
        except RuntimeError as exc:
            refused.append({"scenario_id": int(sc), "reason": str(exc)})
    membership = [p4b_membership(sc) for sc in unseen]
    leak = any(
        m["in_any_train_fold"] or m["in_any_validation_fold"] or m["in_any_test_fold"] or m["in_phase4b_design"]
        for m in membership
    )
    return {
        "independent_scenario_available": bool(unseen) and not leak,
        "independent_kind": "unseen_scenario" if unseen and not leak else "none",
        "do_not_call_lso_independent": True,
        "design": list(design),
        "n_train_registry": len(train),
        "phase4b_labeled": labeled_design,
        "unseen_scenarios": unseen,
        "n_unseen": len(unseen),
        "refused_pools": {
            "phase3_val": list(range(223000, 223064)),
            "phase2_val": list(range(222000, 222064)),
            "phase106_untouched": list(range(106000, 106064)),
        },
        "llv2_train_used_for_low_level_training": True,
        "llv2_train_used_for_b2_fit": False,
        "membership": membership,
        "any_leakage_into_phase4b_folds": bool(leak),
        "reason": (
            "LLV2_train 220000-220063 never entered Phase-4B GroupKFold, feature selection, "
            "model selection, threshold selection, or ExtraTrees hyperparameter selection. "
            "They were Low-Level MAPPO training environments. B2 ExtraTrees has not seen them. "
            "Phase-3/2 validation and Phase-10.6 untouched seeds are refused and are not used."
        ),
    }


def load_design_exec() -> list[dict]:
    rows = [r for r in attach_observables(load_phase4a()) if r["family"] == PRIMARY]
    for row in rows:
        if abs(float(row["C_abs"]) - (float(row["U_traj"]) - float(row["U_single"]))) > ALIGN_ATOL:
            raise RuntimeError("design C_abs identity failed")
    return rows


def fit_frozen_b2(design_rows: list[dict]) -> B2ShadowScorer:
    assert_no_future_keys(design_rows, B2_FEATURE_KEYS)
    x = feature_matrix(design_rows, B2_FEATURE_KEYS)
    y = np.array([r["C_abs"] for r in design_rows], dtype=np.float64)
    return B2ShadowScorer().fit(x, y)


def state_pick(rows: list[dict], idxs: list[int], score) -> dict:
    sc = np.asarray(score, dtype=np.float64)
    ut = np.array([rows[i]["U_traj"] for i in idxs], dtype=np.float64)
    ih = int(np.argmax(sc))
    it = int(np.argmax(ut))
    best = float(np.max(ut))
    gi = idxs[ih]
    return {
        "idx": ih,
        "gidx": gi,
        "oracle_idx": it,
        "candidate_id": int(rows[gi]["candidate_id"]),
        "assignment_hash": rows[gi].get("assignment_hash"),
        "U_single": float(rows[gi]["U_single"]),
        "U_traj": float(ut[ih]),
        "R": float(ut[ih] / best) if abs(best) > 0 else float("nan"),
        "top1": float(ih == it),
        "top3": float(it in set(np.argsort(-sc)[:3].tolist())),
        "best": best,
    }


def mechanism_label(d_u: float, d_c: float) -> str:
    if d_u <= 0.0 and d_c > 0.0:
        return "residual-driven"
    if d_u > 0.0 and d_c > 0.0:
        return "local+residual"
    if d_u > 0.0 and d_c <= 0.0:
        return "local-driven"
    return "conflicting"


def evaluate(rows: list[dict], c_hat: np.ndarray) -> tuple[list[dict], list[dict]]:
    u_hat = np.array([float(rows[i]["U_single"]) + float(c_hat[i]) for i in range(len(rows))], dtype=np.float64)
    decisions = []
    metrics = []
    for sc, idxs in grouped(rows).items():
        loc = state_pick(rows, idxs, [rows[i]["U_single"] for i in idxs])
        b2 = state_pick(rows, idxs, [u_hat[i] for i in idxs])
        ora = state_pick(rows, idxs, [rows[i]["U_traj"] for i in idxs])
        d_u = float(b2["U_single"] - loc["U_single"])
        d_c = float(c_hat[b2["gidx"]] - c_hat[loc["gidx"]])
        d_h = float(u_hat[b2["gidx"]] - u_hat[loc["gidx"]])
        changed = int(b2["candidate_id"] != loc["candidate_id"])
        rec = {
            "scenario_id": sc,
            "LOCAL_candidate": loc["candidate_id"],
            "B2_candidate": b2["candidate_id"],
            "ORACLE_candidate": ora["candidate_id"],
            "hash_LOCAL": loc["assignment_hash"],
            "hash_B2": b2["assignment_hash"],
            "hash_ORACLE": ora["assignment_hash"],
            "U_single_LOCAL": loc["U_single"],
            "U_single_B2": b2["U_single"],
            "U_traj_LOCAL": loc["U_traj"],
            "U_traj_B2": b2["U_traj"],
            "U_traj_ORACLE": ora["U_traj"],
            "C_hat_LOCAL": float(c_hat[loc["gidx"]]),
            "C_hat_B2": float(c_hat[b2["gidx"]]),
            "U_hat_LOCAL": float(u_hat[loc["gidx"]]),
            "U_hat_B2": float(u_hat[b2["gidx"]]),
            "R_LOCAL": loc["R"],
            "R_B2": b2["R"],
            "R_ORACLE": 1.0,
            "delta_R": float(b2["R"] - loc["R"]),
            "B2_changed": changed,
            "B2_matches_oracle": int(b2["candidate_id"] == ora["candidate_id"]),
            "LOCAL_matches_oracle": int(loc["candidate_id"] == ora["candidate_id"]),
            "top1": b2["top1"],
            "top3": b2["top3"],
            "dU_single": d_u,
            "dC": d_c,
            "dU_hat": d_h,
            "mechanism": mechanism_label(d_u, d_c) if changed else "unchanged",
            "spearman_C": spearman([c_hat[i] for i in idxs], [rows[i]["C_abs"] for i in idxs]),
            "spearman_Uhat": spearman([u_hat[i] for i in idxs], [rows[i]["U_traj"] for i in idxs]),
            "mean_U_traj": float(np.mean([rows[i]["U_traj"] for i in idxs])),
            "median_U_traj": float(np.median([rows[i]["U_traj"] for i in idxs])),
        }
        decisions.append(rec)
        metrics.append(dict(rec))
    return decisions, metrics


def permute_independent(rows: list[dict], model: ExtraTreesRegressor) -> dict:
    rng = np.random.default_rng(PERM_SEED)
    x = feature_matrix(rows, B2_FEATURE_KEYS)
    cols = [B2_FEATURE_KEYS.index(k) for k in P3E_CAND_KEYS]
    r_real, r_perm = [], []
    by = grouped(rows)
    c_real = np.asarray(model.predict(x), dtype=np.float64)
    u_real = np.array([float(r["U_single"]) for r in rows], dtype=np.float64) + c_real
    real_mean = float(np.mean([state_pick(rows, idxs, [u_real[i] for i in idxs])["R"] for idxs in by.values()]))
    for _ in range(N_PERM):
        x_perm = x.copy()
        for idxs in by.values():
            sl = np.asarray(idxs, dtype=np.int64)
            order = rng.permutation(len(idxs))
            block = x_perm[sl][:, cols]
            tmp = x_perm[sl]
            tmp[:, cols] = block[order]
            x_perm[sl] = tmp
        c_p = np.asarray(model.predict(x_perm), dtype=np.float64)
        u_p = np.array([float(r["U_single"]) for r in rows], dtype=np.float64) + c_p
        r_perm.append(float(np.mean([state_pick(rows, idxs, [u_p[i] for i in idxs])["R"] for idxs in by.values()])))
        r_real.append(real_mean)
    return {"n_perm": N_PERM, "R_pred_real_repeat": ext_sum(r_real), "R_pred_perm": ext_sum(r_perm)}


def decide(indep: dict, decisions: list[dict], perm: dict, pred: dict) -> dict:
    if not indep["independent_scenario_available"]:
        return {
            "gate": "PARTIAL",
            "text": "PARTIAL — no independent scenario available",
            "planner_ready": False,
            "independent_scenario_available": False,
        }
    dR = np.array([r["delta_R"] for r in decisions], dtype=np.float64)
    n_pos = int(np.sum(dR > 1e-12))
    n_neg = int(np.sum(dR < -1e-12))
    n_zero = int(len(dR) - n_pos - n_neg)
    r_b2 = float(np.mean([r["R_B2"] for r in decisions]))
    r_loc = float(np.mean([r["R_LOCAL"] for r in decisions]))
    std_r = float(np.std([r["R_B2"] for r in decisions], ddof=1)) if len(decisions) > 1 else float("nan")
    perm_drop = float(perm["R_pred_real_repeat"]["mean"] - perm["R_pred_perm"]["mean"])
    best = max(decisions, key=lambda r: r["delta_R"])
    worst = min(decisions, key=lambda r: r["delta_R"])
    dominated = abs(best["delta_R"]) > 3.0 * max(float(np.median(np.abs(dR))), 1e-6) and n_pos <= max(4, len(decisions) // 4)
    improved = r_b2 > r_loc + 1e-12
    stable = improved and n_pos >= n_neg and perm_drop > PERM_DROP_EXISTING and (not dominated)
    if not improved:
        gate, text = "FAIL", "独立数据上无法复现 B2 相对 LOCAL 的改善。"
    elif stable and len(decisions) >= 8 and std_r <= 0.20:
        gate, text = "PASS", "B2 在 independent scenario 上相对 LOCAL 有稳定改善，permutation sanity 通过，且不是少数 scenario 驱动。"
    else:
        gate, text = "PARTIAL", "独立数据上相对 LOCAL 有改善，但样本偏少或跨 scenario 不够稳。"
    return {
        "gate": gate,
        "text": text,
        "planner_ready": False,
        "independent_scenario_available": True,
        "mean_R_B2": r_b2,
        "median_R_B2": float(np.median([r["R_B2"] for r in decisions])),
        "std_R_B2": std_r,
        "mean_R_LOCAL": r_loc,
        "median_R_LOCAL": float(np.median([r["R_LOCAL"] for r in decisions])),
        "std_R_LOCAL": float(np.std([r["R_LOCAL"] for r in decisions], ddof=1)) if len(decisions) > 1 else float("nan"),
        "mean_dR": float(np.mean(dR)),
        "worst_dR": float(worst["delta_R"]),
        "worst_scenario": worst["scenario_id"],
        "best_dR": float(best["delta_R"]),
        "best_scenario": best["scenario_id"],
        "n_improve": n_pos,
        "n_degraded": n_neg,
        "n_unchanged": n_zero,
        "n_changed": int(sum(r["B2_changed"] for r in decisions)),
        "top1": float(np.mean([r["top1"] for r in decisions])),
        "top3": float(np.mean([r["top3"] for r in decisions])),
        "oracle_overlap": int(sum(r["B2_matches_oracle"] for r in decisions)),
        "local_oracle_overlap": int(sum(r["LOCAL_matches_oracle"] for r in decisions)),
        "mean_U_B2": float(np.mean([r["U_traj_B2"] for r in decisions])),
        "median_U_B2": float(np.median([r["U_traj_B2"] for r in decisions])),
        "mean_U_LOCAL": float(np.mean([r["U_traj_LOCAL"] for r in decisions])),
        "median_U_LOCAL": float(np.median([r["U_traj_LOCAL"] for r in decisions])),
        "mean_U_ORACLE": float(np.mean([r["U_traj_ORACLE"] for r in decisions])),
        "payload_gain_vs_LOCAL": float(np.mean([r["U_traj_B2"] - r["U_traj_LOCAL"] for r in decisions])),
        "residual_gap_vs_ORACLE": float(np.mean([r["U_traj_ORACLE"] - r["U_traj_B2"] for r in decisions])),
        "perm_drop": perm_drop,
        "perm_R": float(perm["R_pred_perm"]["mean"]),
        "dominated_by_few": bool(dominated),
        "spearman_C": pred["spearman_C"],
        "within_spearman_C": pred["within_spearman_C"],
        "spearman_Uhat": pred["spearman_Uhat"],
    }


def figures(decisions: list[dict]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    xs = [str(r["scenario_id"])[-3:] for r in decisions]
    fig, ax = plt.subplots(figsize=(max(7.4, 0.18 * len(xs) + 4), 3.8))
    ax.bar(np.arange(len(xs)) - 0.18, [r["R_LOCAL"] for r in decisions], 0.36, label="LOCAL", color="#9e9e9e")
    ax.bar(np.arange(len(xs)) + 0.18, [r["R_B2"] for r in decisions], 0.36, label="B2", color="#4C78A8")
    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels(xs, rotation=90, fontsize=7)
    ax.set_ylabel("R_pred")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "independent_rpred.png", dpi=140)
    plt.close(fig)
    changed = [r for r in decisions if r["B2_changed"]]
    labels = ["residual-driven", "local+residual", "local-driven", "conflicting"]
    counts = Counter(r["mechanism"] for r in changed)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(labels, [counts.get(k, 0) for k in labels], color=["#54A24B", "#4C78A8", "#F58518", "#E45756"])
    ax.set_ylabel("changed states")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG / "independent_mechanism.png", dpi=140)
    plt.close(fig)


def write_report(indep: dict, decisions: list[dict], perm: dict, mech: dict, pred: dict, gate: dict, leak_rows: list[dict]) -> str:
    if not indep["independent_scenario_available"]:
        return "\n".join(
            [
                "# HTW Phase-4C Independent Validation Audit",
                "",
                "本阶段只验证冻结 B2 是否能在真正未参与 Phase-4B / shadow 的独立数据上成立。未调参，未改 planner，未改 Low-Level。",
                "",
                "**PARTIAL — no independent scenario available**",
                "",
                "当前工程没有真正的 unseen scenario，因此不能把 LSO 冒充 independent validation。",
                "",
                "本阶段停止。不进入 planner integration，不进入 Phase-4D。",
                "",
            ]
        )
    lines = [
        "# HTW Phase-4C Independent Validation Audit",
        "",
        "本阶段只验证冻结 B2 candidate-conditioned residual credit 是否能在真正未参与 Phase-4B / Phase-4B shadow 的独立 scenario 上成立。",
        "不是方法开发，不是调参，不是 planner integration。未训练 Low-Level，未改 reward / utility / decoder / pool / checkpoint，未根据 independent test 反向调整 B2。",
        "",
        f"**门控：{gate['gate']}** — {gate['text']}",
        "",
        f"**planner_ready = False**。本阶段结束后停止，不进入 planner integration / Phase-4D。",
        "",
        "## 二、Independent data audit",
        "",
        f"- 真正 unseen scenario：**是**。n={indep['n_unseen']}，IDs={indep['unseen_scenarios']}",
        f"- 来源：`LLV2_train`。这些 ID **从未** 进入 Phase-4B GroupKFold。",
        f"- Low-Level 训练是否用过这些 scenario：是。B2 ExtraTrees 是否用过：否。",
        "- Phase-3 validation 223000-223063 / Phase-2 validation 222000-222063 / Phase-10.6 106000-106063：拒绝使用。",
        "- 不存在则应写“当前工程没有真正的 unseen scenario，因此只能进行 seed-held-out validation。” 本阶段**不**走这条，因为存在 unseen scenario。",
        "- 不用 LSO 冒充 independent validation。",
        "",
        "## 八、防泄漏",
        "",
        "全部 64 个 independent scenario 的泄漏项均为 0：不在 Phase-4B design / train fold / validation fold / test fold，也未用于 model / feature / threshold / hyperparameter selection。明细见 `independent_summary.json` 的 leakage 字段。",
        "",
        "B2 只在原 Phase-4A/4B 的 16 个 design EXEC state 上用冻结 ExtraTrees 拟合一次；independent labels 不进入 fit。",
            "",
            "## 固定 B2",
            "",
            "`C_hat = ExtraTrees(U_single + Phase-3E candidate-conditioned geometry)`，`U_hat = U_single + C_hat`。",
            f"ExtraTrees 固定：{ET}。未改 feature / beta / C 定义。",
            "U_single：5-seed official shadow_h15 均值。U_traj：FirstHoldOverrideWorker 覆盖 first-hold，hold 2/3/4 走正式 EXEC greedy，LL_RUN=0，deterministic=True。",
            "`C_abs = U_traj - U_single`。pool=`generate_pool(env, seed_key=scenario*1009+0)`。",
            "",
            "## 四 / 五、Independent test（primary = R_B2 vs R_LOCAL）",
            "",
            f"- mean/median/std R_B2 = {_fmt(gate['mean_R_B2'])} / {_fmt(gate['median_R_B2'])} / {_fmt(gate['std_R_B2'])}",
            f"- mean/median/std R_LOCAL = {_fmt(gate['mean_R_LOCAL'])} / {_fmt(gate['median_R_LOCAL'])} / {_fmt(gate['std_R_LOCAL'])}",
            f"- mean ΔR = {_fmt(gate['mean_dR'])}；worst-case ΔR = {_fmt(gate['worst_dR'])} (scenario {gate['worst_scenario']})",
            f"- top-1 / top-3 = {_fmt(gate['top1'])} / {_fmt(gate['top3'])}",
            f"- mean/median U_traj(B2) = {_sci(gate['mean_U_B2'])} / {_sci(gate['median_U_B2'])}",
            f"- B2-ORACLE overlap = {gate['oracle_overlap']}/{len(decisions)}",
            f"- B2 改变 first-hold = {gate['n_changed']}/{len(decisions)}",
            f"- improved / degraded / unchanged = {gate['n_improve']} / {gate['n_degraded']} / {gate['n_unchanged']}",
            f"- payload gain vs LOCAL = {_sci(gate['payload_gain_vs_LOCAL'])}；residual gap vs ORACLE = {_sci(gate['residual_gap_vs_ORACLE'])}",
            f"- dominated_by_few = {gate['dominated_by_few']}",
            f"- Spearman(C_hat, C) = {_fmt(pred['spearman_C'])}",
            f"- within-state Spearman(C_hat, C) mean = {_fmt(pred['within_spearman_C'])}",
            f"- Spearman(U_hat, U_traj) = {_fmt(pred['spearman_Uhat'])}",
            "",
            "恶化 scenario（全部保留，不删除）：",
            "",
            "| scenario | R_LOCAL | R_B2 | ΔR |",
            "|---|---:|---:|---:|",
        ]
    worse = [r for r in decisions if r["delta_R"] < -1e-12]
    for r in worse:
        lines.append(f"| {r['scenario_id']} | {_fmt(r['R_LOCAL'])} | {_fmt(r['R_B2'])} | {_fmt(r['delta_R'])} |")
    lines.extend(
        [
            "",
            "逐 scenario 全表见 `scenario_metrics.csv` / `independent_decisions.csv`。",
            "",
            "## 六、Permutation sanity（不重训，只打乱 geometry↔candidate）",
            "",
            f"- R_B2 = {_fmt(perm['R_pred_real_repeat']['mean'])}",
            f"- R_perm = {_fmt(perm['R_pred_perm']['mean'])} ± {_fmt(perm['R_pred_perm']['std'])}",
            f"- drop = {_fmt(gate['perm_drop'])}",
            "",
            "## 七、Residual-driven mechanism",
            "",
            "| class | n | fraction |",
            "|---|---:|---:|",
        ]
    )
    for name in ("residual-driven", "local+residual", "local-driven", "conflicting"):
        lines.append(f"| {name} | {mech['counts'][name]} | {_fmt(mech['fractions'][name])} |")
    lines.extend(
        [
            "",
            f"改变 state 上 mean dU_single={_sci(mech['mean_dU_single'])}，mean dC={_sci(mech['mean_dC'])}，mean dU_hat={_sci(mech['mean_dU_hat'])}。",
            "",
            "## PASS / PARTIAL / FAIL",
            "",
            f"**{gate['gate']}**。{gate['text']}",
            "",
            "本阶段停止。不修改 B2，不修改 planner，不进入 Phase-4D。",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    indep = audit_independence()
    _dump(OUTPUT / "independence_audit.json", indep)
    if not indep["independent_scenario_available"]:
        gate = decide(indep, [], {}, {})
        body = write_report(indep, [], {}, {}, {}, gate, [])
        (OUTPUT / REPORT_NAME).write_text(body + "\n", encoding="utf-8")
        (ROOT / REPORT_NAME).write_text(body + "\n", encoding="utf-8")
        _dump(OUTPUT / "independent_summary.json", {"independence": indep, "gate": gate})
        print(json.dumps({"gate": gate["gate"], "independent": False}, indent=2), flush=True)
        return gate
    design = load_design_exec()
    scorer = fit_frozen_b2(design)
    labels = generate_exec_labels(indep["unseen_scenarios"])
    if not labels:
        raise RuntimeError("independent label generation returned no rows")
    assert_no_future_keys(labels, B2_FEATURE_KEYS)
    x_te = feature_matrix(labels, B2_FEATURE_KEYS)
    c_hat = scorer.predict_credit(x_te)
    u_hat = np.array([float(r["U_single"]) for r in labels], dtype=np.float64) + c_hat
    pred = {
        "spearman_C": spearman(c_hat, [r["C_abs"] for r in labels]),
        "within_spearman_C": float(
            np.nanmean([spearman([c_hat[i] for i in idxs], [labels[i]["C_abs"] for i in idxs]) for idxs in grouped(labels).values()])
        ),
        "spearman_Uhat": spearman(u_hat, [r["U_traj"] for r in labels]),
    }
    decisions, metrics = evaluate(labels, c_hat)
    perm = permute_independent(labels, scorer.model)
    changed = [r for r in decisions if r["B2_changed"]]
    counts = Counter(r["mechanism"] for r in changed)
    n_ch = max(len(changed), 1)
    mech = {
        "n_changed": len(changed),
        "counts": {k: int(counts.get(k, 0)) for k in ("residual-driven", "local+residual", "local-driven", "conflicting")},
        "fractions": {k: counts.get(k, 0) / n_ch for k in ("residual-driven", "local+residual", "local-driven", "conflicting")},
        "mean_dU_single": float(np.mean([r["dU_single"] for r in changed])) if changed else float("nan"),
        "mean_dC": float(np.mean([r["dC"] for r in changed])) if changed else float("nan"),
        "mean_dU_hat": float(np.mean([r["dU_hat"] for r in changed])) if changed else float("nan"),
    }
    gate = decide(indep, decisions, perm, pred)
    figures(decisions)
    _csv_write(OUTPUT / "independent_decisions.csv", decisions)
    _csv_write(OUTPUT / "scenario_metrics.csv", metrics)
    _dump(
        OUTPUT / "independent_summary.json",
        {
            "independence": {k: v for k, v in indep.items() if k != "membership"},
            "leakage": indep["membership"],
            "gate": gate,
            "pred": pred,
            "perm": perm,
            "mechanism": mech,
            "et": ET,
            "B2_keys": list(B2_FEATURE_KEYS),
            "fit": "one ExtraTrees on all 16 Phase-4B EXEC design states; independent labels never used for fit",
            "n_design_rows": len(design),
            "n_independent_rows": len(labels),
        },
    )
    body = write_report(indep, decisions, perm, mech, pred, gate, indep["membership"])
    (OUTPUT / REPORT_NAME).write_text(body + "\n", encoding="utf-8")
    (ROOT / REPORT_NAME).write_text(body + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "gate": gate["gate"],
                "independent": True,
                "n_scenario": len(decisions),
                "R_B2": gate.get("mean_R_B2"),
                "R_LOCAL": gate.get("mean_R_LOCAL"),
                "n_worse": gate.get("n_degraded"),
                "planner_ready": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return gate


if __name__ == "__main__":
    run()
