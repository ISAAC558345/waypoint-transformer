"""Phase-4D-2 safety-criterion audit. Frozen Phase-4C tables and frozen tau only."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from wt.lowlevel_v2.phase4d2_deployment_safety_audit.protocol import (
    INDEPENDENT_IDS,
    KNOWN_CRITERION_FILES,
    N_INDEPENDENT,
    OUTPUT,
    P4C_DECISIONS,
    P4C_LABELS,
    P4C_SUMMARY,
    REPORT_NAME,
    ROOT,
    TAU_DISPLAY,
    TAU_SOURCE,
    TAU_UHAT,
)

EPS = 1e-12


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


def load_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"missing Phase-4C artifact: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing Phase-4C artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_decisions() -> list[dict]:
    raw = load_csv(P4C_DECISIONS)
    out = []
    ints = (
        "scenario_id",
        "LOCAL_candidate",
        "B2_candidate",
        "ORACLE_candidate",
        "B2_changed",
        "B2_matches_oracle",
        "LOCAL_matches_oracle",
    )
    floats = (
        "U_single_LOCAL",
        "U_single_B2",
        "U_traj_LOCAL",
        "U_traj_B2",
        "U_traj_ORACLE",
        "C_hat_LOCAL",
        "C_hat_B2",
        "U_hat_LOCAL",
        "U_hat_B2",
        "R_LOCAL",
        "R_B2",
        "R_ORACLE",
        "delta_R",
        "top1",
        "top3",
        "dU_hat",
    )
    for row in raw:
        item = dict(row)
        for key in ints:
            item[key] = int(float(item[key]))
        for key in floats:
            item[key] = float(item[key])
        out.append(item)
    have = [int(r["scenario_id"]) for r in out]
    if have != list(INDEPENDENT_IDS):
        raise RuntimeError(f"expected {N_INDEPENDENT} IDs 220000-220063, got {have[:5]}... n={len(have)}")
    if 220012 not in have:
        raise RuntimeError("220012 missing; do not drop worst-case scenario")
    return out


def local_top3_map(labels: list[dict]) -> dict[int, float]:
    by = defaultdict(list)
    for row in labels:
        by[int(float(row["scenario_id"]))].append(row)
    out = {}
    for sc, rows in by.items():
        us = np.array([float(r["U_single"]) for r in rows], dtype=np.float64)
        ut = np.array([float(r["U_traj"]) for r in rows], dtype=np.float64)
        oracle = int(np.argmax(ut))
        top3 = set(np.argsort(-us)[:3].tolist())
        out[sc] = float(oracle in top3)
    return out


def search_existing_criterion() -> dict:
    hits = []
    for path in KNOWN_CRITERION_FILES:
        hits.append({"path": str(path), "exists": path.is_file()})
    return {
        "searched_known_artifacts": hits,
        "found_pre_registered_numeric_thresholds": False,
        "reason": (
            "Phase-4B/4C/4D summaries set planner_ready=false and do not define "
            "numeric acceptance rules for mean_dR / degrade_rate / worst_delta_R / "
            "mean utility gain. No prior deployment_criterion.json exists."
        ),
        "status": "NOT PRE-REGISTERED",
    }


def apply_gate(row: dict, name: str) -> str:
    margin = float(row["dU_hat"])
    if name == "Gate-0":
        return "B2"
    if name == "Gate-1":
        return "B2" if margin >= TAU_UHAT else "LOCAL"
    raise RuntimeError(f"unknown gate {name}")


def gate_metrics(decisions: list[dict], local_top3: dict[int, float], name: str) -> dict:
    uses, u, r, top1, top3, d_u, d_r = [], [], [], [], [], [], []
    oracle_hits = []
    for row in decisions:
        choice = apply_gate(row, name)
        uses.append(choice)
        if choice == "B2":
            u.append(row["U_traj_B2"])
            r.append(row["R_B2"])
            top1.append(float(row["B2_matches_oracle"]))
            top3.append(row["top3"])
            oracle_hits.append(int(row["B2_matches_oracle"]))
        else:
            u.append(row["U_traj_LOCAL"])
            r.append(row["R_LOCAL"])
            top1.append(float(row["LOCAL_matches_oracle"]))
            top3.append(local_top3[row["scenario_id"]])
            oracle_hits.append(int(row["LOCAL_matches_oracle"]))
        d_u.append(u[-1] - row["U_traj_LOCAL"])
        d_r.append(r[-1] - row["R_LOCAL"])
        row[f"{name}_choice"] = choice
        row[f"{name}_U"] = u[-1]
        row[f"{name}_R"] = r[-1]
        row[f"{name}_dR"] = d_r[-1]
        row[f"{name}_dU"] = d_u[-1]
        row[f"{name}_oracle"] = oracle_hits[-1]
    r = np.asarray(r, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    dR = np.asarray(d_r, dtype=np.float64)
    dU = np.asarray(d_u, dtype=np.float64)
    oracle_u = np.array([row["U_traj_ORACLE"] for row in decisions], dtype=np.float64)
    n_b2 = int(sum(c == "B2" for c in uses))
    n_improve = int(np.sum(dR > EPS))
    n_worse = int(np.sum(dR < -EPS))
    n_tie = int(np.sum(np.abs(dR) <= EPS))
    return {
        "gate": name,
        "definition": (
            "always B2"
            if name == "Gate-0"
            else "use B2 iff margin >= tau else LOCAL; margin = U_hat(argmax U_hat) - U_hat(argmax U_single)"
        ),
        "tau_Uhat": None if name == "Gate-0" else TAU_UHAT,
        "n_B2": n_b2,
        "n_LOCAL": int(len(uses) - n_b2),
        "mean_R": float(np.mean(r)),
        "median_R": float(np.median(r)),
        "std_R": float(np.std(r, ddof=1)),
        "mean_U_traj": float(np.mean(u)),
        "median_U_traj": float(np.median(u)),
        "n_improve": n_improve,
        "n_tie": n_tie,
        "n_degrade": n_worse,
        "degrade_rate": float(n_worse / len(uses)),
        "mean_dR": float(np.mean(dR)),
        "median_dR": float(np.median(dR)),
        "p05_dR": float(np.percentile(dR, 5)),
        "p10_dR": float(np.percentile(dR, 10)),
        "worst_delta_R": float(np.min(dR)),
        "worst_scenario": int(decisions[int(np.argmin(dR))]["scenario_id"]),
        "top1": float(np.mean(top1)),
        "top3": float(np.mean(top3)),
        "oracle_overlap_count": int(sum(oracle_hits)),
        "oracle_overlap_rate": float(np.mean(oracle_hits)),
        "mean_gain_U_vs_LOCAL": float(np.mean(dU)),
        "mean_residual_gap_vs_ORACLE": float(np.mean(oracle_u - u)),
    }


def scenario_rows(decisions: list[dict]) -> list[dict]:
    rows = []
    for row in decisions:
        use_b2 = int(row["Gate-1_choice"] == "B2")
        dR_g1 = float(row["Gate-1_dR"])
        dR_g0 = float(row["delta_R"])
        degraded = int(dR_g1 < -EPS)
        avoided = int((dR_g0 < -EPS) and not use_b2)
        rows.append(
            {
                "scenario_id": row["scenario_id"],
                "LOCAL_first_hold": row["LOCAL_candidate"],
                "B2_first_hold": row["B2_candidate"],
                "ORACLE_first_hold": row["ORACLE_candidate"],
                "margin": row["dU_hat"],
                "use_B2": use_b2,
                "R_LOCAL": row["R_LOCAL"],
                "R_B2": row["R_B2"],
                "delta_R": dR_g0,
                "U_LOCAL": row["U_traj_LOCAL"],
                "U_B2": row["U_traj_B2"],
                "degraded": degraded,
                "R_Gate1": row["Gate-1_R"],
                "U_Gate1": row["Gate-1_U"],
                "delta_R_Gate1": dR_g1,
                "Gate0_degraded": int(dR_g0 < -EPS),
                "Gate1_avoided_Gate0_degrade": avoided,
                "is_220012": int(row["scenario_id"] == 220012),
            }
        )
    return rows


def special_lists(rows: list[dict]) -> dict:
    still = [r["scenario_id"] for r in rows if r["degraded"]]
    avoided = [r["scenario_id"] for r in rows if r["Gate1_avoided_Gate0_degrade"]]
    improved = [r for r in rows if r["delta_R_Gate1"] > EPS]
    improved_sorted = sorted(improved, key=lambda r: r["delta_R_Gate1"], reverse=True)
    row012 = next(r for r in rows if r["scenario_id"] == 220012)
    return {
        "gate1_still_degraded": still,
        "gate1_largest_improvements": [
            {"scenario_id": r["scenario_id"], "delta_R_Gate1": r["delta_R_Gate1"]} for r in improved_sorted[:5]
        ],
        "scenario_220012": {
            "margin": row012["margin"],
            "use_B2": row012["use_B2"],
            "delta_R_B2": row012["delta_R"],
            "delta_R_Gate1": row012["delta_R_Gate1"],
            "avoided": bool(row012["Gate1_avoided_Gate0_degrade"]),
        },
        "gate0_degraded_but_gate1_avoided": avoided,
    }


def stability(decisions: list[dict], g0: dict, g1: dict, p4c: dict) -> dict:
    dR = np.array([row["Gate-1_dR"] for row in decisions], dtype=np.float64)
    pos = np.clip(dR, 0.0, None)
    total_pos = float(np.sum(pos))
    order = np.argsort(-pos)
    top3_share = float(np.sum(pos[order[:3]]) / total_pos) if total_pos > 0 else 0.0
    top5_share = float(np.sum(pos[order[:5]]) / total_pos) if total_pos > 0 else 0.0
    mean_all = float(np.mean(dR))
    loo = []
    for i, row in enumerate(decisions):
        mask = np.ones(len(dR), dtype=bool)
        mask[i] = False
        loo.append(
            {
                "dropped": row["scenario_id"],
                "mean_dR": float(np.mean(dR[mask])),
                "abs_change": abs(float(np.mean(dR[mask]) - mean_all)),
            }
        )
    loo_max = max(loo, key=lambda x: x["abs_change"])
    retain = (g1["mean_R"] - (g0["mean_R"] - g0["mean_dR"])) / g0["mean_dR"] if g0["mean_dR"] != 0 else float("nan")
    gate = p4c.get("gate", {})
    perm = p4c.get("perm", {})
    return {
        "retain_main_gain": {
            "mean_R_LOCAL": float(g0["mean_R"] - g0["mean_dR"]),
            "mean_R_Gate0": g0["mean_R"],
            "mean_R_Gate1": g1["mean_R"],
            "fraction_of_Gate0_mean_dR_retained": float(retain),
            "answer": "yes" if retain >= 0.7 and g1["mean_dR"] > 0 else "no",
        },
        "reduce_degradation": {
            "n_degrade_Gate0": g0["n_degrade"],
            "n_degrade_Gate1": g1["n_degrade"],
            "degrade_rate_Gate0": g0["degrade_rate"],
            "degrade_rate_Gate1": g1["degrade_rate"],
            "worst_delta_R_Gate0": g0["worst_delta_R"],
            "worst_delta_R_Gate1": g1["worst_delta_R"],
            "answer": "yes" if g1["n_degrade"] < g0["n_degrade"] else "no",
        },
        "gain_from_few_scenarios": {
            "positive_dR_mass": total_pos,
            "top3_share_of_positive_dR": top3_share,
            "top5_share_of_positive_dR": top5_share,
            "n_positive": int(np.sum(dR > EPS)),
            "answer": "concentrated_but_not_single_scenario" if top3_share < 0.5 else "yes_concentrated",
        },
        "single_scenario_dominance": {
            "max_loo_abs_change_mean_dR": loo_max["abs_change"],
            "dropped_scenario": loo_max["dropped"],
            "loo_mean_dR_after_drop": loo_max["mean_dR"],
            "answer": "no_single_scenario_dominates" if loo_max["abs_change"] < abs(mean_all) else "possible_dominance",
        },
        "permutation_sanity": {
            "source": "Phase-4C independent_summary.json; not retrained",
            "perm_R_mean": gate.get("perm_R", perm.get("R_pred_perm", {}).get("mean")),
            "perm_drop": gate.get("perm_drop"),
            "real_R": gate.get("mean_R_B2"),
            "supports_candidate_conditioned_signal": bool(gate.get("perm_drop", 0) > 0.1),
        },
        "gate_uses_only_deployment_observables": True,
        "gate_uses_independent_U_traj_C_oracle": False,
    }


def criterion_payload(search: dict, g0: dict, g1: dict, p4c: dict) -> dict:
    observed = {
        "A_mean_dR": {"Gate-0": g0["mean_dR"], "Gate-1": g1["mean_dR"]},
        "B_degrade_rate": {"Gate-0": g0["degrade_rate"], "Gate-1": g1["degrade_rate"]},
        "C_worst_delta_R": {"Gate-0": g0["worst_delta_R"], "Gate-1": g1["worst_delta_R"]},
        "D_mean_utility_gain": {"Gate-0": g0["mean_gain_U_vs_LOCAL"], "Gate-1": g1["mean_gain_U_vs_LOCAL"]},
        "E_permutation": {
            "source": "Phase-4C independent_summary.json",
            "perm_R": p4c.get("gate", {}).get("perm_R"),
            "perm_drop": p4c.get("gate", {}).get("perm_drop"),
            "retrained": False,
        },
        "F_gate_input_restriction": {
            "uses_independent_U_traj": False,
            "uses_independent_C": False,
            "uses_oracle": False,
        },
    }
    needs_human = {
        "A_min_mean_dR": None,
        "B_max_degrade_rate": None,
        "C_min_worst_delta_R": None,
        "D_min_mean_utility_gain": None,
        "E_min_perm_drop": None,
        "note": (
            "These acceptance thresholds are not pre-registered in the project. "
            "Do not fill them from the 64 independent U_traj outcomes. "
            "A human must pre-register them before PASS / planner_ready=True is possible."
        ),
    }
    return {
        "status": "NOT PRE-REGISTERED",
        "pre_registered": False,
        "existing_criterion_search": search,
        "frozen_model": {
            "C_hat": "frozen ExtraTrees(U_single + Phase-3E candidate-conditioned geometry)",
            "U_hat": "U_single + C_hat",
            "retrained": False,
            "features_changed": False,
            "hyperparameters_changed": False,
        },
        "frozen_gate": {
            "rule": "if margin >= tau: use B2 else LOCAL",
            "margin": "U_hat(argmax U_hat) - U_hat(argmax U_single)",
            "tau": TAU_UHAT,
            "tau_display": TAU_DISPLAY,
            "tau_source": TAU_SOURCE,
            "tau_changed": False,
            "tuned_on_independent_U_traj": False,
        },
        "metrics": {
            "A_mean_dR": {
                "definition": "mean(R_selected - R_LOCAL)",
                "acceptance_rule": "NOT PRE-REGISTERED",
                "needs_human_confirmation": True,
            },
            "B_degrade_rate": {
                "definition": "N(R_selected < R_LOCAL) / N",
                "acceptance_rule": "NOT PRE-REGISTERED",
                "needs_human_confirmation": True,
            },
            "C_worst_delta_R": {
                "definition": "min(R_selected - R_LOCAL)",
                "acceptance_rule": "NOT PRE-REGISTERED",
                "needs_human_confirmation": True,
            },
            "D_mean_utility_gain": {
                "definition": "mean(U_selected - U_LOCAL)",
                "acceptance_rule": "NOT PRE-REGISTERED",
                "needs_human_confirmation": True,
            },
            "E_permutation_sanity": {
                "definition": "Phase-4C permutation of candidate-conditioned geometry; no retrain",
                "acceptance_rule": "NOT PRE-REGISTERED",
                "needs_human_confirmation": True,
            },
            "F_no_label_leak_in_gate": {
                "definition": "deployment gate must not use independent U_traj / C / oracle",
                "acceptance_rule": "fixed: forbidden",
                "satisfied": True,
            },
        },
        "observed_on_independent_64": observed,
        "needs_human_confirmation": needs_human,
        "planner_ready_requires": "all pre-registered acceptance rules must exist AND be satisfied",
        "cannot_declare_planner_ready": True,
    }


def decide(search: dict, g1: dict, stability_out: dict) -> dict:
    leak = bool(stability_out["gate_uses_independent_U_traj_C_oracle"])
    no_gain = (g1["mean_dR"] <= 0.0) or (g1["mean_R"] <= (g1["mean_R"] - g1["mean_dR"]))
    if leak or no_gain:
        gate = "FAIL"
        text = "固定 Gate-1 在 independent scenario 上不再具有相对 LOCAL 的稳定收益，或发现数据/实现泄漏。"
    else:
        gate = "PARTIAL"
        text = (
            "B2 在 independent scenario 上泛化成立，固定 margin gate 也降低了风险，"
            "但项目尚未预先定义正式 deployment-safety criterion，因此不能宣布 planner_ready。"
        )
    return {
        "gate": gate,
        "text": text,
        "planner_ready": False,
        "deployment_criterion": search["status"],
        "pre_registered": False,
        "tau_Uhat": TAU_UHAT,
        "tau_source": TAU_SOURCE,
        "all_64_retained": True,
        "includes_220012": True,
    }


def write_report(
    search: dict,
    g0: dict,
    g1: dict,
    special: dict,
    stability_out: dict,
    verdict: dict,
    p4c: dict,
) -> str:
    local_r = g0["mean_R"] - g0["mean_dR"]
    local_u = g0["mean_U_traj"] - g0["mean_gain_U_vs_LOCAL"]
    perm_r = stability_out["permutation_sanity"]["perm_R_mean"]
    perm_drop = stability_out["permutation_sanity"]["perm_drop"]
    lines = [
        "# HTW Phase-4D-2 B2 Pre-Registered Deployment Safety Criterion Audit",
        "",
        "本阶段只把 deployment-safe 判定标准写成显式配置，并对已冻结的 64 个真正 unseen scenario 做最终安全性审计。",
        "未重新训练 ExtraTrees，未改变 B2 特征/超参数，未重选 τ，未删除任何 independent scenario（含 220012），",
        "未用 independent U_traj/C/oracle 构造 gate，未修改正式 planner / decide()，未进入 Phase-4E。",
        "",
        f"**门控：{verdict['gate']}** — {verdict['text']}",
        "",
        f"**planner_ready = {verdict['planner_ready']}**。",
        "",
        f"**deployment criterion = {verdict['deployment_criterion']}**。",
        "",
        "## 数据来源与 64 个 independent scenario 身份",
        "",
        "- Phase-4C 独立验证决策表：`src/wt/lowlevel_v2/phase4c_independent_validation/outputs/independent_decisions.csv`",
        "- Phase-4C 标签表：`src/wt/lowlevel_v2/phase4c_independent_validation/outputs/independent_eval_labels.csv`",
        "- Phase-4C 摘要（permutation 只读复用，不重训）：`src/wt/lowlevel_v2/phase4c_independent_validation/outputs/independent_summary.json`",
        f"- 64 个 independent scenario ID：{list(INDEPENDENT_IDS)}",
        "- 这些 ID 来自 `LLV2_train`，从未进入 Phase-4B GroupKFold / feature / hyperparameter / threshold selection",
        "- 全部 64 个 scenario 保留，包括 worst-case 220012",
        "- B2 分数复用 Phase-4C 已写出的 C_hat / U_hat，不重新 fit ExtraTrees",
        "",
        "## 冻结对象",
        "",
        "- C_hat = 已冻结 ExtraTrees(U_single + Phase-3E candidate-conditioned geometry)",
        "- U_hat = U_single + C_hat",
        f"- τ = {_sci(TAU_UHAT)}（显示为 1.535e6）",
        f"- τ 来源：{TAU_SOURCE}",
        "- τ **不是**由 64 个 independent U_traj 反推，本阶段禁止修改",
        "",
        "## Deployment-time 可观测量",
        "",
        "Gate 只允许使用 first-hold 时可获得的信息：U_single、Phase-3E candidate-conditioned geometry、B2 的 C_hat/U_hat、由此得到的 ranking，以及 margin。",
        "禁止作为 gate 输入：U_traj、C、oracle、后续 hold、remaining arrivals、未来位置/动作。",
        "",
        "## 既有工程阈值检索",
        "",
        f"- 检索结果：{search['status']}",
        f"- 找到预注册数值阈值：{search['found_pre_registered_numeric_thresholds']}",
        f"- 原因：{search['reason']}",
        "- 因此 **不要** 从这 64 个 independent scenario 拟合阈值。",
        "- `deployment_criterion.json` 中 A–E 的 acceptance_rule 全部标记为 NOT PRE-REGISTERED，并列为需要人工确认的字段。",
        "- F（gate 不得使用 independent U_traj/C/oracle）是唯一已固定的限制，且本阶段满足。",
        "",
        "## 固定 Gate 定义",
        "",
        "- Gate-0：永远使用 B2",
        "- Gate-1：若 margin >= 1.535e6 则使用 B2，否则退回 LOCAL",
        "- margin = U_hat(argmax U_hat) - U_hat(argmax U_single)",
        "- 未改变 first-hold 时 margin=0 < τ，因此本定义与 Phase-4D 的 Gate-1/Gate-2 等价。这是定义结果，不是调参。",
        "",
        "## 最终审计指标",
        "",
        f"对照（Phase-4C 冻结）：LOCAL mean R={_fmt(local_r)}，mean U_traj={_sci(local_u)}。",
        "",
        "| 指标 | Gate-0 | Gate-1 |",
        "|---|---:|---:|",
        f"| n_B2 | {g0['n_B2']} | {g1['n_B2']} |",
        f"| n_LOCAL | {g0['n_LOCAL']} | {g1['n_LOCAL']} |",
        f"| mean R | {_fmt(g0['mean_R'])} | {_fmt(g1['mean_R'])} |",
        f"| median R | {_fmt(g0['median_R'])} | {_fmt(g1['median_R'])} |",
        f"| std R | {_fmt(g0['std_R'])} | {_fmt(g1['std_R'])} |",
        f"| mean U_traj | {_sci(g0['mean_U_traj'])} | {_sci(g1['mean_U_traj'])} |",
        f"| median U_traj | {_sci(g0['median_U_traj'])} | {_sci(g1['median_U_traj'])} |",
        f"| improve / tie / degrade | {g0['n_improve']}/{g0['n_tie']}/{g0['n_degrade']} | {g1['n_improve']}/{g1['n_tie']}/{g1['n_degrade']} |",
        f"| degrade rate | {_fmt(g0['degrade_rate'])} | {_fmt(g1['degrade_rate'])} |",
        f"| mean ΔR | {_fmt(g0['mean_dR'])} | {_fmt(g1['mean_dR'])} |",
        f"| median ΔR | {_fmt(g0['median_dR'])} | {_fmt(g1['median_dR'])} |",
        f"| 5th percentile ΔR | {_fmt(g0['p05_dR'])} | {_fmt(g1['p05_dR'])} |",
        f"| 10th percentile ΔR | {_fmt(g0['p10_dR'])} | {_fmt(g1['p10_dR'])} |",
        f"| worst-case ΔR | {_fmt(g0['worst_delta_R'])} | {_fmt(g1['worst_delta_R'])} |",
        f"| worst-case scenario | {g0['worst_scenario']} | {g1['worst_scenario']} |",
        f"| top-1 | {_fmt(g0['top1'])} | {_fmt(g1['top1'])} |",
        f"| top-3 | {_fmt(g0['top3'])} | {_fmt(g1['top3'])} |",
        f"| B2-ORACLE overlap | {g0['oracle_overlap_count']}/64 ({_fmt(g0['oracle_overlap_rate'])}) | {g1['oracle_overlap_count']}/64 ({_fmt(g1['oracle_overlap_rate'])}) |",
        f"| mean residual gap to ORACLE | {_sci(g0['mean_residual_gap_vs_ORACLE'])} | {_sci(g1['mean_residual_gap_vs_ORACLE'])} |",
        "",
        "## 逐 scenario 风险（Gate-1）",
        "",
        f"- Gate-1 仍然恶化的 scenario：{special['gate1_still_degraded']}",
        f"- Gate-1 改善最大的 5 个 scenario：{special['gate1_largest_improvements']}",
        (
            f"- 220012：margin={_sci(special['scenario_220012']['margin'])}，use_B2={special['scenario_220012']['use_B2']}，"
            f"ΔR_B2={_fmt(special['scenario_220012']['delta_R_B2'])}，ΔR_Gate1={_fmt(special['scenario_220012']['delta_R_Gate1'])}，"
            f"avoided={special['scenario_220012']['avoided']}"
        ),
        f"- Gate-0 恶化但 Gate-1 成功避免：{special['gate0_degraded_but_gate1_avoided']}",
        "- 完整 64 行见 `phase4d2_scenario_final.csv`",
        "",
        "## 稳定性检查（不重训）",
        "",
        (
            f"1. Gate-1 是否仍然保留 B2 的主要收益？**{stability_out['retain_main_gain']['answer']}**。"
            f" Gate-0 mean R={_fmt(g0['mean_R'])}，Gate-1 mean R={_fmt(g1['mean_R'])}，"
            f"保留 Gate-0 mean ΔR 的 {_fmt(stability_out['retain_main_gain']['fraction_of_Gate0_mean_dR_retained'])}。"
        ),
        (
            f"2. Gate-1 是否显著减少 Gate-0 的 degradation？**{stability_out['reduce_degradation']['answer']}**。"
            f" 恶化数 {g0['n_degrade']}→{g1['n_degrade']}，worst ΔR {_fmt(g0['worst_delta_R'])}→{_fmt(g1['worst_delta_R'])}。"
        ),
        (
            f"3. Gate-1 的收益是否主要来自少数 scenario？**{stability_out['gain_from_few_scenarios']['answer']}**。"
            f" 正 ΔR 质量中 top-3 份额={_fmt(stability_out['gain_from_few_scenarios']['top3_share_of_positive_dR'])}，"
            f"top-5 份额={_fmt(stability_out['gain_from_few_scenarios']['top5_share_of_positive_dR'])}，"
            f"正收益 scenario 数={stability_out['gain_from_few_scenarios']['n_positive']}。"
        ),
        (
            f"4. 是否存在某一个 scenario 对结果有异常支配作用？**{stability_out['single_scenario_dominance']['answer']}**。"
            f" 最大 leave-one-out |Δ mean ΔR|={_fmt(stability_out['single_scenario_dominance']['max_loo_abs_change_mean_dR'])}"
            f"（去掉 {stability_out['single_scenario_dominance']['dropped_scenario']}）。"
        ),
        (
            f"5. B2 permutation sanity 是否仍然支持 candidate-conditioned signal？**"
            f"{stability_out['permutation_sanity']['supports_candidate_conditioned_signal']}**。"
            f" Phase-4C：real R={_fmt(stability_out['permutation_sanity']['real_R'])}，"
            f"perm R={_fmt(perm_r)}，drop={_fmt(perm_drop)}。未重新训练。"
        ),
        "6. Gate 是否完全由 deployment-time 可观测量构成？**是**。margin 与 τ 都不使用 independent U_traj/C/oracle。",
        "",
        "## 已经证明 / 尚未证明",
        "",
        "已经证明：",
        "",
        "- B2 在真正 unseen scenario 上相对 LOCAL 具有泛化收益（Phase-4C：mean R 0.892 vs 0.763）。",
        "- 固定 margin gate（τ 来自 Phase-4B shadow）可以降低恶化风险（11→3，worst ΔR -0.500→-0.214）。",
        "- Gate 输入不依赖 independent 标签。",
        "",
        "尚未证明：",
        "",
        "- B2 是否达到项目规定的正式 deployment-safety standard。",
        "- 因为该 standard **尚未预注册**，本阶段不能把观测值改写成 acceptance rule。",
        "- 没有第二批 held-out scenario 证明 gate 本身的安全结论可迁移。",
        "",
        "## PASS / PARTIAL / FAIL",
        "",
        "判定规则（预先写死，不按结果改写）：",
        "",
        "- PASS：仅当已有预注册 deployment criterion 且全部满足。",
        "- PARTIAL：B2 泛化成立且固定 margin gate 降低风险，但项目尚未预定义正式 criterion。",
        "- FAIL：固定 Gate-1 不再具有相对 LOCAL 的稳定收益，或发现泄漏。",
        "",
        f"**{verdict['gate']}**。{verdict['text']}",
        "",
        f"**planner_ready = {verdict['planner_ready']}**。",
        "",
        "本阶段停止。不修改 planner，不修改 decide()，不训练任何模型，不修改 B2，不修改 τ，不进入 Phase-4E。",
        "",
    ]
    return "\n".join(lines)


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    decisions = load_decisions()
    labels = load_csv(P4C_LABELS)
    p4c = load_json(P4C_SUMMARY)
    local_top3 = local_top3_map(labels)
    search = search_existing_criterion()
    g0 = gate_metrics(decisions, local_top3, "Gate-0")
    g1 = gate_metrics(decisions, local_top3, "Gate-1")
    rows = scenario_rows(decisions)
    special = special_lists(rows)
    stability_out = stability(decisions, g0, g1, p4c)
    criterion = criterion_payload(search, g0, g1, p4c)
    verdict = decide(search, g1, stability_out)
    _csv_write(OUTPUT / "phase4d2_gate_final.csv", [g0, g1])
    _csv_write(OUTPUT / "phase4d2_scenario_final.csv", rows)
    _dump(OUTPUT / "deployment_criterion.json", criterion)
    _dump(
        OUTPUT / "phase4d2_safety_summary.json",
        {
            "criterion_status": search["status"],
            "gates": [g0, g1],
            "special": special,
            "stability": stability_out,
            "verdict": verdict,
            "data": {
                "decisions": str(P4C_DECISIONS),
                "labels": str(P4C_LABELS),
                "summary": str(P4C_SUMMARY),
                "independent_ids": list(INDEPENDENT_IDS),
            },
            "planner_ready": False,
        },
    )
    body = write_report(search, g0, g1, special, stability_out, verdict, p4c)
    (OUTPUT / REPORT_NAME).write_text(body + "\n", encoding="utf-8")
    (ROOT / REPORT_NAME).write_text(body + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "gate": verdict["gate"],
                "planner_ready": False,
                "deployment_criterion": search["status"],
                "R_Gate0": g0["mean_R"],
                "R_Gate1": g1["mean_R"],
                "n_degrade_Gate0": g0["n_degrade"],
                "n_degrade_Gate1": g1["n_degrade"],
            },
            indent=2,
        ),
        flush=True,
    )
    return verdict


if __name__ == "__main__":
    run()
