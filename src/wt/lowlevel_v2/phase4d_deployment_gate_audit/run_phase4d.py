"""Phase-4D conservative deployment-gate audit. Frozen Phase-4C tables only."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from wt.lowlevel_v2.phase4d_deployment_gate_audit.protocol import (
    INDEPENDENT_IDS,
    N_INDEPENDENT,
    OUTPUT,
    P4C_DECISIONS,
    P4C_LABELS,
    REPORT_NAME,
    ROOT,
    TAU_SOURCE,
    TAU_UHAT,
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


def load_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"missing Phase-4C artifact: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
        "dU_single",
        "dC",
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


def observables() -> dict:
    return {
        "allowed": [
            "U_single",
            "Phase-3E candidate-conditioned geometry",
            "C_hat / U_hat from frozen ExtraTrees",
            "candidate ranking by U_single or U_hat",
            "score margin = U_hat(B2) - U_hat(LOCAL)",
            "B2_changed = argmax(U_hat) != argmax(U_single)",
        ],
        "forbidden_as_gate_input": [
            "U_traj",
            "C_abs / C / F",
            "oracle / argmax U_traj",
            "later holds",
            "remaining_arrivals",
            "future position / action",
        ],
        "uncertainty": {
            "available_in_frozen_B2": False,
            "reason": "frozen B2 contract only returns C_hat = ExtraTrees.predict; no tree-dispersion / uncertainty field was defined in Phase-4B/4C. Gate-3 is not constructible. Do not invent a proxy.",
        },
    }


def risk_rows(decisions: list[dict]) -> list[dict]:
    out = []
    for row in decisions:
        dR = float(row["delta_R"])
        out.append(
            {
                "scenario_id": row["scenario_id"],
                "delta_R": dR,
                "R_LOCAL": row["R_LOCAL"],
                "R_B2": row["R_B2"],
                "U_traj_LOCAL": row["U_traj_LOCAL"],
                "U_traj_B2": row["U_traj_B2"],
                "U_traj_ORACLE": row["U_traj_ORACLE"],
                "B2_changed": row["B2_changed"],
                "B2_worse": int(dR < -1e-12),
                "B2_better": int(dR > 1e-12),
                "B2_tie": int(abs(dR) <= 1e-12),
                "margin_Uhat": float(row["dU_hat"]),
                "margin_available_at_deploy": True,
                "U_single_LOCAL": row["U_single_LOCAL"],
                "U_single_B2": row["U_single_B2"],
                "C_hat_LOCAL": row["C_hat_LOCAL"],
                "C_hat_B2": row["C_hat_B2"],
                "LOCAL_candidate": row["LOCAL_candidate"],
                "B2_candidate": row["B2_candidate"],
                "ORACLE_candidate": row["ORACLE_candidate"],
            }
        )
    return out


def risk_summary(risk: list[dict]) -> dict:
    dR = np.array([r["delta_R"] for r in risk], dtype=np.float64)
    worse = [r["scenario_id"] for r in risk if r["B2_worse"]]
    return {
        "n": len(risk),
        "n_improve": int(sum(r["B2_better"] for r in risk)),
        "n_tie": int(sum(r["B2_tie"] for r in risk)),
        "n_worse": int(sum(r["B2_worse"] for r in risk)),
        "n_changed": int(sum(r["B2_changed"] for r in risk)),
        "mean_dR": float(np.mean(dR)),
        "median_dR": float(np.median(dR)),
        "p05_dR": float(np.percentile(dR, 5)),
        "p10_dR": float(np.percentile(dR, 10)),
        "worst_dR": float(np.min(dR)),
        "worst_scenario": int(min(risk, key=lambda r: r["delta_R"])["scenario_id"]),
        "worse_scenarios": worse,
        "all_64_retained": True,
        "includes_220012": 220012 in [r["scenario_id"] for r in risk],
    }


def apply_gate(row: dict, name: str) -> str:
    margin = float(row["dU_hat"])
    changed = int(row["B2_changed"]) == 1
    if name == "Gate-0":
        return "B2"
    if name == "Gate-1":
        return "B2" if changed and margin >= TAU_UHAT else "LOCAL"
    if name == "Gate-2":
        return "B2" if margin >= TAU_UHAT else "LOCAL"
    raise RuntimeError(f"unknown constructible gate {name}")


def gate_metrics(decisions: list[dict], local_top3: dict[int, float], name: str, constructible: bool) -> dict:
    if not constructible:
        return {
            "gate": name,
            "constructible": False,
            "definition": "uncertainty rejection; frozen B2 has no defined uncertainty",
            "reason": "不可构造",
        }
    uses, u, r, top1, top3, d_vs_loc = [], [], [], [], [], []
    for row in decisions:
        choice = apply_gate(row, name)
        uses.append(choice)
        if choice == "B2":
            u.append(row["U_traj_B2"])
            r.append(row["R_B2"])
            top1.append(row["B2_matches_oracle"])
            top3.append(row["top3"])
        else:
            u.append(row["U_traj_LOCAL"])
            r.append(row["R_LOCAL"])
            top1.append(row["LOCAL_matches_oracle"])
            top3.append(local_top3[row["scenario_id"]])
        d_vs_loc.append(u[-1] - row["U_traj_LOCAL"])
        r_vs_loc = r[-1] - row["R_LOCAL"]
        row[f"{name}_choice"] = choice
        row[f"{name}_U"] = u[-1]
        row[f"{name}_R"] = r[-1]
        row[f"{name}_dR_vs_LOCAL"] = r_vs_loc
    r = np.asarray(r, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    dR = np.array([row[f"{name}_dR_vs_LOCAL"] for row in decisions], dtype=np.float64)
    oracle_u = np.array([row["U_traj_ORACLE"] for row in decisions], dtype=np.float64)
    n_b2 = int(sum(c == "B2" for c in uses))
    return {
        "gate": name,
        "constructible": True,
        "tau_Uhat": TAU_UHAT if name != "Gate-0" else None,
        "n_use_B2": n_b2,
        "n_use_LOCAL": int(len(uses) - n_b2),
        "mean_U_traj": float(np.mean(u)),
        "median_U_traj": float(np.median(u)),
        "mean_R": float(np.mean(r)),
        "median_R": float(np.median(r)),
        "worst_R": float(np.min(r)),
        "worst_R_scenario": int(decisions[int(np.argmin(r))]["scenario_id"]),
        "top1": float(np.mean(top1)),
        "top3": float(np.mean(top3)),
        "n_improve": int(np.sum(dR > 1e-12)),
        "n_worse": int(np.sum(dR < -1e-12)),
        "n_tie": int(np.sum(np.abs(dR) <= 1e-12)),
        "mean_gain_vs_LOCAL": float(np.mean(d_vs_loc)),
        "worst_change_vs_LOCAL": float(np.min(dR)),
        "worst_change_scenario": int(decisions[int(np.argmin(dR))]["scenario_id"]),
        "residual_gap_vs_ORACLE": float(np.mean(oracle_u - u)),
    }


def decide(risk: dict, gates: list[dict], obs: dict) -> dict:
    g0 = next(g for g in gates if g["gate"] == "Gate-0")
    constructible = [g for g in gates if g.get("constructible")]
    safer = [
        g
        for g in constructible
        if g["gate"] != "Gate-0" and g["n_worse"] < g0["n_worse"] and g["mean_R"] >= g0["mean_R"] - 0.03
    ]
    deployment_defined = False
    overfit_risk = True
    if not safer:
        text = "B2 在 unseen scenario 上有平均收益，但预定义 Gate-0/1/2 没有同时保住主要收益并明显压住恶化；Gate-3 不可构造。不能接入正式 planner。"
        gate = "PARTIAL"
    else:
        text = (
            "存在预定义 margin gate 能减少部分恶化，但阈值来自 16-state Phase-4B shadow，"
            "只在同一批 64 independent scenario 上评价，没有第二批 held-out 证明 gate 本身不过拟合。"
            "工程也没有预先定义 deployment-safety threshold。不能接入正式 planner。"
        )
        gate = "PARTIAL"
    return {
        "gate": gate,
        "text": text,
        "planner_ready": False,
        "deployment_safe_gate": None,
        "deployment_requirement_predefined": deployment_defined,
        "gate_overfit_risk_on_same_64": overfit_risk,
        "uncertainty_constructible": bool(obs["uncertainty"]["available_in_frozen_B2"]),
        "tau_Uhat": TAU_UHAT,
        "tau_source": TAU_SOURCE,
        "n_independent": risk["n"],
        "worse_retained": risk["worse_scenarios"],
    }


def write_report(obs: dict, risk: dict, gates: list[dict], verdict: dict, decisions: list[dict]) -> str:
    g0 = next(g for g in gates if g["gate"] == "Gate-0")
    g1 = next(g for g in gates if g["gate"] == "Gate-1")
    g2 = next(g for g in gates if g["gate"] == "Gate-2")
    g3 = next(g for g in gates if g["gate"] == "Gate-3")
    lines = [
        "# HTW Phase-4D B2 Conservative Deployment Gate Audit",
        "",
        "本阶段只审计冻结 B2 是否具备保守部署条件，并检查只依赖 first-hold 可观测量的预定义 gate。",
        "未修改正式 planner / decide()，未修改 Low-Level，未重新训练 B2，未按 independent U_traj 调阈值，未进入 Phase-4E。",
        "",
        f"**门控：{verdict['gate']}** — {verdict['text']}",
        "",
        f"**planner_ready = {verdict['planner_ready']}**。deployment_safe_gate = {verdict['deployment_safe_gate']}。",
        "",
        "## 数据来源",
        "",
        "- Phase-4C 真正 unseen scenario 独立验证：`src/wt/lowlevel_v2/phase4c_independent_validation/outputs/independent_decisions.csv`",
        f"- 64 个 independent scenario ID：{list(INDEPENDENT_IDS)}",
        "- 这些 ID 来自 `LLV2_train`，从未进入 Phase-4B GroupKFold / feature / hyperparameter / threshold selection",
        "- 全部 64 个 scenario 保留，包括 worst-case 220012，不删除恶化场景",
        "- B2 分数复用 Phase-4C 已写出的 C_hat / U_hat，不重新 fit ExtraTrees",
        "",
        "## A. Deployment-time 可观测量",
        "",
        "允许：",
        "",
    ]
    for item in obs["allowed"]:
        lines.append(f"- {item}")
    lines.extend(["", "禁止作为 gate 输入：", ""])
    for item in obs["forbidden_as_gate_input"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            f"模型不确定性：{obs['uncertainty']['reason']}",
            "",
            "## B. 逐 scenario 风险（相对 LOCAL）",
            "",
            f"- 改善 / 持平 / 恶化 = {risk['n_improve']} / {risk['n_tie']} / {risk['n_worse']}",
            f"- B2 改变 first-hold = {risk['n_changed']}/64",
            f"- mean ΔR = {_fmt(risk['mean_dR'])}",
            f"- median ΔR = {_fmt(risk['median_dR'])}",
            f"- 5th percentile ΔR = {_fmt(risk['p05_dR'])}",
            f"- 10th percentile ΔR = {_fmt(risk['p10_dR'])}",
            f"- worst-case ΔR = {_fmt(risk['worst_dR'])} (scenario {risk['worst_scenario']})",
            f"- B2 < LOCAL 的 scenario：{risk['worse_scenarios']}",
            "- 全表见 `phase4d_scenario_risk.csv`",
            "",
            "## C. 预定义 gate（不优化）",
            "",
            f"- 固定阈值 τ = {_sci(TAU_UHAT)}",
            f"- 阈值来源：{TAU_SOURCE}。**不是**用 64 个 independent U_traj 反选出来的。",
            "- score margin = U_hat(argmax U_hat) - U_hat(argmax U_single)，只依赖 first-hold 分数",
            "",
            "- Gate-0：永远使用 B2",
            "- Gate-1：仅当 first-hold 改变且 margin ≥ τ 时用 B2，否则 LOCAL",
            "- Gate-2：仅当 margin ≥ τ 时用 B2，否则 LOCAL",
            "- Gate-3：uncertainty rejection — **不可构造**",
            "",
            "说明：当 first-hold 未改变时 margin=0 < τ，因此 Gate-1 与 Gate-2 在本定义下等价。这是定义结果，不是调参。",
            "",
            "## D. 各 gate 指标",
            "",
            "| gate | constructible | n_B2 | mean U | mean R | median R | worst R | top-1 | top-3 | improve/worse/tie | mean gain vs LOCAL | worst ΔR vs LOCAL | gap vs ORACLE |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
        ]
    )
    for g in (g0, g1, g2):
        lines.append(
            f"| {g['gate']} | yes | {g['n_use_B2']} | {_sci(g['mean_U_traj'])} | {_fmt(g['mean_R'])} | {_fmt(g['median_R'])} | {_fmt(g['worst_R'])} | {_fmt(g['top1'])} | {_fmt(g['top3'])} | {g['n_improve']}/{g['n_worse']}/{g['n_tie']} | {_sci(g['mean_gain_vs_LOCAL'])} | {_fmt(g['worst_change_vs_LOCAL'])} | {_sci(g['residual_gap_vs_ORACLE'])} |"
        )
    lines.append(f"| {g3['gate']} | **不可构造** |  |  |  |  |  |  |  |  |  |  |  |")
    lines.extend(
        [
            "",
            f"Gate-1/2 使用 B2 的 scenario 数 = {g1['n_use_B2']}；退回 LOCAL = {g1['n_use_LOCAL']}。",
            "",
            "## E. 安全性",
            "",
            "1. 有没有简单 gate 能保留主要平均收益并减少恶化？",
            f"   Gate-0 mean R={_fmt(g0['mean_R'])}，恶化 {g0['n_worse']}，worst ΔR={_fmt(g0['worst_change_vs_LOCAL'])}。",
            f"   Gate-1/2 mean R={_fmt(g1['mean_R'])}，恶化 {g1['n_worse']}，worst ΔR={_fmt(g1['worst_change_vs_LOCAL'])}。",
            "   预定义 τ 来自 Phase-4B shadow，不是在 64 集上搜索得到。",
            "2. gate 是否可能只在这 64 个 scenario 上过拟合？",
            "   **是，有这个风险。** 阈值虽未用 independent U_traj 选择，但 gate 的安全结论只在同一批 64 上评价，没有第二批 unseen scenario。",
            "3. 若没有足够证据证明 gate 安全：planner_ready = False。",
            f"   本阶段 **planner_ready = {verdict['planner_ready']}**。工程仍未定义正式 deployment-safety threshold。",
            "4. 未为了 PASS 修改 gate 定义，未删除 220012，未调 τ。",
            "",
            "## F. PASS / PARTIAL / FAIL",
            "",
            f"**{verdict['gate']}**。{verdict['text']}",
            "",
            "本阶段停止。不修改 planner，不调用正式 decide()，不进入 Phase-4E，不训练任何模型。",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    decisions = load_decisions()
    labels = load_csv(P4C_LABELS)
    local_top3 = local_top3_map(labels)
    obs = observables()
    risk = risk_rows(decisions)
    risk_stat = risk_summary(risk)
    g0 = gate_metrics(decisions, local_top3, "Gate-0", True)
    g1 = gate_metrics(decisions, local_top3, "Gate-1", True)
    g2 = gate_metrics(decisions, local_top3, "Gate-2", True)
    g3 = gate_metrics(decisions, local_top3, "Gate-3", False)
    gates = [g0, g1, g2, g3]
    verdict = decide(risk_stat, gates, obs)
    _csv_write(OUTPUT / "phase4d_scenario_risk.csv", risk)
    _csv_write(
        OUTPUT / "phase4d_gate_results.csv",
        [
            {
                "scenario_id": r["scenario_id"],
                "margin_Uhat": r["dU_hat"],
                "B2_changed": r["B2_changed"],
                "Gate-0": r.get("Gate-0_choice"),
                "Gate-1": r.get("Gate-1_choice"),
                "Gate-2": r.get("Gate-2_choice"),
                "Gate-3": "not_constructible",
                "R_LOCAL": r["R_LOCAL"],
                "R_B2": r["R_B2"],
                "R_Gate-0": r.get("Gate-0_R"),
                "R_Gate-1": r.get("Gate-1_R"),
                "R_Gate-2": r.get("Gate-2_R"),
                "delta_R_B2": r["delta_R"],
                "delta_R_Gate-1": r.get("Gate-1_dR_vs_LOCAL"),
                "delta_R_Gate-2": r.get("Gate-2_dR_vs_LOCAL"),
            }
            for r in decisions
        ],
    )
    _csv_write(OUTPUT / "phase4d_gate_comparison.csv", [g for g in gates])
    _dump(
        OUTPUT / "phase4d_gate_summary.json",
        {
            "observables": obs,
            "risk": risk_stat,
            "gates": gates,
            "verdict": verdict,
            "tau": {"value": TAU_UHAT, "source": TAU_SOURCE, "tuned_on_independent_U_traj": False},
            "data": {
                "decisions": str(P4C_DECISIONS),
                "labels": str(P4C_LABELS),
                "independent_ids": list(INDEPENDENT_IDS),
            },
            "planner_ready": False,
        },
    )
    body = write_report(obs, risk_stat, gates, verdict, decisions)
    (OUTPUT / REPORT_NAME).write_text(body + "\n", encoding="utf-8")
    (ROOT / REPORT_NAME).write_text(body + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "gate": verdict["gate"],
                "planner_ready": False,
                "R_Gate0": g0["mean_R"],
                "R_Gate1": g1["mean_R"],
                "n_worse_Gate0": g0["n_worse"],
                "n_worse_Gate1": g1["n_worse"],
                "Gate3": "不可构造",
            },
            indent=2,
        ),
        flush=True,
    )
    return verdict


if __name__ == "__main__":
    run()
