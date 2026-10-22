"""Phase-4D-5 candidate design. No scoring. No official selection."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from wt.lowlevel_v2.phase4d5_deployment_criterion_candidates.protocol import (
    CANDIDATES,
    FROZEN_GATE1,
    OBSERVED_REFERENCE,
    OUTPUT,
    REPORT_NAME,
    ROOT,
)


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def payload() -> dict:
    return {
        "status": "HUMAN_SPECIFICATION_REQUIRED",
        "deployment_criterion": "HUMAN_SPECIFICATION_REQUIRED",
        "pre_registered": False,
        "official_criterion": None,
        "selected_candidate": None,
        "planner_ready": False,
        "phase_status": "BLOCKED_ON_HUMAN_SPECIFICATION",
        "pass_fail_computed": False,
        "thresholds_fit_from_independent_64": False,
        "wrote_registry": False,
        "gate_vs_criterion": {
            "Gate-1": (
                "Per-scenario switch: use B2 iff score margin >= frozen tau. "
                "Does not decide planner integration."
            ),
            "deployment_criterion": (
                "System-level acceptance of frozen B2 + frozen Gate-1. "
                "Does not change the per-scenario switch."
            ),
            "frozen_gate": FROZEN_GATE1,
        },
        "observed_reference": OBSERVED_REFERENCE,
        "candidates": list(CANDIDATES),
        "selection_rule": "This file does not select a candidate. A human owner must confirm one.",
    }


def write_comparison(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate_id",
        "name",
        "principle",
        "max_degradation_rate",
        "max_worst_case_delta_R",
        "min_mean_delta_R",
        "min_mean_U_gain",
        "scenario_degradation_policy",
        "require_all_scenarios_noninferior",
        "independent_holdout_required",
        "required_independent_batches",
        "rationale",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in CANDIDATES:
            item = {k: row[k] for k in fields if k != "rationale"}
            item["rationale"] = json.dumps(row["rationale"], ensure_ascii=False)
            writer.writerow(item)


def write_report() -> str:
    obs = OBSERVED_REFERENCE
    a, b, c = CANDIDATES
    return "\n".join(
        [
            "# HTW Phase-4D-5 Deployment Criterion Candidate Design",
            "",
            "本阶段只设计三档 deployment criterion candidate，不做 deployment PASS/FAIL。",
            "未训练或修改 B2，未修改 τ / Gate-1 / planner / decide()，未改写 64 个 independent scenario 结果。",
            "未把任何 candidate 写入 `deployment_criterion_registry.json`，未宣布 official criterion，未进入 Phase-4E。",
            "",
            "**deployment_criterion = HUMAN_SPECIFICATION_REQUIRED**",
            "",
            "**planner_ready = False**",
            "",
            "**status = BLOCKED_ON_HUMAN_SPECIFICATION**",
            "",
            "## Gate-1 与 deployment criterion 分离",
            "",
            "Gate-1 只决定**某一个** scenario 是否使用 B2：",
            "",
            "- margin = U_hat(argmax U_hat) − U_hat(argmax U_single)",
            "- τ = 1.535e6（Phase-4B shadow 锁定，本阶段不改）",
            "- 若 margin ≥ τ 则 B2，否则 LOCAL",
            "",
            "Deployment criterion 决定**整套**冻结 B2 + 冻结 Gate-1 是否允许进入正式 planner。",
            "二者不能混为一谈。改 criterion 不改 Gate-1；改 Gate-1 也不等于已经有 criterion。",
            "",
            "## Observed reference（禁止作为阈值来源）",
            "",
            f"“{obs['disclaimer']}”",
            "",
            f"- mean ΔR = {obs['mean_delta_R']}",
            f"- degrade rate = {obs['degrade']} = {obs['degrade_rate']}",
            f"- worst-case ΔR = {obs['worst_case_delta_R']}",
            f"- mean R = {obs['mean_R']}",
            f"- n_B2 = {obs['n_B2']}",
            "",
            "下列 candidate 的数值来自工程设计常数（R 定义在 [0,1] 上的十进制/二分小数、1-in-100 稀有事件配额），",
            "**不是**由 3/64、−0.214 或 0.111 外推出来的 5%、−0.25 或同类填充。",
            "本阶段不计算任何 candidate 在 64 集上的 PASS/FAIL。",
            "",
            "## Candidate-A：Strict Safety",
            "",
            f"- 原则：{a['principle']}",
            f"- C1 max_degradation_rate = {a['max_degradation_rate']}",
            f"- C2 max_worst_case_delta_R = {a['max_worst_case_delta_R']}",
            f"- C3 min_mean_delta_R = {a['min_mean_delta_R']}",
            f"- C4 min_mean_U_gain = {a['min_mean_U_gain']}",
            f"- C5 scenario_degradation_policy = {a['scenario_degradation_policy']}",
            f"- C6 require_all_scenarios_noninferior = {a['require_all_scenarios_noninferior']}",
            f"- C7 independent_holdout_required = {a['independent_holdout_required']}",
            f"- C8 required_independent_batches = {a['required_independent_batches']}",
            "",
            "理由：任何相对 LOCAL 的单场景恶化都视为部署缺陷。LOCAL fallback 不能用来接受有害替换。",
            "额外要求 1 个独立生成的新 batch，因为严格安全性不能只建立在 220000–220063 这一张表上。",
            "",
            "## Candidate-B：Balanced Deployment",
            "",
            f"- 原则：{b['principle']}",
            f"- C1 max_degradation_rate = {b['max_degradation_rate']}（1-in-100，不是 3/64 或 5%）",
            f"- C2 max_worst_case_delta_R = {b['max_worst_case_delta_R']}（R 单位区间的 1/10，不是 −0.214 或 −0.25）",
            f"- C3 min_mean_delta_R = {b['min_mean_delta_R']}（平均不下降，不是 0.111）",
            f"- C4 min_mean_U_gain = {b['min_mean_U_gain']}（平均 payload 不低于 LOCAL，不抄当前增益）",
            f"- C5 scenario_degradation_policy = {b['scenario_degradation_policy']}",
            f"- C6 require_all_scenarios_noninferior = {b['require_all_scenarios_noninferior']}",
            f"- C7 independent_holdout_required = {b['independent_holdout_required']}",
            f"- C8 required_independent_batches = {b['required_independent_batches']}",
            "",
            "理由：允许极少数、且有上界的单场景损失；总体 rank 与 payload 不得变差；再用 1 个新 unseen batch 约束过拟合。",
            "",
            "## Candidate-C：Conservative-with-Fallback",
            "",
            f"- 原则：{c['principle']}",
            f"- C1 max_degradation_rate = {c['max_degradation_rate']}（2-in-100，不是 3/64）",
            f"- C2 max_worst_case_delta_R = {c['max_worst_case_delta_R']}（R 单位区间的 1/8，不是 −0.214 的填充）",
            f"- C3 min_mean_delta_R = {c['min_mean_delta_R']}（平均至少改善 R 单位区间的 0.02，才值得改默认 planner）",
            f"- C4 min_mean_U_gain = {c['min_mean_U_gain']}",
            f"- C5 scenario_degradation_policy = {c['scenario_degradation_policy']}",
            f"- C6 require_all_scenarios_noninferior = {c['require_all_scenarios_noninferior']}",
            f"- C7 independent_holdout_required = {c['independent_holdout_required']}",
            f"- C8 required_independent_batches = {c['required_independent_batches']}",
            "",
            "理由：Gate-1 已经在 margin 不足时退回 LOCAL。系统级部署仍要求有限恶化、最坏损失有界、平均有正收益，",
            "并且用 **2** 个额外独立 batch 确认这是系统性质而不是单表结果。",
            "",
            "## 人工选择表",
            "",
            "| Candidate | 核心原则 | 单场景风险 | 总体收益要求 | 独立验证要求 |",
            "|-----------|----------|------------|--------------|--------------|",
            "| A | Strict Safety | 不允许任何恶化；worst-case ΔR ≥ 0 | 平均 ΔR ≥ 0 且平均 U 不低于 LOCAL | 需要 1 个新 independent batch |",
            "| B | Balanced | 最多 1/100 恶化，且单场景 ΔR ≥ −0.10 | 平均 ΔR ≥ 0 且平均 U 不低于 LOCAL | 需要 1 个新 independent batch |",
            "| C | Conservative + Fallback | 最多 2/100 恶化，且单场景 ΔR ≥ −0.125 | 平均 ΔR ≥ 0.02 且平均 U 不低于 LOCAL | 需要 2 个新 independent batch |",
            "",
            "本报告不选择 candidate。最终 deployment criterion 必须由项目负责人人工确认。",
            "",
            "## 最终状态",
            "",
            "- deployment_criterion = HUMAN_SPECIFICATION_REQUIRED",
            "- planner_ready = False",
            "- status = BLOCKED_ON_HUMAN_SPECIFICATION",
            "- official_criterion = None",
            "- 未写入 deployment_criterion_registry.json",
            "",
            "本阶段停止。不训练、不调参、不改 B2、不改 τ、不改 planner、不改 decide()、",
            "不重新计算 64 个 scenario、不宣布 PASS、不进入 Phase-4E。",
            "",
        ]
    )


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = payload()
    _dump(OUTPUT / "deployment_criterion_candidates.json", data)
    write_comparison(OUTPUT / "deployment_criterion_comparison.csv")
    body = write_report()
    (OUTPUT / REPORT_NAME).write_text(body + "\n", encoding="utf-8")
    (ROOT / REPORT_NAME).write_text(body + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "deployment_criterion": "HUMAN_SPECIFICATION_REQUIRED",
                "planner_ready": False,
                "status": "BLOCKED_ON_HUMAN_SPECIFICATION",
                "n_candidates": 3,
                "selected": None,
                "pass_fail": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return data


if __name__ == "__main__":
    run()
