"""Phase-4D-4 human specification draft. No threshold fitting. No planner change."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from wt.lowlevel_v2.phase4d4_deployment_criterion_specification.protocol import (
    HUMAN_PENDING,
    OBSERVED_REFERENCE,
    OUTPUT,
    REPORT_NAME,
    ROOT,
    TAU_SOURCE,
    TAU_UHAT,
)


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dimensions() -> list[dict]:
    return [
        {
            "criterion_id": "C1",
            "criterion_name": "degradation rate",
            "definition": "Maximum fraction of scenarios allowed to have R_selected < R_LOCAL after the frozen Gate-1.",
            "possible_policy": "choose a pre-registered fraction in [0, 1]; do not copy 3/64 or 0.047",
            "engineering_implication": "A lower cap rejects more B2 substitutions even when average gain is positive.",
            "requires_numeric_value": True,
            "human_decision": HUMAN_PENDING,
            "field": "max_degradation_rate",
            "value": None,
        },
        {
            "criterion_id": "C2",
            "criterion_name": "worst-case delta_R",
            "definition": "Most negative allowed R_selected - R_LOCAL on any retained scenario.",
            "possible_policy": "choose a pre-registered lower bound; do not copy -0.214",
            "engineering_implication": "A tighter floor can fail the current Gate-1 even if mean gain remains positive.",
            "requires_numeric_value": True,
            "human_decision": HUMAN_PENDING,
            "field": "max_worst_case_delta_R",
            "value": None,
        },
        {
            "criterion_id": "C3",
            "criterion_name": "mean delta_R",
            "definition": "Whether the mean of R_selected - R_LOCAL must satisfy a pre-registered inequality.",
            "possible_policy": "none | must_be_greater_than_0 | must_be_at_least_pre_specified_value",
            "engineering_implication": "A numeric floor is a project policy, not a statistic of the current 64 scenarios.",
            "requires_numeric_value": "conditional",
            "human_decision": HUMAN_PENDING,
            "field": "min_mean_delta_R",
            "value": None,
        },
        {
            "criterion_id": "C4",
            "criterion_name": "mean U_traj",
            "definition": "Whether mean selected U_traj must beat LOCAL, optionally by a pre-specified margin.",
            "possible_policy": "none | must_exceed_LOCAL | must_exceed_LOCAL_by_pre_specified_margin",
            "engineering_implication": "Utility-margin policy is independent of the current 1.081e7 vs 9.469e6 observation.",
            "requires_numeric_value": "conditional",
            "human_decision": HUMAN_PENDING,
            "field": "min_mean_U_gain",
            "value": None,
        },
        {
            "criterion_id": "C5",
            "criterion_name": "scenario degradation",
            "definition": "Whether any individual scenario may be worse than LOCAL after the gate.",
            "possible_policy": "forbid_all | allow_some | allow_only_if_worst_case_bound_holds",
            "engineering_implication": "forbid_all is stricter than C1/C2 and would fail any remaining B2-used degradation.",
            "requires_numeric_value": False,
            "human_decision": HUMAN_PENDING,
            "field": "scenario_degradation_policy",
            "value": None,
        },
        {
            "criterion_id": "C6",
            "criterion_name": "non-inferiority",
            "definition": "strict: for every scenario R_selected >= R_LOCAL; aggregate: only summary statistics must pass.",
            "possible_policy": "strict_non_inferiority | aggregate_non_inferiority",
            "engineering_implication": "Strict non-inferiority forbids the current 3 Gate-1 degradations if they remain after audit.",
            "requires_numeric_value": False,
            "human_decision": HUMAN_PENDING,
            "field": "require_all_scenarios_noninferior",
            "value": None,
        },
        {
            "criterion_id": "C7",
            "criterion_name": "independent hold-out",
            "definition": "Whether a second unseen batch, not 220000-220063, must also be evaluated against the same criterion.",
            "possible_policy": "not_required | required_one_new_batch | required_multiple_independent_batches",
            "engineering_implication": "If required, planner_ready stays false until a new unseen batch is generated and audited.",
            "requires_numeric_value": False,
            "human_decision": HUMAN_PENDING,
            "field": "independent_holdout_required",
            "value": None,
        },
        {
            "criterion_id": "C8",
            "criterion_name": "repeated batches",
            "definition": "How many independent batches must pass, and whether each batch must be generated independently.",
            "possible_policy": "required_batch_count is a human integer or null; each batch must be independently generated if required",
            "engineering_implication": "Repeated-batch policy is a project risk decision, not something the current 64 can decide.",
            "requires_numeric_value": "conditional",
            "human_decision": HUMAN_PENDING,
            "field": "required_independent_batches",
            "value": None,
        },
    ]


def specification(dims: list[dict]) -> dict:
    return {
        "status": "HUMAN_SPECIFICATION_REQUIRED",
        "deployment_criterion": "HUMAN_SPECIFICATION_REQUIRED",
        "pre_registered": False,
        "planner_ready": False,
        "phase_status": "BLOCKED_ON_HUMAN_SPECIFICATION",
        "official_standard": None,
        "auto_selected_criterion": None,
        "thresholds_fit_from_independent_64": False,
        "frozen_gate": {
            "rule": "if margin >= tau: use B2 else LOCAL",
            "margin": "U_hat(argmax U_hat) - U_hat(argmax U_single)",
            "tau": TAU_UHAT,
            "tau_display": 1.535e6,
            "tau_source": TAU_SOURCE,
            "tau_changed": False,
        },
        "acceptance_fields": {
            "max_degradation_rate": None,
            "max_worst_case_delta_R": None,
            "min_mean_delta_R": None,
            "min_mean_U_gain": None,
            "require_all_scenarios_noninferior": None,
            "independent_holdout_required": None,
            "required_independent_batches": None,
        },
        "policy_fields": {
            "mean_delta_R_policy": None,
            "mean_U_traj_policy": None,
            "scenario_degradation_policy": None,
            "non_inferiority_policy": None,
            "holdout_policy": None,
            "each_batch_independently_generated": None,
        },
        "dimensions": dims,
        "observed_reference": OBSERVED_REFERENCE,
        "observed_reference_disclaimer": OBSERVED_REFERENCE["disclaimer"],
        "next_audit_after_human_spec": {
            "reuse_frozen_64": True,
            "refit_B2": False,
            "retune_tau": False,
            "procedure": (
                "After a human writes non-null acceptance fields, re-score the same frozen "
                "64-scenario Gate-1 table against those fields. Do not search thresholds on "
                "that table. Do not enter Phase-4E until the audit is requested."
            ),
        },
    }


def write_matrix(path: Path, dims: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "criterion_id",
        "criterion_name",
        "definition",
        "possible_policy",
        "engineering_implication",
        "requires_numeric_value",
        "human_decision",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in dims:
            writer.writerow({k: row[k] for k in fields})


def write_report(spec: dict) -> str:
    obs = spec["observed_reference"]
    return "\n".join(
        [
            "# HTW Phase-4D-4 Deployment Criterion Human Specification Draft",
            "",
            "本阶段只建立供人工确认的 deployment criterion 规格草案。",
            "未重新训练或修改 B2，未修改 τ，未修改 planner / decide()，未进入 Phase-4E。",
            "未用 64 个 independent scenario 的 mean / median / worst-case / degradation rate 反推任何正式阈值。",
            "未自动选择任何候选 criterion，未宣布 planner_ready=True。",
            "",
            "**deployment_criterion = HUMAN_SPECIFICATION_REQUIRED**",
            "",
            "**planner_ready = False**",
            "",
            "**status = BLOCKED_ON_HUMAN_SPECIFICATION**",
            "",
            "## Q1. 当前为什么不能直接进入 planner？",
            "",
            "正式 `decide()` 仍是 joint greedy 解码，不是 B2 shadow score。",
            "Phase-4C/4D 只证明：在冻结 Gate-1 下，B2 对真正 unseen scenario 相对 LOCAL 有平均收益，并能减少部分恶化。",
            "仓库没有预注册的 deployment-safety acceptance criterion。没有该标准，就不能判断当前观测是否“足够安全”，因此不能接入正式 planner。",
            "",
            "## Q2. 当前缺失的究竟是模型证据还是 deployment specification？",
            "",
            "**缺失的是 deployment specification，不是模型是否存在 independent 证据。**",
            "Phase-4C 已经在 64 个真正 unseen scenario（LLV2_train 220000–220063）上验证 B2。",
            "Phase-4D/4D-2 已经用预先固定的 τ=1.535e6 审计 Gate-1。",
            "Phase-4D-3 检索确认：NO PRE-REGISTERED DEPLOYMENT CRITERION FOUND。",
            "阻塞项是人工尚未规定 C1–C8，而不是还没有 independent scenario。",
            "",
            "## Q3. 哪些指标已经有 independent validation？",
            "",
            "以下是已冻结的模型/gate 证据，不是 acceptance rule：",
            "",
            "- B2 在 64 个 unseen scenario 上相对 LOCAL 的平均收益（Phase-4C：mean R 0.892 vs 0.763）",
            "- permutation sanity：candidate-conditioned geometry 被打乱后 R 下降（Phase-4C drop=0.220）",
            "- 固定 margin Gate-1 降低恶化（11→3）并避开 220012（Phase-4D/4D-2）",
            "- Gate 输入只用 first-hold 可观测量，不使用 independent U_traj / C / oracle",
            "",
            "## Q4. 哪些指标只是 observed performance？",
            "",
            "`observed_reference` 单独记录冻结 Gate-1 表现，并明确：",
            f"“{obs['disclaimer']}”",
            "",
            f"- n_B2 = {obs['n_B2']}",
            f"- mean R = {obs['mean_R']}",
            f"- mean ΔR = {obs['mean_delta_R']}",
            f"- degrade = {obs['degrade']}",
            f"- degrade_rate = {obs['degrade_rate']}",
            f"- worst_case_delta_R = {obs['worst_case_delta_R']}",
            "",
            "这些数字不得填入 acceptance_fields。",
            "",
            "## Q5. 哪些参数必须由人工/项目负责人预先规定？",
            "",
            "C1 最大允许 degradation rate → `max_degradation_rate = null`",
            "C2 最大允许 worst-case ΔR → `max_worst_case_delta_R = null`",
            "C3 mean ΔR 要求：不要求 / 必须 > 0 / 必须 ≥ 预先指定值 → `min_mean_delta_R = null`",
            "C4 mean U_traj 要求：不要求 / 必须 > LOCAL / 必须超过预先指定 improvement margin → `min_mean_U_gain = null`",
            "C5 是否允许 scenario degradation：完全不允许 / 允许少量 / 允许但受 worst-case 限制",
            "C6 非劣：strict（∀ scenario, R_selected ≥ R_LOCAL）或 aggregate → `require_all_scenarios_noninferior = null`",
            "C7 是否需要第二批 deployment hold-out → `independent_holdout_required = null`",
            "C8 跨 batch 重复验证：`required_independent_batches = null`；若需要，每批是否必须独立生成也待定",
            "",
            "决策矩阵见 `deployment_criterion_decision_matrix.csv`。`human_decision` 全部为 PENDING_HUMAN_DECISION，未自动选择。",
            "",
            "## Q6. criterion 确定后，下一步应该如何重新审计 64 个 independent scenario？",
            "",
            "1. 人工把 C1–C8 写成非 null 的预注册字段，且不得用这 64 个 scenario 的统计量回填。",
            "2. 保持 B2、特征、超参数、τ=1.535e6、Gate-1 定义不变。",
            "3. 复用已冻结的 Phase-4C/4D Gate-1 逐 scenario 表，按新 criterion 逐条判定 PASS/FAIL。",
            "4. 不在这 64 个 scenario 上搜索更优 τ 或更优阈值。",
            "5. 若 C7/C8 要求新的 independent batch，则在新 batch 上重复同一审计；在此之前 planner_ready 仍为 False。",
            "6. 即使 64 个 scenario 全部满足人工标准，也只有在预注册 criterion 被书面锁定后才能讨论 planner 接入；那属于后续指令，不是本阶段。",
            "",
            "## 规格字段（全部未填正式数值）",
            "",
            "- max_degradation_rate = null",
            "- max_worst_case_delta_R = null",
            "- min_mean_delta_R = null",
            "- min_mean_U_gain = null",
            "- require_all_scenarios_noninferior = null",
            "- independent_holdout_required = null",
            "- required_independent_batches = null",
            "",
            "## 最终状态",
            "",
            "- deployment_criterion = HUMAN_SPECIFICATION_REQUIRED",
            "- planner_ready = False",
            "- status = BLOCKED_ON_HUMAN_SPECIFICATION",
            "",
            "本阶段停止。不修改任何模型、planner、decide()、B2、τ。不进入 Phase-4E。",
            "",
        ]
    )


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dims = dimensions()
    spec = specification(dims)
    _dump(OUTPUT / "deployment_criterion_specification.json", spec)
    write_matrix(OUTPUT / "deployment_criterion_decision_matrix.csv", dims)
    body = write_report(spec)
    (OUTPUT / REPORT_NAME).write_text(body + "\n", encoding="utf-8")
    (ROOT / REPORT_NAME).write_text(body + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "deployment_criterion": "HUMAN_SPECIFICATION_REQUIRED",
                "planner_ready": False,
                "status": "BLOCKED_ON_HUMAN_SPECIFICATION",
                "numeric_fields_filled": False,
                "auto_selected": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return spec


if __name__ == "__main__":
    run()
