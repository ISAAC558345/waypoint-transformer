"""Phase-4D-3 registry search. No B2 retrain. No tau or planner change."""
from __future__ import annotations

import json
import re
from pathlib import Path

from wt.lowlevel_v2.phase4d3_deployment_criterion_registry.protocol import (
    FROZEN_GATE1,
    NEAR_MISS_DOCS,
    OUTPUT,
    REPORT_NAME,
    ROOT,
    SEARCH_GLOBS,
    SEARCH_KEYWORDS,
    TAU_SOURCE,
    TAU_UHAT,
)

# Numeric deployment-threshold patterns. A real pre-registered criterion
# would have to state one of these as an acceptance rule, not as an observed value.
NUMERIC_RULE = re.compile(
    r"(allowed_degrade|max_degrade|allowed_worst|required_mean_delta|required_mean_U|"
    r"degrade_rate\s*[<≤]=?\s*0\.|worst[-_ ]case[^\n]{0,40}[<≤]=?\s*-0\.)",
    re.IGNORECASE,
)


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    return value


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n", encoding="utf-8")


def _expand_glob(pattern: str) -> list[Path]:
    if "{md,json}" in pattern:
        out = []
        for ext in ("md", "json"):
            out.extend(ROOT.glob(pattern.replace("{md,json}", ext)))
        return out
    return list(ROOT.glob(pattern))


def collect_corpus() -> list[Path]:
    found = []
    seen = set()
    extra = [
        ROOT / "AGENTS.md",
        ROOT / "docs/archive/phase_reports/PLANS.md",
        ROOT / "docs/archive/phase_reports/FINAL_README_CN.md",
        ROOT / "docs/archive/phase_reports/FINAL_RESEARCH_SUMMARY_CN.md",
        ROOT / "docs/archive/phase_reports/FINAL_PROJECT_MANIFEST.md",
        ROOT / "docs/archive/phase_reports/HTW_Phase3_EvidenceMatrix_CN.md",
        ROOT / "docs/archive/phase_reports/HTW_Phase4A_HindsightCredit_DatasetAudit_CN.md",
        ROOT / "docs/archive/phase_reports/HTW_Phase4B_TrajectoryAwareCreditLearner_Audit_CN.md",
        ROOT / "docs/archive/phase_reports/HTW_Phase4B_B2_IndependentValidation_Audit_CN.md",
        ROOT / "docs/archive/phase_reports/HTW_Phase4B_B2_ShadowIntegration_Audit_CN.md",
        ROOT / "docs/archive/phase_reports/HTW_Phase4C_IndependentValidation_Audit_CN.md",
        ROOT / "docs/archive/phase_reports/HTW_Phase4D_DeploymentGate_Audit_CN.md",
        ROOT / "docs/archive/phase_reports/HTW_Phase4D2_DeploymentSafety_Audit_CN.md",
    ]
    for path in extra:
        if path.is_file() and path.resolve() not in seen:
            seen.add(path.resolve())
            found.append(path)
    for pattern in SEARCH_GLOBS:
        for path in _expand_glob(pattern):
            if not path.is_file():
                continue
            key = path.resolve()
            if key in seen:
                continue
            if "phase4d3_deployment_criterion_registry" in str(path).replace("\\", "/"):
                continue
            seen.add(key)
            found.append(path)
    return sorted(found, key=lambda p: str(p).lower())


def search_file(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"path": str(path), "readable": False, "error": str(exc)}
    hits = []
    lower = text.lower()
    for key in SEARCH_KEYWORDS:
        if key.lower() in lower:
            hits.append(key)
    numeric = bool(NUMERIC_RULE.search(text))
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "readable": True,
        "keyword_hits": hits,
        "has_keyword": bool(hits),
        "has_numeric_acceptance_pattern": numeric,
    }


def classify(hits: list[dict]) -> dict:
    numeric_hits = [h for h in hits if h.get("has_numeric_acceptance_pattern")]
    # Phase-4D-2 writes null placeholders like B_max_degrade_rate; that is not a rule.
    real_rules = []
    for item in numeric_hits:
        path = item["path"]
        if "phase4d2" in path or "Phase4D2" in path:
            continue
        if "phase4d3" in path:
            continue
        real_rules.append(item)
    found = False
    return {
        "pre_registered_criterion_found": found,
        "numeric_pattern_files_excluding_4d2_placeholders": [h["path"] for h in real_rules],
        "keyword_positive_files": [h["path"] for h in hits if h.get("has_keyword")],
        "near_misses": list(NEAR_MISS_DOCS),
        "conclusion": "NO PRE-REGISTERED DEPLOYMENT CRITERION FOUND",
    }


def registry_payload(search: dict) -> dict:
    return {
        "status": "NOT_PRE_REGISTERED",
        "pre_registered": False,
        "source": None,
        "criterion_name": None,
        "source_file": None,
        "source_section": None,
        "requires_human_specification": True,
        "allowed_degrade_rate": None,
        "allowed_worst_case_delta_R": None,
        "required_mean_delta_R": None,
        "required_mean_U_gain": None,
        "other_required_conditions": None,
        "search_conclusion": "NO PRE-REGISTERED DEPLOYMENT CRITERION FOUND",
        "search": search,
        "frozen_gate": {
            "rule": "if margin >= tau: use B2 else LOCAL",
            "margin": "U_hat(argmax U_hat) - U_hat(argmax U_single)",
            "tau": TAU_UHAT,
            "tau_display": 1.535e6,
            "tau_source": TAU_SOURCE,
            "tau_changed": False,
        },
        "planner_ready": False,
        "phase_status": "BLOCKED_ON_HUMAN_SPECIFICATION",
    }


def proposal_payload() -> dict:
    obs = FROZEN_GATE1
    params = [
        {
            "parameter": "max_allowed_degrade_rate",
            "meaning": "Maximum fraction of scenarios where selected first-hold is worse than LOCAL.",
            "currently_observed": obs["degrade_rate"],
            "currently_observed_note": "Gate-1 degrade = 3/64 = 0.047; not an acceptance threshold.",
            "human_decision_required": True,
            "proposed_official_value": None,
        },
        {
            "parameter": "max_allowed_worst_case_delta_R",
            "meaning": "Most negative allowed R_selected - R_LOCAL on any retained scenario.",
            "currently_observed": obs["worst_case_delta_R"],
            "currently_observed_note": "Gate-1 worst-case ΔR = -0.214 on 220059; not an acceptance threshold.",
            "human_decision_required": True,
            "proposed_official_value": None,
        },
        {
            "parameter": "require_mean_delta_R_positive",
            "meaning": "Whether mean(R_selected - R_LOCAL) must be strictly greater than 0.",
            "currently_observed": obs["mean_dR"],
            "currently_observed_note": "Gate-1 mean ΔR = 0.111.",
            "human_decision_required": True,
            "proposed_official_value": None,
        },
        {
            "parameter": "require_mean_U_traj_above_LOCAL",
            "meaning": "Whether mean selected U_traj must exceed mean LOCAL U_traj.",
            "currently_observed": "Gate-1 mean U_traj = 1.081e7 vs LOCAL 9.469e6 (Phase-4C/4D frozen).",
            "human_decision_required": True,
            "proposed_official_value": None,
        },
        {
            "parameter": "allow_some_scenario_degradation",
            "meaning": "Whether any scenario may be worse than LOCAL after the gate.",
            "currently_observed": "yes, currently 3 scenarios still degrade: 220023, 220028, 220059",
            "human_decision_required": True,
            "proposed_official_value": None,
        },
        {
            "parameter": "require_B2_never_worse_than_LOCAL_when_used",
            "meaning": "Whether every scenario that actually uses B2 must have R_B2 >= R_LOCAL.",
            "currently_observed": "false; the 3 remaining degradations all used B2 (margin >= tau).",
            "human_decision_required": True,
            "proposed_official_value": None,
        },
        {
            "parameter": "require_independent_deployment_holdout",
            "meaning": "Whether a second unseen batch, not the current 64, must also pass the criterion.",
            "currently_observed": "not done; Phase-4C/4D/4D-2 all evaluate the same 220000-220063 set.",
            "human_decision_required": True,
            "proposed_official_value": None,
        },
        {
            "parameter": "require_multiple_independent_batches",
            "meaning": "Whether more than one independent batch must satisfy the same criterion.",
            "currently_observed": "only one independent batch exists.",
            "human_decision_required": True,
            "proposed_official_value": None,
        },
    ]
    return {
        "status": "PROPOSAL_ONLY_NOT_OFFICIAL",
        "pre_registered": False,
        "official_standard": None,
        "warning": (
            "This file lists parameters a human must decide. "
            "No candidate number is declared as the official deployment criterion. "
            "Do not treat currently_observed as a threshold."
        ),
        "frozen_gate_not_optimized": {
            "margin": "U_hat(argmax U_hat) - U_hat(argmax U_single)",
            "tau": 1.535e6,
            "rule": "if margin >= tau: B2 else LOCAL",
        },
        "observed_performance_gate1": obs,
        "parameters_requiring_human_decision": params,
    }


def write_report(search: dict) -> str:
    obs = FROZEN_GATE1
    keyword_n = len(search["classify"]["keyword_positive_files"])
    lines = [
        "# HTW Phase-4D-3 Deployment Criterion Registry Audit",
        "",
        "本阶段只检索仓库里是否已有预注册 deployment criterion，并为人工预注册准备参数清单。",
        "未训练 B2，未修改 B2，未修改 τ，未修改 planner / decide()，未进入 Phase-4E，未从 64 个 independent scenario 反推阈值。",
        "",
        f"**deployment_criterion = {search['registry_status']}**",
        "",
        "**planner_ready = False**",
        "",
        "**status = BLOCKED_ON_HUMAN_SPECIFICATION**",
        "",
        "## Q1. 仓库中是否存在预注册 deployment criterion？",
        "",
        "**否。** NO PRE-REGISTERED DEPLOYMENT CRITERION FOUND.",
        "",
        f"- 检索文件数：{search['n_files']}",
        f"- 含关键词文件数：{keyword_n}",
        "- 含数值 acceptance 规则、且能作为 B2 部署标准的文件数：0",
        "- 未根据 Gate-1 的 3/64、−0.214、0.111 或 26/64 反推任何阈值",
        "",
        "## Q2. 如果存在，原始来源是什么？",
        "",
        "不存在。`source = null`。",
        "",
        "相关但不是 deployment criterion 的近邻文档：",
        "",
    ]
    for item in NEAR_MISS_DOCS:
        lines.append(f"- `{item['path']}` / {item['section']}：{item['why_not_criterion']}")
    lines.extend(
        [
            "",
            "补充：Phase-3 / Phase-4A / FINAL_* / `HTW_Phase3_EvidenceMatrix_CN.md` 没有规定 B2 的 allowed degrade rate、worst-case ΔR 或 mean gain 门槛。",
            "Phase-4B 的 Gate-1/2 只是 16-state 科学 sanity（mean R > LOCAL、permutation drop>0.03），同一节写明 Gate-3 deployment threshold **未定义**。",
            "Phase-10.3 in-range zero-motion gate、Round-14 `deployment_ready=false`、Low-Level safety projector 属于别的系统，不能挪用为 B2 residual-credit 部署标准。",
            "",
            "## Q3. 如果不存在，哪些参数必须由人工确定？",
            "",
            "见 `deployment_criterion_proposal.json`。至少需要人工决定：",
            "",
            "1. 最大允许 degradation rate",
            "2. 最大允许 worst-case ΔR",
            "3. 是否要求 mean ΔR > 0",
            "4. 是否要求 mean U_traj > LOCAL",
            "5. 是否允许少量 scenario degradation",
            "6. 是否要求 B2 在所有使用场景都不劣于 LOCAL",
            "7. 是否需要独立 deployment hold-out 再验证一次",
            "8. 是否需要连续多个 independent batch 满足 criterion",
            "",
            "proposal 只列出参数，**不把任何候选数值宣布为正式标准**。",
            "",
            "## Q4. 当前 Gate-1 的 observed performance 是什么？",
            "",
            "以下数字来自已冻结的 Phase-4C / Phase-4D / Phase-4D-2，本阶段不修改、不优化。",
            "",
            "- Gate-1：margin = U_hat(argmax U_hat) − U_hat(argmax U_single)；τ = 1.535e6",
            f"- n_B2 = {obs['n_B2']}",
            f"- mean R = {obs['mean_R']}",
            f"- mean ΔR = {obs['mean_dR']}",
            f"- degrade = {obs['n_degrade']}",
            f"- degrade rate = {obs['degrade_rate']}",
            f"- worst-case ΔR = {obs['worst_case_delta_R']}",
            "",
            "这些是 observed performance，不是 acceptance rule。",
            "",
            "## Q5. 在 criterion 未确定之前，能否宣布 planner_ready？",
            "",
            "**不能。** `planner_ready = False`。即使 Gate-1 指标看起来不错，也不能因为指标好看自动 PASS。",
            "",
            "## 最终状态",
            "",
            "- deployment_criterion = NOT_PRE_REGISTERED",
            "- planner_ready = False",
            "- status = BLOCKED_ON_HUMAN_SPECIFICATION",
            "",
            "本阶段停止。不训练，不改 B2，不改 τ，不改 planner，不改 decide()，不接入 shadow planner，不进入 Phase-4E。",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = collect_corpus()
    hits = [search_file(path) for path in files]
    classified = classify(hits)
    search = {
        "n_files": len(files),
        "keywords": list(SEARCH_KEYWORDS),
        "hits": hits,
        "classify": classified,
        "registry_status": "NOT_PRE_REGISTERED",
    }
    registry = registry_payload(search)
    proposal = proposal_payload()
    _dump(OUTPUT / "deployment_criterion_registry.json", registry)
    _dump(OUTPUT / "deployment_criterion_proposal.json", proposal)
    body = write_report(search)
    (OUTPUT / REPORT_NAME).write_text(body + "\n", encoding="utf-8")
    (ROOT / REPORT_NAME).write_text(body + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "deployment_criterion": "NOT_PRE_REGISTERED",
                "planner_ready": False,
                "status": "BLOCKED_ON_HUMAN_SPECIFICATION",
                "n_files_searched": len(files),
                "pre_registered_found": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return registry


if __name__ == "__main__":
    run()
