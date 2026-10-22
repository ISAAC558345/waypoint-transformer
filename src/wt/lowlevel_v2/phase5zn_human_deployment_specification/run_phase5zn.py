"""Freeze the candidate specification. Does not run fresh confirmation."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from wt.htw_phase9.const import HOLD
from wt.lowlevel_v2.phase1.protocol import ALPHA_RESIDUAL
from wt.lowlevel_v2.phase5g_closed_loop.protocol import MAPPO_CKPT
from wt.lowlevel_v2.phase5k_joint_decoder.protocol import OUTPUT as PHASE5K_OUTPUT
from wt.lowlevel_v2.phase5l_joint_encoder.protocol import OUTPUT as PHASE5L_OUTPUT
from wt.lowlevel_v2.phase5l_joint_encoder.protocol import SCENARIOS as PHASE5L_SCENARIOS
from wt.lowlevel_v2.phase5l_joint_encoder.protocol import WIDTH
from wt.lowlevel_v2.phase5s_multiscenario_nonadditive.protocol import (
    CONFIRM_SCENARIOS,
    MODEL_POOL_WIDTH,
    TRAIN_SCENARIOS,
)
from wt.lowlevel_v2.phase5v_temporal_representation.protocol import CHECKPOINTS as TEMPORAL_CHECKPOINTS
from wt.lowlevel_v2.phase5zd_executor_response_value.protocol import CHECKPOINTS as VALUE_CHECKPOINTS
from wt.lowlevel_v2.phase5zd_executor_response_value.protocol import RESPONSE_FEATURES, SEEDS
from wt.lowlevel_v2.phase5zd_executor_response_value.protocol import OUTPUT as VALUE_OUTPUT
from wt.lowlevel_v2.phase5zm_multiseed_model_policy.consensus import rank_mean_select
from wt.lowlevel_v2.phase5zm_multiseed_model_policy.protocol import AGGREGATION, VALUE_SEEDS
from wt.lowlevel_v2.phase5zm_multiseed_model_policy.run_phase5zm import _consensus_once
from wt.lowlevel_v2.phase5zj_true_h1_acceleration.fast_path import shipped_h1
from wt.lowlevel_v2.protocol import FINAL_CKPT, HOLD_STEPS

from .protocol import (
    AGGREGATION_APPROVAL,
    AGGREGATION_RULE,
    DEPLOYMENT_LATENCY_REQUIREMENT,
    DEPLOYMENT_VERIFIED,
    FRESH_CONFIRMATION_AUTHORIZED,
    FRESH_POOL,
    FROZEN_FOR_FRESH_CONFIRMATION,
    MODEL_R1_AGGREGATION_APPROVED,
    PHASE,
    PHASE4E_REPORT,
    PHASE4E_STATUS,
    PLANNER_READY,
    RECOMMEND_APPROVAL,
    REPORT,
    RUNTIME_FALLBACK_POLICY,
    TIMEOUT_POLICY,
    choose_gate,
    dump_protocol,
)

PROTOTYPE_LATENCY = {"median_ms": 334.28, "p95_ms": 450.93, "p99_ms": 458.34, "source": "phase5zm_model_selection_gate.json"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file(path: Path) -> dict:
    resolved = Path(path)
    return {
        "path": str(resolved),
        "exists": resolved.is_file(),
        "sha256": _sha256(resolved) if resolved.is_file() else None,
    }


def _registries() -> dict:
    from wt.lowlevel_v2.phase5g_closed_loop.protocol import SCENARIOS as G
    from wt.lowlevel_v2.phase5i_joint_utility.protocol import SCENARIOS as I
    from wt.lowlevel_v2.phase5k_joint_decoder.protocol import SCENARIOS as K
    from wt.lowlevel_v2.phase5m_joint_planner.protocol import SCENARIOS as M
    from wt.lowlevel_v2.phase5n_joint_planner.protocol import SCENARIOS as N
    from wt.lowlevel_v2.phase5o_executor_joint_utility_audit.protocol import SCENARIOS as O
    from wt.lowlevel_v2.phase5p_executor_joint_planner.protocol import SCENARIOS as P
    from wt.lowlevel_v2.phase5q_encoder_scorer_audit.protocol import SCENARIOS as Q
    from wt.lowlevel_v2.phase5r_nonadditive_generalization.protocol import SCENARIOS as R
    from wt.lowlevel_v2.phase5t_scorer_failure_audit.protocol import SCENARIOS as T

    named = {
        "train_310100_310109": TRAIN_SCENARIOS,
        "confirm_310110_310117": CONFIRM_SCENARIOS,
        "phase5g": G,
        "phase5i": I,
        "phase5k": K,
        "phase5l": PHASE5L_SCENARIOS,
        "phase5m": M,
        "phase5n": N,
        "phase5o": O,
        "phase5p": P,
        "phase5q": Q,
        "phase5r": R,
        "phase5t": T,
    }
    fresh = set(FRESH_POOL)
    intersections = {name: sorted(fresh & set(values)) for name, values in named.items()}
    consumed = {name: items for name, items in intersections.items() if items}
    return {
        "checked_registries_only": True,
        "listed_data_directories": False,
        "loaded_snapshots": False,
        "checked_labels": False,
        "ran_rollout": False,
        "ran_inference": False,
        "pool": list(FRESH_POOL),
        "intersections": intersections,
        "consumed_registries": consumed,
        "integrity": "PASS" if not consumed else "FAIL",
    }


def _pipeline_consistent() -> tuple[bool, list[str]]:
    problems = []
    source = inspect.getsource(_consensus_once)
    if "shipped_h1(" not in source or "optimized_h1(" in source:
        problems.append("true_h1_call")
    if "rank_mean_select(" not in source or "build_task_option(" not in source:
        problems.append("consensus_or_taskoption_call")
    if AGGREGATION != AGGREGATION_RULE or tuple(VALUE_SEEDS) != tuple(SEEDS) or len(VALUE_SEEDS) != 5:
        problems.append("aggregation_or_seeds")
    if int(MODEL_POOL_WIDTH) != 3 or int(HOLD) != 15 or int(HOLD_STEPS) != 15:
        problems.append("topk_or_hold")
    if len(RESPONSE_FEATURES) != 14:
        problems.append("response_schema")
    values = np.asarray(
        [
            [1.0, 5.0],
            [5.0, 1.0],
            [5.0, 1.0],
            [5.0, 1.0],
            [5.0, 1.0],
        ],
        dtype=np.float64,
    )
    assignments = np.asarray([[0, -1], [-1, 0]], dtype=np.int64)
    selected = rank_mean_select(values, assignments)
    if selected["key"] != (0, -1) or selected["stage"] != "mean_rank":
        problems.append("rank_mean_rule")
    if shipped_h1.__name__ != "shipped_h1":
        problems.append("true_h1_symbol")
    return not problems, problems


def _manifest(consistent: bool) -> dict:
    root = Path(__file__).resolve().parents[4]
    components = {
        "stage1_encoder": _file(PHASE5L_OUTPUT / "joint_utility_encoder.pt"),
        "matching_decoder": _file(PHASE5K_OUTPUT / "matching_decoder.pt"),
        "official_deepsets_weights": _file(FINAL_CKPT),
        "mappo_checkpoint": _file(MAPPO_CKPT),
        "temporal_normalization_seed0": _file(TEMPORAL_CHECKPOINTS / "phase5v_temp_aug_seed0.pt"),
        "response_normalization": _file(VALUE_OUTPUT / "phase5zd_response_normalization.json"),
        "response_feature_spec": _file(VALUE_OUTPUT / "phase5zd_response_feature_spec.json"),
        "rank_mean_source": _file(root / "src" / "wt" / "lowlevel_v2" / "phase5zm_multiseed_model_policy" / "consensus.py"),
        "true_h1_source": _file(root / "src" / "wt" / "lowlevel_v2" / "phase5zj_true_h1_acceleration" / "fast_path.py"),
        "candidate_pool_source": _file(root / "src" / "wt" / "lowlevel_v2" / "phase5r_nonadditive_generalization" / "world.py"),
        "candidate_lists_source": _file(root / "src" / "wt" / "lowlevel_v2" / "phase5m_joint_planner" / "planner.py"),
        "exclusive_pool_source": _file(root / "src" / "wt" / "lowlevel_v2" / "phase5n_joint_planner" / "matchings.py"),
        "safety_projector_source": _file(root / "src" / "wt" / "lowlevel_v2" / "safety_projector.py"),
        "taskoption_source": _file(root / "src" / "wt" / "htw_phase100" / "contract.py"),
        "response_assembler_source": _file(root / "src" / "wt" / "lowlevel_v2" / "phase5zd_executor_response_value" / "response.py"),
        "value_scorer_source": _file(root / "src" / "wt" / "lowlevel_v2" / "phase5zd_executor_response_value" / "scorer.py"),
    }
    for seed in SEEDS:
        components[f"model_r1_seed_{seed}"] = _file(VALUE_CHECKPOINTS / f"phase5zd_response_seed{int(seed)}.pt")
    missing = [name for name, item in components.items() if not item["exists"]]
    payload = {
        "role": "candidate_final_architecture",
        "installed_in_official_planner": False,
        "modifiable_after_freeze": False,
        "encoder": "Phase-5L JointUtilityHGNN",
        "encoder_width": int(WIDTH),
        "candidate_generator": "stage-1 matching logits, per-UAV top-3, exclusive joint pool",
        "top_k": int(MODEL_POOL_WIDTH),
        "candidate_count": 120,
        "true_h1": "shipped_h1: original executor act, one clone, one environment step, no counterfactual rate grid",
        "true_h1_excludes_unshipped_batch_gemm": True,
        "model_r1_heads": 5,
        "aggregation": AGGREGATION_RULE,
        "aggregation_uses_j15": False,
        "aggregation_uses_scenario_performance": False,
        "aggregation_uses_seed_weights": False,
        "selects_best_seed": False,
        "tie_break": ["majority_top1_count", "lexicographic assignment tuple"],
        "rank_definition": "competition rank, rank 1 is best",
        "taskoption_schema": ["targets", "decision_time", "remaining_hold", "uav_mask", "buoy_mask", "association", "urgency"],
        "association_equals_targets": True,
        "urgency": None,
        "hold": int(HOLD_STEPS),
        "idle": -1,
        "official_deepsets_is_controller_encoder": False,
        "safety_projector": "SafetyProjector",
        "alpha_residual": float(ALPHA_RESIDUAL),
        "response_schema": list(RESPONSE_FEATURES),
        "components": components,
        "missing_components": missing,
        "pipeline_consistent": consistent and not missing,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    payload["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _approval_sheet() -> str:
    return "\n".join(
        [
            "# Human Approval Sheet",
            "",
            "### H1. Rank-Mean Consensus",
            "",
            "Recommended:",
            "",
            "APPROVE",
            "",
            "Human decision:",
            "",
            "[ ] APPROVE",
            "[ ] REJECT",
            "",
            "### H2. Max high-level inference latency",
            "",
            "Human decision:",
            "",
            "[ ] 500 ms",
            "[ ] 750 ms",
            "[ ] 1000 ms",
            "[ ] NOT_YET_SPECIFIED",
            "[ ] OTHER: ______",
            "",
            "### H3. Timeout policy",
            "",
            "Human decision:",
            "",
            "[ ] KEEP_CURRENT_TASKOPTION",
            "[ ] USE_LAST_VALID_ASSIGNMENT",
            "[ ] ABORT_NEW_DECISION",
            "[ ] OTHER: ______",
            "",
            "### H4. Runtime fallback",
            "",
            "Human decision:",
            "",
            "[ ] KEEP_CURRENT_VALID_TASKOPTION",
            "[ ] ABORT_DECISION",
            "[ ] OTHER: ______",
            "",
            "### H5. Fresh confirmation",
            "",
            "Pool:",
            "",
            "310118–310125",
            "",
            "Human decision:",
            "",
            "[ ] AUTHORIZE",
            "[ ] DO_NOT_AUTHORIZE",
            "",
        ]
    )


def _report(gate: dict, integrity: dict, manifest: dict) -> str:
    return "\n".join(
        [
            "# HTW Phase-5ZN：人工部署规范冻结",
            "",
            f"**Gate = {gate['gate']} {gate['gate_name']}**",
            "",
            "**planner_ready = False**",
            "",
            "**deployment_verified = False**",
            "",
            "**FROZEN_FOR_FRESH_CONFIRMATION = False**",
            "",
            "**fresh_confirmation_authorized = False**",
            "",
            "本阶段没有改算法，没有训练，没有 rollout，没有打开 310118–310125。Phase-4E 仍是 DEPLOYMENT_VERIFICATION_FAILED。",
            "",
            "## 候选架构",
            "",
            "高层是：当前状态、冻结的 Phase-5L encoder 和 Stage-1 top-3 候选、120 个合法联合分配、shipped optimized True-H1 的一步响应、五个冻结 Model-R1、Rank-Mean Consensus V1、TaskOption。",
            "",
            "低层是：TaskOption、冻结 Residual MAPPO、SafetyProjector、环境执行。低层不重新选择浮标。",
            "",
            "True-H1 是价值评估器。它不是第二个 planner，不是轨迹 planner，也不是 15 步 rollout。未交付的 batch GEMM 不在冻结结构里。正式 DeepSets 权重被登记并保持不动，但它不是这条候选控制器的 encoder。",
            "",
            f"清单哈希是 `{manifest['manifest_sha256']}`。缺失组件数是 {len(manifest['missing_components'])}。",
            "",
            "## H1. 聚合规则",
            "",
            "技术建议是批准 RANK_MEAN_CONSENSUS_V1。理由只来自 Phase-5ZM：seed 和 candidate permutation invariance 都是 1.0，24 个已消费状态里 23 个五 seed 一致、1 个是 4/5，Consensus 的 selected J15 仍是 2.7917e6 bit，相对 Model-S 仍是 +4.5000e5 bit。选择不使用 J15、场景收益或 seed 权重。",
            "",
            "model_r1_aggregation_approved 仍然是 false。批准状态是 HUMAN_APPROVAL_REQUIRED。RECOMMEND_APPROVAL = true 只是建议，不是批准。",
            "",
            "## H2. 时延",
            "",
            "原型测量是 median 334.28 ms、P95 450.93 ms、P99 458.34 ms。deployment_latency_requirement = HUMAN_SPECIFICATION_REQUIRED。",
            "",
            "参考选项是 500 ms、750 ms、1000 ms 和 NOT_YET_SPECIFIED。本阶段没有选择。500 ms 只是当前计算平台上略高于已消费开发状态的 P99，不是真实 UAV 部署 SLA。",
            "",
            "## H3. 超时",
            "",
            "timeout_policy = HUMAN_SPECIFICATION_REQUIRED。",
            "",
            "当前控制器在 hold 计数到 0 时发布新 option，并把 hold 设为 15；每个已执行步把计数减 1。",
            "",
            "T1 KEEP_CURRENT_TASKOPTION 不发布新分配，不把 hold 重新设为 15，沿用当前合法 option，并按现有规则递减 remaining_hold。它最接近现有递减语义，但在 hold 已经到期的决策边界上，现有控制器会开始新的 15 步窗口，所以 T1 仍需人工批准。",
            "",
            "T2 USE_LAST_VALID_ASSIGNMENT 会把上一次合法分配重新包成 TaskOption。按现有构造方式，这会重新打开任务窗口，因此不是默认安全等价行为。",
            "",
            "T3 ABORT_NEW_DECISION 不改变当前 executor 状态，也不发布新结果。它和 T1 的差别是：T1 仍允许 hold 计数按原规则递减，T3 连 executor 状态也不改。",
            "",
            "## H4. 运行时回退",
            "",
            "runtime_fallback_policy = HUMAN_SPECIFICATION_REQUIRED。",
            "",
            "True-H1 异常、Model-R1 的 NaN 或 Inf、空候选池、非法联合分配、SafetyProjector 失败、checkpoint 缺失、TaskOption 构造失败，都没有被批准的自动 fallback。DeepSets、B2、Oracle、单个 R1 seed、Phase-5L 直接分配和 fly-toward 都不是 fallback。",
            "",
            "如果已经存在一个合法 TaskOption，KEEP_CURRENT_VALID_TASKOPTION 是非灾难性推理失败时最小的候选行为。如果当前没有合法 TaskOption，状态是 NO_VALID_FALLBACK_DEFINED。本阶段不补一条新策略。",
            "",
            "SafetyProjector 失败时，冻结执行器内部会把动作置零。那是已经冻结的执行器行为，不是新的部署 fallback。",
            "",
            "## H5. Fresh confirmation",
            "",
            f"预留池完整性是 {integrity['integrity']}。检查只读了协议里的场景登记，没有列数据目录，没有加载 snapshot，没有读标签，没有 rollout，也没有 inference。",
            "",
            "310100–310109 是训练登记，310110–310117 是已消费确认登记。310118–310125 与这些登记没有交集，因此没有参与训练、checkpoint 选择、损失设计、候选设计、阈值调整、Rank-Mean 选择、True-H1 优化或时延优化。",
            "",
            "fresh_confirmation_authorized = false。authorization_source = HUMAN_APPROVAL_REQUIRED。",
            "",
            "## 未来确认草案",
            "",
            "只有人工授权之后，310118–310125 才能按这份冻结系统一次性评估。不允许根据前面几个 scenario 改模型、聚合、超时、候选、seed 或 checkpoint。",
            "",
            "主比较是冻结的 Rank-Mean True-H1 控制器对冻结 Model-S。Oracle 不是部署基线。正式 DeepSets 只有在不生成新数据、并且已有产物能给出同一 payload 时才能作为次比较；不能为了补齐它而重新 rollout。",
            "",
            "成功标准草案只要求平均配对 payload 改进大于 0，并同时报告场景聚类 95% 区间、中位数和改进的正、平、负数量。同时要求非法分配、状态污染、崩溃和最终决策的 NaN/Inf 都是 0。没有规定必须达到某个百分比。时延门槛写成 LATENCY_GATE_PENDING_HUMAN_SPECIFICATION。",
            "",
            "这些标准没有被批准，也不能用 310118–310125 来调整。",
            "",
        ]
    )


def _dump(name: str, payload: dict) -> None:
    (PHASE / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    dump_protocol()
    if not PHASE4E_REPORT.is_file() or "DEPLOYMENT_VERIFICATION_FAILED" not in PHASE4E_REPORT.read_text(encoding="utf-8"):
        raise RuntimeError("phase-4e failure status is missing")
    integrity = _registries()
    consistent, problems = _pipeline_consistent()
    manifest = _manifest(consistent)
    freeze_ok = bool(manifest["pipeline_consistent"]) and not problems
    gate = choose_gate(
        integrity_pass=integrity["integrity"] == "PASS",
        freeze_consistent=freeze_ok,
        human_approval_pending=True,
    )
    _dump("phase5zn_algorithm_freeze_manifest.json", manifest)
    _dump(
        "phase5zn_latency_reference.json",
        {
            "prototype_latency": PROTOTYPE_LATENCY,
            "deployment_latency_requirement": DEPLOYMENT_LATENCY_REQUIREMENT,
            "options_not_selected": ["500 ms", "750 ms", "1000 ms", "NOT_YET_SPECIFIED"],
            "selected_option": None,
            "note_500ms": "500 ms is only slightly above the consumed development P99 on this computer. It is not a UAV deployment SLA.",
        },
    )
    _dump(
        "phase5zn_timeout_policy_options.json",
        {
            "timeout_policy": TIMEOUT_POLICY,
            "selected_option": None,
            "current_rule": "A new option is published when the hold counter reaches zero and the counter is set to 15. Each executed step decrements the counter by 1.",
            "options": {
                "KEEP_CURRENT_TASKOPTION": {
                    "publishes_new_assignment": False,
                    "restarts_hold_15": False,
                    "decrements_existing_hold": True,
                    "calls_backup_planner": False,
                    "compatible_with_decrement_rule": True,
                    "identical_at_expired_hold_boundary": False,
                },
                "USE_LAST_VALID_ASSIGNMENT": {
                    "rebuilds_taskoption": True,
                    "restarts_task_window": True,
                    "default_safe_equivalent": False,
                },
                "ABORT_NEW_DECISION": {
                    "changes_executor_state": False,
                    "publishes_new_assignment": False,
                    "same_as_keep_current": False,
                },
            },
        },
    )
    _dump(
        "phase5zn_runtime_fallback_options.json",
        {
            "runtime_fallback_policy": RUNTIME_FALLBACK_POLICY,
            "selected_option": None,
            "cases": [
                "true_h1_candidate_evaluation_exception",
                "model_r1_nan",
                "model_r1_inf",
                "candidate_pool_empty",
                "invalid_joint_assignment",
                "safety_projector_failure",
                "checkpoint_missing",
                "taskoption_construction_failure",
            ],
            "forbidden_fallbacks": [
                "deepsets_direct_assignment",
                "b2",
                "oracle",
                "single_r1_seed",
                "phase5l_direct_hgnn",
                "fly_toward",
            ],
            "non_catastrophic_candidate": "KEEP_CURRENT_VALID_TASKOPTION",
            "when_no_current_legal_option": "NO_VALID_FALLBACK_DEFINED",
            "safety_projector_zero_action": "frozen executor behavior, not a new deployment fallback",
        },
    )
    _dump("phase5zn_fresh_pool_integrity.json", integrity)
    _dump(
        "phase5zn_fresh_confirmation_protocol_draft.json",
        {
            "status": "draft_not_authorized",
            "pool": "310118-310125",
            "execution": "one_shot_frozen_system_after_human_authorization",
            "forbidden_after_seeing_results": [
                "modify_model",
                "modify_aggregation",
                "modify_timeout",
                "modify_candidates",
                "select_seed",
                "select_checkpoint",
            ],
            "metrics": [
                "cumulative_payload",
                "residual_queue",
                "completion",
                "normalized_regret",
                "selected_j15_or_realized_payload",
                "assignment_legality",
                "conflict_rate",
                "sinr",
                "inference_latency",
                "seed_consensus_strength",
            ],
            "primary_comparison": "frozen_rank_mean_true_h1_versus_frozen_model_s",
            "secondary_comparison": "official_deepsets_only_from_existing_artifacts_without_a_new_rollout",
            "oracle_is_deployment_baseline": False,
            "performance_criterion_draft": "mean_paired_payload_improvement > 0",
            "also_report": ["scenario_cluster_95_ci", "median_improvement", "positive_tie_negative_scenario_count"],
            "hard_conditions_draft": {
                "illegal_assignment": 0,
                "state_contamination": 0,
                "crash": 0,
                "nan_inf_final_decision": 0,
            },
            "percent_improvement_threshold": None,
            "latency_gate": "LATENCY_GATE_PENDING_HUMAN_SPECIFICATION",
            "criterion_approved": False,
        },
    )
    (PHASE / "phase5zn_human_approval_sheet.md").write_text(_approval_sheet(), encoding="utf-8")
    _dump(
        "phase5zn_human_specification_gate.json",
        {
            "gate": gate["gate"],
            "gate_name": gate["gate_name"],
            "aggregation_rule": AGGREGATION_RULE,
            "model_r1_aggregation_approved": MODEL_R1_AGGREGATION_APPROVED,
            "model_r1_aggregation_approval": AGGREGATION_APPROVAL,
            "recommend_approval": RECOMMEND_APPROVAL,
            "deployment_latency_requirement": DEPLOYMENT_LATENCY_REQUIREMENT,
            "timeout_policy": TIMEOUT_POLICY,
            "runtime_fallback_policy": RUNTIME_FALLBACK_POLICY,
            "fresh_pool_integrity": integrity["integrity"],
            "fresh_confirmation_authorized": FRESH_CONFIRMATION_AUTHORIZED,
            "authorization_source": "HUMAN_APPROVAL_REQUIRED",
            "manifest_sha256": manifest["manifest_sha256"],
            "missing_components": manifest["missing_components"],
            "consistency_problems": problems,
            "planner_ready": PLANNER_READY,
            "deployment_verified": DEPLOYMENT_VERIFIED,
            "frozen_for_fresh_confirmation": FROZEN_FOR_FRESH_CONFIRMATION,
            "phase4e_status": PHASE4E_STATUS,
            "phase4e_overwritten": False,
            "opens_310118_310125": False,
            "deployment_pass": False,
        },
    )
    REPORT.write_text(_report(gate, integrity, manifest), encoding="utf-8")
    print(
        json.dumps(
            {
                "gate": gate["gate"],
                "integrity": integrity["integrity"],
                "missing": manifest["missing_components"],
                "problems": problems,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
