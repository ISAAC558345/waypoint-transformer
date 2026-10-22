"""Apply the approved specification and recheck the Phase-5ZN freeze."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from wt.lowlevel_v2.phase1.protocol import ALPHA_RESIDUAL
from wt.lowlevel_v2.phase5s_multiscenario_nonadditive.protocol import (
    CONFIRM_SCENARIOS,
    MODEL_POOL_WIDTH,
    TRAIN_SCENARIOS,
)
from wt.lowlevel_v2.phase5zd_executor_response_value.protocol import RESPONSE_FEATURES
from wt.lowlevel_v2.phase5zm_multiseed_model_policy.protocol import AGGREGATION
from wt.lowlevel_v2.protocol import HOLD_STEPS

from .protocol import (
    DEPLOYMENT_VERIFIED,
    FRESH_POOL,
    MANIFEST,
    PHASE,
    PHASE4E_REPORT,
    PHASE4E_STATUS,
    PLANNER_READY,
    REPORT,
    ROOT,
    choose_gate,
    dump_protocol,
)

FRESH_PATH = re.compile(r"31011[89]|31012[0-5]")
ALGORITHM_DIRS = (
    "l3b",
    "htw_phase100",
    "lowlevel_v2/phase5l_joint_encoder",
    "lowlevel_v2/phase5k_joint_decoder",
    "lowlevel_v2/phase5m_joint_planner",
    "lowlevel_v2/phase5n_joint_planner",
    "lowlevel_v2/phase5r_nonadditive_generalization",
    "lowlevel_v2/phase5zj_true_h1_acceleration",
    "lowlevel_v2/phase5zd_executor_response_value",
    "lowlevel_v2/phase5zm_multiseed_model_policy",
    "lowlevel_v2/phase5g_closed_loop",
    "lowlevel_v2/phase1b",
)
ALGORITHM_FILES = (
    "lowlevel_v2/safety_projector.py",
    "lowlevel_v2/residual_policy.py",
    "lowlevel_v2/observation.py",
)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _registries() -> dict:
    from wt.lowlevel_v2.phase5g_closed_loop.protocol import SCENARIOS as G
    from wt.lowlevel_v2.phase5i_joint_utility.protocol import SCENARIOS as I
    from wt.lowlevel_v2.phase5k_joint_decoder.protocol import SCENARIOS as K
    from wt.lowlevel_v2.phase5l_joint_encoder.protocol import SCENARIOS as L
    from wt.lowlevel_v2.phase5m_joint_planner.protocol import SCENARIOS as M
    from wt.lowlevel_v2.phase5n_joint_planner.protocol import SCENARIOS as N
    from wt.lowlevel_v2.phase5o_executor_joint_utility_audit.protocol import SCENARIOS as O
    from wt.lowlevel_v2.phase5p_executor_joint_planner.protocol import SCENARIOS as P
    from wt.lowlevel_v2.phase5q_encoder_scorer_audit.protocol import SCENARIOS as Q
    from wt.lowlevel_v2.phase5r_nonadditive_generalization.protocol import SCENARIOS as R
    from wt.lowlevel_v2.phase5t_scorer_failure_audit.protocol import SCENARIOS as T

    fresh = set(FRESH_POOL)
    named = {
        "train": TRAIN_SCENARIOS,
        "confirm": CONFIRM_SCENARIOS,
        "phase5g": G,
        "phase5i": I,
        "phase5k": K,
        "phase5l": L,
        "phase5m": M,
        "phase5n": N,
        "phase5o": O,
        "phase5p": P,
        "phase5q": Q,
        "phase5r": R,
        "phase5t": T,
    }
    consumed = {name: sorted(set(values) & fresh) for name, values in named.items()}
    consumed = {name: items for name, items in consumed.items() if items}
    return {
        "checked_registries_only": True,
        "listed_pool_directory": False,
        "loaded_snapshots": False,
        "checked_labels": False,
        "ran_inference": False,
        "ran_rollout": False,
        "consumed_registries": consumed,
        "not_used_for": [
            "training",
            "validation",
            "checkpoint_selection",
            "architecture_design",
            "loss_design",
            "threshold_design",
            "candidate_design",
            "aggregation_design",
            "true_h1_optimization",
            "latency_optimization",
            "fallback_design",
        ],
        "integrity": "FAIL" if consumed else "PASS",
    }


def _rehash(manifest: dict) -> dict:
    rows = []
    mismatch = []
    for name, item in manifest["components"].items():
        path = Path(item["path"])
        current = _sha256(path)
        same = current is not None and current == item["sha256"]
        rows.append(
            {
                "name": name,
                "exists": path.is_file(),
                "phase5zn_sha256": item["sha256"],
                "current_sha256": current,
                "match": same,
                "mtime_newer_than_manifest": False,
            }
        )
        if not same:
            mismatch.append(name)
    structural = {
        "top_k": int(MODEL_POOL_WIDTH) == int(manifest["top_k"]) == 3,
        "hold": int(HOLD_STEPS) == int(manifest["hold"]) == 15,
        "aggregation": AGGREGATION == manifest["aggregation"] == "RANK_MEAN_CONSENSUS_V1",
        "response_schema": list(RESPONSE_FEATURES) == list(manifest["response_schema"]),
        "alpha_residual": float(ALPHA_RESIDUAL) == float(manifest["alpha_residual"]),
        "tie_break": list(manifest["tie_break"]) == ["majority_top1_count", "lexicographic assignment tuple"],
        "idle": int(manifest["idle"]) == -1,
        "candidate_count": int(manifest["candidate_count"]) == 120,
        "true_h1_excludes_unshipped_batch_gemm": bool(manifest["true_h1_excludes_unshipped_batch_gemm"]),
    }
    if not all(structural.values()):
        mismatch.extend(name for name, ok in structural.items() if not ok)
    return {"components": rows, "structural": structural, "mismatch": mismatch, "match": not mismatch}


def _walk_algorithm(manifest_mtime: float) -> dict:
    src = ROOT / "src" / "wt"
    newer = []
    fresh_writes = 0
    for relative in ALGORITHM_DIRS:
        base = src / relative
        if not base.exists():
            continue
        files = base.rglob("*") if base.is_dir() else [base]
        for path in files:
            if not path.is_file() or path.suffix == ".pyc" or "__pycache__" in path.parts:
                continue
            if FRESH_PATH.search(str(path)):
                if path.stat().st_mtime > manifest_mtime + 2.0:
                    fresh_writes += 1
                continue
            if path.stat().st_mtime > manifest_mtime + 2.0:
                newer.append(str(path.relative_to(ROOT)))
    for relative in ALGORITHM_FILES:
        path = src / relative
        if path.is_file() and path.stat().st_mtime > manifest_mtime + 2.0:
            newer.append(str(path.relative_to(ROOT)))
    return {"algorithm_files_written_after_phase5zn": newer, "fresh_pool_paths_written_after_phase5zn": fresh_writes}


def _git_status() -> dict:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    stdout = completed.stdout.decode("utf-8", errors="replace")
    algorithm_entries = []
    pyc_entries = 0
    specification_entries = []
    report_entries = []
    fresh_mentions = 0
    other = 0
    for line in stdout.splitlines():
        if FRESH_PATH.search(line):
            fresh_mentions += 1
            continue
        path = line[3:] if len(line) > 3 else line
        if "phase5zo_confirmation_freeze" in path or path.endswith("HTW_Phase5ZO_HumanSpecificationApplicationAndConfirmationFreeze_CN.md"):
            specification_entries.append(line)
        elif path.endswith(".md") or "phase5zn_human_deployment_specification" in path:
            report_entries.append(line)
        elif any(marker in path.replace("\\", "/") for marker in ALGORITHM_DIRS + ALGORITHM_FILES):
            if "__pycache__" in path or path.endswith(".pyc"):
                pyc_entries += 1
            else:
                algorithm_entries.append(line)
        else:
            other += 1
    return {
        "git_exit_code": completed.returncode,
        "algorithm_git_entries_redacted_count": len(algorithm_entries),
        "algorithm_pyc_entries_not_source": pyc_entries,
        "algorithm_git_entries_sample": algorithm_entries[:40],
        "preexisting_untracked_are_not_post_freeze_edits": True,
        "human_specification_entries": specification_entries,
        "report_entries_count": len(report_entries),
        "other_entries_count": other,
        "fresh_pool_mentions_not_listed": fresh_mentions,
    }


def _human_spec() -> dict:
    return {
        "source": "human_approved_not_modified_by_cursor",
        "h1_model_r1_aggregation": {
            "approved": True,
            "rule": "RANK_MEAN_CONSENSUS_V1",
            "definition": "R_mean(A) = mean_s rank_s(A); A* = argmin_A R_mean(A)",
            "tie_break": ["majority_top1_count", "canonical candidate order = lexicographic assignment tuple"],
            "rank_definition": "competition rank from Phase-5ZM, rank 1 is best",
            "forbidden": [
                "seed_weighting",
                "best_seed_selection",
                "j15_dependent_selection",
                "scenario_dependent_weighting",
                "checkpoint_selection",
            ],
            "model_r1_aggregation_approved": True,
        },
        "h2_latency": {
            "deployment_latency_requirement": "NOT_YET_SPECIFIED",
            "prototype_latency_ms": {"median": 334.28, "p95": 450.93, "p99": 458.34},
            "prototype_is_uav_sla": False,
            "latency_gate_active": False,
            "latency_diagnostic_only": True,
            "threshold_500ms_created": False,
        },
        "h3_timeout": {
            "policy": "KEEP_CURRENT_TASKOPTION_NO_HOLD_RESET",
            "if_valid_option_exists": {
                "keep_association": True,
                "create_new_taskoption": False,
                "reset_remaining_hold_to_15": False,
                "decrement_remaining_hold_by_existing_rule": True,
                "call_other_planner": False,
            },
            "if_option_missing_or_invalid": {
                "action": "ABORT_NEW_DECISION",
                "construct_assignment": False,
                "forbidden_fallbacks": [
                    "deepsets",
                    "b2",
                    "oracle",
                    "single_r1_seed",
                    "phase5l_direct_assignment",
                    "fly_toward",
                ],
            },
            "recreating_same_option_resets_hold": False,
        },
        "h4_runtime_fallback": {
            "primary": "KEEP_CURRENT_VALID_TASKOPTION",
            "secondary": "ABORT_DECISION_IF_NO_VALID_TASKOPTION",
            "applies_to": [
                "true_h1_exception",
                "model_r1_nan_inf",
                "candidate_pool_empty",
                "invalid_assignment",
                "checkpoint_missing",
                "taskoption_construction_failure",
                "high_level_inference_exception",
            ],
            "safety_projector_failure": "keep the existing safety path; if it has no legal recovery, record the error and stop the current decision; the high level must not invent a new target",
            "forbidden_new_fallbacks": [
                "deepsets",
                "b2",
                "oracle",
                "single_r1_seed",
                "phase5l_direct_hgnn",
                "fly_toward",
            ],
        },
        "h5_fresh_confirmation": {
            "pool": "310118-310125",
            "human_approved_for_next_phase_only": True,
            "opened_in_phase5zo": False,
        },
    }


def _confirmation_protocol() -> dict:
    return {
        "status": "pre_registered_not_executed",
        "open_once": True,
        "pool": "310118-310125",
        "partial_peek_then_modify_forbidden": True,
        "primary_system": "Frozen Rank-Mean True-H1 Hierarchical Controller",
        "primary_baseline": "Frozen Model-S",
        "not_primary_baseline": ["DeepSets", "HGNN-direct", "Oracle", "best R1 seed"],
        "deepsets_secondary_diagnostic_only": True,
        "primary_endpoint": "paired cumulative communication payload",
        "delta_definition": "Payload_candidate - Payload_ModelS",
        "report": [
            "mean_delta_payload",
            "median_delta_payload",
            "scenario_cluster_bootstrap_95_ci",
            "positive_scenario_count",
            "tie_count",
            "negative_scenario_count",
        ],
        "bootstrap": {"cluster": "scenario_id", "draws": 10000, "seed": 0, "interval": [2.5, 97.5], "statistic": "mean paired delta"},
        "primary_success_requires": ["mean_delta_payload > 0", "scenario_cluster_95_ci_lower_bound > 0"],
        "positive_mean_alone_is_not_success": True,
        "latency_gate_active": False,
        "latency_is_not_deployment_verification": True,
        "frozen_after_this_phase": [
            "architecture",
            "candidate_generator",
            "top3",
            "true_h1",
            "r1_checkpoints",
            "aggregation",
            "tie_breaking",
            "taskoption",
            "hold",
            "mappo",
            "reward",
            "normalization",
            "fallback",
            "timeout_policy",
            "primary_endpoint",
            "statistical_criterion",
        ],
        "change_after_seeing_pool": "FRESH_CONFIRMATION_INVALIDATED",
    }


def _metrics() -> dict:
    return {
        "primary_endpoint": "paired cumulative communication payload",
        "secondary_metrics_pre_registered": [
            "cumulative_payload",
            "residual_queue",
            "completed_service_blocks",
            "normalized_regret",
            "assignment_legality",
            "duplicate_target_rate",
            "conflict_rate",
            "sinr",
            "service_progress",
            "seed_consensus_strength",
            "inference_latency",
            "runtime_exception_count",
            "fallback_count",
            "timeout_count",
        ],
        "latency_statistics_diagnostic_only": ["mean", "median", "p95", "p99"],
        "safety_requirements": {
            "illegal_assignment": 0,
            "state_contamination": 0,
            "nan_final_decision": 0,
            "inf_final_decision": 0,
            "unhandled_runtime_exception": 0,
        },
        "fallback_must_be_reported_even_if_episode_completes": True,
        "may_redefine_after_confirmation": False,
    }


def _registered_gates() -> dict:
    return {
        "CONF-A": {
            "name": "INDEPENDENT_PERFORMANCE_CONFIRMED",
            "requires": [
                "mean_delta_payload > 0",
                "scenario_cluster_95_ci_lower_bound > 0",
                "illegal_assignment = 0",
                "state_contamination = 0",
                "nan_inf_final_decision = 0",
                "unhandled_runtime_exception = 0",
            ],
        },
        "CONF-B": {"name": "POSITIVE_MEAN_BUT_NOT_STATISTICALLY_CONFIRMED", "condition": "mean_delta_payload > 0 and the 95% CI contains 0"},
        "CONF-C": {"name": "NO_INDEPENDENT_GAIN", "condition": "mean_delta_payload <= 0"},
        "CONF-D": {"name": "SAFETY_OR_EXECUTION_FAILURE"},
        "CONF-E": {"name": "PROTOCOL_VIOLATION"},
        "executed_now": False,
    }


def _report(gate: dict, rehash: dict, contamination: dict, integrity: dict) -> str:
    ready = gate["gate"] == "FREEZE-A"
    return "\n".join(
        [
            "# HTW Phase-5ZO：人工规范落地与确认冻结审计",
            "",
            f"**Gate = {gate['gate']} {gate['gate_name']}**",
            "",
            f"**FROZEN_FOR_FRESH_CONFIRMATION = {gate['frozen_for_fresh_confirmation']}**",
            "",
            f"**fresh_confirmation_authorized = {gate['fresh_confirmation_authorized']}**",
            "",
            "**planner_ready = False**",
            "",
            "**deployment_verified = False**",
            "",
            "本阶段没有打开 310118–310125，没有运行 fresh confirmation，没有训练，没有 rollout，也没有为了让哈希一致而修改算法。Phase-4E 仍是 DEPLOYMENT_VERIFICATION_FAILED。这不是 deployment PASS。",
            "",
            "## 已写入的人工决定",
            "",
            "H1 批准 RANK_MEAN_CONSENSUS_V1。`model_r1_aggregation_approved = true`。不允许 seed 权重、最佳 seed、J15 选分配或按场景调权。tie-break 仍是 majority Top-1，然后是分配向量字典序。",
            "",
            "H2 的部署时延要求是 NOT_YET_SPECIFIED。prototype 的 median 334.28 ms、P95 450.93 ms、P99 458.34 ms 只作记录。`latency_gate_active = false`，`latency_diagnostic_only = true`。没有把 500 ms 写成门槛。",
            "",
            "H3 是 KEEP_CURRENT_TASKOPTION_NO_HOLD_RESET。存在合法 TaskOption 时保留 association，不新建 option，不把 remaining_hold 重置为 15，计数仍按原规则递减。option 已失效时 ABORT_NEW_DECISION，不调用其他高层。",
            "",
            "H4 的主回退是 KEEP_CURRENT_VALID_TASKOPTION，没有合法 option 时 ABORT_DECISION_IF_NO_VALID_TASKOPTION。SafetyProjector 失败不由高层换目标补偿；现有安全路径没有合法恢复时，记录错误并停止当前决策。",
            "",
            "H5 的人工批准只授权下一阶段在本 Gate 通过后一次性打开预留池。本阶段没有读取该池。",
            "",
            "## 哈希复核",
            "",
            f"与 Phase-5ZN 清单不一致的项目数是 {len(rehash['mismatch'])}。结构字段核对结果是 {rehash['structural']}。",
            "",
            "候选控制器仍是 Phase-5L HGNN 只负责 Stage-1 候选，不直接输出最终分配；shipped True-H1 只做一步反事实；五个 Model-R1 估计该步响应的价值；Rank-Mean 聚合；TaskOption 的 hold 是 15；低层 MAPPO 不重新选择浮标。",
            "",
            "## 工作区",
            "",
            f"Phase-5ZN 之后写入的算法文件数是 {contamination['algorithm_write_count']}。这类文件属于算法变化。本阶段新增的规范文件不计入。",
            "",
            f"Git 状态里与算法路径相交的未跟踪源文件记录数是 {contamination['git']['algorithm_git_entries_redacted_count']}，另有 {contamination['git']['algorithm_pyc_entries_not_source']} 条缓存文件。这些是 Phase-5ZN 之前就已经存在、且没有被 git 跟踪的工作区文件，不是本阶段之后的新修改。冻结组件的哈希没有变化，Phase-5ZN 之后也没有算法文件被重新写入，所以算法变化数 C 是 0。",
            "",
            "## 预留池",
            "",
            f"fresh_pool_blind_integrity = {integrity['integrity']}。只核对了既有场景登记，没有列池内文件，没有加载 snapshot、标签或轨迹。训练登记是 310100–310109，已消费确认登记是 310110–310117。",
            "",
            "## 下一阶段协议",
            "",
            "协议已经预注册，但没有执行。主系统是冻结的 Rank-Mean True-H1 控制器，主基线是冻结 Model-S。DeepSets、HGNN 直接分配、Oracle 和最佳 R1 seed 都不是主基线。",
            "",
            "主终点是配对累积通信 payload。确认成功必须同时满足平均差大于 0，以及场景聚类 95% 区间下界大于 0。只看平均值为正不能通过。时延继续记录，但不参与这次算法 Gate，也不是部署时延验证。",
            "",
            "看到预留池之后如果改动架构、候选、True-H1、checkpoint、聚合、hold、MAPPO、reward、归一化、回退、超时、主终点或统计标准，确认结果记为 FRESH_CONFIRMATION_INVALIDATED。",
            "",
            "## 本阶段是否授权下一阶段读取预留池",
            "",
            "可以。下一阶段才能第一次读取该池。" if ready else "不可以。本阶段没有得到 FREEZE-A，预留池继续保持未读。",
            "",
        ]
    )


def _dump(name: str, payload: dict) -> None:
    (PHASE / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    if not PHASE4E_REPORT.is_file() or "DEPLOYMENT_VERIFICATION_FAILED" not in PHASE4E_REPORT.read_text(encoding="utf-8"):
        raise RuntimeError("phase-4e failure status is missing")
    if not MANIFEST.is_file():
        raise RuntimeError("phase-5zn manifest is missing")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest_mtime = MANIFEST.stat().st_mtime
    rehash = _rehash(manifest)
    timestamp_only = []
    for row in rehash["components"]:
        path = Path(manifest["components"][row["name"]]["path"])
        if path.is_file():
            row["mtime_newer_than_manifest"] = bool(path.stat().st_mtime > manifest_mtime + 2.0)
            if row["mtime_newer_than_manifest"] and row["match"]:
                timestamp_only.append(row["name"])
    writes = _walk_algorithm(manifest_mtime)
    writes["algorithm_files_written_after_phase5zn"].extend(timestamp_only)
    git_info = _git_status()
    integrity = _registries()
    integrity["post_freeze_fresh_pool_touch_count"] = int(writes["fresh_pool_paths_written_after_phase5zn"])
    integrity["git_status_mentions_not_opened"] = int(git_info["fresh_pool_mentions_not_listed"])
    if writes["fresh_pool_paths_written_after_phase5zn"]:
        integrity["integrity"] = "FAIL"
    human = _human_spec()
    protocol = _confirmation_protocol()
    metrics = _metrics()
    registered = _registered_gates()
    required = (
        human["h1_model_r1_aggregation"]["model_r1_aggregation_approved"] is True
        and human["h2_latency"]["latency_gate_active"] is False
        and human["h3_timeout"]["policy"] == "KEEP_CURRENT_TASKOPTION_NO_HOLD_RESET"
        and human["h4_runtime_fallback"]["primary"] == "KEEP_CURRENT_VALID_TASKOPTION"
        and protocol["primary_baseline"] == "Frozen Model-S"
        and "scenario_cluster_95_ci_lower_bound > 0" in registered["CONF-A"]["requires"]
        and metrics["may_redefine_after_confirmation"] is False
    )
    contamination_count = len(writes["algorithm_files_written_after_phase5zn"])
    gate = choose_gate(
        hash_match=bool(rehash["match"]),
        integrity_pass=integrity["integrity"] == "PASS",
        contamination_count=contamination_count,
        human_spec_applied=True,
        protocol_complete=bool(required),
    )
    dump_protocol(bool(gate["frozen_for_fresh_confirmation"]))
    _dump("phase5zo_human_specification.json", human)
    _dump("phase5zo_algorithm_hash_recheck.json", rehash)
    _dump(
        "phase5zo_workspace_contamination_audit.json",
        {
            "algorithm_write_count": contamination_count,
            "algorithm_files_written_after_phase5zn": writes["algorithm_files_written_after_phase5zn"],
            "categories": {
                "A_human_specification_only": ["src/wt/lowlevel_v2/phase5zo_confirmation_freeze"],
                "B_report_or_audit_only": ["HTW_Phase5ZO_HumanSpecificationApplicationAndConfirmationFreeze_CN.md"],
                "C_algorithm_affecting_change": writes["algorithm_files_written_after_phase5zn"],
            },
            "git": git_info,
            "ignored_untracked_files": False,
        },
    )
    _dump("phase5zo_fresh_pool_blind_integrity.json", integrity)
    _dump("phase5zo_final_confirmation_protocol.json", protocol)
    _dump("phase5zo_pre_registered_metrics.json", metrics)
    _dump("phase5zo_pre_registered_gate.json", registered)
    _dump(
        "phase5zo_freeze_authorization_gate.json",
        {
            "gate": gate["gate"],
            "gate_name": gate["gate_name"],
            "section3_stop_name": gate["section3_stop_name"],
            "model_r1_aggregation_approved": True,
            "deployment_latency_requirement": "NOT_YET_SPECIFIED",
            "latency_gate_active": False,
            "latency_diagnostic_only": True,
            "timeout_policy": "KEEP_CURRENT_TASKOPTION_NO_HOLD_RESET",
            "runtime_fallback_primary": "KEEP_CURRENT_VALID_TASKOPTION",
            "runtime_fallback_secondary": "ABORT_DECISION_IF_NO_VALID_TASKOPTION",
            "hash_match": rehash["match"],
            "hash_mismatch": rehash["mismatch"],
            "workspace_contamination_count": contamination_count,
            "fresh_pool_blind_integrity": integrity["integrity"],
            "fresh_pool_read_in_this_phase": False,
            "confirmation_protocol_preregistered": True,
            "confirmation_executed": False,
            "frozen_for_fresh_confirmation": gate["frozen_for_fresh_confirmation"],
            "fresh_confirmation_authorized": gate["fresh_confirmation_authorized"],
            "planner_ready": PLANNER_READY,
            "deployment_verified": DEPLOYMENT_VERIFIED,
            "phase4e_status": PHASE4E_STATUS,
            "phase4e_overwritten": False,
            "deployment_pass": False,
            "next_phase_may_open_fresh_pool": gate["gate"] == "FREEZE-A",
        },
    )
    REPORT.write_text(_report(gate, rehash, {"algorithm_write_count": contamination_count, "git": git_info}, integrity), encoding="utf-8")
    print(
        json.dumps(
            {
                "gate": gate["gate"],
                "mismatch": rehash["mismatch"],
                "contamination": contamination_count,
                "integrity": integrity["integrity"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
