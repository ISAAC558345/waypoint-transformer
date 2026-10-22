"""One-shot fresh confirmation. The pool is opened only after the freeze check."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from wt.l3b.config import L3BConfig
from wt.lowlevel_v2.phase5g_closed_loop.collect import load_frozen_actor
from wt.lowlevel_v2.phase5l_joint_encoder.run_phase5l import _load_matching
from wt.lowlevel_v2.phase5m_joint_planner.run_phase5m import _load_proposed
from wt.lowlevel_v2.phase5r_nonadditive_generalization.metrics import cluster_bootstrap
from wt.lowlevel_v2.phase5v_temporal_representation.run_phase5v import _checkpoint as _temporal_checkpoint
from wt.lowlevel_v2.phase5x_executor_lookahead.shadow import max_abs_diff
from wt.lowlevel_v2.phase5ze_response_distillation.run_phase5ze import _load_response_norm, _load_value
from wt.lowlevel_v2.phase5zo_confirmation_freeze.protocol import MANIFEST
from wt.lowlevel_v2.phase5zo_confirmation_freeze.run_phase5zo import _rehash, _sha256

from .episode import run_episode
from .protocol import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    DEPLOYMENT_VERIFIED,
    FRESH_POOL,
    MODEL_S_PAYLOAD,
    MODEL_S_SELECTION,
    PHASE,
    PHASE4E_REPORT,
    PHASE4E_STATUS,
    PLANNER_READY,
    PRIMARY_BASELINE,
    PRIMARY_SYSTEM,
    REPORT,
    VALUE_SEEDS,
    choose_gate,
    confirmation_may_start,
    model_s_payload,
)

ZO = PHASE.parent / "phase5zo_confirmation_freeze"
PROTOCOL_FILES = (
    "phase5zo_final_confirmation_protocol.json",
    "phase5zo_pre_registered_metrics.json",
    "phase5zo_pre_registered_gate.json",
    "phase5zo_algorithm_hash_recheck.json",
    "phase5zo_protocol.json",
    "phase5zo_human_specification.json",
)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    return value


def _dump(name: str, payload: dict) -> None:
    PHASE.mkdir(parents=True, exist_ok=True)
    (PHASE / name).write_text(json.dumps(_plain(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(name: str, rows: list[dict], header: list[str]) -> None:
    path = PHASE / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_json(name: str) -> dict:
    return json.loads((ZO / name).read_text(encoding="utf-8"))


def _actor_checksum(actor) -> tuple[float, int]:
    total = 0.0
    count = 0
    for parameter in actor.parameters():
        total += float(parameter.detach().abs().sum().item())
        count += int(parameter.numel())
    return total, count


def _protocol_ok() -> tuple[bool, dict]:
    protocol = _load_json("phase5zo_protocol.json")
    final = _load_json("phase5zo_final_confirmation_protocol.json")
    metrics = _load_json("phase5zo_pre_registered_metrics.json")
    gate = _load_json("phase5zo_pre_registered_gate.json")
    human = _load_json("phase5zo_human_specification.json")
    problems = []
    if protocol["model_r1_aggregation_rule"] != "RANK_MEAN_CONSENSUS_V1" or protocol["model_r1_aggregation_approved"] is not True:
        problems.append("aggregation")
    if protocol["deployment_latency_requirement"] != "NOT_YET_SPECIFIED" or protocol["latency_gate_active"] is not False:
        problems.append("latency")
    if protocol["timeout_policy"] != "KEEP_CURRENT_TASKOPTION_NO_HOLD_RESET":
        problems.append("timeout")
    if protocol["runtime_fallback_primary"] != "KEEP_CURRENT_VALID_TASKOPTION":
        problems.append("fallback_primary")
    if protocol["runtime_fallback_secondary"] != "ABORT_DECISION_IF_NO_VALID_TASKOPTION":
        problems.append("fallback_secondary")
    if final["primary_baseline"] != PRIMARY_BASELINE or final["primary_system"] != PRIMARY_SYSTEM:
        problems.append("baseline")
    if final["primary_endpoint"] != "paired cumulative communication payload":
        problems.append("endpoint")
    if list(final["primary_success_requires"]) != ["mean_delta_payload > 0", "scenario_cluster_95_ci_lower_bound > 0"]:
        problems.append("success_rule")
    if int(final["bootstrap"]["draws"]) != BOOTSTRAP_DRAWS or int(final["bootstrap"]["seed"]) != BOOTSTRAP_SEED:
        problems.append("bootstrap")
    if final["latency_gate_active"] is not False:
        problems.append("latency_gate")
    if metrics["may_redefine_after_confirmation"] is not False:
        problems.append("metrics_locked")
    if "scenario_cluster_95_ci_lower_bound > 0" not in gate["CONF-A"]["requires"]:
        problems.append("conf_a")
    if human["h2_latency"]["threshold_500ms_created"] is not False:
        problems.append("threshold_500ms")
    if human["h1_model_r1_aggregation"]["model_r1_aggregation_approved"] is not True:
        problems.append("human_aggregation")
    return not problems, {"problems": problems}


def _pre_access() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rehash = _rehash(manifest)
    rules_ok, rules = _protocol_ok()
    phase4e_ok = PHASE4E_REPORT.is_file() and "DEPLOYMENT_VERIFICATION_FAILED" in PHASE4E_REPORT.read_text(encoding="utf-8")
    match = bool(rehash["match"] and rules_ok and phase4e_ok and len(rehash["components"]) == 21)
    return {
        "match": match,
        "component_count": len(rehash["components"]),
        "hash_mismatch": list(rehash["mismatch"]),
        "structural": rehash["structural"],
        "protocol_problems": rules["problems"],
        "phase4e_ok": phase4e_ok,
        "opened_pool": False,
    }


def _load_bundle() -> dict:
    config = L3BConfig()
    actor = load_frozen_actor()
    temporal = torch.load(_temporal_checkpoint(0), map_location="cpu", weights_only=False)
    return {
        "config": config,
        "frozen": _load_proposed(),
        "matching": _load_matching(),
        "actor": actor,
        "actor_checksum": _actor_checksum(actor),
        "norm": _load_response_norm(),
        "center": np.asarray(temporal["center"], dtype=np.float32),
        "scale": np.asarray(temporal["scale"], dtype=np.float32),
        "response_models": {int(seed): _load_value("response", int(seed)) for seed in VALUE_SEEDS},
        "static_models": {int(seed): _load_value("static", int(seed)) for seed in VALUE_SEEDS},
    }


def _mark_first_access() -> str:
    stamp = datetime.now(timezone.utc).isoformat()
    _dump(
        "phase5zp_first_access_record.json",
        {
            "fresh_confirmation_first_access_utc": stamp,
            "fresh_confirmation_pool_consumed": True,
            "fresh_confirmation_pool_role": "ONE_SHOT_FRESH_CONFIRMATION_CONSUMED",
            "pool": [int(item) for item in FRESH_POOL],
            "untouched_after_this_record": False,
        },
    )
    return stamp


def _evaluate(bundle: dict) -> list[dict]:
    """All eight scenarios, with no branch on intermediate payload."""
    grouped = []
    for scenario in FRESH_POOL:
        print(f"scenario {int(scenario)} started", flush=True)
        primary = run_episode(bundle, int(scenario), "primary")
        static = [run_episode(bundle, int(scenario), "model_s", static_seed=int(seed)) for seed in VALUE_SEEDS]
        for item in static:
            if max_abs_diff(primary["initial"], item["initial"]) != 0.0:
                primary["state_contamination"] = int(primary["state_contamination"]) + 1
        grouped.append({"primary": primary, "static": static})
        print(f"scenario {int(scenario)} finished", flush=True)
    return grouped


def _mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _percentile(values, q: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, q)) if array.size else float("nan")


def _fmt(value: float) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return format(float(value), ".6g")


def _scenario_rows(grouped: list[dict]) -> list[dict]:
    rows = []
    for item in grouped:
        primary = item["primary"]
        static = item["static"]
        payload_s = model_s_payload([float(row["payload"]) for row in static])
        delta = float(primary["payload"] - payload_s)
        relative = float(delta / payload_s) if payload_s != 0.0 else float("nan")
        if delta > 0.0:
            outcome = "win"
        elif delta < 0.0:
            outcome = "loss"
        else:
            outcome = "tie"
        rows.append(
            {
                "scenario_id": int(primary["scenario_id"]),
                "primary_cumulative_payload": float(primary["payload"]),
                "primary_residual_queue": float(primary["residual_queue"]),
                "primary_completed_blocks": float(primary["completed_blocks"]),
                "primary_average_sinr": float(primary["average_sinr"]),
                "primary_conflict_rate": float(primary["conflict_rate"]),
                "primary_service_progress": float(primary["service_progress"]),
                "primary_inference_latency_ms": float(primary["inference_latency_ms"]),
                "primary_fallback_count": int(primary["fallback_count"]),
                "primary_timeout_count": int(primary["timeout_count"]),
                "primary_mean_interference": float(primary["mean_interference"]),
                "primary_mean_queue": float(primary["mean_queue"]),
                "primary_dropped_blocks": float(primary["dropped_blocks"]),
                "models_cumulative_payload": payload_s,
                "models_residual_queue": _mean(row["residual_queue"] for row in static),
                "models_completed_blocks": _mean(row["completed_blocks"] for row in static),
                "models_average_sinr": _mean(row["average_sinr"] for row in static),
                "models_conflict_rate": _mean(row["conflict_rate"] for row in static),
                "models_service_progress": _mean(row["service_progress"] for row in static),
                "models_inference_latency_ms": _mean(row["inference_latency_ms"] for row in static),
                "models_fallback_count": _mean(row["fallback_count"] for row in static),
                "models_timeout_count": _mean(row["timeout_count"] for row in static),
                "models_dropped_blocks": _mean(row["dropped_blocks"] for row in static),
                "delta_payload": delta,
                "relative_delta": relative,
                "normalized_improvement": relative,
                "outcome": outcome,
            }
        )
    return rows


def _half(rows: list[dict], key: str) -> dict:
    med = float(np.median([float(row[key]) for row in rows]))
    high = [row for row in rows if float(row[key]) >= med]
    low = [row for row in rows if float(row[key]) < med]
    return {
        "split_median": med,
        "high_mean_delta": _mean(row["delta_payload"] for row in high),
        "low_mean_delta": _mean(row["delta_payload"] for row in low),
        "high_count": len(high),
        "low_count": len(low),
        "used_by_gate": False,
    }


def _write_report(gate: dict, rows: list[dict], summary: dict, access: dict) -> None:
    lines = [
        "# HTW Phase-5ZP：一次性 Fresh Confirmation",
        "",
        f"Gate 是 {gate['gate']}（{gate['gate_name']}）。",
        "",
        "这不是 deployment verification，也没有进入正式 planner。`planner_ready = False`，`deployment_verified = False`。Phase-4E 仍是 DEPLOYMENT_VERIFICATION_FAILED。",
        "",
        "## Q1. 执行前 310118–310125 是否仍未使用？",
        "",
        "是。访问前的冻结检查只读取 Phase-5ZO 已经写好的协议和哈希清单，没有重置这些 scenario。Phase-5ZO 的盲审计结论是这些编号没有进入训练、验证、checkpoint 选择、结构设计、损失设计、阈值设计、候选设计、聚合设计、True-H1 优化、时延优化或 fallback 设计。",
        "",
        "## Q2. 第一次访问时间和 consumed 标记",
        "",
        f"第一次访问的 UTC 时间是 {access.get('fresh_confirmation_first_access_utc', '未访问')}。访问记录在第一次重置之前写入，`fresh_confirmation_pool_consumed = {access.get('fresh_confirmation_pool_consumed', False)}`，角色是 {access.get('fresh_confirmation_pool_role', 'UNOPENED')}。",
        "",
        "## Q3. 候选系统和 Model-S 是否完全 paired？",
        "",
        f"是。每个 scenario 的主系统和五个冻结 Model-S seed 都从同一个 scenario seed 的初始状态各自复制环境。初始状态不一致的次数是 {summary['paired_initial_mismatches']}。Model-S 的 payload 是五个冻结 static seed 整局 payload 的平均值，不是 payload 最高的 seed。选择规则在打开预留池之前已经固定为每个 seed 在同一 120 候选上对零响应做 stable argmax。",
        "",
        "## Q4. 八个 scenario 的 paired payload",
        "",
    ]
    for row in rows:
        lines.append(
            f"- {int(row['scenario_id'])}：主系统 {_fmt(row['primary_cumulative_payload'])} bit，Model-S {_fmt(row['models_cumulative_payload'])} bit，Δ {_fmt(row['delta_payload'])} bit，{row['outcome']}。"
        )
    lines.extend(
        [
            "",
            "## Q5. mean Δpayload",
            "",
            f"{_fmt(summary['mean_delta'])} bit。",
            "",
            "## Q6. median Δpayload",
            "",
            f"{_fmt(summary['median_delta'])} bit。",
            "",
            "## Q7. scenario-cluster 95% CI",
            "",
            f"[{_fmt(summary['ci_low'])}, {_fmt(summary['ci_high'])}] bit。统计单位是 scenario。bootstrap 抽 10000 次，种子是 0。决策时刻、UAV、seed 和时间步没有被当成独立样本。",
            "",
            "## Q8. 正、平、负 scenario",
            "",
            f"正 {summary['positive']}，平 {summary['tie']}，负 {summary['negative']}。",
            "",
            "## Q9. 是否达到 CONF-A 的两个性能条件？",
            "",
            f"平均差大于 0：{summary['mean_positive']}。95% 区间下界大于 0：{summary['ci_positive']}。两个条件同时成立才算性能确认。只凭平均值为正不够。",
            "",
            "## Q10. illegal assignment",
            "",
            f"{summary['illegal_assignment']}。",
            "",
            "## Q11. state contamination",
            "",
            f"{summary['state_contamination']}。",
            "",
            "## Q12. NaN、Inf 和未处理运行异常",
            "",
            f"NaN {summary['nan_final_decision']}，Inf {summary['inf_final_decision']}，未处理运行异常 {summary['unhandled_runtime_exception']}。已记录的运行异常次数是 {summary['runtime_exception_count']}。",
            "",
            "## Q13. fallback 和 timeout",
            "",
            f"fallback {summary['fallback_count']} 次，timeout {summary['timeout_count']} 次。timeout 只统计新决策没有产出、因而保留当前合法 TaskOption 且不把 hold 重置为 15 的次数。没有使用 500 ms 作为超时门槛。SafetyProjector 失败沿用原有零动作恢复，次数是 {summary['projector_failures']}，高层没有因此改目标。",
            "",
            "## Q14. 五个 R1 head 的一致性",
            "",
            f"主系统高层决策 {summary['primary_decisions']} 次。五 seed Top-1 完全相同的比例是 {_fmt(summary['full_agreement'])}。平均 consensus strength 是 {_fmt(summary['mean_strength'])}。平均 pairwise Spearman 是 {_fmt(summary['mean_spearman'])}。被选候选的平均 mean rank 是 {_fmt(summary['mean_selected_rank'])}。Top-1 与 Top-2 的平均 rank margin 是 {_fmt(summary['mean_rank_margin'])}。出现不一致时仍执行冻结的 Rank-Mean V1，没有改聚合。",
            "",
            "## Q15. secondary metrics 相对 Model-S",
            "",
            f"平均剩余队列：主系统 {_fmt(summary['primary_queue'])}，Model-S {_fmt(summary['models_queue'])}。",
            f"平均完成块数：主系统 {_fmt(summary['primary_blocks'])}，Model-S {_fmt(summary['models_blocks'])}。",
            f"平均 SINR：主系统 {_fmt(summary['primary_sinr'])}，Model-S {_fmt(summary['models_sinr'])}。",
            f"平均冲突率：主系统 {_fmt(summary['primary_conflict'])}，Model-S {_fmt(summary['models_conflict'])}。",
            f"平均服务进度：主系统 {_fmt(summary['primary_progress'])}，Model-S {_fmt(summary['models_progress'])}。",
            "normalized regret 沿用已登记的候选池 J15 oracle 定义。本阶段不运行 15 步 oracle rollout，因此这项没有新的数值。丢包块数作为环境已有诊断：主系统平均 "
            f"{_fmt(summary['primary_dropped'])}，Model-S 平均 {_fmt(summary['models_dropped'])}。环境没有 fairness 记录。",
            "",
            "## Q16. 高层时延",
            "",
            f"主系统每次高层决策的 total：mean {_fmt(summary['latency_mean'])} ms，median {_fmt(summary['latency_median'])} ms，P95 {_fmt(summary['latency_p95'])} ms，P99 {_fmt(summary['latency_p99'])} ms，max {_fmt(summary['latency_max'])} ms。",
            f"分项平均：encoder {_fmt(summary['encoder_ms'])} ms，candidate {_fmt(summary['candidate_ms'])} ms，True-H1 {_fmt(summary['h1_ms'])} ms，五个 R1 {_fmt(summary['r1_ms'])} ms，Rank-Mean {_fmt(summary['agg_ms'])} ms，TaskOption {_fmt(summary['option_ms'])} ms。",
            "这些数字只是当前研究计算机上的诊断。`latency_gate_active = false`。它们不参与 CONF-A、CONF-B 或 CONF-C，也不是 deployment latency verification。",
            "",
            "## Q17. 最终 Gate",
            "",
            f"{gate['gate']}。`independent_algorithm_confirmation = {gate['independent_algorithm_confirmation']}`。",
            "",
            "## Q18. 这 8 个 scenario 还能不能再当 fresh confirmation？",
            "",
            "YES。只要本阶段已经打开它们，它们就永久是 consumed confirmation data。以后同一套系统的重复运行只能称为 REPRODUCTION_ON_CONSUMED_CONFIRMATION_DATA。",
            "",
            "## 事后描述",
            "",
            "下面的分组是 Gate 计算完成之后的解释，不构成新门槛，也不改变主总体。",
            f"干扰中位数分组：高干扰组平均 Δ {_fmt(summary['interference_split']['high_mean_delta'])} bit（{summary['interference_split']['high_count']} 个），低干扰组 {_fmt(summary['interference_split']['low_mean_delta'])} bit（{summary['interference_split']['low_count']} 个）。",
            f"冲突中位数分组：高冲突组平均 Δ {_fmt(summary['conflict_split']['high_mean_delta'])} bit，低冲突组 {_fmt(summary['conflict_split']['low_mean_delta'])} bit。",
            f"队列中位数分组：高队列组平均 Δ {_fmt(summary['queue_split']['high_mean_delta'])} bit，低队列组 {_fmt(summary['queue_split']['low_mean_delta'])} bit。",
            "",
            "True-H1 仍是一步 executor-conditioned response，不是 15 步 rollout。低层 MAPPO 只执行已经给出的 TaskOption，不重新选择 buoy。",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _empty_report(reason: str) -> None:
    text = "\n".join(
        [
            "# HTW Phase-5ZP：一次性 Fresh Confirmation",
            "",
            "Gate 是 CONF-E（PROTOCOL_VIOLATION）。",
            "",
            f"原因是 {reason}。预留池没有打开，310118–310125 仍然没有被这次确认读取。",
            "",
            "Q18 在预留池未被打开时不是 YES。它们还没有被这次确认消耗。",
            "",
            "`planner_ready = False`，`deployment_verified = False`。Phase-4E 仍是 DEPLOYMENT_VERIFICATION_FAILED。",
        ]
    )
    REPORT.write_text(text + "\n", encoding="utf-8")


def _finish_without_access(pre: dict) -> None:
    _dump("phase5zp_pre_access_hash_check.json", pre)
    _dump(
        "phase5zp_first_access_record.json",
        {
            "fresh_confirmation_first_access_utc": None,
            "fresh_confirmation_pool_consumed": False,
            "fresh_confirmation_pool_role": "UNOPENED",
            "pool": [int(item) for item in FRESH_POOL],
        },
    )
    _dump(
        "phase5zp_data_lineage_update.json",
        {
            "pool": "310118-310125",
            "historical_role": "UNOPENED",
            "eligible_for_future_development": False,
            "consumed": False,
        },
    )
    gate = choose_gate(
        protocol_violation=True,
        illegal_assignment=0,
        state_contamination=0,
        nan_final_decision=0,
        inf_final_decision=0,
        unhandled_runtime_exception=0,
        mean_delta=float("nan"),
        ci_low=float("nan"),
    )
    gate["reason"] = "PRE_ACCESS_FREEZE_MISMATCH"
    gate["fresh_confirmation_authorized"] = "not_opened"
    gate["frozen_for_fresh_confirmation"] = True
    gate["planner_ready"] = PLANNER_READY
    gate["deployment_verified"] = DEPLOYMENT_VERIFIED
    gate["phase4e_status"] = PHASE4E_STATUS
    gate["deployment_pass"] = False
    _dump("phase5zp_fresh_confirmation_gate.json", gate)
    _empty_report("PRE_ACCESS_FREEZE_MISMATCH")
    print({"gate": "CONF-E", "reason": "PRE_ACCESS_FREEZE_MISMATCH"}, flush=True)


def main() -> None:
    PHASE.mkdir(parents=True, exist_ok=True)
    if os.environ.get("PHASE5ZP_SMOKE") == "1":
        bundle = _load_bundle()
        episode = run_episode(bundle, 310110, "primary", max_steps=16)
        print(
            {
                "smoke": "ok",
                "candidates": int(episode["latency_rows"][0]["candidate_count"]) if episode["latency_rows"] else 0,
                "decisions": int(episode["decision_count"]),
                "illegal": int(episode["illegal_assignment"]),
                "unhandled": int(episode["unhandled_runtime_exception"]),
                "contamination": int(episode["state_contamination"]),
                "fallback": int(episode["fallback_count"]),
                "hold_ok": bool(episode["hold_decrement_ok"]),
                "targets_ok": bool(episode["low_level_target_unchanged"]),
            },
            flush=True,
        )
        return
    pre = _pre_access()
    file_hashes = {name: _sha256(ZO / name) for name in PROTOCOL_FILES}
    _dump(
        "phase5zp_protocol_snapshot.json",
        {
            "primary_system": PRIMARY_SYSTEM,
            "primary_baseline": PRIMARY_BASELINE,
            "model_s_payload": MODEL_S_PAYLOAD,
            "model_s_selection": MODEL_S_SELECTION,
            "not_primary_baseline": ["DeepSets", "direct HGNN", "Oracle", "single R1 seed", "fly-toward"],
            "protocol_file_sha256": file_hashes,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "latency_gate_active": False,
            "deployment_latency_requirement": "NOT_YET_SPECIFIED",
        },
    )
    _dump("phase5zp_pre_access_hash_check.json", pre)
    if not confirmation_may_start(bool(pre["match"])):
        _finish_without_access(pre)
        return
    bundle = _load_bundle()
    stamp = _mark_first_access()
    grouped = _evaluate(bundle)
    post = _rehash(json.loads(MANIFEST.read_text(encoding="utf-8")))
    actor_after = _actor_checksum(bundle["actor"])
    protocol_unchanged = all(_sha256(ZO / name) == file_hashes[name] for name in PROTOCOL_FILES)
    rows = _scenario_rows(grouped)
    deltas = [float(row["delta_payload"]) for row in rows]
    ci = cluster_bootstrap([{"scenario_id": int(row["scenario_id"]), "delta_payload": float(row["delta_payload"])} for row in rows], "delta_payload", draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    primary_latency = [row for item in grouped for row in item["primary"]["latency_rows"]]
    totals = [float(row["total_ms"]) for row in primary_latency]
    consensus = [row for item in grouped for row in item["primary"]["consensus_rows"]]
    strengths = []
    spearmans = []
    margins = []
    selected_ranks = []
    full_agree = 0
    for row in consensus:
        strengths.append(float(row["consensus_strength"]))
        spearmans.append(float(row["pairwise_spearman"]))
        margins.append(float(row["top1_top2_rank_margin"]))
        selected_ranks.append(float(row["selected_mean_rank"]))
        tops = str(row["seed_top1"]).split("|")
        full_agree += int(len(set(tops)) == 1)
    episodes = [item["primary"] for item in grouped] + [row for item in grouped for row in item["static"]]
    illegal = int(sum(item["illegal_assignment"] for item in episodes))
    contamination = int(sum(item["state_contamination"] for item in episodes))
    nan_count = int(sum(item["nan_final_decision"] for item in episodes))
    inf_count = int(sum(item["inf_final_decision"] for item in episodes))
    unhandled = int(sum(item["unhandled_runtime_exception"] for item in episodes))
    h1_steps_ok = all(int(row["true_h1_steps"]) == 1 for row in primary_latency) if primary_latency else False
    hold_ok = all(bool(item["hold_decrement_ok"]) and bool(item["low_level_target_unchanged"]) for item in episodes)
    option_preview = [row for item in episodes for row in item["option_rows"]]
    association_ok = bool(option_preview) and all(
        bool(row["association_equals_assignment"]) and int(row["duplicate_targets"]) == 0 and int(row["remaining_hold_initialized"]) == 15
        for row in option_preview
    )
    protocol_violation = not (
        post["match"] and protocol_unchanged and actor_after == bundle["actor_checksum"] and h1_steps_ok and hold_ok and association_ok
    )
    gate = choose_gate(
        protocol_violation=protocol_violation,
        illegal_assignment=illegal,
        state_contamination=contamination,
        nan_final_decision=nan_count,
        inf_final_decision=inf_count,
        unhandled_runtime_exception=unhandled,
        mean_delta=float(np.mean(deltas)),
        ci_low=float(ci["low"]),
    )
    positive = int(sum(row["outcome"] == "win" for row in rows))
    tie = int(sum(row["outcome"] == "tie" for row in rows))
    negative = int(sum(row["outcome"] == "loss" for row in rows))
    summary = {
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "ci_low": float(ci["low"]),
        "ci_high": float(ci["high"]),
        "positive": positive,
        "tie": tie,
        "negative": negative,
        "mean_positive": bool(float(np.mean(deltas)) > 0.0),
        "ci_positive": bool(float(ci["low"]) > 0.0),
        "illegal_assignment": illegal,
        "state_contamination": contamination,
        "nan_final_decision": nan_count,
        "inf_final_decision": inf_count,
        "unhandled_runtime_exception": unhandled,
        "runtime_exception_count": int(sum(item["runtime_exception_count"] for item in episodes)),
        "fallback_count": int(sum(item["fallback_count"] for item in episodes)),
        "timeout_count": int(sum(item["timeout_count"] for item in episodes)),
        "projector_failures": int(sum(item["projector_failures"] for item in episodes)),
        "paired_initial_mismatches": contamination,
        "primary_decisions": len(consensus),
        "full_agreement": float(full_agree / max(len(consensus), 1)),
        "mean_strength": _mean(strengths),
        "mean_spearman": _mean(spearmans),
        "mean_selected_rank": _mean(selected_ranks),
        "mean_rank_margin": _mean(margins),
        "primary_queue": _mean(row["primary_residual_queue"] for row in rows),
        "models_queue": _mean(row["models_residual_queue"] for row in rows),
        "primary_blocks": _mean(row["primary_completed_blocks"] for row in rows),
        "models_blocks": _mean(row["models_completed_blocks"] for row in rows),
        "primary_sinr": _mean(row["primary_average_sinr"] for row in rows),
        "models_sinr": _mean(row["models_average_sinr"] for row in rows),
        "primary_conflict": _mean(row["primary_conflict_rate"] for row in rows),
        "models_conflict": _mean(row["models_conflict_rate"] for row in rows),
        "primary_progress": _mean(row["primary_service_progress"] for row in rows),
        "models_progress": _mean(row["models_service_progress"] for row in rows),
        "primary_dropped": _mean(row["primary_dropped_blocks"] for row in rows),
        "models_dropped": _mean(row["models_dropped_blocks"] for row in rows),
        "latency_mean": _mean(totals),
        "latency_median": _percentile(totals, 50),
        "latency_p95": _percentile(totals, 95),
        "latency_p99": _percentile(totals, 99),
        "latency_max": float(np.max(totals)) if totals else float("nan"),
        "encoder_ms": _mean(row["encoder_ms"] for row in primary_latency),
        "candidate_ms": _mean(row["candidate_ms"] for row in primary_latency),
        "h1_ms": _mean(row["true_h1_ms"] for row in primary_latency),
        "r1_ms": _mean(row["r1_ms"] for row in primary_latency),
        "agg_ms": _mean(row["aggregation_ms"] for row in primary_latency),
        "option_ms": _mean(row["taskoption_ms"] for row in primary_latency),
        "interference_split": _half(rows, "primary_mean_interference"),
        "conflict_split": _half(rows, "primary_conflict_rate"),
        "queue_split": _half(rows, "primary_mean_queue"),
    }
    # The paired mismatch counter above also includes H1 fingerprint misses.
    # Report the reset-only mismatches separately.
    reset_mismatches = 0
    for item in grouped:
        for static in item["static"]:
            if max_abs_diff(item["primary"]["initial"], static["initial"]) != 0.0:
                reset_mismatches += 1
    summary["paired_initial_mismatches"] = int(reset_mismatches)
    scenario_header = list(rows[0].keys())
    _write_csv("phase5zp_scenario_results.csv", rows, scenario_header)
    _write_csv(
        "phase5zp_primary_payload_comparison.csv",
        rows,
        ["scenario_id", "primary_cumulative_payload", "models_cumulative_payload", "delta_payload", "relative_delta", "outcome"],
    )
    secondary = []
    for item, row in zip(grouped, rows):
        strengths_here = [float(record["consensus_strength"]) for record in item["primary"]["consensus_rows"]]
        secondary.append(
            {
                "scenario_id": int(row["scenario_id"]),
                "system": "primary",
                "cumulative_payload": row["primary_cumulative_payload"],
                "residual_queue": row["primary_residual_queue"],
                "completed_service_blocks": row["primary_completed_blocks"],
                "normalized_regret": "",
                "assignment_legality": item["primary"]["assignment_legality"],
                "duplicate_target_rate": item["primary"]["duplicate_target_rate"],
                "conflict_rate": row["primary_conflict_rate"],
                "sinr": row["primary_average_sinr"],
                "service_progress": row["primary_service_progress"],
                "seed_consensus_strength": _mean(strengths_here),
                "inference_latency_ms": row["primary_inference_latency_ms"],
                "runtime_exception_count": item["primary"]["runtime_exception_count"],
                "fallback_count": row["primary_fallback_count"],
                "timeout_count": row["primary_timeout_count"],
            }
        )
        secondary.append(
            {
                "scenario_id": int(row["scenario_id"]),
                "system": "model_s_mean",
                "cumulative_payload": row["models_cumulative_payload"],
                "residual_queue": row["models_residual_queue"],
                "completed_service_blocks": row["models_completed_blocks"],
                "normalized_regret": "",
                "assignment_legality": _mean(ep["assignment_legality"] for ep in item["static"]),
                "duplicate_target_rate": _mean(ep["duplicate_target_rate"] for ep in item["static"]),
                "conflict_rate": row["models_conflict_rate"],
                "sinr": row["models_average_sinr"],
                "service_progress": row["models_service_progress"],
                "seed_consensus_strength": "",
                "inference_latency_ms": row["models_inference_latency_ms"],
                "runtime_exception_count": _mean(ep["runtime_exception_count"] for ep in item["static"]),
                "fallback_count": row["models_fallback_count"],
                "timeout_count": row["models_timeout_count"],
            }
        )
    _write_csv("phase5zp_secondary_metrics.csv", secondary, list(secondary[0].keys()))
    _write_csv("phase5zp_seed_consensus.csv", consensus, list(consensus[0].keys()) if consensus else ["scenario_id"])
    option_rows = [row for item in episodes for row in item["option_rows"]]
    _write_csv("phase5zp_taskoption_audit.csv", option_rows, list(option_rows[0].keys()) if option_rows else ["scenario_id"])
    exception_rows = [row for item in episodes for row in item["exception_rows"]]
    exception_header = ["scenario_id", "time_step", "system", "static_seed", "exception_type", "fallback_action", "episode_continued", "hold_reset"]
    _write_csv("phase5zp_runtime_exception_log.csv", exception_rows, exception_header)
    _write_csv("phase5zp_fallback_log.csv", [row for item in episodes for row in item["fallback_rows"]], exception_header)
    latency_rows = [row for item in episodes for row in item["latency_rows"]]
    _write_csv("phase5zp_latency.csv", latency_rows, list(latency_rows[0].keys()) if latency_rows else ["scenario_id"])
    _dump(
        "phase5zp_cluster_bootstrap.json",
        {
            "cluster": "scenario_id",
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "statistic": "mean paired delta",
            "mean": summary["mean_delta"],
            "median": summary["median_delta"],
            "low": summary["ci_low"],
            "high": summary["ci_high"],
            "positive": positive,
            "tie": tie,
            "negative": negative,
            "n_scenarios": len(rows),
        },
    )
    _dump(
        "phase5zp_protocol_compliance.json",
        {
            "pre_access_match": True,
            "post_access_hash_match": bool(post["match"]),
            "post_access_mismatch": list(post["mismatch"]),
            "protocol_files_unchanged": protocol_unchanged,
            "actor_checksum_unchanged": actor_after == bundle["actor_checksum"],
            "true_h1_is_one_step": h1_steps_ok,
            "hold_and_low_level_targets_ok": hold_ok,
            "primary_baseline": PRIMARY_BASELINE,
            "baseline_switched_after_results": False,
            "latency_gate_active": False,
            "threshold_500ms_created": False,
            "normalized_regret_note": "pre-registered formula needs an oracle J15 label; no 15-step oracle rollout was run",
            "partial_peek_then_modify": False,
        },
    )
    _dump(
        "phase5zp_data_lineage_update.json",
        {
            "pool": "310118-310125",
            "historical_role": "FRESH_CONFIRMATION_CONSUMED",
            "used_for": ["final frozen algorithm independent confirmation"],
            "not_for_future": [
                "training",
                "feature_selection",
                "hyperparameter_tuning",
                "architecture_tuning",
                "loss_design",
                "threshold_tuning",
                "candidate_redesign",
                "aggregation_redesign",
                "fallback_redesign",
            ],
            "eligible_for_future_development": False,
            "rerun_label": "REPRODUCTION_ON_CONSUMED_CONFIRMATION_DATA",
        },
    )
    gate_payload = dict(gate)
    gate_payload.update(
        {
            "reason": None if gate["gate"] != "CONF-E" else "POST_ACCESS_OR_EXECUTION_PROTOCOL",
            "fresh_confirmation_authorized": "consumed",
            "frozen_for_fresh_confirmation": True,
            "planner_ready": False,
            "deployment_verified": False,
            "phase4e_status": PHASE4E_STATUS,
            "deployment_pass": False,
            "mean_delta_payload": summary["mean_delta"],
            "median_delta_payload": summary["median_delta"],
            "ci_low": summary["ci_low"],
            "ci_high": summary["ci_high"],
            "positive_scenarios": positive,
            "tie_scenarios": tie,
            "negative_scenarios": negative,
            "illegal_assignment": illegal,
            "state_contamination": contamination,
            "nan_final_decision": nan_count,
            "inf_final_decision": inf_count,
            "unhandled_runtime_exception": unhandled,
            "fallback_count": summary["fallback_count"],
            "timeout_count": summary["timeout_count"],
            "latency_gate_active": False,
            "first_access_utc": stamp,
        }
    )
    _dump("phase5zp_fresh_confirmation_gate.json", gate_payload)
    access = json.loads((PHASE / "phase5zp_first_access_record.json").read_text(encoding="utf-8"))
    _write_report(gate, rows, summary, access)
    print({"gate": gate["gate"], "mean_delta": summary["mean_delta"], "ci_low": summary["ci_low"], "ci_high": summary["ci_high"]}, flush=True)


if __name__ == "__main__":
    main()
