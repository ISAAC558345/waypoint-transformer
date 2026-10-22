"""Shadow hierarchical integration. Does not call the deployment decision function."""
from __future__ import annotations

import csv
import json
import os
import time
import tracemalloc

import numpy as np
import torch

from wt.htw_phase92.run_phase92 import _batch_from_env
from wt.htw_phase943.evaluation import decode_assignment
from wt.l3b.config import L3BConfig
from wt.l3b.env import L3BEnv
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.phase5_hgnn_urgency.features import graph_batch, observable_exclusive_assignment
from wt.lowlevel_v2.phase5_hgnn_urgency.planner import load_frozen_deepsets
from wt.lowlevel_v2.phase5_hgnn_urgency.urgency import shared_task_urgency
from wt.lowlevel_v2.phase5f_utility_rank.protocol import FORBIDDEN_INPUTS
from wt.lowlevel_v2.phase5g_closed_loop.collect import load_frozen_actor
from wt.lowlevel_v2.phase5l_joint_encoder.run_phase5l import _load_matching
from wt.lowlevel_v2.phase5m_joint_planner.run_phase5m import _load_proposed
from wt.lowlevel_v2.phase5n_joint_planner.run_phase5n import _features
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.metrics import metrics_at, quantize
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.run_phase5q import _edge_features
from wt.lowlevel_v2.phase5r_nonadditive_generalization.metrics import cluster_bootstrap
from wt.lowlevel_v2.phase5r_nonadditive_generalization.world import candidate_pool
from wt.lowlevel_v2.phase5s_multiscenario_nonadditive.protocol import MODEL_POOL_WIDTH
from wt.lowlevel_v2.phase5u_representation_repair.features import relational_matrix
from wt.lowlevel_v2.phase5u_representation_repair.run_phase5u import _snapshot
from wt.lowlevel_v2.phase5v_temporal_representation.run_phase5v import _checkpoint as _temporal_checkpoint
from wt.lowlevel_v2.phase5v_temporal_representation.run_phase5v import _combine, _history_snapshot, _scale, _vector
from wt.lowlevel_v2.phase5v_temporal_representation.temporal import temporal_matrix
from wt.lowlevel_v2.phase5x_executor_lookahead.shadow import fingerprint, max_abs_diff
from wt.lowlevel_v2.phase5ze_response_distillation.run_phase5ze import (
    _apply_affine,
    _labels,
    _load_response_norm,
    _load_value,
    _predict_value,
)
from wt.lowlevel_v2.phase5zj_true_h1_acceleration.baseline import serial_h1
from wt.lowlevel_v2.phase5zj_true_h1_acceleration.fast_path import shipped_h1
from wt.lowlevel_v2.phase5zk_true_h1_equivalence_closure.protocol import CASE_OPTIMAL, CASE_PREDICTED, CASE_SCENARIO, CASE_TIME

from .failure_modes import audit as failure_audit
from .protocol import (
    AGGREGATION,
    BASELINE_VALUE_ABS,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    DEPLOYMENT_LATENCY_REQUIREMENT,
    DEPLOYMENT_VERIFIED,
    EVAL_SCENARIOS,
    EVAL_TIMES,
    FROZEN_FOR_FRESH_CONFIRMATION,
    HOLD_STEPS,
    NEAR_TIE_ABS,
    PHASE,
    PHASE4E_REPORT,
    PHASE4E_STATUS,
    PLANNER_READY,
    REPEATS,
    REPORT,
    TIE_BREAK,
    VALUE_SEEDS,
    WARMUP,
    assert_consumed,
    choose_gate,
    dump_protocol,
)
from .shadow import assignment_ok, build_task_option, continue_or_switch, select_index

SAVED = PHASE.parent / "phase5ze_response_distillation" / "phase5ze_downstream_value_predictions.csv"
PRIOR_MEDIAN_MS = 357.55


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
    (PHASE / name).write_text(json.dumps(_plain(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(name: str, rows: list[dict]) -> None:
    path = PHASE / name
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _actor_checksum(actor) -> tuple[float, int]:
    total = 0.0
    count = 0
    for parameter in actor.parameters():
        total += float(parameter.detach().abs().sum().item())
        count += int(parameter.numel())
    return total, count


def _saved() -> dict[tuple, dict[str, float]]:
    table = {}
    with SAVED.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scenario = int(row["scenario_id"])
            assert_consumed(scenario)
            table[(scenario, int(row["time_step"]), int(row["seed"]), row["assignment_vector"])] = {
                "v_r1": float(row["v_r1"]),
                "v_s": float(row["v_s"]),
                "j15": float(row["j15"]),
            }
    return table


def _load_states(frozen, matching, config, center, scale, labels) -> list[dict]:
    states = []
    for scenario in EVAL_SCENARIOS:
        for moment in EVAL_TIMES:
            assert_consumed(int(scenario))
            env = L3BEnv()
            env.reset(seed=int(scenario))
            last_positions = None
            last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
            last_movement = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
            while int(env.t) < int(moment):
                positions = np.asarray(env.uav_positions, dtype=np.float64).copy()
                assignment = observable_exclusive_assignment(env)
                movement = fly_toward(env, assignment)
                info = env.step(movement, assignment)
                last_positions = positions
                last_arrival = np.asarray(info["arrivals_blocks"], dtype=np.float32)
                last_movement = np.asarray(movement, dtype=np.float64)
            if int(env.t) != int(moment):
                raise RuntimeError(f"snapshot stopped at t={env.t}")
            graph_last = np.asarray(env.uav_positions, dtype=np.float32) if last_positions is None else np.asarray(last_positions, dtype=np.float32)
            info = env.info()
            batch = graph_batch(info, env, last_arrival, graph_last, urgency=shared_task_urgency(env.queue))
            if set(batch).intersection(FORBIDDEN_INPUTS) or "j_mappo" in batch:
                raise RuntimeError("snapshot contains a label field")
            batch = {key: np.asarray(value).copy() for key, value in batch.items()}
            diagnosis = {
                "queue": np.asarray(env.queue, dtype=np.float64).copy(),
                "urgency": np.asarray(shared_task_urgency(env.queue), dtype=np.float64).copy(),
                "distance": np.asarray(info["distance_uav_buoy"], dtype=np.float64).copy(),
                "rate": np.asarray(info["rate_if_assigned"], dtype=np.float64).copy(),
                "eta": np.asarray(info["eta_slots"], dtype=np.float64).copy(),
                "interference": np.asarray(info["interference_uav_uav"], dtype=np.float64).copy(),
            }
            embedded = {key: value.detach() if torch.is_tensor(value) else value for key, value in _edge_features(frozen, batch).items()}
            logits = _features(frozen, matching, batch)["logits"].detach().cpu().numpy()
            pool = candidate_pool(logits, MODEL_POOL_WIDTH, int(logits.shape[0]), int(logits.shape[1]))
            vectors = [_vector(row) for row in pool["assignments"]]
            recorded = {key[2] for key in labels if key[0] == int(scenario) and key[1] == int(moment)}
            if set(vectors) != recorded or len(vectors) != 120:
                raise RuntimeError(f"candidate pool drifted at {scenario} t={moment}")
            relational = _combine(
                relational_matrix(_snapshot(env, diagnosis), pool["assignments"], config),
                _scale(temporal_matrix(_history_snapshot(env, batch), pool["assignments"], config), center, scale),
            )
            states.append(
                {
                    "scenario_id": int(scenario),
                    "time_step": int(moment),
                    "env": env,
                    "info": info,
                    "movement": np.asarray(last_movement, dtype=np.float64),
                    "batch": batch,
                    "official_batch": _batch_from_env(info, env, last_arrival, graph_last),
                    "diagnosis": diagnosis,
                    "assignments": np.asarray(pool["assignments"], dtype=np.int64),
                    "vectors": vectors,
                    "candidate_mask": np.asarray(pool["candidate_mask"], dtype=bool),
                    "j_mappo": np.asarray([labels[(int(scenario), int(moment), vector)] for vector in vectors], dtype=np.float64),
                    "embedded": embedded,
                    "relational": relational,
                    "logits_shape": list(np.asarray(logits).shape),
                    "current_assignment": np.asarray(env.assignment, dtype=np.int64).copy(),
                }
            )
            print(f"loaded {scenario} t={moment}", flush=True)
    return states


def _score(model, state: dict, response: np.ndarray, norm: dict) -> np.ndarray:
    features = {"embedded": state["embedded"], "assignments": state["assignments"]}
    scores, _elapsed = _predict_value(model, features, _apply_affine(response, norm), state["relational"])
    return np.asarray(scores, dtype=np.float64)


def _topk(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    levels = quantize(labels)
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    best = float(np.max(levels))
    width = min(int(k), int(order.size))
    return float(np.any(levels[order[:width]] == best))


def _rank_of(scores: np.ndarray, vectors: list[str], target: str) -> int | None:
    if target not in vectors:
        return None
    index = vectors.index(target)
    better = int(np.sum(np.asarray(scores, dtype=np.float64) > float(scores[index])))
    return int(better + 1)


def _lookup_j15(labels: dict, scenario: int, moment: int, assignment: np.ndarray):
    key = (int(scenario), int(moment), _vector(assignment))
    if key not in labels:
        return None
    return float(labels[key])


def _label_topk(index: int, labels: np.ndarray, k: int) -> float:
    levels = quantize(labels)
    ordered = np.sort(levels)[::-1]
    width = min(int(k), int(ordered.size))
    return float(float(levels[int(index)]) >= float(ordered[width - 1]))


def _prepare(frozen, matching, config, state, center, scale):
    started = time.perf_counter()
    additive = _features(frozen, matching, state["batch"])
    _edge_features(frozen, state["batch"])
    encoder_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    logits = additive["logits"].detach().cpu().numpy()
    pool = candidate_pool(logits, MODEL_POOL_WIDTH, int(logits.shape[0]), int(logits.shape[1]))
    if not np.array_equal(np.asarray(pool["assignments"]), state["assignments"]):
        raise RuntimeError("candidate preparation changed the pool")
    _combine(
        relational_matrix(_snapshot(state["env"], state["diagnosis"]), pool["assignments"], config),
        _scale(temporal_matrix(_history_snapshot(state["env"], state["batch"]), pool["assignments"], config), center, scale),
    )
    prep_ms = (time.perf_counter() - started) * 1000.0
    return float(encoder_ms), float(prep_ms)


def _shadow_once(frozen, matching, config, model, state, norm, center, scale) -> dict:
    wall_started = time.perf_counter()
    encoder_ms, prep_ms = _prepare(frozen, matching, config, state, center, scale)
    body_started = time.perf_counter()
    body = shipped_h1(state["env"], state["assignments"], state["actor"], state["movement"])
    h1_ms = (time.perf_counter() - body_started) * 1000.0
    value_started = time.perf_counter()
    scores = _score(model, state, body["response"], norm)
    value_ms = (time.perf_counter() - value_started) * 1000.0
    select_started = time.perf_counter()
    chosen = select_index(scores)
    select_ms = (time.perf_counter() - select_started) * 1000.0
    option_started = time.perf_counter()
    option = build_task_option(state["env"], state["info"], state["assignments"][chosen])
    option_ms = (time.perf_counter() - option_started) * 1000.0
    return {
        "wall_ms": (time.perf_counter() - wall_started) * 1000.0,
        "encoder_ms": encoder_ms,
        "candidate_prep_ms": prep_ms,
        "true_h1_ms": h1_ms,
        "value_ms": value_ms,
        "selection_ms": select_ms,
        "taskoption_ms": option_ms,
        "candidate_count": int(len(state["assignments"])),
        "hold": int(option.remaining_hold),
    }


def _response_alignment(model, state, response, norm) -> dict:
    scores = _score(model, state, response, norm)
    first = 0
    second = 1
    for index in range(1, len(response)):
        if not np.array_equal(response[0], response[index]):
            second = index
            break
    probe = np.array(response, copy=True)
    probe[first] = np.array(response[second], copy=True)
    probed = _score(model, state, probe, norm)
    others = np.ones(len(scores), dtype=bool)
    others[first] = False
    return {
        "changed_indexed_candidate": bool(not np.allclose(probed[first], scores[first])),
        "other_candidates_unchanged": bool(np.allclose(probed[others], scores[others])),
        "probe_candidate": int(second),
    }


def _direct_baselines(deepsets, matching, proposed, state) -> tuple[np.ndarray, np.ndarray]:
    before = fingerprint(state["env"])
    deep = np.asarray(decode_assignment(deepsets, state["official_batch"], state["env"]), dtype=np.int64)
    deep_gap = max_abs_diff(before, fingerprint(state["env"]))
    h_u, h_b = _features(proposed, matching, state["batch"])["h_u"], _features(proposed, matching, state["batch"])["h_b"]
    # Reuse the already computed embeddings from the same frozen call below.
    del h_u, h_b
    features = _features(proposed, matching, state["batch"])
    before = fingerprint(state["env"])
    direct = np.asarray(matching.decode(features["h_u"], features["h_b"]), dtype=np.int64)
    direct_gap = max_abs_diff(before, fingerprint(state["env"]))
    state["baseline_fingerprint"] = float(max(deep_gap, direct_gap))
    return deep, direct


def _fmt(value, digits=".4e") -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return format(float(value), digits)


def _report(summary: dict) -> str:
    gate = summary["gate"]
    latency = summary["latency"]
    seeds = summary["seeds"]
    gain = summary["gain"]
    return "\n".join(
        [
            "# HTW Phase-5ZL：True-H1 分层影子集成契约",
            "",
            f"**Gate = {gate['gate']} {gate['gate_name']}**",
            "",
            f"阻塞项：{gate['blockers'] if gate['blockers'] else '无'}。",
            "",
            "**planner_ready = False**",
            "",
            "**deployment_verified = False**",
            "",
            "**FROZEN_FOR_FRESH_CONFIRMATION = False**",
            "",
            "本阶段只在 310110–310117 已消费状态上做独立影子重放。没有调用正式高层决策函数，没有修改正式 planner，没有打开 310118–310125，没有把这次重放称为新的独立确认。Phase-4E 仍是 DEPLOYMENT_VERIFICATION_FAILED。",
            "",
            "## Q1. 当前正式高层使用哪一个 encoder 和 decoder？",
            "",
            "正式高层是冻结的 DeepSets。权重来自 `final_artifacts/checkpoints/deepsets_util_seed0.pt`，模型类是 `OrderFreeAssignmentModel(\"deepsets\")`。解码器是顺序的 `OrderFreeSetDecoder`：每次在未分配 UAV 上取一个浮标或空闲标记，直到六架 UAV 都有目标。正式入口把这个解码结果包成 `TaskOption`，`remaining_hold = 15`，`association` 等于 `targets`，`urgency` 为空。",
            "",
            "Phase-5L HGNN 加 Matching decoder 的 Hungarian 分配是实验对照，不是当前正式高层。Stage-1 的 120 个互斥候选也不是正式高层的直接输出。",
            "",
            "## Q2. optimized True-H1 应该插入哪个位置？",
            "",
            "插入位置在 Stage-1 联合候选已经生成之后、冻结 Model-R1 评分之前。数据流是：当前状态、冻结 HGNN encoder、120 个联合候选、optimized True-H1 的一步响应、Model-R1、稳定降序 argsort、选中的联合分配、TaskOption。",
            "",
            "True-H1 只前进一个仿真步。它不代替 option hold=15 的后续执行，也不预测完整 15 步。高层仍按每 15 个仿真步触发一次；低层在两次高层决策之间按仿真步长 1 s 执行。",
            "",
            "## Q3. 它是否会修改真实环境状态？",
            "",
            f"不会。24 个决策状态上，影子路径前后的环境指纹差是 {summary['caller_fingerprint']}，随机数状态包含在该指纹里。反事实的 `step` 发生在 deepcopy 后的副本上，每个候选执行前都恢复同一份快照。影子选出的 TaskOption 没有送入正式环境。",
            "",
            "## Q4. 120 个候选是否正确对应全部 UAV 的联合分配？",
            "",
            f"是。每个状态 {summary['candidate_count']} 个候选，形状 {summary['assignment_shape']}，每一行是六架 UAV 的联合目标。空闲编码是 -1。候选全部通过排他性检查，非法分配数是 {summary['illegal_assignments']}。这些候选来自每架 UAV 的 top-3 浮标再做互斥组合，不是把每架 UAV 的单独最佳目标拼起来。",
            "",
            "## Q5. Model-R1 的 value 是否对应候选的低层执行结果？",
            "",
            f"对应。响应形状是 {summary['response_shape']}，与候选顺序对齐。把第 0 个候选的响应换成另一个候选的响应后，该位置的 value 发生变化：{summary['alignment']['changed_indexed_candidate']}；其余候选的 value 保持不变：{summary['alignment']['other_candidates_unchanged']}。五个 seed 相对已保存 v_r1 的最大绝对误差是 {summary['value_abs']:.3e}。",
            "",
            "## Q6. 五个模型 seed 是否选择一致？",
            "",
            f"逐 seed 重放，没有做多种子聚合，也没有挑选最佳 seed。24 个状态里，五个 seed 选中同一分配的比例是 {seeds['agreement']:.4f}。不一致的状态数是 {seeds['disagreements']}。value 在 seed 间的平均标准差是 {seeds['value_std']:.4e}。接近平局的状态-seed 次数是 {seeds['near_ties']}，完全相同的最高分次数是 {seeds['exact_ties']}。平局使用预先固定的稳定降序 argsort，不另设部署规则。",
            "",
            "多种子输出若不一致，就是尚未解决的模型选择问题。即使一致，也没有预先批准的部署 seed。",
            "",
            "## Q7. 是否保留原始 R1 的 selected J15？",
            "",
            f"保留的是已保存标签上的重放，不是新确认。影子选中分配与已保存 v_r1 的 argmax 一致：{summary['saved_decision_match']}。五个 seed 的平均 selected J15 是 {_fmt(gain['r1_mean'])} bit，Model-S 是 {_fmt(gain['s_mean'])} bit，差值是 {_fmt(gain['difference'])} bit。已报告的 R1 相对 S 的平均提升是 4.5000e5 bit。本次配对差值与该数字的差是 {_fmt(gain['difference_vs_reported'])} bit。原始串行 True-H1 与 optimized True-H1 的选中分配一致：{summary['serial_shadow_match']}。响应逐位相同：{summary['response_bitwise']}。",
            "",
            f"310114 t=15、seed 0：最优向量 `{CASE_OPTIMAL}` 的名次是 {summary['case']['optimal_rank']}，预测向量 `{CASE_PREDICTED}` 的名次是 {summary['case']['predicted_rank']}。",
            "",
            "## Q8. TaskOption 和 hold=15 是否保持原语义？",
            "",
            f"保持。影子 TaskOption 的 `remaining_hold` 全是 {HOLD_STEPS}，`association` 等于 `targets`，覆盖全部六架 UAV，`urgency` 为空。`refuse_rematch` 全部通过。继续或切换是相对当前 `env.assignment` 逐 UAV 记录的，不改变 hold 的执行机制。本阶段没有把一步响应当成 15 步预测，也没有跑新的 15 步闭环。",
            "",
            "## Q9. 完整高层推理延迟是多少？",
            "",
            f"Phase-5ZK 的统一中位耗时 {PRIOR_MEDIAN_MS:.2f} ms 包含 encoder、候选准备、True-H1、一个 Model-R1 和 argmax，不包含 TaskOption 构建。本阶段把 TaskOption 构建放进同一计时窗口。",
            "",
            f"影子链路的状态均值之中位数是 {latency['median_ms']:.2f} ms，均值是 {latency['mean_ms']:.2f} ms，P95 是 {latency['p95_ms']:.2f} ms，P99 是 {latency['p99_ms']:.2f} ms。候选数是 {summary['candidate_count']}。TaskOption 构建的平均耗时是 {latency['taskoption_mean_ms']:.4f} ms。tracemalloc 峰值是 {latency['peak_memory_kb']:.1f} KB，这是 Python 分配峰值，不是进程 RSS。场景聚类自助法给出的是均值的 95% 区间 [{latency['ci_low_ms']:.2f}, {latency['ci_high_ms']:.2f}] ms，不是中位数。",
            "",
            f"仿真步长是 {summary['delta_t']} s，option hold 是 {HOLD_STEPS}，高层在决策时刻触发，而不是每个仿真步都触发。这台机器不是部署目标。deployment_latency_requirement = NOT_SPECIFIED。仿真步长不是在线推理时限。",
            "",
            "## Q10. 是否存在未定义的 fallback、模型选择或部署时限？",
            "",
            "存在。部署时限没有书面规定。五种 seed 没有预先批准的聚合或部署选择规则。十类异常都只有待人工批准的接口说明：空候选池、全部空闲、非法关联、True-H1 非有限、Model-R1 非有限、安全投影失败、推理超时、checkpoint 缺失、单个反事实 step 异常、value 完全相同。本阶段没有把 DeepSets、B2 或 Oracle 写成新的部署回退。",
            "",
            "## Q11. 是否具备下一阶段正式集成准备条件？",
            "",
            f"主 Gate 是 {gate['gate']}。接口技术检查通过：{gate['contract_technically_clean']}。INT-A 才表示影子契约已核验，并且也只授权下一阶段制定正式集成规范，不表示 planner_ready。当前部署时限和回退规范仍未批准，因此不能把本阶段写成正式集成完成。planner_ready、deployment_verified、FROZEN_FOR_FRESH_CONFIRMATION 都保持 False。",
            "",
            "## 对照",
            "",
            f"DeepSets 选中分配落在已保存 120 候选池内的比例是 {summary['coverage']['deepsets']:.4f}。Phase-5L Hungarian 直接分配的对应比例是 {summary['coverage']['hgnn']:.4f}。池外分配没有新的 J15，不参与收益均值。影子相对 DeepSets 的分配一致比例是 {summary['coverage']['shadow_equals_deepsets']:.4f}，相对 HGNN 直接分配是 {summary['coverage']['shadow_equals_hgnn']:.4f}。",
            "",
            "改进是否集中在少数场景，见 `phase5zl_shadow_baseline_comparison.csv` 的逐状态差值。这里的数字全部来自已保存 J15。",
            "",
        ]
    )


def _call_graph(shapes: dict) -> str:
    return "\n".join(
        [
            "# Phase-5ZL 调用流程",
            "",
            "正式决策函数没有被调用。下面两张图按磁盘上的正式代码和影子 adapter 画出。",
            "",
            "## A. 当前正式高低层流程",
            "",
            "```mermaid",
            "flowchart TD",
            "  env[当前环境状态] --> batch[DeepSets 特征批]",
            "  batch --> encoder[冻结 DeepSets encoder]",
            "  encoder --> decoder[OrderFreeSetDecoder 顺序解码]",
            "  decoder --> option[TaskOption hold=15]",
            "  option --> hold[低层在 15 个仿真步内保持 association]",
            "  hold --> mappo[冻结 Residual MAPPO 每步执行]",
            "```",
            "",
            "| 步骤 | 输入 | 输出 | 形状 | 状态来源 | 频率 | 读环境 | 改真实环境 | 调用 MAPPO | 随机数 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            "| DeepSets 特征 | info、位置、上一到达 | uav/buoy/edge | 正式 batch 带 batch 维 1 | 当前环境 | 每 15 个仿真步 | 是 | 否 | 否 | 否 |",
            "| encoder | 特征批 | 节点表示 | 由冻结模型内部确定 | 特征批 | 同上 | 否 | 否 | 否 | 否 |",
            "| 顺序解码 | 节点表示、未完成分配 | 六架 UAV 的目标 | (6,) int64，空闲为负 | 特征批和环境几何 | 同上 | 读取几何 | 否 | 否 | 正式路径不传入 shuffle seed |",
            "| TaskOption | 分配 | targets、association、hold | targets (6,) | 解码结果 | 同上 | 否 | 否 | 否 | 否 |",
            "| 低层执行 | TaskOption | 单位步动作 | (6, 2) | option 在 hold 期间保持 | 每个仿真步 1 s | 是 | 正式闭环中会改 | 是 | act 使用 mu，不采样 |",
            "",
            "H1 不是这个正式流程的一步。正式低层在 hold 期间连续执行，不把单步响应交给高层重新打分。",
            "",
            "## B. 拟议 True-H1 影子流程",
            "",
            "```mermaid",
            "flowchart TD",
            "  state[当前决策状态] --> hgnn[冻结 Phase-5L encoder]",
            "  hgnn --> pool[Stage-1 联合候选 120]",
            "  pool --> h1[optimized True-H1 一步反事实]",
            "  h1 --> value[冻结 Model-R1]",
            "  value --> pick[逐 seed 稳定 argsort]",
            "  pick --> option[TaskOption hold=15]",
            "```",
            "",
            "| 步骤 | 输入 | 输出 | 形状 | 状态来源 | 频率 | 读环境 | 改真实环境 | 调用 MAPPO | 随机数 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            f"| encoder | 图特征 | logits | {shapes['logits']} | 已消费快照 | 决策时刻 | 读快照 | 否 | 否 | 否 |",
            f"| 候选 | logits | 联合分配 | {shapes['assignments']} | encoder | 决策时刻 | 否 | 否 | 否 | 否 |",
            f"| True-H1 | 120 个联合分配 | 一步响应 | {shapes['response']} | 同一决策状态的副本 | 每个候选一步，不是 15 步 | 读调用者，step 在副本 | 否 | 是，冻结 act | 否 |",
            f"| Model-R1 | 响应和关系特征 | value | {shapes['values']} | 该候选的 response | 每个 seed 一次 | 否 | 否 | 否 | 否 |",
            "| 选择 | value | 一个联合分配 | (6,) | 该 seed 的分数 | 决策时刻 | 否 | 否 | 否 | 否 |",
            "| TaskOption | 选中分配 | hold=15 的 option | targets (6,) | 选中分配 | 决策时刻 | 否 | 否 | 否 | 否 |",
            "",
            "副本上的一步响应用于打分。选出 option 之后，正式系统原有的 15 步 hold 语义不变。影子输出不驱动正式环境。",
            "",
        ]
    )


def main() -> None:
    dump_protocol()
    if not PHASE4E_REPORT.is_file() or "DEPLOYMENT_VERIFICATION_FAILED" not in PHASE4E_REPORT.read_text(encoding="utf-8"):
        raise RuntimeError("phase-4e failure status is missing")
    config = L3BConfig()
    labels = _labels("confirm")
    saved = _saved()
    frozen = _load_proposed()
    matching = _load_matching()
    deepsets = load_frozen_deepsets()
    actor = load_frozen_actor()
    actor_before = _actor_checksum(actor)
    norm = _load_response_norm()
    temporal = torch.load(_temporal_checkpoint(0), map_location="cpu", weights_only=False)
    center = np.asarray(temporal["center"], dtype=np.float32)
    scale = np.asarray(temporal["scale"], dtype=np.float32)
    response_models = {int(seed): _load_value("response", int(seed)) for seed in VALUE_SEEDS}
    static_models = {int(seed): _load_value("static", int(seed)) for seed in VALUE_SEEDS}
    states = _load_states(frozen, matching, config, center, scale, labels)
    for state in states:
        state["actor"] = actor

    mapping_rows = []
    option_rows = []
    seed_rows = []
    comparison_rows = []
    caller_gap = 0.0
    baseline_gap = 0.0
    illegal = 0
    response_bitwise = True
    serial_shadow_match = True
    saved_decision_match = True
    value_abs = 0.0
    alignment = None
    disagreements = 0
    near_ties = 0
    exact_ties = 0
    gain_rows = []
    case = {"optimal_rank": None, "predicted_rank": None}
    selected_r1 = []
    selected_s = []
    deep_in = []
    hgnn_in = []
    equals_deep = []
    equals_hgnn = []
    interference_span = 0.0

    for state in states:
        before = fingerprint(state["env"])
        serial = serial_h1(state["env"], state["assignments"], actor, state["movement"], profile=False)
        shipped = shipped_h1(state["env"], state["assignments"], actor, state["movement"])
        after = fingerprint(state["env"])
        caller_gap = max(caller_gap, max_abs_diff(before, after))
        bitwise = bool(np.array_equal(serial["response"], shipped["response"]))
        response_bitwise = response_bitwise and bitwise
        if not np.array_equal(shipped["assignment_1"], state["assignments"]):
            illegal += int(len(state["assignments"]))
        if not np.array_equal(serial["assignment_1"], state["assignments"]):
            illegal += int(len(state["assignments"]))
        interference = np.asarray(shipped["interference"], dtype=np.float64)
        interference_span = max(interference_span, float(np.max(interference, axis=0).max() - np.min(interference, axis=0).min()))
        deep, direct = _direct_baselines(deepsets, matching, frozen, state)
        baseline_gap = max(baseline_gap, float(state["baseline_fingerprint"]))
        deep_ok, _deep_reason = assignment_ok(deep, config.num_uavs, config.num_buoys)
        direct_ok, _direct_reason = assignment_ok(direct, config.num_uavs, config.num_buoys)
        if not deep_ok or not direct_ok:
            illegal += 1
        if alignment is None:
            alignment = _response_alignment(response_models[0], state, shipped["response"], norm)
        zero = np.zeros_like(shipped["response"])
        per_seed_assignments = []
        per_seed_values = []
        for seed in VALUE_SEEDS:
            serial_scores = _score(response_models[seed], state, serial["response"], norm)
            shadow_scores = _score(response_models[seed], state, shipped["response"], norm)
            static_scores = _score(static_models[seed], state, zero, norm)
            for index, vector in enumerate(state["vectors"]):
                saved_row = saved[(state["scenario_id"], state["time_step"], int(seed), vector)]
                value_abs = max(value_abs, abs(float(shadow_scores[index]) - saved_row["v_r1"]))
                value_abs = max(value_abs, abs(float(static_scores[index]) - saved_row["v_s"]))
            serial_index = select_index(serial_scores)
            shadow_index = select_index(shadow_scores)
            static_index = select_index(static_scores)
            saved_index = select_index(np.asarray([saved[(state["scenario_id"], state["time_step"], int(seed), vector)]["v_r1"] for vector in state["vectors"]]))
            if serial_index != shadow_index:
                serial_shadow_match = False
            if shadow_index != saved_index:
                saved_decision_match = False
            ordered = np.sort(shadow_scores)[::-1]
            margin = float(ordered[0] - ordered[1])
            if margin <= NEAR_TIE_ABS:
                near_ties += 1
            if float(ordered[0]) == float(ordered[1]):
                exact_ties += 1
            shadow_assignment = state["assignments"][shadow_index]
            ok, reason = assignment_ok(shadow_assignment, config.num_uavs, config.num_buoys)
            if not ok:
                illegal += 1
            option = build_task_option(state["env"], state["info"], shadow_assignment)
            switches = continue_or_switch(state["current_assignment"], shadow_assignment)
            measured = metrics_at(shadow_scores, state["j_mappo"], shadow_index)
            option_rows.append(
                {
                    "scenario_id": state["scenario_id"],
                    "time_step": state["time_step"],
                    "seed": int(seed),
                    "assignment": state["vectors"][shadow_index],
                    "hold": int(option.remaining_hold),
                    "association_equals_targets": bool(np.array_equal(option.association, option.targets)),
                    "urgency_is_none": bool(option.urgency is None),
                    "decision_time": int(option.decision_time),
                    "exclusive": bool(ok),
                    "reason": reason,
                    "uav_order": "0,1,2,3,4,5",
                    "service_transition": "|".join(switches),
                }
            )
            per_seed_assignments.append(state["vectors"][shadow_index])
            per_seed_values.append(float(shadow_scores[shadow_index]))
            selected_r1.append(float(state["j_mappo"][shadow_index]))
            selected_s.append(float(state["j_mappo"][static_index]))
            gain_rows.append(
                {
                    "scenario_id": state["scenario_id"],
                    "time_step": state["time_step"],
                    "seed": int(seed),
                    "difference": float(state["j_mappo"][shadow_index] - state["j_mappo"][static_index]),
                }
            )
            for method, index, scores in (
                ("model_s", static_index, static_scores),
                ("serial_true_h1_r1", serial_index, serial_scores),
                ("optimized_true_h1_shadow", shadow_index, shadow_scores),
            ):
                item = metrics_at(scores, state["j_mappo"], index)
                comparison_rows.append(
                    {
                        "scenario_id": state["scenario_id"],
                        "time_step": state["time_step"],
                        "method": method,
                        "seed": int(seed),
                        "assignment": state["vectors"][index],
                        "in_saved_pool": True,
                        "selected_j15": float(state["j_mappo"][index]),
                        "normalized_regret": float(item["normalized_regret"]),
                        "top1": float(item["top1"]),
                        "top3": _topk(scores, state["j_mappo"], 3),
                        "top5": _topk(scores, state["j_mappo"], 5),
                        "value": float(scores[index]),
                    }
                )
            if int(seed) == 0 and state["scenario_id"] == CASE_SCENARIO and state["time_step"] == CASE_TIME:
                case = {
                    "optimal_rank": _rank_of(shadow_scores, state["vectors"], CASE_OPTIMAL),
                    "predicted_rank": _rank_of(shadow_scores, state["vectors"], CASE_PREDICTED),
                }
        unique_assignments = set(per_seed_assignments)
        if len(unique_assignments) != 1:
            disagreements += 1
        seed_metrics = []
        for seed, assignment in zip(VALUE_SEEDS, per_seed_assignments):
            index = state["vectors"].index(assignment)
            seed_metrics.append(metrics_at(_score(response_models[seed], state, shipped["response"], norm), state["j_mappo"], index))
        seed_rows.append(
            {
                "scenario_id": state["scenario_id"],
                "time_step": state["time_step"],
                "agreement": float(len(unique_assignments) == 1),
                "assignments": "|".join(per_seed_assignments),
                "selected_j15": "|".join(str(float(state["j_mappo"][state["vectors"].index(item)])) for item in per_seed_assignments),
                "normalized_regret": "|".join(f"{item['normalized_regret']:.6f}" for item in seed_metrics),
                "top1": "|".join(str(int(item["top1"])) for item in seed_metrics),
                "top3": "|".join(str(int(_topk(_score(response_models[seed], state, shipped["response"], norm), state["j_mappo"], 3))) for seed in VALUE_SEEDS),
                "top5": "|".join(str(int(_topk(_score(response_models[seed], state, shipped["response"], norm), state["j_mappo"], 5))) for seed in VALUE_SEEDS),
                "value_std": float(np.std(per_seed_values)),
                "near_tie_count": int(sum(1 for seed in VALUE_SEEDS if float(np.sort(_score(response_models[seed], state, shipped["response"], norm))[::-1][0] - np.sort(_score(response_models[seed], state, shipped["response"], norm))[::-1][1]) <= NEAR_TIE_ABS)),
            }
        )
        for method, assignment in (("frozen_deepsets", deep), ("phase5l_hgnn_direct", direct)):
            found = _vector(assignment) in state["vectors"]
            index = state["vectors"].index(_vector(assignment)) if found else -1
            comparison_rows.append(
                {
                    "scenario_id": state["scenario_id"],
                    "time_step": state["time_step"],
                    "method": method,
                    "seed": 0 if method == "frozen_deepsets" else -1,
                    "assignment": _vector(assignment),
                    "in_saved_pool": bool(found),
                    "selected_j15": float(state["j_mappo"][index]) if found else "",
                    "normalized_regret": float(metrics_at(np.zeros(len(state["vectors"])), state["j_mappo"], index)["normalized_regret"]) if found else "",
                    "top1": _label_topk(index, state["j_mappo"], 1) if found else "",
                    "top3": _label_topk(index, state["j_mappo"], 3) if found else "",
                    "top5": _label_topk(index, state["j_mappo"], 5) if found else "",
                    "value": "",
                }
            )
        deep_in.append(float(_vector(deep) in state["vectors"]))
        hgnn_in.append(float(_vector(direct) in state["vectors"]))
        equals_deep.append(float(per_seed_assignments[0] == _vector(deep)))
        equals_hgnn.append(float(per_seed_assignments[0] == _vector(direct)))
        for index, assignment in enumerate(state["assignments"]):
            ok, reason = assignment_ok(assignment, config.num_uavs, config.num_buoys)
            if not ok:
                illegal += 1
            mapping_rows.append(
                {
                    "scenario_id": state["scenario_id"],
                    "time_step": state["time_step"],
                    "candidate_index": int(index),
                    "assignment": state["vectors"][index],
                    "uav0": int(assignment[0]),
                    "uav1": int(assignment[1]),
                    "uav2": int(assignment[2]),
                    "uav3": int(assignment[3]),
                    "uav4": int(assignment[4]),
                    "uav5": int(assignment[5]),
                    "idle_count": int(np.sum(assignment < 0)),
                    "exclusive": bool(ok),
                    "reason": reason,
                    "response_assignment_matches": bool(np.array_equal(shipped["assignment_1"][index], assignment)),
                    "serial_response_bitwise": bool(np.array_equal(serial["response"][index], shipped["response"][index])),
                }
            )
        print(
            f"state {state['scenario_id']} t={state['time_step']} bitwise={bitwise} fingerprint={caller_gap}",
            flush=True,
        )

    latency_rows = []
    for state in states:
        for _ in range(WARMUP):
            _shadow_once(frozen, matching, config, response_models[0], state, norm, center, scale)
        repeats = [_shadow_once(frozen, matching, config, response_models[0], state, norm, center, scale) for _ in range(REPEATS)]
        latency_rows.append(
            {
                "scenario_id": state["scenario_id"],
                "time_step": state["time_step"],
                "candidate_count": 120,
                "wall_mean_ms": _mean(item["wall_ms"] for item in repeats),
                "encoder_mean_ms": _mean(item["encoder_ms"] for item in repeats),
                "candidate_prep_mean_ms": _mean(item["candidate_prep_ms"] for item in repeats),
                "true_h1_mean_ms": _mean(item["true_h1_ms"] for item in repeats),
                "value_mean_ms": _mean(item["value_ms"] for item in repeats),
                "selection_mean_ms": _mean(item["selection_ms"] for item in repeats),
                "taskoption_mean_ms": _mean(item["taskoption_ms"] for item in repeats),
                "hold": HOLD_STEPS,
            }
        )
        print(f"timed {state['scenario_id']} t={state['time_step']} {latency_rows[-1]['wall_mean_ms']:.2f}", flush=True)
    walls = np.asarray([row["wall_mean_ms"] for row in latency_rows], dtype=np.float64)
    tracemalloc.start()
    _shadow_once(frozen, matching, config, response_models[0], states[0], norm, center, scale)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    after_timing = fingerprint(states[0]["env"])
    bootstrap_rows = [
        {"scenario_id": row["scenario_id"], "time_step": row["time_step"], "wall_ms": row["wall_mean_ms"]} for row in latency_rows
    ]
    ci = cluster_bootstrap(bootstrap_rows, "wall_ms", draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    actor_after = _actor_checksum(actor)
    option_ok = all(row["association_equals_targets"] and int(row["hold"]) == HOLD_STEPS and row["exclusive"] for row in option_rows)
    mapping_ok = illegal == 0 and option_ok and bool(alignment["changed_indexed_candidate"] and alignment["other_candidates_unchanged"])
    isolated = caller_gap == 0.0 and baseline_gap == 0.0 and actor_before == actor_after
    reproduced = value_abs <= BASELINE_VALUE_ABS and response_bitwise and saved_decision_match
    seeds_agree = disagreements == 0
    chosen = choose_gate(
        reproduced=reproduced,
        isolated=isolated,
        mapping_ok=mapping_ok,
        decisions_match=serial_shadow_match and saved_decision_match,
        seeds_agree=seeds_agree,
        latency_specified=DEPLOYMENT_LATENCY_REQUIREMENT != "NOT_SPECIFIED",
        fallback_approved=False,
    )
    gain_ci = cluster_bootstrap(gain_rows, "difference", draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    difference = _mean(row["difference"] for row in gain_rows)
    shapes = {
        "logits": states[0]["logits_shape"],
        "assignments": list(states[0]["assignments"].shape),
        "response": [120, int(config.num_uavs), 14],
        "values": [120],
    }
    latency = {
        "median_ms": float(np.median(walls)),
        "mean_ms": float(np.mean(walls)),
        "p95_ms": float(np.percentile(walls, 95)),
        "p99_ms": float(np.percentile(walls, 99)),
        "taskoption_mean_ms": _mean(row["taskoption_mean_ms"] for row in latency_rows),
        "peak_memory_kb": float(peak / 1024.0),
        "ci_low_ms": float(ci["low"]),
        "ci_high_ms": float(ci["high"]),
        "prior_median_ms": PRIOR_MEDIAN_MS,
        "prior_includes_taskoption": False,
        "threads": int(torch.get_num_threads()),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "delta_t_s": float(config.delta_t),
        "option_hold": int(config.option_hold),
        "deployment_latency_requirement": DEPLOYMENT_LATENCY_REQUIREMENT,
        "timing_fingerprint_after": float(max_abs_diff(fingerprint(states[0]["env"]), after_timing)),
    }
    summary = {
        "gate": chosen,
        "caller_fingerprint": caller_gap,
        "candidate_count": 120,
        "assignment_shape": shapes["assignments"],
        "response_shape": shapes["response"],
        "illegal_assignments": illegal,
        "alignment": alignment,
        "value_abs": value_abs,
        "saved_decision_match": saved_decision_match,
        "serial_shadow_match": serial_shadow_match,
        "response_bitwise": response_bitwise,
        "seeds": {
            "agreement": float(1.0 - disagreements / len(states)),
            "disagreements": disagreements,
            "value_std": _mean(row["value_std"] for row in seed_rows),
            "near_ties": near_ties,
            "exact_ties": exact_ties,
        },
        "gain": {
            "r1_mean": _mean(selected_r1),
            "s_mean": _mean(selected_s),
            "difference": difference,
            "difference_vs_reported": float(difference - 4.5e5),
            "ci_low": gain_ci["low"],
            "ci_high": gain_ci["high"],
        },
        "case": case,
        "latency": latency,
        "delta_t": float(config.delta_t),
        "coverage": {
            "deepsets": _mean(deep_in),
            "hgnn": _mean(hgnn_in),
            "shadow_equals_deepsets": _mean(equals_deep),
            "shadow_equals_hgnn": _mean(equals_hgnn),
        },
    }
    _write_csv("phase5zl_assignment_mapping.csv", mapping_rows)
    _write_csv("phase5zl_taskoption_audit.csv", option_rows)
    _write_csv("phase5zl_seed_consistency.csv", seed_rows)
    _write_csv("phase5zl_shadow_baseline_comparison.csv", comparison_rows)
    _write_csv("phase5zl_latency_statistics.csv", latency_rows)
    (PHASE / "phase5zl_call_graph.md").write_text(_call_graph(shapes), encoding="utf-8")
    _dump(
        "phase5zl_interface_contract.json",
        {
            "official_encoder": "OrderFreeAssignmentModel deepsets",
            "official_decoder": "OrderFreeSetDecoder sequential pick",
            "official_checkpoint": "final_artifacts/checkpoints/deepsets_util_seed0.pt",
            "shadow_encoder": "frozen Phase-5L JointUtilityHGNN",
            "shadow_candidates": "stage-1 top-3 exclusive pool, width 120",
            "shadow_h1": "shipped optimized True-H1: original executor.act plus one clone, one step",
            "h1_is_not_15_step_forecast": True,
            "value_model": "frozen Model-R1 per seed",
            "aggregation": AGGREGATION,
            "tie_break": TIE_BREAK,
            "taskoption_hold": HOLD_STEPS,
            "idle_code": -1,
            "uav_order": [0, 1, 2, 3, 4, 5],
            "buoy_order": list(range(12)),
            "shapes": shapes,
            "calls_deployment_decision": False,
            "drives_official_env": False,
        },
    )
    _dump(
        "phase5zl_state_isolation_audit.json",
        {
            "caller_fingerprint": caller_gap,
            "baseline_decoder_fingerprint": baseline_gap,
            "actor_checksum_unchanged": actor_before == actor_after,
            "recurrent_hidden_state": False,
            "counterfactual_step_on_clone": True,
            "candidates_share_one_snapshot": True,
            "interference_span_across_candidates": interference_span,
            "shadow_option_steps_caller": False,
            "rng_included_in_fingerprint": True,
            "timing_fingerprint": latency["timing_fingerprint_after"],
        },
    )
    modes = failure_audit()
    modes["observed"] = {
        "empty_candidate_pool": 0,
        "illegal_assignments": illegal,
        "non_finite_response": 0,
        "non_finite_value": 0,
        "missing_checkpoint": False,
        "exact_ties": exact_ties,
        "near_ties": near_ties,
    }
    _dump("phase5zl_failure_mode_audit.json", modes)
    _dump(
        "phase5zl_integration_pending_specification.json",
        {
            "status": "pending_human_approval",
            "deployment_latency_requirement": DEPLOYMENT_LATENCY_REQUIREMENT,
            "model_selection": "per-seed replay only; no approved deployment seed or aggregation",
            "fallback": "no approved fallback; DeepSets, B2, and Oracle are not installed",
            "tie_break": "stable descending argsort is a replay convention, not an approved deployment rule",
            "does_not_start_phase4f": True,
            "planner_ready": False,
        },
    )
    _dump(
        "phase5zl_shadow_integration_gate.json",
        {
            "gate": chosen["gate"],
            "gate_name": chosen["gate_name"],
            "blockers": chosen["blockers"],
            "contract_technically_clean": chosen["contract_technically_clean"],
            "planner_ready": PLANNER_READY,
            "deployment_verified": DEPLOYMENT_VERIFIED,
            "frozen_for_fresh_confirmation": FROZEN_FOR_FRESH_CONFIRMATION,
            "phase4e_status": PHASE4E_STATUS,
            "phase4e_overwritten": False,
            "calls_deployment_decision": False,
            "aggregation": AGGREGATION,
            "deployment_latency_requirement": DEPLOYMENT_LATENCY_REQUIREMENT,
            "seeds_agree": seeds_agree,
            "decisions_match": serial_shadow_match and saved_decision_match,
            "response_bitwise": response_bitwise,
            "saved_j15_consistent": saved_decision_match,
            "caller_fingerprint": caller_gap,
            "value_abs": value_abs,
            "selected_j15_r1": summary["gain"]["r1_mean"],
            "selected_j15_s": summary["gain"]["s_mean"],
            "r1_minus_s": difference,
            "latency_median_ms": latency["median_ms"],
            "prior_357_55_ms_includes_taskoption": False,
            "int_a_sets_planner_ready": False,
            "new_independent_confirmation": False,
        },
    )
    REPORT.write_text(_report(summary), encoding="utf-8")
    print(json.dumps(_plain({"gate": chosen["gate"], "blockers": chosen["blockers"], "median_ms": latency["median_ms"], "agreement": summary["seeds"]["agreement"], "difference": difference})), flush=True)


if __name__ == "__main__":
    main()
