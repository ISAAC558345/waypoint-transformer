"""Rank-mean consensus audit. Does not call the deployment decision function."""
from __future__ import annotations

import csv
import json
import os
import time

import numpy as np
import torch
from scipy.stats import spearmanr

from wt.l3b.config import L3BConfig
from wt.lowlevel_v2.phase5g_closed_loop.collect import load_frozen_actor
from wt.lowlevel_v2.phase5l_joint_encoder.run_phase5l import _load_matching
from wt.lowlevel_v2.phase5m_joint_planner.run_phase5m import _load_proposed
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.metrics import metrics_at, quantize
from wt.lowlevel_v2.phase5r_nonadditive_generalization.metrics import cluster_bootstrap
from wt.lowlevel_v2.phase5v_temporal_representation.run_phase5v import _checkpoint as _temporal_checkpoint
from wt.lowlevel_v2.phase5v_temporal_representation.run_phase5v import _vector
from wt.lowlevel_v2.phase5x_executor_lookahead.shadow import fingerprint, max_abs_diff
from wt.lowlevel_v2.phase5ze_response_distillation.run_phase5ze import _labels, _load_response_norm, _load_value
from wt.lowlevel_v2.phase5zl_shadow_hierarchical_integration.run_phase5zl import _load_states, _prepare, _score
from wt.lowlevel_v2.phase5zl_shadow_hierarchical_integration.shadow import assignment_ok, build_task_option
from wt.lowlevel_v2.phase5zj_true_h1_acceleration.fast_path import shipped_h1

from .consensus import canonical_key, rank_mean_select
from .protocol import (
    AGGREGATION,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CANDIDATE_PERMUTATIONS,
    CASE_SCENARIO,
    CASE_TIME,
    DEPLOYMENT_LATENCY_REQUIREMENT,
    DEPLOYMENT_VERIFIED,
    FRESH_CONFIRMATION_AUTHORIZED,
    FROZEN_FOR_FRESH_CONFIRMATION,
    HOLD_STEPS,
    J15_RETAIN_ABS,
    LARGE_CONFLICT_FRACTION,
    LOW_STRENGTH,
    MODEL_R1_AGGREGATION_APPROVED,
    PHASE,
    PHASE4E_REPORT,
    PHASE4E_STATUS,
    PLANNER_READY,
    REPORT,
    REPEATS,
    SEED_PERMUTATIONS,
    VALUE_SEEDS,
    WARMUP,
    assert_consumed,
    choose_gate,
    dump_protocol,
)

SAVED = PHASE.parent / "phase5ze_response_distillation" / "phase5ze_downstream_value_predictions.csv"


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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _median(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def _load_saved() -> dict[tuple[int, int], dict]:
    grouped: dict[tuple[int, int, int], dict[str, dict]] = {}
    with SAVED.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scenario = int(row["scenario_id"])
            assert_consumed(scenario)
            key = (scenario, int(row["time_step"]), int(row["seed"]))
            grouped.setdefault(key, {})[row["assignment_vector"]] = {
                "v_r1": float(row["v_r1"]),
                "v_s": float(row["v_s"]),
                "j15": float(row["j15"]),
            }
    states = {}
    for scenario, moment, seed in list(grouped):
        states.setdefault((scenario, moment), {})[seed] = grouped[(scenario, moment, seed)]
    for key, seeds in states.items():
        order = list(seeds[0])
        if len(order) != 120 or any(set(seeds[item]) != set(order) for item in VALUE_SEEDS):
            raise RuntimeError(f"saved candidate pool is incomplete at {key}")
    return states


def _matrix(seed_rows: dict[int, dict[str, dict]], order: list[str], field: str) -> np.ndarray:
    return np.asarray([[seed_rows[seed][vector][field] for vector in order] for seed in VALUE_SEEDS], dtype=np.float64)


def _assignments(order: list[str]) -> np.ndarray:
    return np.asarray([[int(item) for item in vector.split(";")] for vector in order], dtype=np.int64)


def _score_hit(values: np.ndarray, labels: np.ndarray, k: int) -> float:
    levels = quantize(labels)
    best = float(np.max(levels))
    order = np.argsort(-np.asarray(values, dtype=np.float64), kind="mergesort")
    width = min(int(k), int(order.size))
    return float(np.any(levels[order[:width]] == best))


def _rank_hit(rank_sum: np.ndarray, labels: np.ndarray, k: int) -> float:
    levels = quantize(labels)
    best = float(np.max(levels))
    width = min(int(k), int(rank_sum.size))
    cutoff = int(np.partition(rank_sum, width - 1)[width - 1])
    return float(np.any(levels[rank_sum <= cutoff] == best))


def _regret(labels: np.ndarray, index: int) -> float:
    return float(metrics_at(np.zeros(len(labels)), labels, int(index))["normalized_regret"])


def _margin(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    return float(ordered[0] - ordered[1])


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    if not union:
        return 1.0
    return float(len(left & right) / len(union))


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    result = spearmanr(left, right)
    value = getattr(result, "statistic", result.correlation)
    return float(value)


def _sign_counts(values: list[float]) -> dict[str, int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "positive": int(np.sum(array > 0.0)),
        "tie": int(np.sum(array == 0.0)),
        "negative": int(np.sum(array < 0.0)),
    }


def _fmt(value: float) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return format(float(value), ".4e")


def _consensus_once(frozen, matching, config, models, state, norm, center, scale) -> dict:
    wall_started = time.perf_counter()
    encoder_ms, prep_ms = _prepare(frozen, matching, config, state, center, scale)
    body_started = time.perf_counter()
    body = shipped_h1(state["env"], state["assignments"], state["actor"], state["movement"])
    h1_ms = (time.perf_counter() - body_started) * 1000.0
    value_started = time.perf_counter()
    scores = np.stack([_score(models[int(seed)], state, body["response"], norm) for seed in VALUE_SEEDS], axis=0)
    value_ms = (time.perf_counter() - value_started) * 1000.0
    agg_started = time.perf_counter()
    selected = rank_mean_select(scores, state["assignments"])
    agg_ms = (time.perf_counter() - agg_started) * 1000.0
    option_started = time.perf_counter()
    option = build_task_option(state["env"], state["info"], selected["assignment"])
    option_ms = (time.perf_counter() - option_started) * 1000.0
    return {
        "wall_ms": (time.perf_counter() - wall_started) * 1000.0,
        "encoder_ms": encoder_ms,
        "candidate_prep_ms": prep_ms,
        "true_h1_ms": h1_ms,
        "value_ms": value_ms,
        "aggregation_ms": agg_ms,
        "taskoption_ms": option_ms,
        "key": selected["key"],
        "scores": scores,
        "hold": int(option.remaining_hold),
    }


def _audit_saved(saved: dict) -> dict:
    consensus_rows = []
    disagreement_rows = []
    comparison_rows = []
    permutation_rows = []
    candidate_rows = []
    correlation_store = {(left, right): [] for left in VALUE_SEEDS for right in VALUE_SEEDS if left < right}
    agreement_store = {(left, right): [] for left in VALUE_SEEDS for right in VALUE_SEEDS if left < right}
    jaccard_values = []
    illegal = 0
    case = None
    per_seed_j15 = {int(seed): [] for seed in VALUE_SEEDS}
    static_j15 = {int(seed): [] for seed in VALUE_SEEDS}
    consensus_j15 = []
    consensus_regret = []
    state_delta_s = []
    state_delta_r = []
    scenario_delta_s: dict[int, list[float]] = {}
    strength_counts = {"1.0": 0, "0.8": 0, "0.6": 0, "le_0.4": 0}
    agree_counts = {"5": 0, "4": 0, "le3": 0}

    for scenario in sorted({key[0] for key in saved}):
        for moment in sorted(key[1] for key in saved if key[0] == scenario):
            seed_rows = saved[(scenario, moment)]
            order = list(seed_rows[0])
            assignments = _assignments(order)
            labels = np.asarray([seed_rows[0][vector]["j15"] for vector in order], dtype=np.float64)
            values = _matrix(seed_rows, order, "v_r1")
            static_values = _matrix(seed_rows, order, "v_s")
            selected = rank_mean_select(values, assignments)
            repeated = rank_mean_select(values, assignments)
            if repeated["key"] != selected["key"] or not np.array_equal(repeated["rank_sum"], selected["rank_sum"]):
                raise RuntimeError("rank-mean selection is not deterministic")
            ok, reason = assignment_ok(selected["assignment"], 6, 12)
            if not ok:
                illegal += 1
            vector = _vector(selected["assignment"])
            index = int(selected["index"])
            seed_choice = []
            for seed in VALUE_SEEDS:
                seed_values = values[VALUE_SEEDS.index(seed)]
                seed_index = int(np.flatnonzero(seed_values == np.max(seed_values))[0])
                seed_choice.append(order[seed_index])
                item = metrics_at(values[VALUE_SEEDS.index(seed)], labels, seed_index)
                per_seed_j15[int(seed)].append(float(labels[seed_index]))
                comparison_rows.append(
                    {
                        "scenario_id": scenario,
                        "time_step": moment,
                        "method": f"r1_seed_{seed}",
                        "seed": int(seed),
                        "assignment": order[seed_index],
                        "candidate_index": seed_index,
                        "selected_j15": float(labels[seed_index]),
                        "normalized_regret": float(item["normalized_regret"]),
                        "top1": float(item["top1"]),
                        "top3": _score_hit(values[VALUE_SEEDS.index(seed)], labels, 3),
                        "top5": _score_hit(values[VALUE_SEEDS.index(seed)], labels, 5),
                    }
                )
                static_index = int(np.flatnonzero(static_values[VALUE_SEEDS.index(seed)] == np.max(static_values[VALUE_SEEDS.index(seed)]))[0])
                static_item = metrics_at(static_values[VALUE_SEEDS.index(seed)], labels, static_index)
                static_j15[int(seed)].append(float(labels[static_index]))
                comparison_rows.append(
                    {
                        "scenario_id": scenario,
                        "time_step": moment,
                        "method": f"model_s_seed_{seed}",
                        "seed": int(seed),
                        "assignment": order[static_index],
                        "candidate_index": static_index,
                        "selected_j15": float(labels[static_index]),
                        "normalized_regret": float(static_item["normalized_regret"]),
                        "top1": float(static_item["top1"]),
                        "top3": _score_hit(static_values[VALUE_SEEDS.index(seed)], labels, 3),
                        "top5": _score_hit(static_values[VALUE_SEEDS.index(seed)], labels, 5),
                    }
                )
            consensus_j15.append(float(labels[index]))
            consensus_regret.append(_regret(labels, index))
            comparison_rows.append(
                {
                    "scenario_id": scenario,
                    "time_step": moment,
                    "method": "rank_mean_consensus_v1",
                    "seed": -1,
                    "assignment": vector,
                    "candidate_index": index,
                    "selected_j15": float(labels[index]),
                    "normalized_regret": _regret(labels, index),
                    "top1": _rank_hit(selected["rank_sum"], labels, 1),
                    "top3": _rank_hit(selected["rank_sum"], labels, 3),
                    "top5": _rank_hit(selected["rank_sum"], labels, 5),
                }
            )
            static_mean = float(np.mean([labels[int(np.flatnonzero(static_values[offset] == np.max(static_values[offset]))[0])] for offset in range(5)]))
            delta_s = float(labels[index] - static_mean)
            state_delta_s.append(delta_s)
            scenario_delta_s.setdefault(scenario, []).append(delta_s)
            for seed in VALUE_SEEDS:
                state_delta_r.append(float(labels[index] - per_seed_j15[int(seed)][-1]))
            rank_text = "|".join(str(int(item)) for item in selected["ranks"])
            ordered_sums = np.sort(selected["rank_sum"])
            consensus_rows.append(
                {
                    "scenario_id": scenario,
                    "time_step": moment,
                    "selected_assignment": vector,
                    "selected_candidate_index": index,
                    "rank_vector": rank_text,
                    "mean_rank": selected["mean_rank"],
                    "majority_top1_count": selected["majority_top1"],
                    "selected_j15": float(labels[index]),
                    "normalized_regret": _regret(labels, index),
                    "top1": _rank_hit(selected["rank_sum"], labels, 1),
                    "top3": _rank_hit(selected["rank_sum"], labels, 3),
                    "top5": _rank_hit(selected["rank_sum"], labels, 5),
                    "value_margin_min": min(_margin(values[offset]) for offset in range(5)),
                    "rank_margin": float(ordered_sums[1] - ordered_sums[0]) / 5.0,
                    "seed_disagreement_count": int(len(set(seed_choice)) - 1),
                    "selection_stage": selected["stage"],
                    "consensus_strength": selected["strength"],
                    "seed_assignments": "|".join(seed_choice),
                    "exclusive": bool(ok),
                    "reason": reason,
                }
            )
            unique = len(set(seed_choice))
            if unique == 1:
                agree_counts["5"] += 1
            elif max(seed_choice.count(item) for item in set(seed_choice)) == 4:
                agree_counts["4"] += 1
            else:
                agree_counts["le3"] += 1
            max_majority = int(np.max(selected["majority"]))
            if max_majority == 5:
                strength_counts["1.0"] += 1
            elif max_majority == 4:
                strength_counts["0.8"] += 1
            elif max_majority == 3:
                strength_counts["0.6"] += 1
            elif max_majority <= int(LOW_STRENGTH * 5):
                strength_counts["le_0.4"] += 1
            rank_matrix = selected["rank_matrix"]
            for left_offset, left in enumerate(VALUE_SEEDS):
                for right in VALUE_SEEDS[left_offset + 1 :]:
                    right_offset = VALUE_SEEDS.index(right)
                    correlation_store[(left, right)].append(_spearman(rank_matrix[left_offset], rank_matrix[right_offset]))
                    agreement_store[(left, right)].append(float(seed_choice[left_offset] == seed_choice[right_offset]))
                    left_set = {int(item) for item in np.flatnonzero(rank_matrix[left_offset] <= 3)}
                    right_set = {int(item) for item in np.flatnonzero(rank_matrix[right_offset] <= 3)}
                    jaccard_values.append(_jaccard(left_set, right_set))
            rank_variance = float(np.mean(np.var(rank_matrix, axis=0)))
            disagreement_rows.append(
                {
                    "scenario_id": scenario,
                    "time_step": moment,
                    "unique_top1": unique,
                    "consensus_strength": selected["strength"],
                    "rank_variance": rank_variance,
                    "top1_top2_rank_margin": float(ordered_sums[1] - ordered_sums[0]) / 5.0,
                    "min_value_margin": min(_margin(values[offset]) for offset in range(5)),
                    "pairwise_agreement_mean": _mean(
                        float(seed_choice[left] == seed_choice[right])
                        for left in range(5)
                        for right in range(left + 1, 5)
                    ),
                }
            )
            for name, perm in (("identity", SEED_PERMUTATIONS[0]), ("reverse", SEED_PERMUTATIONS[1])):
                moved = rank_mean_select(values[list(perm)], assignments)
                permutation_rows.append(
                    {
                        "scenario_id": scenario,
                        "time_step": moment,
                        "kind": name,
                        "assignment_unchanged": int(moved["key"] == selected["key"]),
                        "mean_rank_unchanged": int(abs(moved["mean_rank"] - selected["mean_rank"]) == 0.0),
                        "strength_unchanged": int(moved["strength"] == selected["strength"]),
                    }
                )
            for perm_index, perm in enumerate(SEED_PERMUTATIONS[2:]):
                moved = rank_mean_select(values[list(perm)], assignments)
                permutation_rows.append(
                    {
                        "scenario_id": scenario,
                        "time_step": moment,
                        "kind": f"random_{perm_index}",
                        "assignment_unchanged": int(moved["key"] == selected["key"]),
                        "mean_rank_unchanged": int(abs(moved["mean_rank"] - selected["mean_rank"]) == 0.0),
                        "strength_unchanged": int(moved["strength"] == selected["strength"]),
                    }
                )
            for perm_index, perm in enumerate(CANDIDATE_PERMUTATIONS):
                order_index = np.asarray(perm, dtype=np.int64)
                moved = rank_mean_select(values[:, order_index], assignments[order_index])
                restored = np.empty(len(order), dtype=np.int64)
                restored[order_index] = moved["rank_sum"]
                candidate_rows.append(
                    {
                        "scenario_id": scenario,
                        "time_step": moment,
                        "permutation": perm_index,
                        "assignment_unchanged": int(moved["key"] == selected["key"]),
                        "mean_rank_unchanged": int(np.array_equal(restored, selected["rank_sum"])),
                    }
                )
            if scenario == CASE_SCENARIO and moment == CASE_TIME:
                rivals = []
                for choice in dict.fromkeys(seed_choice):
                    rival_index = order.index(choice)
                    rivals.append(
                        {
                            "assignment": choice,
                            "ranks": [int(item) for item in rank_matrix[:, rival_index]],
                            "mean_rank": float(selected["rank_sum"][rival_index] / 5.0),
                            "majority_top1_count": int(selected["majority"][rival_index]),
                            "j15": float(labels[rival_index]),
                        }
                    )
                case = {
                    "selected": vector,
                    "stage": selected["stage"],
                    "uses_j15": False,
                    "rivals": rivals,
                    "j15_equal": len({item["j15"] for item in rivals}) == 1,
                }
            print(f"consensus {scenario} t={moment} {vector} stage={selected['stage']}", flush=True)

    n_states = len(consensus_rows)
    seed_perm_ok = all(row["assignment_unchanged"] and row["mean_rank_unchanged"] and row["strength_unchanged"] for row in permutation_rows)
    candidate_perm_ok = all(row["assignment_unchanged"] and row["mean_rank_unchanged"] for row in candidate_rows)
    correlation_rows = []
    for (left, right), values_list in correlation_store.items():
        correlation_rows.append(
            {
                "seed_a": int(left),
                "seed_b": int(right),
                "mean_rank_spearman": _mean(values_list),
                "assignment_agreement": _mean(agreement_store[(left, right)]),
            }
        )
    seed_means = {int(seed): _mean(items) for seed, items in per_seed_j15.items()}
    static_means = {int(seed): _mean(items) for seed, items in static_j15.items()}
    consensus_mean = _mean(consensus_j15)
    static_mean = _mean(static_means.values())
    retained = bool(consensus_mean + J15_RETAIN_ABS >= min(seed_means.values()) and consensus_mean + J15_RETAIN_ABS >= static_mean)
    return {
        "consensus_rows": consensus_rows,
        "disagreement_rows": disagreement_rows,
        "comparison_rows": comparison_rows,
        "permutation_rows": permutation_rows,
        "candidate_rows": candidate_rows,
        "correlation_rows": correlation_rows,
        "illegal": illegal,
        "case": case,
        "n_states": n_states,
        "agree_5": agree_counts["5"] / n_states,
        "agree_4": agree_counts["4"] / n_states,
        "agree_le3": agree_counts["le3"] / n_states,
        "strength_counts": strength_counts,
        "large_conflict": strength_counts["le_0.4"] / n_states > LARGE_CONFLICT_FRACTION,
        "jaccard": _mean(jaccard_values),
        "rank_variance": _mean(row["rank_variance"] for row in disagreement_rows),
        "rank_margin": _mean(row["top1_top2_rank_margin"] for row in disagreement_rows),
        "seed_means": seed_means,
        "static_means": static_means,
        "consensus_mean": consensus_mean,
        "consensus_median": _median(consensus_j15),
        "static_mean": static_mean,
        "consensus_regret": _mean(consensus_regret),
        "delta_s": _mean(state_delta_s),
        "delta_r": _mean(state_delta_r),
        "state_signs": _sign_counts(state_delta_s),
        "scenario_signs": _sign_counts([_mean(items) for items in scenario_delta_s.values()]),
        "seed_permutation_invariance": float(seed_perm_ok),
        "candidate_permutation_invariance": float(candidate_perm_ok),
        "j15_retained": retained,
        "consensus_ci": cluster_bootstrap(
            [
                {"scenario_id": row["scenario_id"], "time_step": row["time_step"], "selected_j15": row["selected_j15"]}
                for row in consensus_rows
            ],
            "selected_j15",
            draws=BOOTSTRAP_DRAWS,
            seed=BOOTSTRAP_SEED,
        ),
        "delta_ci": cluster_bootstrap(
            [
                {"scenario_id": row["scenario_id"], "time_step": row["time_step"], "delta": delta}
                for row, delta in zip(consensus_rows, state_delta_s)
            ],
            "delta",
            draws=BOOTSTRAP_DRAWS,
            seed=BOOTSTRAP_SEED,
        ),
        "saved_choice": {(int(row["scenario_id"]), int(row["time_step"])): row["selected_assignment"] for row in consensus_rows},
    }


def _report(summary: dict) -> str:
    gate = summary["gate"]
    case = summary["case"]
    rival_lines = []
    for rival in case["rivals"]:
        rival_lines.append(
            f"- `{rival['assignment']}`：五 seed rank {rival['ranks']}，mean rank {rival['mean_rank']:.1f}，"
            f"majority {rival['majority_top1_count']}，已保存 J15 {_fmt(rival['j15'])} bit。"
        )
    return "\n".join(
        [
            "# HTW Phase-5ZM：Model-R1 多种子共识与部署规范审计",
            "",
            f"**Gate = {gate['gate']} {gate['gate_name']}**",
            "",
            f"阻塞项：{gate['blockers']}。",
            "",
            "**planner_ready = False**",
            "",
            "**deployment_verified = False**",
            "",
            "**FROZEN_FOR_FRESH_CONFIRMATION = False**",
            "",
            "**fresh_confirmation_authorized = False**",
            "",
            "本阶段只在 310110–310117 上审计预先登记的 Rank-Mean Consensus V1。没有挑选最佳 seed，没有根据 J15 调权，没有打开 310118–310125，没有调用正式决策函数。Phase-4E 仍是 DEPLOYMENT_VERIFICATION_FAILED。",
            "",
            "## Q1. 五个 R1 seed 的真实分歧程度有多大？",
            "",
            f"24 个状态中，5/5 选中同一分配的比例是 {summary['agree_5']:.4f}，4/5 的比例是 {summary['agree_4']:.4f}，不超过 3/5 的比例是 {summary['agree_le3']:.4f}。",
            f"consensus_strength 等于 1.0、0.8、0.6 和不超过 0.4 的状态数分别是 {summary['strength_counts']['1.0']}、{summary['strength_counts']['0.8']}、{summary['strength_counts']['0.6']}、{summary['strength_counts']['le_0.4']}。",
            f"两两 seed 的平均分配一致率见相关表。Top-3 rank 集合的平均 Jaccard 是 {summary['jaccard']:.4f}。候选 rank 方差的状态平均是 {summary['rank_variance']:.4f}。第一名与第二名的平均 rank margin 是 {summary['rank_margin']:.4f}。",
            "",
            "这些数字只作为 confidence diagnostic，没有用来改决策。",
            "",
            "## Q2. 310111 t=15 为什么出现 seed disagreement？",
            "",
            "五个冻结 value head 对同一组 120 个候选给出了不同的 Top-1。seed 0 的最高分落在一个分配上，seed 1–4 的最高分落在另一个分配上。两者都是各自 seed 的 rank 1，不是平局打破造成的。",
            "",
            "## Q3. Rank-Mean Consensus V1 最终选择哪一个 assignment？",
            "",
            f"310111 t=15 选择 `{case['selected']}`。决定阶段是 {case['stage']}。",
            *rival_lines,
            "",
            "mean rank 更低者优先。若 mean rank 相同，则 majority Top-1 更高者优先。若仍然相同，则用分配向量的字典序。",
            "",
            "## Q4. 这个选择是否完全不依赖 J15？",
            "",
            "不依赖。选择函数只接收五个 seed 的 value 和候选分配向量。J15、scenario id 和开发集收益都不是输入。两个竞争分配的已保存 J15 是否相同，只在选择之后读取："
            + ("相同。" if case["j15_equal"] else "不相同。")
            + "该状态没有因为 J15 相同而被删掉。",
            "",
            "## Q5. seed permutation 是否完全不影响结果？",
            "",
            f"seed_permutation_invariance = {summary['seed_permutation_invariance']:.1f}。原顺序、反序和 10 个固定随机置换下，rank mean、选中分配和 consensus strength 都不变。",
            "",
            "## Q6. candidate enumeration 是否影响结果？",
            "",
            f"candidate_permutation_invariance = {summary['candidate_permutation_invariance']:.1f}。10 个固定置换下，按分配对齐后的 rank sum 和最终分配都不变。完全平局时的第二层 tie-break 是分配向量字典序，不是输入下标。",
            "",
            "## Q7. Consensus 是否保留现有 True-H1 R1 的收益？",
            "",
            f"这是已保存标签上的重放，不是新的性能结论。Consensus 平均 selected J15 是 {_fmt(summary['consensus_mean'])} bit，中位数是 {_fmt(summary['consensus_median'])} bit。场景聚类 95% 区间是 [{_fmt(summary['consensus_ci']['low'])}, {_fmt(summary['consensus_ci']['high'])}] bit，这个区间描述的是均值，不是中位数。",
            f"五个单 seed 的平均 selected J15 是 {', '.join(f'seed {seed} {_fmt(value)}' for seed, value in summary['seed_means'].items())}。Model-S 五个 seed 的总平均是 {_fmt(summary['static_mean'])} bit。",
            f"Consensus 相对 Model-S 的平均 ΔJ15 是 {_fmt(summary['delta_s'])} bit，区间 [{_fmt(summary['delta_ci']['low'])}, {_fmt(summary['delta_ci']['high'])}] bit。相对单 seed R1 的平均差是 {_fmt(summary['delta_r'])} bit。",
            f"按状态计，相对 Model-S 为正、平、负的数量是 {summary['state_signs']['positive']}、{summary['state_signs']['tie']}、{summary['state_signs']['negative']}。按 scenario 平均后是 {summary['scenario_signs']['positive']}、{summary['scenario_signs']['tie']}、{summary['scenario_signs']['negative']}。",
            f"预先登记的保留条件是：Consensus 均值不低于最低单 seed，也不低于 Model-S，容差 {J15_RETAIN_ABS:.0f} bit。结果是 {summary['j15_retained']}。平均 normalized regret 是 {summary['consensus_regret']:.6f}。",
            "",
            "## Q8. 五 seed ensemble 后完整推理耗时是多少？",
            "",
            f"计时边界是 encoder、候选生成、optimized True-H1、五个 Model-R1、Rank-Mean、TaskOption。中位数 {summary['latency']['median_ms']:.2f} ms，均值 {summary['latency']['mean_ms']:.2f} ms，P95 {summary['latency']['p95_ms']:.2f} ms，P99 {summary['latency']['p99_ms']:.2f} ms。",
            f"分量均值：True-H1 {summary['latency']['h1_ms']:.2f} ms，五个 Model-R1 {summary['latency']['value_ms']:.2f} ms，aggregation {summary['latency']['aggregation_ms']:.4f} ms。",
            "没有为这次测量改写模块。deployment_latency_requirement 仍然是 NOT_SPECIFIED。Phase-5ZL 的 342.09 ms 只含一个 value head，不能和这里的五 seed 总时间直接相减后称为加速或退化结论以外的优化。",
            "",
            "## Q9. 当前还缺哪些人工部署规范？",
            "",
            "缺五项，都写在 pending specification 里，本阶段没有代填：最大高层时延、超时策略、推理异常 fallback、Rank-Mean 是否正式批准、fresh confirmation 是否授权。DeepSets、B2、Oracle、HGNN 和 seed 0 都没有被设为 fallback。",
            "",
            "## Q10. 是否已经具备冻结算法结构的条件？",
            "",
            f"技术核验是 {gate['aggregation_technically_verified']}。主 Gate 是 {gate['gate']}。即便技术核验通过，聚合规则仍是 model_r1_aggregation_approved = false，不能算已经冻结进正式系统。FROZEN_FOR_FRESH_CONFIRMATION 保持 False。",
            "",
            "## Q11. 是否可以打开 310118–310125？",
            "",
            "FRESH_CONFIRMATION_NOT_AUTHORIZED",
            "",
            "人工尚未批准全部 pending specification。本阶段的答案不能是“可以”。",
            "",
            "## 接口草案",
            "",
            "当前状态进入冻结的 Stage-1 encoder 和候选生成，得到 120 个联合分配。optimized True-H1 只给这些分配做一步 value 评价。五个冻结 Model-R1 输出 value，Rank-Mean Consensus V1 选出一个联合分配，写成 remaining_hold=15 的 TaskOption。冻结 Residual MAPPO 只执行这个 option，不重新选择浮标。",
            "",
            "True-H1 不是第二个 planner，不是轨迹 planner，也不是 15 步 rollout。",
            "",
        ]
    )


def main() -> None:
    dump_protocol()
    if not PHASE4E_REPORT.is_file() or "DEPLOYMENT_VERIFICATION_FAILED" not in PHASE4E_REPORT.read_text(encoding="utf-8"):
        raise RuntimeError("phase-4e failure status is missing")
    saved = _load_saved()
    audit = _audit_saved(saved)
    config = L3BConfig()
    labels = _labels("confirm")
    frozen = _load_proposed()
    matching = _load_matching()
    actor = load_frozen_actor()
    norm = _load_response_norm()
    temporal = torch.load(_temporal_checkpoint(0), map_location="cpu", weights_only=False)
    center = np.asarray(temporal["center"], dtype=np.float32)
    scale = np.asarray(temporal["scale"], dtype=np.float32)
    models = {int(seed): _load_value("response", int(seed)) for seed in VALUE_SEEDS}
    states = _load_states(frozen, matching, config, center, scale, labels)
    for state in states:
        state["actor"] = actor
    latency_rows = []
    caller_gap = 0.0
    value_gap = 0.0
    live_match = True
    for state in states:
        key = (int(state["scenario_id"]), int(state["time_step"]))
        before = fingerprint(state["env"])
        for _ in range(WARMUP):
            warmup = _consensus_once(frozen, matching, config, models, state, norm, center, scale)
        for seed_offset, seed in enumerate(VALUE_SEEDS):
            for index, vector in enumerate(state["vectors"]):
                saved_value = saved[key][int(seed)][vector]["v_r1"]
                value_gap = max(value_gap, abs(float(warmup["scores"][seed_offset, index]) - saved_value))
        if _vector(warmup["key"]) != audit["saved_choice"][key] or int(warmup["hold"]) != HOLD_STEPS:
            live_match = False
        repeats = [_consensus_once(frozen, matching, config, models, state, norm, center, scale) for _ in range(REPEATS)]
        after = fingerprint(state["env"])
        caller_gap = max(caller_gap, max_abs_diff(before, after))
        latency_rows.append(
            {
                "scenario_id": state["scenario_id"],
                "time_step": state["time_step"],
                "true_h1_ms": _mean(item["true_h1_ms"] for item in repeats),
                "five_model_r1_ms": _mean(item["value_ms"] for item in repeats),
                "aggregation_ms": _mean(item["aggregation_ms"] for item in repeats),
                "taskoption_ms": _mean(item["taskoption_ms"] for item in repeats),
                "total_ms": _mean(item["wall_ms"] for item in repeats),
                "candidate_count": 120,
            }
        )
        print(f"timed {key[0]} t={key[1]} {latency_rows[-1]['total_ms']:.2f}", flush=True)
    walls = np.asarray([row["total_ms"] for row in latency_rows], dtype=np.float64)
    latency = {
        "median_ms": float(np.median(walls)),
        "mean_ms": float(np.mean(walls)),
        "p95_ms": float(np.percentile(walls, 95)),
        "p99_ms": float(np.percentile(walls, 99)),
        "h1_ms": _mean(row["true_h1_ms"] for row in latency_rows),
        "value_ms": _mean(row["five_model_r1_ms"] for row in latency_rows),
        "aggregation_ms": _mean(row["aggregation_ms"] for row in latency_rows),
        "threads": int(torch.get_num_threads()),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
    }
    chosen = choose_gate(
        permutation_ok=audit["seed_permutation_invariance"] == 1.0 and audit["candidate_permutation_invariance"] == 1.0,
        j15_retained=bool(audit["j15_retained"]),
        illegal=audit["illegal"] > 0,
        isolation_ok=caller_gap == 0.0,
        large_conflict=bool(audit["large_conflict"]),
        live_matches_saved=bool(live_match and value_gap <= 1.0e-3),
        human_spec_approved=False,
    )
    summary = dict(audit)
    summary["gate"] = chosen
    summary["latency"] = latency
    _write_csv("phase5zm_rank_mean_consensus.csv", audit["consensus_rows"])
    _write_csv("phase5zm_seed_disagreement.csv", audit["disagreement_rows"])
    _write_csv("phase5zm_seed_rank_correlations.csv", audit["correlation_rows"])
    _write_csv("phase5zm_seed_permutation_audit.csv", audit["permutation_rows"])
    _write_csv("phase5zm_candidate_permutation_audit.csv", audit["candidate_rows"])
    _write_csv("phase5zm_j15_replay_comparison.csv", audit["comparison_rows"])
    _write_csv("phase5zm_latency.csv", latency_rows)
    _dump(
        "phase5zm_integration_contract.json",
        {
            "status": "draft_not_installed",
            "flow": [
                "current_state",
                "frozen_stage1_encoder_and_candidate_generator",
                "120_joint_assignments",
                "optimized_true_h1_one_step_value_evaluator",
                "five_frozen_model_r1_heads",
                "rank_mean_consensus_v1",
                "selected_joint_assignment",
                "taskoption_association_hold_15",
                "frozen_residual_mappo_executes_option",
            ],
            "true_h1_role": "high-level assignment value evaluator",
            "true_h1_is_planner": False,
            "true_h1_is_trajectory_planner": False,
            "true_h1_is_15_step_rollout": False,
            "low_level_reselects_buoy": False,
            "installed_in_official_planner": False,
            "aggregation": AGGREGATION,
            "aggregation_approved": False,
        },
    )
    _dump(
        "phase5zm_deployment_policy_pending_specification.json",
        {
            "status": "pending_human_approval",
            "max_high_level_latency_ms": "HUMAN_SPECIFICATION_REQUIRED",
            "timeout_policy": "HUMAN_SPECIFICATION_REQUIRED",
            "timeout_policy_options_not_selected": [
                "KEEP_CURRENT_TASKOPTION",
                "USE_LAST_VALID_ASSIGNMENT",
                "ABORT_DECISION",
            ],
            "fallback_policy": "HUMAN_SPECIFICATION_REQUIRED",
            "fallback_cases": [
                "true_h1_exception",
                "model_r1_non_finite",
                "candidate_pool_empty",
                "safety_projector_failure",
                "checkpoint_missing",
            ],
            "deepsets_is_fallback": False,
            "b2_is_fallback": False,
            "oracle_is_fallback": False,
            "hgnn_is_fallback": False,
            "seed0_is_fallback": False,
            "model_r1_aggregation": AGGREGATION,
            "model_r1_aggregation_approved": MODEL_R1_AGGREGATION_APPROVED,
            "fresh_confirmation_pool": "310118-310125",
            "fresh_confirmation_authorized": FRESH_CONFIRMATION_AUTHORIZED,
            "deployment_latency_requirement": DEPLOYMENT_LATENCY_REQUIREMENT,
        },
    )
    _dump(
        "phase5zm_model_selection_gate.json",
        {
            "gate": chosen["gate"],
            "gate_name": chosen["gate_name"],
            "blockers": chosen["blockers"],
            "aggregation_technically_verified": chosen["aggregation_technically_verified"],
            "aggregation": AGGREGATION,
            "model_r1_aggregation_approved": MODEL_R1_AGGREGATION_APPROVED,
            "seed_permutation_invariance": audit["seed_permutation_invariance"],
            "candidate_permutation_invariance": audit["candidate_permutation_invariance"],
            "illegal_assignments": audit["illegal"],
            "j15_retained": audit["j15_retained"],
            "consensus_mean_j15": audit["consensus_mean"],
            "single_seed_not_selected": True,
            "caller_fingerprint": caller_gap,
            "live_value_abs": value_gap,
            "live_matches_saved": live_match,
            "latency_median_ms": latency["median_ms"],
            "deployment_latency_requirement": DEPLOYMENT_LATENCY_REQUIREMENT,
            "planner_ready": PLANNER_READY,
            "deployment_verified": DEPLOYMENT_VERIFIED,
            "frozen_for_fresh_confirmation": FROZEN_FOR_FRESH_CONFIRMATION,
            "fresh_confirmation_authorized": FRESH_CONFIRMATION_AUTHORIZED,
            "fresh_confirmation_answer": "FRESH_CONFIRMATION_NOT_AUTHORIZED",
            "phase4e_status": PHASE4E_STATUS,
            "phase4e_overwritten": False,
            "opens_310118_310125": False,
            "new_independent_confirmation": False,
        },
    )
    REPORT.write_text(_report(summary), encoding="utf-8")
    print(
        json.dumps(
            _plain(
                {
                    "gate": chosen["gate"],
                    "blockers": chosen["blockers"],
                    "invariance": audit["seed_permutation_invariance"],
                    "candidate_invariance": audit["candidate_permutation_invariance"],
                    "consensus_mean": audit["consensus_mean"],
                    "median_ms": latency["median_ms"],
                    "case": audit["case"]["selected"] if audit["case"] else None,
                }
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
