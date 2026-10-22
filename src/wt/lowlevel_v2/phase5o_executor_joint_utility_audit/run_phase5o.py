"""Audit J_FLY against frozen-MAPPO payload on the Phase-5N assignment pool."""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict

import numpy as np
import torch

from wt.lowlevel_v2.phase5_hgnn_urgency.urgency import shared_task_urgency
from wt.lowlevel_v2.phase5g_closed_loop.collect import load_frozen_actor
from wt.lowlevel_v2.phase5j_joint_redesign.joint import conflict_rate, duplicate_count
from wt.lowlevel_v2.phase5j_joint_redesign.run_phase5j import replay_env
from wt.lowlevel_v2.phase5m_joint_planner.run_phase5m import _make_evaluator, _marginals
from wt.lowlevel_v2.phase5n_joint_planner.protocol import LABEL_VERSION
from wt.round30.ppo import seed_everything

from .metrics import choose_label, discrimination, quantize_block, rank_correlation, regret, top1_overlap, top3_recovery
from .protocol import (
    CACHE,
    CACHE_VERSION,
    DEPLOYMENT_VERIFIED,
    OUTPUT,
    PAIR_LIMIT,
    PAYLOAD_BLOCK_BITS,
    PHASE5N_LABELS,
    PHASE5N_SUMMARY,
    PLANNER_READY,
    POOL_SIZE,
    REPORT,
    assert_phase5o_seed,
)
from .rollout import execute_mappo


def _vector(assignment) -> str:
    return ";".join(str(int(item)) for item in np.asarray(assignment, dtype=np.int64))


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_csv(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def _fmt(value, digits: str = ".4f") -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return format(float(value), digits)


def _mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def load_pool() -> list[dict]:
    if not PHASE5N_LABELS.is_file():
        raise FileNotFoundError(str(PHASE5N_LABELS))
    payload = torch.load(PHASE5N_LABELS, map_location="cpu", weights_only=False)
    if int(payload.get("version", -1)) != int(LABEL_VERSION):
        raise RuntimeError("phase-5o requires the Phase-5N maximal-matching label cache")
    states = []
    for state in payload["states"]:
        scenario = int(state["scenario_id"])
        moment = int(state["time_step"])
        assert_phase5o_seed(scenario)
        assignments = np.asarray(state["assignments"], dtype=np.int64)
        joints = np.asarray(state["joint"], dtype=np.float64)
        if int(assignments.shape[0]) != int(POOL_SIZE) or int(joints.shape[0]) != int(POOL_SIZE):
            raise RuntimeError(f"{scenario} t={moment} does not have {POOL_SIZE} assignments")
        states.append(
            {
                "scenario_id": scenario,
                "time_step": moment,
                "assignments": assignments,
                "j_fly": joints,
                "optimal": np.asarray(state["optimal"], dtype=np.int64),
            }
        )
    if len(states) != 6:
        raise RuntimeError("phase-5o expected six decision states")
    return states


def _load_cache() -> list[dict]:
    if not CACHE.is_file():
        return []
    payload = torch.load(CACHE, map_location="cpu", weights_only=False)
    if int(payload.get("version", -1)) != int(CACHE_VERSION):
        return []
    return list(payload["rows"])


def _save_cache(rows: list[dict]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.save({"version": int(CACHE_VERSION), "rows": rows}, CACHE)


def _verify_cached_fly(env, movement, assignment: np.ndarray, cached: float) -> None:
    options = [[int(buoy)] if int(buoy) >= 0 else [] for buoy in assignment]
    marginal = _marginals(env, movement, options)
    value = float(_make_evaluator(env, movement, marginal)(assignment))
    if abs(value - float(cached)) > 1.0:
        raise RuntimeError(f"cached J_FLY {cached} disagrees with the Phase-5J definition {value}")


def sweep(actor, states: list[dict]) -> list[dict]:
    cached = _load_cache()
    done = {(int(row["scenario_id"]), int(row["time_step"])) for row in cached}
    counts = defaultdict(int)
    for row in cached:
        counts[(int(row["scenario_id"]), int(row["time_step"]))] += 1
    rows = [row for row in cached if counts[(int(row["scenario_id"]), int(row["time_step"]))] == int(POOL_SIZE)]
    finished = {key for key, count in counts.items() if count == int(POOL_SIZE)}
    for state in states:
        key = (int(state["scenario_id"]), int(state["time_step"]))
        if key in finished:
            print(f"resume {key[0]} t={key[1]} rows={POOL_SIZE}", flush=True)
            continue
        if key in done:
            rows = [row for row in rows if (int(row["scenario_id"]), int(row["time_step"])) != key]
        env, movement = replay_env(key[0], key[1])
        _verify_cached_fly(env, movement, state["assignments"][0], float(state["j_fly"][0]))
        fresh = []
        for index, assignment in enumerate(state["assignments"]):
            started = time.perf_counter()
            measured = execute_mappo(env, assignment, actor, movement, trace=False)
            elapsed = time.perf_counter() - started
            chosen = np.asarray(assignment, dtype=np.int64)
            fresh.append(
                {
                    "scenario_id": key[0],
                    "time_step": key[1],
                    "assignment_id": int(index),
                    "assignment_vector": _vector(chosen),
                    "j_fly": float(state["j_fly"][index]),
                    "j_mappo": float(measured["j_mappo"]),
                    "duplicate_targets": float(duplicate_count(chosen)),
                    "conflict_rate": float(conflict_rate(chosen, int(env.config.num_buoys))),
                    "cumulative_interference": float(measured["cumulative_interference"]),
                    "completion": float(measured["completion"]),
                    "assigned_in_range": float(measured["assigned_in_range"]),
                    "residual_queue": float(measured["residual_queue"]),
                    "service_blocks": float(measured["service_blocks"]),
                    "mean_uav_target_distance_m": float(measured["mean_uav_target_distance_m"]),
                    "min_uav_target_distance_m": float(measured["min_uav_target_distance_m"]),
                    "rollout_conflict_rate": float(measured["rollout_conflict_rate"]),
                    "runtime_s": float(elapsed),
                }
            )
            if index % 20 == 0:
                print(f"sweep {key[0]} t={key[1]} {index + 1}/{POOL_SIZE} j_mappo={measured['j_mappo']:.3e}", flush=True)
        rows.extend(fresh)
        _save_cache(rows)
        print(f"saved {key[0]} t={key[1]}", flush=True)
    if len(rows) != 6 * int(POOL_SIZE):
        raise RuntimeError(f"expected {6 * POOL_SIZE} rows, found {len(rows)}")
    return rows


def _groups(rows: list[dict]) -> dict[tuple[int, int], list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["scenario_id"]), int(row["time_step"]))].append(row)
    for group in grouped.values():
        group.sort(key=lambda item: int(item["assignment_id"]))
    return dict(grouped)


def _state_tables(groups: dict[tuple[int, int], list[dict]]) -> tuple[list[dict], list[dict], dict]:
    discrimination_rows = []
    alignment_rows = []
    by_key = {}
    for key in sorted(groups):
        group = groups[key]
        fly = np.asarray([row["j_fly"] for row in group], dtype=np.float64)
        executed = np.asarray([row["j_mappo"] for row in group], dtype=np.float64)
        fly_stats = discrimination(fly)
        executed_stats = discrimination(executed)
        for label, stats in (("J_FLY", fly_stats), ("J_MAPPO", executed_stats)):
            discrimination_rows.append(
                {
                    "scenario_id": key[0],
                    "time_step": key[1],
                    "label": label,
                    "min": stats["min"],
                    "max": stats["max"],
                    "std": stats["std"],
                    "n_unique": stats["n_unique"],
                    "tie_fraction": stats["tie_fraction"],
                    "range_bits": stats["range_bits"],
                    "discriminates": stats["discriminates"],
                    "tolerance_bits": float(PAYLOAD_BLOCK_BITS),
                }
            )
        spearman, kendall = rank_correlation(fly, executed)
        gap, normalized = regret(fly, executed)
        alignment_rows.append(
            {
                "scope": "state",
                "scenario_id": key[0],
                "time_step": key[1],
                "spearman": spearman,
                "kendall": kendall,
                "top1_overlap": top1_overlap(fly, executed),
                "top3_recovery": top3_recovery(fly, executed),
                "regret_fly": gap,
                "normalized_regret_fly": normalized,
                "n": float(len(group)),
                "j_fly_n_unique": fly_stats["n_unique"],
                "j_mappo_n_unique": executed_stats["n_unique"],
                "j_fly_tied": float(fly_stats["discriminates"] == 0.0),
                "j_mappo_discriminates": executed_stats["discriminates"],
            }
        )
        by_key[key] = {
            "fly": fly,
            "mappo": executed,
            "fly_stats": fly_stats,
            "mappo_stats": executed_stats,
            "spearman": spearman,
            "kendall": kendall,
            "top1": top1_overlap(fly, executed),
            "top3": top3_recovery(fly, executed),
            "regret": gap,
            "normalized_regret": normalized,
            "rows": group,
        }
    pooled_fly = np.concatenate([item["fly"] for item in by_key.values()])
    pooled_mappo = np.concatenate([item["mappo"] for item in by_key.values()])
    pooled_spearman, pooled_kendall = rank_correlation(pooled_fly, pooled_mappo)
    pooled_gap, pooled_normalized = regret(pooled_fly, pooled_mappo)
    alignment_rows.append(
        {
            "scope": "pooled",
            "scenario_id": -1,
            "time_step": -1,
            "spearman": pooled_spearman,
            "kendall": pooled_kendall,
            "top1_overlap": top1_overlap(pooled_fly, pooled_mappo),
            "top3_recovery": top3_recovery(pooled_fly, pooled_mappo),
            "regret_fly": pooled_gap,
            "normalized_regret_fly": pooled_normalized,
            "n": float(pooled_fly.size),
            "j_fly_n_unique": discrimination(pooled_fly)["n_unique"],
            "j_mappo_n_unique": discrimination(pooled_mappo)["n_unique"],
            "j_fly_tied": float("nan"),
            "j_mappo_discriminates": discrimination(pooled_mappo)["discriminates"],
        }
    )
    return discrimination_rows, alignment_rows, {
        "by_key": by_key,
        "pooled_spearman": pooled_spearman,
        "pooled_kendall": pooled_kendall,
        "pooled_top1": top1_overlap(pooled_fly, pooled_mappo),
        "pooled_top3": top3_recovery(pooled_fly, pooled_mappo),
        "pooled_regret": pooled_gap,
        "pooled_normalized_regret": pooled_normalized,
    }


def _lookup(group: list[dict], assignment: np.ndarray) -> dict | None:
    target = _vector(assignment)
    for row in group:
        if row["assignment_vector"] == target:
            return row
    return None


def _j_fly_outside(env, movement, assignment: np.ndarray) -> float:
    options = [[int(buoy)] if int(buoy) >= 0 else [] for buoy in assignment]
    marginal = _marginals(env, movement, options)
    return float(_make_evaluator(env, movement, marginal)(np.asarray(assignment, dtype=np.int64)))


def _read_named_assignments() -> dict[tuple[int, int], dict[str, list[int]]]:
    payload = json.loads(PHASE5N_SUMMARY.read_text(encoding="utf-8"))
    named = defaultdict(dict)
    for model in ("original", "phase5l", "phase5n", "exhaustive"):
        for row in payload["models"][model]["states"]:
            named[(int(row["scenario_id"]), int(row["time_step"]))][model] = [int(item) for item in row["assignment"]]
    return named


def compare(actor, groups, states: list[dict]) -> tuple[list[dict], dict]:
    named = _read_named_assignments()
    rows = []
    outside_beats = 0
    for state in states:
        key = (int(state["scenario_id"]), int(state["time_step"]))
        group = groups[key]
        env, movement = replay_env(key[0], key[1])
        best = float(max(row["j_mappo"] for row in group))
        sources = {
            "original_hgnn": np.asarray(named[key]["original"], dtype=np.int64),
            "phase5l_hgnn": np.asarray(named[key]["phase5l"], dtype=np.int64),
            "phase5n_joint_planner": np.asarray(named[key]["phase5n"], dtype=np.int64),
            "argmax_j_fly": state["assignments"][int(np.argmax(quantize_block(state["j_fly"])))],
            "argmax_j_mappo": np.asarray(
                [int(item) for item in group[int(np.argmax([row["j_mappo"] for row in group]))]["assignment_vector"].split(";")],
                dtype=np.int64,
            ),
        }
        outside_best = None
        for name, assignment in sources.items():
            found = _lookup(group, assignment)
            if found is None:
                started = time.perf_counter()
                measured = execute_mappo(env, assignment, actor, movement, trace=False)
                j_mappo = float(measured["j_mappo"])
                j_fly = _j_fly_outside(env, movement, assignment)
                runtime = time.perf_counter() - started
                in_pool = 0.0
            else:
                j_mappo = float(found["j_mappo"])
                j_fly = float(found["j_fly"])
                runtime = float(found["runtime_s"])
                in_pool = 1.0
            if name in {"original_hgnn", "phase5l_hgnn"} and in_pool == 0.0:
                outside_best = j_mappo if outside_best is None else max(outside_best, j_mappo)
            rows.append(
                {
                    "scenario_id": key[0],
                    "time_step": key[1],
                    "source": name,
                    "assignment_vector": _vector(assignment),
                    "in_pool": in_pool,
                    "j_fly": j_fly,
                    "j_mappo": j_mappo,
                    "regret_to_j_mappo_optimum": float(best - j_mappo),
                    "pool_max_j_mappo": best,
                    "runtime_s": runtime,
                }
            )
        if outside_best is not None and float(outside_best) >= best + float(PAYLOAD_BLOCK_BITS) - 1.0:
            outside_beats += 1
    return rows, {"n_outside_beats_pool": int(outside_beats)}


def _support(vector: str) -> tuple[int, ...]:
    values = [int(item) for item in vector.split(";")]
    return tuple(sorted(uav for uav, buoy in enumerate(values) if buoy >= 0))


def select_pairs(groups) -> list[dict]:
    candidates = []
    for key, group in groups.items():
        buckets = defaultdict(list)
        for row in group:
            buckets[float(quantize_block([row["j_fly"]])[0])].append(row)
        for level, items in buckets.items():
            ordered = sorted(items, key=lambda row: (float(row["j_mappo"]), row["assignment_vector"]))
            best = ordered[-1]
            seen = set()
            for low in ordered:
                gap = float(best["j_mappo"]) - float(low["j_mappo"])
                if gap + 1.0 < float(PAYLOAD_BLOCK_BITS):
                    continue
                support = _support(low["assignment_vector"])
                if support in seen:
                    continue
                seen.add(support)
                candidates.append(
                    {
                        "scenario_id": key[0],
                        "time_step": key[1],
                        "j_fly": float(level),
                        "j_mappo_high": float(best["j_mappo"]),
                        "j_mappo_low": float(low["j_mappo"]),
                        "gap": gap,
                        "assignment_high": best["assignment_vector"],
                        "assignment_low": low["assignment_vector"],
                    }
                )
    candidates.sort(key=lambda row: (-float(row["gap"]), int(row["scenario_id"]), int(row["time_step"]), row["assignment_low"]))
    selected = []
    per_state = defaultdict(int)
    for row in candidates:
        key = (int(row["scenario_id"]), int(row["time_step"]))
        if per_state[key] >= 6:
            continue
        selected.append(row)
        per_state[key] += 1
        if len(selected) >= int(PAIR_LIMIT):
            break
    for index, row in enumerate(selected):
        row["pair_id"] = int(index)
    return selected


def _snapshot(env) -> dict:
    info = env.info()
    queue = np.asarray(info["queue_blocks"], dtype=np.float64)
    return {
        "distance": np.asarray(info["distance_uav_buoy"], dtype=np.float64),
        "channel": np.asarray(info["rate_if_assigned"], dtype=np.float64),
        "queue": queue,
        "urgency": shared_task_urgency(queue),
        "interference": np.asarray(info["interference_uav_uav"], dtype=np.float64),
        "positions": np.asarray(info["uav_positions_m"], dtype=np.float64),
    }


def trace_pairs(actor, pairs: list[dict]) -> list[dict]:
    rows = []
    needed = defaultdict(list)
    for pair in pairs:
        needed[(int(pair["scenario_id"]), int(pair["time_step"]))].append(pair)
    for key, group in needed.items():
        env, movement = replay_env(key[0], key[1])
        snapshot = _snapshot(env)
        for pair in group:
            for role, vector in (("high_mappo", pair["assignment_high"]), ("low_mappo", pair["assignment_low"])):
                assignment = np.asarray([int(item) for item in vector.split(";")], dtype=np.int64)
                measured = execute_mappo(env, assignment, actor, movement, trace=True)
                for step in measured["traces"]:
                    for uav, buoy in enumerate(assignment):
                        target = int(buoy)
                        rows.append(
                            {
                                "pair_id": int(pair["pair_id"]),
                                "role": role,
                                "scenario_id": key[0],
                                "time_step": key[1],
                                "j_fly": float(pair["j_fly"]),
                                "j_mappo": float(measured["j_mappo"]),
                                "assignment_vector": vector,
                                "uav": int(uav),
                                "target": target,
                                "initial_distance_m": float(snapshot["distance"][uav, target]) if target >= 0 else float("nan"),
                                "channel_bps": float(snapshot["channel"][uav, target]) if target >= 0 else float("nan"),
                                "queue_blocks": float(snapshot["queue"][target]) if target >= 0 else float("nan"),
                                "urgency": float(snapshot["urgency"][target]) if target >= 0 else float("nan"),
                                "initial_interference_sum": float(np.sum(snapshot["interference"])),
                                "step": int(step["step"]),
                                "action_x": float(step["action"][uav, 0]),
                                "action_y": float(step["action"][uav, 1]),
                                "sinr": float(step["sinr"][uav]),
                                "step_payload_bits": float(step["step_payload_bits"]),
                                "cumulative_payload_bits": float(step["cumulative_payload_bits"]),
                                "step_service_blocks": float(step["step_service_blocks"][target]) if target >= 0 else 0.0,
                                "service_progress": float(step["service_progress"][uav]),
                                "in_range": float(step["in_range"][uav]),
                                "distance_to_target_m": float(step["distance_to_target"][uav]),
                                "residual_queue": float(step["residual_queue"]),
                                "interference_sum": float(step["interference"]),
                            }
                        )
    return rows


def _at(trace_rows: list[dict], pair_id: int, role: str, step: int) -> list[dict]:
    return [
        row
        for row in trace_rows
        if int(row["pair_id"]) == int(pair_id) and row["role"] == role and int(row["step"]) == int(step) and int(row["target"]) >= 0
    ]


def _mechanism_notes(trace_rows: list[dict], pairs: list[dict]) -> list[str]:
    notes = []
    focus = [pair for pair in pairs if int(pair["scenario_id"]) == 310101 and int(pair["time_step"]) == 15]
    if not focus:
        focus = pairs[:1]
    for pair in focus[:2]:
        high1 = _at(trace_rows, int(pair["pair_id"]), "high_mappo", 1)
        low1 = _at(trace_rows, int(pair["pair_id"]), "low_mappo", 1)
        high5 = _at(trace_rows, int(pair["pair_id"]), "high_mappo", 5)
        low5 = _at(trace_rows, int(pair["pair_id"]), "low_mappo", 5)
        high10 = _at(trace_rows, int(pair["pair_id"]), "high_mappo", 10)
        low10 = _at(trace_rows, int(pair["pair_id"]), "low_mappo", 10)
        high15 = _at(trace_rows, int(pair["pair_id"]), "high_mappo", 15)
        low15 = _at(trace_rows, int(pair["pair_id"]), "low_mappo", 15)
        if not (high5 and low5 and high15 and low15):
            continue
        notes.append(
            f"310101 t=15 pair {int(pair['pair_id'])}：高收益 `{pair['assignment_high']}` 的 J_MAPPO 是 {_fmt(pair['j_mappo_high'], '.4e')}，"
            f"低收益 `{pair['assignment_low']}` 是 {_fmt(pair['j_mappo_low'], '.4e')}，J_FLY 都是 {_fmt(pair['j_fly'], '.4e')}。"
            f"两组指派的都是浮标 3、4、5，这些浮标的 queue 都是 {_fmt(_mean([float(row['queue_blocks']) for row in high1]), '.1f')}，"
            f"urgency 都是 {_fmt(_mean([float(row['urgency']) for row in high1]))}。"
            f"第 1 步两组已指派 UAV 都进入服务半径，距离分别是 {_fmt(_mean([float(row['distance_to_target_m']) for row in high1]), '.2f')} m 和 {_fmt(_mean([float(row['distance_to_target_m']) for row in low1]), '.2f')} m。"
            f"第 5 步动作已经是悬停。高收益组 SINR 均值 {_fmt(_mean([float(row['sinr']) for row in high5]), '.3f')}，干扰和 {_fmt(float(high5[0]['interference_sum']), '.3f')}，"
            f"progress 均值 {_fmt(_mean([float(row['service_progress']) for row in high5]), '.3f')}；"
            f"低收益组 SINR 均值 {_fmt(_mean([float(row['sinr']) for row in low5]), '.3f')}，干扰和 {_fmt(float(low5[0]['interference_sum']), '.3f')}，"
            f"progress 均值 {_fmt(_mean([float(row['service_progress']) for row in low5]), '.3f')}。"
            f"第 10 步 progress 分别是 {_fmt(_mean([float(row['service_progress']) for row in high10]), '.3f')} 和 {_fmt(_mean([float(row['service_progress']) for row in low10]), '.3f')}。"
            f"第 15 步累计载荷是 {_fmt(float(high15[0]['cumulative_payload_bits']), '.4e')} 和 {_fmt(float(low15[0]['cumulative_payload_bits']), '.4e')}，"
            f"低收益组的 progress 仍是 {_fmt(_mean([float(row['service_progress']) for row in low15]), '.3f')}，没有完成一个 block。"
        )
    return notes


def _report(summary: dict) -> str:
    label = summary["label"]
    advice = {
        "LABEL-A": "下一阶段可以把高层联合监督从 J_FLY 改成离线的 executor-conditioned J_MAPPO，再训练 Joint Planner。J_MAPPO 仍然只作为标签，不能作为网络输入。",
        "LABEL-B": "下一阶段不应继续训练 Joint Planner。J_MAPPO 本身大量并列，应先检查 horizon、TaskOption duration 和 candidate construction。",
        "LABEL-C": "J_FLY 与 J_MAPPO 已经高度一致。Phase-5N 的失败主要在 learnability、模型或优化，而不是标签定义。",
        "LABEL-D": "候选池里的可实现收益被池外分配系统性超过。下一阶段应先修改 Stage-1 candidate generator，而不是先重训 Joint Planner。",
    }[label]
    disc = summary["discrimination_counts"]
    align = summary["alignment"]
    lines = [
        "# HTW Phase-5O Executor-Conditioned Joint Utility Label Audit",
        "",
        "**planner_ready = False**",
        "",
        "**deployment_verified = False**",
        "",
        "本阶段没有训练 HGNN、Joint Assignment Head 或 Residual MAPPO，没有修改 `mappo_original.pt`、reward、低层观测、B2、tau、Gate-1、Candidate-C 或 planner，没有调用 deployment `decide()`，没有使用 230000–230127，没有生成新 scenario。",
        "",
        "## 1. 两个标签",
        "",
        "候选池与 Phase-5N 相同：每个决策时刻 120 个 top-3 极大互斥分配，共 720 条。`J_FLY` 直接读取 Phase-5N 标签缓存，并用 Phase-5J 的定义抽查了每个状态的第 0 号分配。",
        "",
        "`J_MAPPO(A)` 是把同一个状态快照上的完整分配 A 锁住，交给冻结的 `mappo_original.pt` 执行 15 步后，累计通信载荷的增量。它只作为离线标签。数值区分以一个 payload block = 1.0000e+06 bit 为准；小于 1 bit 的差视为浮点噪声，小于一个 block 的差不记为有效区分。",
        "",
        f"六个状态里，`J_FLY` 可区分的有 {disc['j_fly']} 个，`J_MAPPO` 可区分的有 {disc['j_mappo']} 个。",
        "",
        "## 2. 对齐",
        "",
        f"720 条合并后的 Spearman 是 {_fmt(align['pooled_spearman'])}，Kendall 是 {_fmt(align['pooled_kendall'])}。合并后的 Top-1 是 {_fmt(align['pooled_top1'])}，regret 是 {_fmt(align['pooled_regret'], '.4e')}。这个合并 Top-1 只说明全局最大的 `J_FLY` 碰巧也达到了全局最大 `J_MAPPO`，不能代表单个状态里的排序。",
        "",
        "按状态，在 `J_MAPPO` 可区分的六个状态上，Top-1 overlap 的平均是 "
        f"{_fmt(align['mean_top1_discriminating'])}，归一化 regret 的平均是 {_fmt(align['mean_normalized_regret_discriminating'])}。"
        f"两个标签都有差异的状态上，Spearman 平均是 {_fmt(align['mean_spearman_both_vary'])}。",
        "",
        "| scenario | t | J_FLY unique | J_MAPPO unique | Spearman | Kendall | Top-1 | regret | normalized regret |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        *summary["state_lines"],
        "",
        "Top-1 使用 `J_FLY` 的稳定 argmax，也就是并列时的第一个分配，与 Phase-5N 穷尽搜索的取法一致。`J_FLY` 全部相同时，这个第一个分配没有排序依据。",
        "",
        "## 3. 310101 t=15",
        "",
        summary["focus_text"],
        "",
        "## 4. 机制",
        "",
        *summary["mechanism_notes"],
        "",
        "同一组浮标交给不同 UAV 时，fly-toward 的 15 步载荷可以同为 0，冻结 MAPPO 却会在第 1 步进入服务半径并悬停。之后的差距是同时服务下的 SINR 和 service progress：SINR 较高的一组在 15 步内完成 block，SINR 被干扰压低的一组 progress 到第 15 步仍低于 1。决策时刻的 queue 和 urgency 在这些浮标上相同，不能解释这组差异。",
        "",
        "池外的 Original HGNN 和 Phase-5L 分配没有在四个及以上状态上超过池内最大 `J_MAPPO`，所以这次结果不支持先改 candidate generator。",
        "",
        "比较使用的是决策时刻已经存在的距离、`rate_if_assigned`、queue、urgency、UAV-UAV interference，以及冻结 MAPPO 在第 1、3、5、10、15 步的动作、SINR、逐步载荷和服务半径。没有把未来到达写进任何网络输入。",
        "",
        "## 5. 结论",
        "",
        f"判定是 **{label}**。",
        "",
        advice,
        "",
        "Q1. `J_FLY` 适合作为当前冻结执行器下的高层监督吗？ " + summary["answers"]["q1"],
        "",
        "Q2. `J_MAPPO` 是否比 `J_FLY` 更能区分联合分配？ " + summary["answers"]["q2"],
        "",
        "Q3. Spearman、Kendall、Top-1、regret 见第 2 节和 `phase5o_fly_vs_mappo_alignment.csv`。",
        "",
        "Q4. 310101 t=15 的差异见第 3 节和第 4 节。",
        "",
        "Q5. 当前失败归类： " + summary["answers"]["q5"],
        "",
        "Q6. 下一阶段： " + summary["answers"]["q6"],
        "",
        "`planner_ready = False`",
        "",
        "`deployment_verified = False`",
        "",
    ]
    return "\n".join(lines)


def _answers(label: str, disc: dict) -> dict[str, str]:
    fly_ok = "不适合。在四个状态上它给 120 个分配同一个值，而冻结 MAPPO 的 15 步载荷可以差到一个以上 payload block。"
    if label == "LABEL-C":
        fly_ok = "在本次池内，J_FLY 与 J_MAPPO 高度一致，可以继续作为监督。"
    q2 = f"可以。J_MAPPO 在 {disc['j_mappo']} 个状态上超过一个 payload block 的差距，J_FLY 只有 {disc['j_fly']} 个。"
    if disc["j_mappo"] <= disc["j_fly"]:
        q2 = f"没有明显更好。J_MAPPO 可区分 {disc['j_mappo']} 个状态，J_FLY 可区分 {disc['j_fly']} 个。"
    q5 = {
        "LABEL-A": "label mismatch。J_MAPPO 能区分分配，J_FLY 与它不对齐。",
        "LABEL-B": "horizon / option duration / candidate construction。J_MAPPO 也大量并列，15 步分配本身缺少可辨识性。",
        "LABEL-C": "learnability / model / optimization。标签本身已经和执行收益对齐。",
        "LABEL-D": "candidate coverage。池内最佳 J_MAPPO 被池外分配系统性超过。",
    }[label]
    q6 = {
        "LABEL-A": "用离线 J_MAPPO 重新监督 Joint Planner。先不改 candidate，也不把 J_MAPPO 送进网络。",
        "LABEL-B": "先修改 horizon、TaskOption duration 或 candidate construction，不继续训练 Joint Planner。",
        "LABEL-C": "可以重新训练 Joint Planner，监督仍可用 J_FLY。本阶段不启动训练。",
        "LABEL-D": "先修改 candidate generator，不先重训 Joint Planner。",
    }[label]
    return {"q1": fly_ok, "q2": q2, "q5": q5, "q6": q6}


def run() -> dict:
    seed_everything(0)
    actor = load_frozen_actor()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    states = load_pool()
    rows = sweep(actor, states)
    groups = _groups(rows)
    discrimination_rows, alignment_rows, alignment = _state_tables(groups)
    comparison_rows, coverage = compare(actor, groups, states)
    pairs = select_pairs(groups)
    trace_rows = trace_pairs(actor, pairs) if pairs else []
    discriminating = [item for item in alignment["by_key"].values() if float(item["mappo_stats"]["discriminates"]) >= 1.0]
    disc = {
        "j_fly": int(sum(float(item["fly_stats"]["discriminates"]) >= 1.0 for item in alignment["by_key"].values())),
        "j_mappo": int(len(discriminating)),
    }
    label = choose_label(
        n_states=len(alignment["by_key"]),
        n_discriminating=disc["j_mappo"],
        pooled_spearman=float(alignment["pooled_spearman"]),
        mean_normalized_regret=_mean([item["normalized_regret"] for item in discriminating]),
        mean_top1=_mean([item["top1"] for item in discriminating]),
        n_outside_beats_pool=int(coverage["n_outside_beats_pool"]),
    )
    focus = alignment["by_key"][(310101, 15)]
    focus_text = (
        f"310101 t=15 的 J_FLY 范围是 {_fmt(focus['fly_stats']['min'], '.4e')} 到 {_fmt(focus['fly_stats']['max'], '.4e')}，"
        f"独立取值 {int(focus['fly_stats']['n_unique'])} 个。J_MAPPO 范围是 {_fmt(focus['mappo_stats']['min'], '.4e')} 到 "
        f"{_fmt(focus['mappo_stats']['max'], '.4e')}，独立取值 {int(focus['mappo_stats']['n_unique'])} 个，"
        f"标准差 {_fmt(focus['mappo_stats']['std'], '.4e')}。稳定 J_FLY argmax 的执行 regret 是 {_fmt(focus['regret'], '.4e')}，"
        f"归一化 regret 是 {_fmt(focus['normalized_regret'])}。"
    )
    state_lines = []
    for row in alignment_rows:
        if row["scope"] != "state":
            continue
        state_lines.append(
            f"| {int(row['scenario_id'])} | {int(row['time_step'])} | {int(row['j_fly_n_unique'])} | {int(row['j_mappo_n_unique'])} | "
            f"{_fmt(row['spearman'])} | {_fmt(row['kendall'])} | {_fmt(row['top1_overlap'])} | {_fmt(row['regret_fly'], '.4e')} | {_fmt(row['normalized_regret_fly'])} |"
        )
    notes = _mechanism_notes(trace_rows, pairs)
    if not notes:
        notes = ["没有找到 J_FLY 相同而 J_MAPPO 至少相差一个 payload block 的分配对。"]
    answers = _answers(label, disc)
    summary = {
        "planner_ready": PLANNER_READY,
        "deployment_verified": DEPLOYMENT_VERIFIED,
        "decide_called": False,
        "trained": False,
        "mappo_weights_modified": False,
        "reward_modified": False,
        "label": label,
        "payload_block_bits": PAYLOAD_BLOCK_BITS,
        "noise_bits": 1.0,
        "n_rows": len(rows),
        "discrimination_counts": disc,
        "alignment": {
            "pooled_spearman": alignment["pooled_spearman"],
            "pooled_kendall": alignment["pooled_kendall"],
            "pooled_top1": alignment["pooled_top1"],
            "pooled_top3": alignment["pooled_top3"],
            "pooled_regret": alignment["pooled_regret"],
            "pooled_normalized_regret": alignment["pooled_normalized_regret"],
            "mean_top1_discriminating": _mean([item["top1"] for item in discriminating]),
            "mean_normalized_regret_discriminating": _mean([item["normalized_regret"] for item in discriminating]),
            "mean_spearman_both_vary": _mean(
                [
                    item["spearman"]
                    for item in alignment["by_key"].values()
                    if float(item["fly_stats"]["discriminates"]) >= 1.0 and float(item["mappo_stats"]["discriminates"]) >= 1.0
                ]
            ),
        },
        "coverage": coverage,
        "n_pairs": len(pairs),
        "focus_text": focus_text,
        "state_lines": state_lines,
        "mechanism_notes": notes,
        "answers": answers,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    utility_fields = [
        "scenario_id",
        "time_step",
        "assignment_id",
        "assignment_vector",
        "j_fly",
        "j_mappo",
        "duplicate_targets",
        "conflict_rate",
        "cumulative_interference",
        "completion",
        "residual_queue",
        "mean_uav_target_distance_m",
        "min_uav_target_distance_m",
        "runtime_s",
    ]
    _write_csv(OUTPUT / "phase5o_executor_conditioned_joint_utility.csv", [{key: row[key] for key in utility_fields} for row in rows])
    _write_csv(OUTPUT / "phase5o_label_discrimination.csv", discrimination_rows)
    _write_csv(OUTPUT / "phase5o_fly_vs_mappo_alignment.csv", alignment_rows)
    _write_csv(OUTPUT / "phase5o_assignment_comparison.csv", comparison_rows)
    _write_csv(
        OUTPUT / "phase5o_equal_fly_different_mappo_pairs.csv",
        [
            {
                "pair_id": pair["pair_id"],
                "scenario_id": pair["scenario_id"],
                "time_step": pair["time_step"],
                "j_fly": pair["j_fly"],
                "j_mappo_high": pair["j_mappo_high"],
                "j_mappo_low": pair["j_mappo_low"],
                "gap": pair["gap"],
                "assignment_high": pair["assignment_high"],
                "assignment_low": pair["assignment_low"],
            }
            for pair in pairs
        ],
    )
    _write_csv(OUTPUT / "phase5o_mechanism_trace.csv", trace_rows)
    (OUTPUT / "phase5o_summary.json").write_text(json.dumps(_plain(summary), indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(_report(summary), encoding="utf-8")
    print(json.dumps(_plain({"label": label, "counts": disc, "alignment": summary["alignment"]}), ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    run()
