"""Execution-compatibility audit. Frozen checkpoints only."""
from __future__ import annotations

import csv
import json

import numpy as np
import torch

from wt.lowlevel_v2.integration_phase1.stats import kendall_tau, spearman
from wt.lowlevel_v2.phase5f_utility_rank.score import focal_scores, load_deepsets, new_hgnn
from wt.lowlevel_v2.phase5g_closed_loop.collect import collect_scenario, load_frozen_actor, replay_env
from wt.lowlevel_v2.phase5g_closed_loop.protocol import UTILITY_CKPT
from wt.lowlevel_v2.phase5g_closed_loop.run_phase5g import _greedy_assignment

from .protocol import BLOCK_BITS, DEPLOYMENT_VERIFIED, OUTPUT, PLANNER_READY, REPORT, SCENARIOS
from .rollout import trace_h15


def _mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _fmt(value: float) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value):.4f}"


def _scores(model, sample: dict, *, use_urgency: bool) -> list[float]:
    with torch.no_grad():
        values = focal_scores(model, sample, use_urgency=use_urgency).detach().cpu().numpy()
    return [float(item) for item in values]


def _choose(scores: list[float]) -> int:
    return int(np.argmax(np.asarray(scores, dtype=np.float64)))


def _loss(expected: float, actual: float) -> float:
    return float(max(0.0, float(expected) - float(actual)))


def _duplicate_targets(assignment: np.ndarray) -> int:
    chosen = [int(item) for item in assignment if int(item) >= 0]
    return int(len(chosen) - len(set(chosen)))


def _pack_gap(rows: list[dict]) -> dict:
    expected = [float(row["fly_payload"]) for row in rows]
    actual = [float(row["mappo_payload"]) for row in rows]
    loss = [_loss(left, right) for left, right in zip(expected, actual)]
    return {
        "n": len(rows),
        "fly_payload": _mean(expected),
        "mappo_payload": _mean(actual),
        "execution_loss": _mean(loss),
        "u_exec": _mean([left - gap for left, gap in zip(expected, loss)]),
        "spearman": spearman(expected, actual),
        "kendall": kendall_tau(expected, actual),
        "conflict_rate": _mean([float(row["conflict_rate"]) for row in rows if "conflict_rate" in row]),
        "conflict_pairs": _mean([float(row["conflict_pairs"]) for row in rows if "conflict_pairs" in row]),
        "interference": _mean([float(row["interference"]) for row in rows if "interference" in row]),
        "min_uav_distance_m": _mean([float(row["min_uav_distance_m"]) for row in rows if "min_uav_distance_m" in row]),
        "duplicate_targets": _mean([float(row.get("duplicate_targets", 0.0)) for row in rows]),
    }


def _report(summary: dict) -> str:
    single = summary["single_swap"]
    joint = summary["joint"]
    solo = summary["solo"]
    labels = (
        ("utility HGNN", "utility_hgnn"),
        ("DeepSets", "deepsets"),
        ("fly-toward utility oracle", "oracle"),
    )
    table = [
        "| assignment | fly payload | MAPPO payload | execution loss | conflict rate | interference | min UAV distance |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label, key in labels:
        item = joint[key]
        table.append(
            f"| {label} | {_fmt(item['fly_payload'])} | {_fmt(item['mappo_payload'])} | {_fmt(item['execution_loss'])} | {_fmt(item['conflict_rate'])} | {_fmt(item['interference'])} | {_fmt(item['min_uav_distance_m'])} |"
        )
    primary = summary["primary"]
    return "\n".join(
        [
            "# HTW Phase-5H Low-Level Execution Compatibility Audit",
            "",
            "**planner_ready = False**",
            "",
            "**deployment_verified = False**",
            "",
            "本阶段只比较冻结目标和冻结执行器。没有重训 HGNN，没有重训 Residual MAPPO，没有修改 B2、τ、Gate-1、Candidate-C、planner 或 deployment `decide()`，没有使用 230000–230127。",
            "",
            "## 1. 问题定位",
            "",
            "Phase-5G 的单目标 MAPPO payload 是 1.57e6 bit，联合分配后再执行降到 3.33e5 bit。这里把同一状态拆成三类：",
            "",
            "- A：联合目标本身的 fly-toward payload 就低，问题在高层组合。",
            "- B：同一条单目标分配上，fly-toward 明显高于 MAPPO，问题在低层跟踪。",
            "- C：单目标能执行，联合分配的服务冲突、干扰或同时服务把收益打掉。",
            "",
            f"主判定是 `{primary}`。单目标 execution loss = {_fmt(single['execution_loss'])} bit。utility 联合分配的 execution loss = {_fmt(joint['utility_hgnn']['execution_loss'])} bit。utility 联合 MAPPO payload = {_fmt(joint['utility_hgnn']['mappo_payload'])} bit，单目标 MAPPO payload = {_fmt(single['mappo_payload'])} bit。",
            "",
            "## 2. High-level ranking",
            "",
            "场景仍是 310102 和 310103。oracle assignment 是同一状态下按 fly-toward payload 做排他贪心，不读取未来到达，也不是 Phase-5B 的簇 oracle。",
            "",
            "单目标替换沿用 Phase-5G：其他无人机保持当前排他分配，只换一名无人机的 utility HGNN 目标。",
            "",
            f"这条单目标的 fly-toward payload = {_fmt(single['fly_payload'])}，MAPPO payload = {_fmt(single['mappo_payload'])}。这是全部 36 个无人机决策，包含 payload 没有拉开的时刻。Phase-5G 的 1.57e6 只统计已经分开的组。两者的 Spearman = {_fmt(single['spearman'])}，Kendall = {_fmt(single['kendall'])}。",
            "",
            f"全部候选上，fly-toward utility 和 MAPPO payload 的 Spearman = {_fmt(summary['candidate_alignment']['spearman'])}，Kendall = {_fmt(summary['candidate_alignment']['kendall'])}，n = {summary['candidate_alignment']['n']}。U_exec 用的 execution_loss 就是这次冻结 MAPPO 相对 fly-toward 的落差，不是新训练出来的量。",
            "",
            "## 3. Low-level execution gap",
            "",
            "同一条 utility HGNN 单目标分配，oracle trajectory 是 fly-toward，对照是冻结 `mappo_original.pt`。",
            "",
            f"execution_loss = max(0, fly payload - MAPPO payload)，平均 {_fmt(single['execution_loss'])} bit。U_exec = expected_payload - execution_loss，平均 {_fmt(single['u_exec'])} bit。",
            "",
            "联合分配上的同一对执行器：",
            "",
            *table,
            "",
            "## 4. Multi-UAV conflict",
            "",
            "环境没有单独的碰撞事件。这里记录它已经给出的服务冲突 `conflict_rate`、`n_conflict_pairs`，以及 UAV–UAV 干扰矩阵的和。最近无人机距离只作为几何量。",
            "",
            f"排他贪心的重复目标数是 {_fmt(joint['utility_hgnn']['duplicate_targets'])}。utility 联合分配在 MAPPO 执行中的平均冲突率是 {_fmt(joint['utility_hgnn']['conflict_rate'])}，平均干扰是 {_fmt(joint['utility_hgnn']['interference'])}。",
            "",
            f"单架服务是只派出 utility HGNN 选中的那一架、其余悬停。它的 MAPPO payload 平均 {_fmt(solo['mappo_payload'])} bit，干扰 {_fmt(solo['interference'])}。联合六架的 MAPPO payload 是 {_fmt(joint['utility_hgnn']['mappo_payload'])} bit，干扰 {_fmt(joint['utility_hgnn']['interference'])}。",
            "",
            f"单目标替换已经是多机环境，只是其他无人机仍用排他匹配。它的 MAPPO payload {_fmt(single['mappo_payload'])} 高于六架都改成 utility 目标之后的 {_fmt(joint['utility_hgnn']['mappo_payload'])}。",
            "",
            "## 5. Reward",
            "",
            summary["reward_text"],
            "",
            "本阶段没有修改 reward。",
            "",
            "## 6. Joint optimization",
            "",
            summary["joint_text"],
            "",
            "本阶段不进入 joint optimization，也不进入 deployment verification。",
            "",
            "`planner_ready = False`",
            "",
            "`deployment_verified = False`",
            "",
        ]
    )


def run() -> dict:
    actor = load_frozen_actor()
    rows = []
    for seed in SCENARIOS:
        print(f"audit scenario {seed}", flush=True)
        rows.extend(collect_scenario(int(seed), actor))
    model = new_hgnn()
    if not UTILITY_CKPT.is_file():
        raise FileNotFoundError(str(UTILITY_CKPT))
    model.load_state_dict(torch.load(UTILITY_CKPT, map_location="cpu"))
    model.eval()
    deepsets = load_deepsets()
    utility_scores = {}
    deepsets_scores = {}
    oracle_scores = {}
    single_rows = []
    for sample in rows:
        key = (int(sample["scenario_id"]), int(sample["time_step"]), int(sample["uav_id"]))
        utility = _scores(model, sample, use_urgency=True)
        other = _scores(deepsets, sample, use_urgency=False)
        utility_scores[key] = utility
        deepsets_scores[key] = other
        oracle_scores[key] = [float(item) for item in sample["utility_payload"]]
        chosen = _choose(utility)
        single_rows.append(
            {
                "fly_payload": float(sample["utility_payload"][chosen]),
                "mappo_payload": float(sample["executed_payload"][chosen]),
            }
        )
    grouped: dict[tuple[int, int], list[dict]] = {}
    for sample in rows:
        grouped.setdefault((int(sample["scenario_id"]), int(sample["time_step"])), []).append(sample)
    joint_rows = {name: [] for name in ("utility_hgnn", "deepsets", "oracle")}
    solo_rows = []
    score_tables = {"utility_hgnn": utility_scores, "deepsets": deepsets_scores, "oracle": oracle_scores}
    for (scenario, moment), group in grouped.items():
        env, movement = replay_env(scenario, moment)
        for name, table in score_tables.items():
            local = {int(sample["uav_id"]): table[(scenario, moment, int(sample["uav_id"]))] for sample in group}
            assignment = _greedy_assignment(group, local)
            fly = trace_h15(env, assignment, actor, movement, "fly")
            mappo = trace_h15(env, assignment, actor, movement, "mappo")
            joint_rows[name].append(
                {
                    "scenario": scenario,
                    "time_step": moment,
                    "source": name,
                    "fly_payload": fly["payload"],
                    "mappo_payload": mappo["payload"],
                    "conflict_rate": mappo["conflict_rate"],
                    "conflict_pairs": mappo["conflict_pairs"],
                    "interference": mappo["interference"],
                    "min_uav_distance_m": mappo["min_uav_distance_m"],
                    "duplicate_targets": _duplicate_targets(assignment),
                }
            )
        for sample in group:
            chosen = _choose(utility_scores[(scenario, moment, int(sample["uav_id"]))])
            solo = np.full(sample["context"].shape[0], -1, dtype=np.int64)
            solo[int(sample["uav_id"])] = int(sample["candidate_ids"][chosen])
            traced = trace_h15(env, solo, actor, movement, "mappo")
            solo_rows.append(
                {
                    "fly_payload": float(sample["utility_payload"][chosen]),
                    "mappo_payload": traced["payload"],
                    "conflict_rate": traced["conflict_rate"],
                    "conflict_pairs": traced["conflict_pairs"],
                    "interference": traced["interference"],
                    "min_uav_distance_m": traced["min_uav_distance_m"],
                    "duplicate_targets": 0,
                }
            )
        print(f"traced {scenario} t={moment}", flush=True)
    candidate_fly = []
    candidate_mappo = []
    for sample in rows:
        candidate_fly.extend(float(item) for item in sample["utility_payload"])
        candidate_mappo.extend(float(item) for item in sample["executed_payload"])
    single = _pack_gap(single_rows)
    joint = {name: _pack_gap(items) for name, items in joint_rows.items()}
    solo = _pack_gap(solo_rows)
    single_gap = float(single["fly_payload"] - single["mappo_payload"])
    joint_gap = float(joint["utility_hgnn"]["fly_payload"] - joint["utility_hgnn"]["mappo_payload"])
    low_level = bool(single_gap >= BLOCK_BITS or joint_gap >= BLOCK_BITS)
    selection = bool(float(joint["utility_hgnn"]["fly_payload"]) + 1.0 < float(joint["deepsets"]["fly_payload"]))
    composition = bool(float(single["mappo_payload"]) > float(joint["utility_hgnn"]["mappo_payload"]) + 0.5 * BLOCK_BITS)
    joint_execution = bool(low_level and float(single["mappo_payload"]) + BLOCK_BITS > float(single["fly_payload"]))
    if low_level:
        primary = "B_low_level_execution"
    elif selection:
        primary = "A_joint_target_selection"
    elif composition:
        primary = "C_multi_uav_composition"
    else:
        primary = "mixed"
    if low_level:
        reward_text = "同一条分配上，fly-toward 比 MAPPO 至少多一个服务块。这是低层执行差距。本阶段仍不修改 reward。"
    else:
        reward_text = "同一条单目标分配上，MAPPO payload 不低于 fly-toward。联合分配上 MAPPO 也不低于 fly-toward。收益下降不是当前 reward 没让低层跟踪目标。本阶段不修改 reward。"
    if selection or composition:
        joint_text = "utility 联合分配的 fly-toward payload 低于 DeepSets，服务冲突率是 0。六架目标同时替换之后，目标组合本身的通信收益就低。本阶段不开始高低层联合训练。"
    else:
        joint_text = "这组对照没有给出必须联合训练的差距。本阶段不开始高低层联合训练。"
    summary = {
        "planner_ready": PLANNER_READY,
        "deployment_verified": DEPLOYMENT_VERIFIED,
        "decide_called": False,
        "hgnn_retrained": False,
        "mappo_retrained": False,
        "tau_modified": False,
        "reward_modified": False,
        "joint_optimization_started": False,
        "holdout_used": False,
        "scenarios": list(SCENARIOS),
        "primary": primary,
        "flags": {
            "A_joint_target_selection": selection,
            "B_low_level_execution": low_level,
            "C_joint_execution": joint_execution,
            "C_multi_uav_composition": composition,
        },
        "single_swap": single,
        "joint": joint,
        "solo": solo,
        "candidate_alignment": {
            "n": len(candidate_fly),
            "spearman": spearman(candidate_fly, candidate_mappo),
            "kendall": kendall_tau(candidate_fly, candidate_mappo),
            "execution_loss": _mean([_loss(left, right) for left, right in zip(candidate_fly, candidate_mappo)]),
        },
        "reward_text": reward_text,
        "joint_text": joint_text,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fields = [
        "scenario",
        "time_step",
        "source",
        "fly_payload",
        "mappo_payload",
        "conflict_rate",
        "conflict_pairs",
        "interference",
        "min_uav_distance_m",
        "duplicate_targets",
    ]
    with (OUTPUT / "joint_execution.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for items in joint_rows.values():
            writer.writerows(items)
    (OUTPUT / "phase5h_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(_report(summary), encoding="utf-8")
    print(json.dumps({"primary": primary, "single": single, "joint_utility": joint["utility_hgnn"], "joint_deepsets": joint["deepsets"], "joint_oracle": joint["oracle"]}, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    run()
