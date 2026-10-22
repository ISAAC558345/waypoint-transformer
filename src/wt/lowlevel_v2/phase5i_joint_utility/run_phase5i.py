"""Joint-assignment audit. Frozen models, no training."""
from __future__ import annotations

import csv
import json

import numpy as np
import torch

from wt.htw_phase943.evaluation import decode_assignment
from wt.htw_phase945.utility_rollout import shadow_h15_payload
from wt.lowlevel_v2.integration_phase1.stats import kendall_tau, spearman
from wt.lowlevel_v2.phase5f_utility_rank.score import focal_scores, load_deepsets, new_hgnn
from wt.lowlevel_v2.phase5g_closed_loop.collect import collect_scenario, load_frozen_actor, mappo_h15, replay_env
from wt.lowlevel_v2.phase5g_closed_loop.protocol import UTILITY_CKPT
from wt.lowlevel_v2.phase5g_closed_loop.run_phase5g import _greedy_assignment

from .protocol import DEPLOYMENT_VERIFIED, OUTPUT, PLANNER_READY, REPORT, SCENARIOS, SEARCH_LIMIT
from .search import largest_complete_pool, top_options


def _mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _fmt(value: float) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value):.4f}"


def _key(assignment: np.ndarray) -> tuple[int, ...]:
    return tuple(int(item) for item in assignment)


def _marginal(sample: dict, buoy: int) -> float:
    for candidate, value in zip(sample["candidate_ids"], sample["utility_payload"]):
        if int(candidate) == int(buoy):
            return float(value)
    return 0.0


def _score_of(sample: dict, scores: list[float], buoy: int) -> float:
    for candidate, value in zip(sample["candidate_ids"], scores):
        if int(candidate) == int(buoy):
            return float(value)
    return -1.0e9


def _sum_marginal(group: list[dict], assignment: np.ndarray) -> float:
    total = 0.0
    by_uav = {int(sample["uav_id"]): sample for sample in group}
    for uav, buoy in enumerate(assignment):
        if int(buoy) < 0:
            continue
        total += _marginal(by_uav[int(uav)], int(buoy))
    return float(total)


def _sum_score(group: list[dict], assignment: np.ndarray, table: dict[int, list[float]]) -> float:
    total = 0.0
    by_uav = {int(sample["uav_id"]): sample for sample in group}
    for uav, buoy in enumerate(assignment):
        if int(buoy) < 0:
            continue
        total += _score_of(by_uav[int(uav)], table[int(uav)], int(buoy))
    return float(total)


def _raw_argmax(group: list[dict], table: dict[int, list[float]]) -> np.ndarray:
    assignment = np.full(len(group), -1, dtype=np.int64)
    by_uav = {int(sample["uav_id"]): sample for sample in group}
    for uav, scores in table.items():
        sample = by_uav[int(uav)]
        assignment[int(uav)] = int(sample["candidate_ids"][int(np.argmax(np.asarray(scores, dtype=np.float64)))])
    return assignment


def _collision_matrix(assignment: np.ndarray) -> np.ndarray:
    size = int(assignment.shape[0])
    matrix = np.zeros((size, size), dtype=np.float64)
    for left in range(size):
        for right in range(left + 1, size):
            if int(assignment[left]) >= 0 and int(assignment[left]) == int(assignment[right]):
                matrix[left, right] = 1.0
                matrix[right, left] = 1.0
    return matrix


def _coverage(queue: np.ndarray, assignment: np.ndarray) -> float:
    hot = [index for index, value in enumerate(queue) if float(value) > 0.0]
    if not hot:
        return 1.0
    assigned = {int(item) for item in assignment if int(item) >= 0}
    return float(sum(int(index in assigned) for index in hot) / len(hot))


def _report(summary: dict) -> str:
    search = summary["search"]
    labels = (
        ("sum of individual utility", "sum_individual"),
        ("utility HGNN greedy", "utility_greedy"),
        ("utility HGNN decoder", "utility_decoder"),
        ("DeepSets greedy", "deepsets_greedy"),
        ("raw individual argmax", "raw_argmax"),
        ("best joint fly-toward", "best_joint"),
    )
    table = [
        "| assignment | joint fly payload | joint MAPPO payload |",
        "| --- | --- | --- |",
    ]
    for label, key in labels:
        item = summary["executed"][key]
        table.append(f"| {label} | {_fmt(item['fly'])} | {_fmt(item['mappo'])} |")
    enter = summary["enter_joint_training"]
    enter_text = "单个目标的效用最大组合和联合 fly-toward 最大组合不是同一个。联合效用标签有证据。" if enter else "在这个搜索池里，单个目标效用最大的组合已经是联合 fly-toward 最大组合。现在没有新的联合训练证据。"
    decoder_text = summary["decoder_text"]
    head_text = summary["head_text"]
    return "\n".join(
        [
            "# HTW Phase-5I Joint Assignment Utility Audit",
            "",
            "**planner_ready = False**",
            "",
            "**deployment_verified = False**",
            "",
            "本阶段只比较单个目标效用和完整分配效用。没有改模型，没有训练 HGNN，没有改 Residual MAPPO，没有修改 B2、τ、Gate-1、Candidate-C、planner 或 deployment `decide()`，没有使用 230000–230127。",
            "",
            "## 1. Individual utility 的局限",
            "",
            "Phase-5F 的标签是固定其他无人机之后，一名无人机换目标的 15 步 fly-toward payload。Phase-5G 和 Phase-5H 把六架飞机的最高分拼成一条分配之后，团队 payload 下降。",
            "",
            f"搜索池是每架无人机边际 fly-toward 效用的前 {summary['pool_width']} 个目标，再做排他组合。完整的 12 选 6 有 665280 条，不能逐条做 15 步执行。本阶段把池子收在可穷尽的宽度内，最宽而且不超过 {summary['search_limit']} 条。六个决策时刻平均穷尽 { _fmt(search['mean_assignments']) } 条。",
            "",
            f"池内 sum of individual utility 和联合 fly-toward payload 的 Spearman = {_fmt(search['spearman_sum_vs_joint_fly'])}，Kendall = {_fmt(search['kendall_sum_vs_joint_fly'])}。HGNN 分数之和与联合 fly-toward 的 Spearman = {_fmt(search['spearman_score_vs_joint_fly'])}。",
            "",
            f"单个效用最高的组合就是联合 fly-toward 最高组合的比例是 {_fmt(search['top1_same_assignment'])}。",
            "",
            "## 2. Joint mismatch",
            "",
            "oracle joint payload 是这个搜索池里联合 fly-toward 的最大 payload。MAPPO 只在少数代表分配上执行，执行器仍是冻结的 `mappo_original.pt`。",
            "",
            *table,
            "",
            f"sum-individual 最大分配的 MAPPO payload = {_fmt(summary['executed']['sum_individual']['mappo'])}。联合 fly-toward 最大分配的 MAPPO payload = {_fmt(summary['executed']['best_joint']['mappo'])}。两者分配相同的比例是 {_fmt(search['top1_same_assignment'])}。",
            "",
            "## 3. HGNN decoder",
            "",
            decoder_text,
            "",
            "## 4. Global assignment head",
            "",
            head_text,
            "",
            "## 5. Assignment coupling",
            "",
            "冲突矩阵是六架无人机在取各自分数最高目标时，两两选中同一浮标的比例。贪心排他会拆开这个冲突，所以 Phase-5H 看到的重复目标数是 0；冲突发生在组合之前。",
            "",
            summary["matrix_text"],
            "",
            f"原始最高分分配的重复目标数平均 {_fmt(summary['coupling']['raw_duplicate_targets'])}。贪心后的重复目标数平均 {_fmt(summary['coupling']['greedy_duplicate_targets'])}。正队列浮标被贪心分配覆盖的比例是 {_fmt(summary['coupling']['greedy_coverage'])}。原始最高分分配的联合 fly-toward payload 是 {_fmt(summary['executed']['raw_argmax']['fly'])}。",
            "",
            "## 6. 新的监督接口",
            "",
            "不改网络。三个标签只作接口：",
            "",
            "- A assignment similarity：当前排他匹配，Phase-5 已经证明它和 payload 排序不一致。",
            "- B individual trajectory utility：固定其他无人机后的单目标 15 步 payload 排序。它能提高单目标替换，不能代表六架同时换目标。",
            "- C joint trajectory utility：`J(A) = sum_i payload_gain(i, A_i) - lambda * conflict_loss(A)`。",
            "",
            "lambda 不拟合，接口取 1。conflict_loss 定义为边际 payload 之和减去这条分配真正的联合 fly-toward payload，小于 0 时记 0。这样 J(A) 等于联合 payload。服务冲突率在贪心之后是 0，所以不能再用抢同一浮标的计数充当这个损失。",
            "",
            enter_text,
            "",
            "本阶段不开始 joint utility HGNN 训练，也不进入 deployment verification。",
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
        print(f"joint audit scenario {seed}", flush=True)
        rows.extend(collect_scenario(int(seed), actor))
    model = new_hgnn()
    if not UTILITY_CKPT.is_file():
        raise FileNotFoundError(str(UTILITY_CKPT))
    model.load_state_dict(torch.load(UTILITY_CKPT, map_location="cpu"))
    model.eval()
    deepsets = load_deepsets()
    grouped: dict[tuple[int, int], list[dict]] = {}
    utility_table: dict[tuple[int, int, int], list[float]] = {}
    deepsets_table: dict[tuple[int, int, int], list[float]] = {}
    for sample in rows:
        grouped.setdefault((int(sample["scenario_id"]), int(sample["time_step"])), []).append(sample)
        key = (int(sample["scenario_id"]), int(sample["time_step"]), int(sample["uav_id"]))
        with torch.no_grad():
            utility_table[key] = [float(item) for item in focal_scores(model, sample, use_urgency=True).detach().cpu().numpy()]
            deepsets_table[key] = [float(item) for item in focal_scores(deepsets, sample, use_urgency=False).detach().cpu().numpy()]
    sum_values = []
    joint_values = []
    score_values = []
    top1_hits = []
    executed = {name: {"fly": [], "mappo": []} for name in ("sum_individual", "utility_greedy", "utility_decoder", "deepsets_greedy", "raw_argmax", "best_joint")}
    matrices = []
    coupling = {"raw_duplicate_targets": [], "greedy_duplicate_targets": [], "greedy_coverage": []}
    widths = []
    counts = []
    state_rows = []
    for (scenario, moment), group in grouped.items():
        group = sorted(group, key=lambda sample: int(sample["uav_id"]))
        local_utility = {int(sample["uav_id"]): utility_table[(scenario, moment, int(sample["uav_id"]))] for sample in group}
        local_deepsets = {int(sample["uav_id"]): deepsets_table[(scenario, moment, int(sample["uav_id"]))] for sample in group}
        pools = {}
        for width in (1, 2, 3):
            pools[width] = [top_options(sample["candidate_ids"], sample["utility_payload"], width) for sample in group]
        width, assignments = largest_complete_pool(pools, SEARCH_LIMIT)
        widths.append(width)
        known = {_key(item): item for item in assignments}
        named = {
            "utility_greedy": _greedy_assignment(group, local_utility),
            "deepsets_greedy": _greedy_assignment(group, local_deepsets),
            "raw_argmax": _raw_argmax(group, local_utility),
        }
        env, movement = replay_env(scenario, moment)
        named["utility_decoder"] = np.asarray(decode_assignment(model, group[0]["urgency_batch"], env), dtype=np.int64)
        for assignment in named.values():
            known.setdefault(_key(assignment), np.asarray(assignment, dtype=np.int64))
        evaluated = []
        for assignment in known.values():
            evaluated.append(
                {
                    "assignment": assignment,
                    "sum_individual": _sum_marginal(group, assignment),
                    "score_sum": _sum_score(group, assignment, local_utility),
                    "joint_fly": float(shadow_h15_payload(env, assignment)),
                }
            )
        counts.append(len(evaluated))
        for item in evaluated:
            sum_values.append(item["sum_individual"])
            joint_values.append(item["joint_fly"])
            score_values.append(item["score_sum"])
        best_sum = max(evaluated, key=lambda item: (item["sum_individual"], -int(np.sum(np.abs(item["assignment"])))))
        best_joint = max(evaluated, key=lambda item: (item["joint_fly"], item["sum_individual"]))
        top1_hits.append(float(_key(best_sum["assignment"]) == _key(best_joint["assignment"])))
        chosen = {
            "sum_individual": best_sum["assignment"],
            "best_joint": best_joint["assignment"],
            **named,
        }
        cached = {_key(item["assignment"]): item["joint_fly"] for item in evaluated}
        for name, assignment in chosen.items():
            fly = cached.get(_key(assignment), float(shadow_h15_payload(env, assignment)))
            mappo = float(mappo_h15(env, assignment, actor, movement))
            executed[name]["fly"].append(fly)
            executed[name]["mappo"].append(mappo)
        raw = named["raw_argmax"]
        greedy = named["utility_greedy"]
        matrices.append(_collision_matrix(raw))
        coupling["raw_duplicate_targets"].append(float(len([item for item in raw if int(item) >= 0]) - len({int(item) for item in raw if int(item) >= 0})))
        coupling["greedy_duplicate_targets"].append(float(len([item for item in greedy if int(item) >= 0]) - len({int(item) for item in greedy if int(item) >= 0})))
        coupling["greedy_coverage"].append(_coverage(np.asarray(env.queue, dtype=np.float64), greedy))
        state_rows.append(
            {
                "scenario": scenario,
                "time_step": moment,
                "pool_width": width,
                "n_assignments": len(evaluated),
                "top1_same": top1_hits[-1],
                "sum_fly": best_sum["joint_fly"],
                "best_joint_fly": best_joint["joint_fly"],
                "utility_greedy_fly": cached.get(_key(greedy), float("nan")),
            }
        )
        print(f"state {scenario} t={moment} width={width} assignments={len(evaluated)} top1={top1_hits[-1]}", flush=True)
    matrix = np.mean(np.stack(matrices, axis=0), axis=0) if matrices else np.zeros((6, 6))
    matrix_lines = ["| UAV | " + " | ".join(str(index) for index in range(matrix.shape[0])) + " |", "| --- | " + " | ".join("---" for _ in range(matrix.shape[0])) + " |"]
    for index, row in enumerate(matrix):
        matrix_lines.append(f"| {index} | " + " | ".join(f"{float(value):.2f}" for value in row) + " |")
    executed_mean = {name: {"fly": _mean(values["fly"]), "mappo": _mean(values["mappo"])} for name, values in executed.items()}
    decoder_fly = executed_mean["utility_decoder"]["fly"]
    greedy_fly = executed_mean["utility_greedy"]["fly"]
    best_fly = executed_mean["best_joint"]["fly"]
    if decoder_fly + 1.0 < best_fly and abs(decoder_fly - greedy_fly) <= 1.0:
        decoder_text = "现有 OrderFreeSetDecoder 的联合解码和外部贪心一样，都低于池内最好的联合 fly-toward。缺的是联合标签，不是先换一个 decoder。"
        head_text = "现有 decoder 已经能输出一条完整分配。本阶段不增加 global assignment head。"
    elif decoder_fly > greedy_fly + 1.0:
        decoder_text = "现有 decoder 的联合解码高于外部边际贪心。Phase-5G 的失败来自贪心拼接，不是 decoder 缺少联合输出。"
        head_text = "不增加 global assignment head。先使用 decoder 自己的联合解码，而不是再加一个头。"
    else:
        decoder_text = "现有 decoder 的联合 fly-toward 没有明显高于外部贪心，也没有达到池内最好的联合分配。本阶段不修改 decoder。"
        head_text = "不增加 global assignment head。联合效用还没有作为标签训练过，先改标签，不先加结构。"
    enter = bool(_mean(top1_hits) < 1.0 - 1.0e-9)
    summary = {
        "planner_ready": PLANNER_READY,
        "deployment_verified": DEPLOYMENT_VERIFIED,
        "decide_called": False,
        "hgnn_retrained": False,
        "mappo_modified": False,
        "joint_training_started": False,
        "holdout_used": False,
        "scenarios": list(SCENARIOS),
        "pool_width": int(round(_mean([float(item) for item in widths]))),
        "search_limit": SEARCH_LIMIT,
        "search": {
            "mean_assignments": _mean([float(item) for item in counts]),
            "spearman_sum_vs_joint_fly": spearman(sum_values, joint_values),
            "kendall_sum_vs_joint_fly": kendall_tau(sum_values, joint_values),
            "spearman_score_vs_joint_fly": spearman(score_values, joint_values),
            "top1_same_assignment": _mean(top1_hits),
        },
        "executed": executed_mean,
        "coupling": {key: _mean(values) for key, values in coupling.items()},
        "conflict_matrix": matrix.tolist(),
        "matrix_text": "\n".join(matrix_lines),
        "decoder_text": decoder_text,
        "head_text": head_text,
        "enter_joint_training": enter,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "assignment_search.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(state_rows[0]))
        writer.writeheader()
        writer.writerows(state_rows)
    with (OUTPUT / "conflict_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["uav", *[str(index) for index in range(matrix.shape[0])]])
        for index, row in enumerate(matrix):
            writer.writerow([index, *[f"{float(value):.6f}" for value in row]])
    (OUTPUT / "phase5i_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(_report(summary), encoding="utf-8")
    print(json.dumps({"width": summary["pool_width"], "search": summary["search"], "executed": executed_mean, "coupling": summary["coupling"]}, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    run()
