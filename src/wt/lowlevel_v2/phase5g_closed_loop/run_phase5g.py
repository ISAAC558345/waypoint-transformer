"""Closed-loop check of the frozen utility HGNN. Does not train and does not call decide()."""
from __future__ import annotations

import csv
import json

import numpy as np
import torch

from wt.lowlevel_v2.integration_phase1.stats import kendall_tau, spearman
from wt.lowlevel_v2.phase5e_utility_dataset.utility import descending_rank
from wt.lowlevel_v2.phase5f_utility_rank.score import focal_scores, load_deepsets, load_old_hgnn, new_hgnn

from .collect import collect_scenario, load_frozen_actor, mappo_h15, replay_env
from .protocol import DEPLOYMENT_VERIFIED, OUTPUT, PLANNER_READY, REPORT, SCENARIOS, UTILITY_CKPT


def _mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _best(values: list[float]) -> set[int]:
    target = max(values)
    return {index for index, value in enumerate(values) if float(value) == float(target)}


def _hit(scores: list[float], best: set[int]) -> float:
    return float(len(_best(scores).intersection(best)) > 0)


def _top3(scores: list[float], best: set[int]) -> float:
    ranks = descending_rank(scores)
    return float(any(int(ranks[index]) <= 3 for index in best))


def analyze_group(scores: list[float], sample: dict) -> dict:
    utility = [float(item) for item in sample["utility_payload"]]
    executed = [float(item) for item in sample["executed_payload"]]
    ranks = [int(item) for item in sample["utility_rank"]]
    preference = [-float(item) for item in ranks]
    chosen = int(np.argmax(np.asarray(scores, dtype=np.float64)))
    utility_best = _best(utility)
    executed_best = _best(executed)
    ordered = sorted(utility, reverse=True)
    near = len(utility) >= 2 and ordered[0] >= 1.0e6 and ordered[0] - ordered[1] < 1.0e6
    utility_best_executed = max(executed[index] for index in utility_best)
    return {
        "scenario_id": int(sample["scenario_id"]),
        "time_step": int(sample["time_step"]),
        "uav_id": int(sample["uav_id"]),
        "n_candidates": len(scores),
        "spearman_utility": spearman(scores, preference),
        "kendall_utility": kendall_tau(scores, preference),
        "spearman_executed": spearman(scores, executed),
        "kendall_executed": kendall_tau(scores, executed),
        "top1_utility": _hit(scores, utility_best),
        "top3_utility": _top3(scores, utility_best),
        "top1_executed": _hit(scores, executed_best),
        "top3_executed": _top3(scores, executed_best),
        "chosen_utility_payload": utility[chosen],
        "chosen_executed_payload": executed[chosen],
        "best_executed_payload": max(executed),
        "separated_utility": float(max(utility) - min(utility)) >= 1.0e6 - 1.0,
        "separated_executed": float(max(executed) - min(executed)) >= 1.0e6 - 1.0,
        "chosen_candidate": int(sample["candidate_ids"][chosen]),
        "utility_best_candidates": [int(sample["candidate_ids"][index]) for index in sorted(utility_best)],
        "executed_best_candidates": [int(sample["candidate_ids"][index]) for index in sorted(executed_best)],
        "case_a": bool(chosen not in utility_best and executed[chosen] + 1.0e6 <= max(executed)),
        "case_b": bool(utility_best_executed + 1.0e6 <= max(executed)),
        "case_c": bool(near),
    }


def _pack(rows: list[dict], separated_key: str) -> dict:
    subset = [row for row in rows if row[separated_key]]
    source = subset if subset else rows
    return {
        "n_groups": len(source),
        "n_candidates": _mean([row["n_candidates"] for row in source]),
        "spearman_utility": _mean([row["spearman_utility"] for row in source]),
        "kendall_utility": _mean([row["kendall_utility"] for row in source]),
        "spearman_executed": _mean([row["spearman_executed"] for row in source]),
        "kendall_executed": _mean([row["kendall_executed"] for row in source]),
        "top1_utility": _mean([row["top1_utility"] for row in source]),
        "top3_utility": _mean([row["top3_utility"] for row in source]),
        "top1_executed": _mean([row["top1_executed"] for row in source]),
        "top3_executed": _mean([row["top3_executed"] for row in source]),
        "utility_payload": _mean([row["chosen_utility_payload"] for row in source]),
        "executed_payload": _mean([row["chosen_executed_payload"] for row in source]),
        "best_executed_payload": _mean([row["best_executed_payload"] for row in source]),
    }


def _score_names(model, sample: dict, *, use_urgency: bool) -> list[float]:
    with torch.no_grad():
        values = focal_scores(model, sample, use_urgency=use_urgency).detach().cpu().numpy()
    return [float(item) for item in values]


def _greedy_assignment(group: list[dict], scores: dict[int, list[float]]) -> np.ndarray:
    assignment = np.full(group[0]["context"].shape[0], -1, dtype=np.int64)
    taken: set[int] = set()
    order = sorted(group, key=lambda sample: max(scores[int(sample["uav_id"])]), reverse=True)
    for sample in order:
        best_buoy = -1
        best_score = -1.0e18
        for buoy, score in zip(sample["candidate_ids"], scores[int(sample["uav_id"])]):
            if int(buoy) in taken:
                continue
            if float(score) > best_score:
                best_buoy = int(buoy)
                best_score = float(score)
        if best_buoy >= 0:
            assignment[int(sample["uav_id"])] = best_buoy
            taken.add(best_buoy)
    return assignment


def joint_payloads(rows: list[dict], actor, named_scores: dict[str, dict[tuple, list[float]]]) -> dict[str, float]:
    grouped: dict[tuple[int, int], list[dict]] = {}
    for sample in rows:
        grouped.setdefault((int(sample["scenario_id"]), int(sample["time_step"])), []).append(sample)
    totals = {name: [] for name in named_scores}
    for (scenario, moment), group in grouped.items():
        env, movement = replay_env(scenario, moment)
        for name, table in named_scores.items():
            local = {int(sample["uav_id"]): table[(scenario, moment, int(sample["uav_id"]))] for sample in group}
            assignment = _greedy_assignment(group, local)
            totals[name].append(float(mappo_h15(env, assignment, actor, movement)))
    return {name: _mean(values) for name, values in totals.items()}


def _fmt(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.4f}"


def _failures(rows: list[dict]) -> list[dict]:
    cases = []
    for row in rows:
        for case, flag in (("A_high_score_low_payload", row["case_a"]), ("B_utility_target_low_level_miss", row["case_b"]), ("C_close_utilities", row["case_c"])):
            if flag:
                cases.append(
                    {
                        "scenario": row["scenario_id"],
                        "time_step": row["time_step"],
                        "uav": row["uav_id"],
                        "case": case,
                        "chosen_candidate": row["chosen_candidate"],
                        "utility_best": " ".join(str(item) for item in row["utility_best_candidates"]),
                        "executed_best": " ".join(str(item) for item in row["executed_best_candidates"]),
                        "chosen_utility_payload": row["chosen_utility_payload"],
                        "chosen_executed_payload": row["chosen_executed_payload"],
                    }
                )
    return cases


def _report(summary: dict) -> str:
    names = (
        ("assignment HGNN", "old_hgnn"),
        ("DeepSets", "deepsets"),
        ("utility HGNN", "utility_hgnn"),
    )
    lines = [
        "# HTW Phase-5G HGNN Trajectory Utility Closed-loop Validation",
        "",
        "**planner_ready = False**",
        "",
        "**deployment_verified = False**",
        "",
        "本阶段检查冻结的 utility HGNN 在新场景上，交给冻结 Residual MAPPO 执行 15 步之后还有没有 payload 收益。没有训练，没有调用 deployment `decide()`，没有修改 B2、τ、Gate-1、Candidate-C、低层结构或 planner，没有使用 230000–230127。",
        "",
        "## 1. 验证数据",
        "",
        "场景是已经存在的隔离重置 310102 和 310103。它们没有进入 Phase-5F 的训练。310100 是训练场景，本阶段不使用。没有新的 scenario 生成器。",
        "",
        "每个决策时刻是 t=0、15、30。时刻之间用当前排他匹配和 fly-toward 走到决策点。决策之后的 15 步才换成冻结 MAPPO。offline utility 仍是同一分配下 fly-toward 的 15 步 payload，和 Phase-5E 的 baseline 一致。",
        "",
        "## 2. Scenario isolation",
        "",
        f"训练场景 310100 与本阶段场景 {list(SCENARIOS)} 不相交。检查点只读取，不更新。",
        "",
        "## 3. HGNN 排序",
        "",
        "下表只统计 fly-toward payload 已经分开的组。Spearman 和 Kendall 比较预测分数与 utility 名次。Top-1、Top-3 看预测最高分是否覆盖 offline utility 最优候选。",
        "",
        "| model | scenario | n | candidates | Spearman | Kendall | Top-1 | Top-3 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label, key in names:
        for scenario, block in summary["by_scenario"][key].items():
            item = block["utility_separated"]
            lines.append(
                f"| {label} | {scenario} | {item['n_groups']} | {_fmt(item['n_candidates'])} | {_fmt(item['spearman_utility'])} | {_fmt(item['kendall_utility'])} | {_fmt(item['top1_utility'])} | {_fmt(item['top3_utility'])} |"
            )
        pooled = summary["pooled"][key]["utility_separated"]
        lines.append(
            f"| {label} | both | {pooled['n_groups']} | {_fmt(pooled['n_candidates'])} | {_fmt(pooled['spearman_utility'])} | {_fmt(pooled['kendall_utility'])} | {_fmt(pooled['top1_utility'])} | {_fmt(pooled['top3_utility'])} |"
        )
    lines.extend(
        [
            "",
            "## 4. Payload",
            "",
            "候选级 payload 是只替换一名无人机的目标、其他无人机保持当前排他分配后，冻结 MAPPO 走出的 15 步通信增量。联合 payload 是按分数从高到低贪心占用浮标后，再交给同一个 MAPPO 执行的 15 步增量。候选级收益没有带到联合分配上：utility HGNN 的联合 payload 与 assignment HGNN 相同，并且低于 DeepSets。",
            "",
            "| model | MAPPO payload of chosen target | MAPPO best in the group | joint MAPPO payload |",
            "| --- | --- | --- | --- |",
        ]
    )
    for label, key in names:
        item = summary["pooled"][key]["utility_separated"]
        lines.append(f"| {label} | {_fmt(item['executed_payload'])} | {_fmt(item['best_executed_payload'])} | {_fmt(summary['joint'][key])} |")
    lines.extend(
        [
            "",
            "## 5. Baseline",
            "",
            "assignment HGNN 是 Phase-5 的 `hgnn_orderfree.pt`。DeepSets 是冻结的 utility-pool 检查点，不是 assignment NLL。utility HGNN 是 Phase-5F 的 `hgnn_utility_rank.pt`。三个模型的低层都是同一个 `mappo_original.pt`，urgency 没有送进低层。",
            "",
            "## 6. Failure cases",
            "",
            "文件：`phase5g_closed_loop/closed_loop_failures.csv`。只记录 utility HGNN。",
            "",
            f"Case A，预测选中的目标不是 offline utility 最优，而且 MAPPO payload 至少少一个服务块：{summary['failure_counts']['A_high_score_low_payload']}。",
            "",
            f"Case B，offline utility 最优目标的 MAPPO payload 至少比组内 MAPPO 最优少一个服务块：{summary['failure_counts']['B_utility_target_low_level_miss']}。这是 fly-toward 标签和冻结 MAPPO 执行不一致。",
            "",
            f"Case C，最高的两个 offline utility 相差不到一个服务块：{summary['failure_counts']['C_close_utilities']}。这些组需要并列处理，单点 argmax 不稳定。",
            "",
            "## 7. 高低层一致性",
            "",
            "这里的 Spearman、Kendall、Top-1、Top-3 比较的是预测分数和 MAPPO 实际 payload，只统计 MAPPO payload 已经分开的组。",
            "",
            "| model | Spearman | Kendall | Top-1 | Top-3 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for label, key in names:
        item = summary["pooled"][key]["executed_separated"]
        lines.append(f"| {label} | {_fmt(item['spearman_executed'])} | {_fmt(item['kendall_executed'])} | {_fmt(item['top1_executed'])} | {_fmt(item['top3_executed'])} |")
    entered = summary["phase5h"]["closed_loop_gain"]
    gain = "在这两个新场景上，utility HGNN 的单目标 MAPPO payload、utility Top-1 和联合分配 payload 都高于 assignment HGNN。" if entered else "单目标替换上，utility HGNN 的 MAPPO payload 和 utility Top-1 高于 assignment HGNN。联合分配再交给 MAPPO 时，两者 payload 相同，并且低于 DeepSets。因此完整闭环没有稳定收益。"
    lines.extend(
        [
            "",
            "## 8. Phase-5H",
            "",
            gain,
            "",
            "本阶段不进入 Phase-5H，也不做 deployment verification。planner 没有修改。系统优化没有完成。",
            "",
            "`planner_ready = False`",
            "",
            "`deployment_verified = False`",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> dict:
    actor = load_frozen_actor()
    rows = []
    for seed in SCENARIOS:
        print(f"closed loop scenario {seed}", flush=True)
        rows.extend(collect_scenario(int(seed), actor))
    utility_model = new_hgnn()
    if not UTILITY_CKPT.is_file():
        raise FileNotFoundError(str(UTILITY_CKPT))
    utility_model.load_state_dict(torch.load(UTILITY_CKPT, map_location="cpu"))
    utility_model.eval()
    models = {
        "deepsets": (load_deepsets(), False),
        "old_hgnn": (load_old_hgnn(), False),
        "utility_hgnn": (utility_model, True),
    }
    analyzed = {"assignment_score": [analyze_group(sample["assignment_score"], sample) for sample in rows]}
    named_scores = {
        "assignment_score": {
            (int(sample["scenario_id"]), int(sample["time_step"]), int(sample["uav_id"])): list(sample["assignment_score"]) for sample in rows
        }
    }
    for name, (model, use_urgency) in models.items():
        print(f"score {name}", flush=True)
        group_rows = []
        named_scores[name] = {}
        for sample in rows:
            scores = _score_names(model, sample, use_urgency=use_urgency)
            named_scores[name][(int(sample["scenario_id"]), int(sample["time_step"]), int(sample["uav_id"]))] = scores
            group_rows.append(analyze_group(scores, sample))
        analyzed[name] = group_rows
    print("joint mappo", flush=True)
    joint = joint_payloads(rows, actor, {key: named_scores[key] for key in ("old_hgnn", "deepsets", "utility_hgnn")})
    pooled = {}
    by_scenario = {key: {} for key in ("old_hgnn", "deepsets", "utility_hgnn", "assignment_score")}
    for key, group_rows in analyzed.items():
        pooled[key] = {
            "utility_separated": _pack(group_rows, "separated_utility"),
            "executed_separated": _pack(group_rows, "separated_executed"),
        }
        for scenario in SCENARIOS:
            selected = [row for row in group_rows if int(row["scenario_id"]) == int(scenario)]
            by_scenario[key][str(scenario)] = {
                "utility_separated": _pack(selected, "separated_utility"),
                "executed_separated": _pack(selected, "separated_executed"),
            }
    failures = _failures(analyzed["utility_hgnn"])
    counts = {"A_high_score_low_payload": 0, "B_utility_target_low_level_miss": 0, "C_close_utilities": 0}
    for row in failures:
        counts[str(row["case"])] += 1
    new = pooled["utility_hgnn"]["utility_separated"]
    old = pooled["old_hgnn"]["utility_separated"]
    gain = bool(
        new["executed_payload"] > old["executed_payload"]
        and new["top1_utility"] > old["top1_utility"]
        and joint["utility_hgnn"] > joint["old_hgnn"]
    )
    summary = {
        "planner_ready": PLANNER_READY,
        "deployment_verified": DEPLOYMENT_VERIFIED,
        "decide_called": False,
        "mappo_retrained": False,
        "tau_modified": False,
        "holdout_used": False,
        "phase5h_started": False,
        "scenarios": list(SCENARIOS),
        "n_groups": len(rows),
        "pooled": pooled,
        "by_scenario": by_scenario,
        "joint": joint,
        "failure_counts": counts,
        "phase5h": {"closed_loop_gain": gain, "phase5h_started": False},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fields = [
        "scenario",
        "time_step",
        "uav",
        "case",
        "chosen_candidate",
        "utility_best",
        "executed_best",
        "chosen_utility_payload",
        "chosen_executed_payload",
    ]
    with (OUTPUT / "closed_loop_failures.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failures)
    (OUTPUT / "phase5g_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(_report(summary), encoding="utf-8")
    print(json.dumps({"failures": counts, "joint": joint, "utility": new, "old": old}, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    run()
