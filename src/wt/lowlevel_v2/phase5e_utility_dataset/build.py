"""Offline trajectory-aware utility dataset. Does not train and does not call decide()."""
from __future__ import annotations

import csv
import json

import numpy as np

from wt.htw_phase9.const import HOLD
from wt.htw_phase945.utility_rollout import shadow_h15_payload
from wt.l3b.env import L3BEnv
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.integration_phase1.stats import kendall_tau, spearman
from wt.lowlevel_v2.phase5_hgnn_urgency.features import observable_exclusive_assignment

from .protocol import (
    DATASET_COLUMNS,
    DATASET_SEEDS,
    DECISION_TIMES,
    DEPLOYMENT_VERIFIED,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    OUTPUT,
    PLANNER_READY,
    REPORT,
    assert_dataset_seed,
)
from .state import assignment_similarity, candidate_features, current_observation, energy_cost
from .utility import BASELINE_CONFIG, descending_rank, utility_score


def _advance(env: L3BEnv, target: int, last_positions: np.ndarray | None) -> np.ndarray | None:
    while int(env.t) < int(target):
        positions = np.asarray(env.uav_positions, dtype=np.float64).copy()
        assignment = observable_exclusive_assignment(env)
        env.step(fly_toward(env, assignment), assignment)
        last_positions = positions
    if int(env.t) != int(target):
        raise RuntimeError(f"phase-5e stopped at t={env.t}, expected {target}")
    return last_positions


def _legal_buoys(base: np.ndarray, focal: int, num_buoys: int) -> list[int]:
    taken = {int(base[uav]) for uav in range(base.shape[0]) if uav != int(focal) and int(base[uav]) >= 0}
    return [buoy for buoy in range(int(num_buoys)) if buoy not in taken]


def collect_seed(seed: int) -> list[dict]:
    assert_dataset_seed(int(seed))
    if int(HOLD) != 15:
        raise RuntimeError("phase-5e expected the frozen hold of 15")
    env = L3BEnv()
    env.reset(seed=int(seed))
    last_positions = None
    rows: list[dict] = []
    for moment in DECISION_TIMES:
        last_positions = _advance(env, int(moment), last_positions)
        if energy_cost(env) != 0.0:
            raise RuntimeError("phase-5e energy cost must stay 0")
        obs = current_observation(env, last_positions)
        base = observable_exclusive_assignment(env)
        for focal in range(int(env.config.num_uavs)):
            legal = _legal_buoys(base, focal, int(env.config.num_buoys))
            group: list[dict] = []
            for buoy in legal:
                features = candidate_features(obs, focal, buoy)
                leaked = set(FEATURE_COLUMNS).intersection(LABEL_COLUMNS)
                if leaked:
                    raise RuntimeError(f"feature/label overlap: {sorted(leaked)}")
                trial = np.asarray(base, dtype=np.int64).copy()
                trial[focal] = int(buoy)
                payload = float(shadow_h15_payload(env, trial))
                score = utility_score(
                    payload,
                    features["urgency"],
                    features["travel_cost"],
                    features["interference_cost"],
                    features["energy_cost"],
                )
                group.append(
                    {
                        "scenario_id": int(seed),
                        "time_step": int(env.t),
                        "uav_id": int(focal),
                        "candidate_id": int(buoy),
                        "queue": features["queue"],
                        "channel": features["channel"],
                        "distance": features["distance"],
                        "ETA": features["ETA"],
                        "interference": features["interference"],
                        "urgency": features["urgency"],
                        "travel_cost": features["travel_cost"],
                        "assignment_score": assignment_similarity(features),
                        "payload_gain": payload,
                        "utility_score": score,
                        "energy_cost": features["energy_cost"],
                    }
                )
            utility_ranks = descending_rank([row["utility_score"] for row in group])
            assignment_ranks = descending_rank([row["assignment_score"] for row in group])
            for row, util_rank, assign_rank in zip(group, utility_ranks, assignment_ranks):
                row["utility_rank"] = int(util_rank)
                row["assignment_rank"] = int(assign_rank)
            rows.extend(group)
    return rows


def _groups(rows: list[dict]) -> list[list[dict]]:
    grouped: dict[tuple[int, int, int], list[dict]] = {}
    for row in rows:
        key = (int(row["scenario_id"]), int(row["time_step"]), int(row["uav_id"]))
        grouped.setdefault(key, []).append(row)
    return list(grouped.values())


def _max_set(group: list[dict], key: str) -> set[int]:
    values = [float(row[key]) for row in group]
    best = max(values)
    return {int(row["candidate_id"]) for row, value in zip(group, values) if value == best}


def _top3(group: list[dict], rank_key: str) -> float:
    payload_best = _max_set(group, "payload_gain")
    covered = [row for row in group if int(row["candidate_id"]) in payload_best]
    return float(any(int(row[rank_key]) <= 3 for row in covered))


def _discordant_pairs(score: list[float], payload: list[float]) -> int:
    left = np.asarray(score, dtype=np.float64)
    right = np.asarray(payload, dtype=np.float64)
    discordant = 0
    for index in range(left.size):
        prod = np.sign(left[index + 1 :] - left[index]) * np.sign(right[index + 1 :] - right[index])
        discordant += int(np.sum(prod < 0))
    return discordant


def _pair_metrics(group: list[dict], score_key: str, rank_key: str) -> dict[str, float]:
    if len(group) < 3:
        return {"spearman": float("nan"), "kendall": float("nan"), "top1": float("nan"), "top3": float("nan"), "discordant": float("nan")}
    payload = [float(row["payload_gain"]) for row in group]
    score = [float(row[score_key]) for row in group]
    agree = float(len(_max_set(group, score_key).intersection(_max_set(group, "payload_gain"))) > 0)
    return {
        "spearman": spearman(score, payload),
        "kendall": kendall_tau(score, payload),
        "top1": agree,
        "top3": _top3(group, rank_key),
        "discordant": float(_discordant_pairs(score, payload)),
    }


def _mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def ranking_summary(rows: list[dict]) -> dict:
    groups = [group for group in _groups(rows) if len(group) >= 3]
    separated = [group for group in groups if max(float(row["payload_gain"]) for row in group) - min(float(row["payload_gain"]) for row in group) >= 1.0e6 - 1.0]
    names = ("assignment", "utility")
    keys = (("assignment_score", "assignment_rank"), ("utility_score", "utility_rank"))
    out: dict[str, dict] = {}
    for label, (score_key, rank_key) in zip(names, keys):
        block = {}
        for subset_name, subset in (("all", groups), ("payload_separated", separated)):
            metrics = [_pair_metrics(group, score_key, rank_key) for group in subset]
            block[subset_name] = {
                "n_groups": len(subset),
                "spearman": _mean([item["spearman"] for item in metrics]),
                "kendall": _mean([item["kendall"] for item in metrics]),
                "top1": _mean([item["top1"] for item in metrics]),
                "top3": _mean([item["top3"] for item in metrics]),
                "discordant_pairs": int(np.nansum([item["discordant"] for item in metrics])) if metrics else 0,
            }
        out[label] = block
    return out


def failure_rows(rows: list[dict]) -> list[dict]:
    failures: list[dict] = []
    for group in _groups(rows):
        payload_best = _max_set(group, "payload_gain")
        assignment_best = _max_set(group, "assignment_score")
        utility_best = _max_set(group, "utility_score")
        by_id = {int(row["candidate_id"]): row for row in group}
        if not assignment_best.intersection(payload_best):
            for candidate in sorted(assignment_best):
                failures.append(_failure(by_id[candidate], "assignment_selection_error", payload_best))
        if not utility_best.intersection(payload_best):
            for candidate in sorted(utility_best):
                failures.append(_failure(by_id[candidate], "utility_selection_error", payload_best))
        for candidate in sorted(payload_best):
            row = by_id[candidate]
            if int(row["utility_rank"]) > 1:
                failures.append(_failure(row, "payload_high_utility_low", payload_best))
    return failures


def _failure(row: dict, case: str, payload_best: set[int]) -> dict:
    return {
        "scenario": int(row["scenario_id"]),
        "time_step": int(row["time_step"]),
        "uav": int(row["uav_id"]),
        "candidate": int(row["candidate_id"]),
        "case": case,
        "payload_gain": float(row["payload_gain"]),
        "utility_score": float(row["utility_score"]),
        "utility_rank": int(row["utility_rank"]),
        "assignment_score": float(row["assignment_score"]),
        "assignment_rank": int(row["assignment_rank"]),
        "payload_best_candidates": " ".join(str(item) for item in sorted(payload_best)),
    }


def _write_csv(path, rows: list[dict], columns: tuple[str, ...] | list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in columns})


def _fmt(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.4f}"


def _report(summary: dict) -> str:
    rank = summary["ranking"]
    assign_all = rank["assignment"]["all"]
    util_all = rank["utility"]["all"]
    assign_sep = rank["assignment"]["payload_separated"]
    util_sep = rank["utility"]["payload_separated"]
    counts = summary["failure_counts"]
    return "\n".join(
        [
            "# HTW Phase-5E Trajectory-Aware Utility Dataset Audit",
            "",
            "**planner_ready = False**",
            "",
            "**deployment_verified = False**",
            "",
            "本阶段只构造离线 utility 标签并比较两种排序。没有训练 HGNN，没有重训 Residual MAPPO，没有调用 `decide()`，没有修改 B2、τ、Gate-1、Candidate-C 或 planner，没有使用 230000–230127，也没有进入 Phase-5F。",
            "",
            "## 1. 当前 assignment supervision 的问题",
            "",
            "Phase-5 HGNN 的标签是当前时刻的排他匹配：`queue * 当前假设速率 / (distance + 1)`。这个分数不含未来轨迹。Phase-5C 在 t=15 上已经看到它和 15 步 payload 的 Spearman 为 -0.39，Top-1 相同比例为 0。",
            "",
            "冻结 DeepSets 检查点用的是另一套候选池上的 15 步 payload 软标签。本数据集不读取那个检查点，也不改它的 τ。这里要验证的是：候选级的 assignment similarity 是否该换成 15 步 trajectory utility 排序。",
            "",
            "## 2. Offline rollout",
            "",
            "场景只用已有重置 310100 和 310101。决策时刻是 t=0、15、30，对应冻结的 option hold。时刻之间用当前排他匹配加 fly-toward 推进，不调用 `decide()`，也不调用 MAPPO。",
            "",
            "在每个决策时刻，先记下当前状态，再固定其他无人机的当前排他 assignment。对一名无人机的每个未占用浮标，只替换它的目标，在环境副本上用 fly-toward 走 15 步，读取通信 payload 增量。副本轨迹不写回特征。",
            "",
            "输入只含当前量：UAV 位置、由上一步位移得到的速度（t=0 为 0）、当前是否已分配、被分配浮标的队列、当前速率；浮标位置、队列、容量、当前假设速率；UAV–UAV 干扰和 UAV–Buoy 的 distance 与 ETA。没有读取 `remaining_arrivals`。",
            "",
            "## 3. Utility 公式",
            "",
            "`U_k = α * payload_gain + β * urgency - γ * travel_cost - δ * interference_cost - η * energy_cost`",
            "",
            "系数没有拟合。baseline 写在 `phase5e_utility_dataset/utility_config.json`：α=1，β=0，γ=0，δ=0，η=0。因此这条 baseline 的 `utility_score` 等于 `payload_gain`。",
            "",
            "β 保持 0，因为 urgency 只有当前队列比例，没有可标定的第二套时钟。γ 保持 0，因为 ETA 定义为 distance / step_length，所以 distance/ETA 等于固定步长，不能区分候选。δ 保持 0，因为 15 步 payload 已经包含 SINR。η 保持 0，因为环境没有能耗状态；本数据集里 energy_cost 全部是 0。",
            "",
            f"实测 travel_cost 的标准差是 {_fmt(summary['travel_cost_std'])}，组内 interference 是否随候选变化的比例是 {_fmt(summary['interference_varies_within_group'])}。",
            "",
            "## 4. Dataset 字段",
            "",
            "文件：`phase5e_utility_dataset/trajectory_utility_dataset.csv`。",
            "",
            "标识：scenario_id、time_step、uav_id、candidate_id。",
            "",
            "当前特征：queue、channel、distance、ETA、interference、urgency。",
            "",
            "标签：payload_gain、utility_score、utility_rank。utility_rank 从 1 开始，1 是最高 utility，相同分数并列。",
            "",
            f"行数 = {summary['n_rows']}。决策组数 = {summary['n_groups']}。payload 被拉开至少一个服务块的组数 = {summary['n_separated']}。",
            "",
            "## 5. Feature 与 label 隔离",
            "",
            "特征在调用 15 步 shadow 之前从当前 `env.info()` 取出。payload_gain、utility_score、utility_rank 只出现在标签列。特征列与标签列的交集为空，检查结果是通过。",
            "",
            "完整未来位置和动作没有进入 CSV。15 步结果只允许出现在标签里。",
            "",
            "## 6. Ranking 指标",
            "",
            "两种分数都去对 payload_gain 排序。Top-1 用最高分集合是否相交，避免并列时 argmax 碰到第一个下标。Top-3 recovery 看 payload 最高的候选里是否有 utility 或 assignment 名次不差于 3。",
            "",
            "| ranking | subset | n | Spearman | Kendall | Top-1 | Top-3 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            f"| assignment similarity | 全部 | {assign_all['n_groups']} | {_fmt(assign_all['spearman'])} | {_fmt(assign_all['kendall'])} | {_fmt(assign_all['top1'])} | {_fmt(assign_all['top3'])} |",
            f"| trajectory utility | 全部 | {util_all['n_groups']} | {_fmt(util_all['spearman'])} | {_fmt(util_all['kendall'])} | {_fmt(util_all['top1'])} | {_fmt(util_all['top3'])} |",
            f"| assignment similarity | payload 已分开 | {assign_sep['n_groups']} | {_fmt(assign_sep['spearman'])} | {_fmt(assign_sep['kendall'])} | {_fmt(assign_sep['top1'])} | {_fmt(assign_sep['top3'])} |",
            f"| trajectory utility | payload 已分开 | {util_sep['n_groups']} | {_fmt(util_sep['spearman'])} | {_fmt(util_sep['kendall'])} | {_fmt(util_sep['top1'])} | {_fmt(util_sep['top3'])} |",
            "",
            "trajectory utility 在 baseline 下等于 payload_gain。Spearman、Top-1 和 Top-3 都是 1。Kendall 低于 1，是因为现有实现把并列候选对留在分母里。payload 已经分开的组里，utility 与 payload 的反向候选对是 "
            f"{util_sep['discordant_pairs']}，assignment similarity 的反向候选对是 {assign_sep['discordant_pairs']}。",
            "",
            "因此更接近 payload 最优目标的是 trajectory utility ranking。assignment similarity 在这些组上的 Top-1 是 0；payload 最高候选仍经常落在它的前 3 名里，所以它作为短名单还留有重叠，但不能当作最优目标。",
            "",
            "## 7. Failure case",
            "",
            "文件：`phase5e_utility_dataset/utility_failure_cases.csv`。",
            "",
            f"assignment 选错 = {counts['assignment_selection_error']}。utility 选错 = {counts['utility_selection_error']}。payload 最高但 utility 名次低于 1 = {counts['payload_high_utility_low']}。",
            "",
            "assignment 选错表示当前匹配分数的最高候选和 15 步 payload 的最高候选不是同一个。utility 选错和 payload-utility 倒挂在这条 baseline 上是 0，因为 utility 就是 payload，而且没有再减 travel、interference 或 energy。若以后把 β、γ、δ 改成非零，这三列就是用来检查新系数会不会把 payload 最高的候选排低。当前不改系数。",
            "",
            "## 8. 能否进入 HGNN 重新训练",
            "",
            "标签验证支持把 high-level 监督从 assignment similarity 换成这份 trajectory utility ranking。特征里没有 15 步结果，低层也不是这次要动的部分。",
            "",
            "本阶段不满足开始训练的执行条件。HGNN 没有训练，Phase-5F 没有开始，系统优化没有完成。",
            "",
            "`planner_ready = False`",
            "",
            "`deployment_verified = False`",
            "",
        ]
    )


def run() -> dict:
    for seed in DATASET_SEEDS:
        assert_dataset_seed(int(seed))
    rows: list[dict] = []
    for seed in DATASET_SEEDS:
        print(f"utility dataset seed={seed}", flush=True)
        rows.extend(collect_seed(int(seed)))
    if not rows:
        raise RuntimeError("phase-5e produced no rows")
    failures = failure_rows(rows)
    travel = np.asarray([row["travel_cost"] for row in rows], dtype=np.float64)
    vary = []
    for group in _groups(rows):
        values = {float(row["interference"]) for row in group}
        vary.append(float(len(values) > 1))
    counts: dict[str, int] = {
        "assignment_selection_error": 0,
        "utility_selection_error": 0,
        "payload_high_utility_low": 0,
    }
    for row in failures:
        counts[str(row["case"])] += 1
    summary = {
        "planner_ready": PLANNER_READY,
        "deployment_verified": DEPLOYMENT_VERIFIED,
        "training": False,
        "hgnn_trained": False,
        "mappo_retrained": False,
        "decide_called": False,
        "tau_modified": False,
        "holdout_used": False,
        "phase5f_started": False,
        "seeds": list(DATASET_SEEDS),
        "decision_times": list(DECISION_TIMES),
        "n_rows": len(rows),
        "n_groups": len(_groups(rows)),
        "n_separated": int(sum(1 for group in _groups(rows) if max(float(row["payload_gain"]) for row in group) - min(float(row["payload_gain"]) for row in group) >= 1.0e6 - 1.0)),
        "travel_cost_std": float(np.std(travel)),
        "energy_cost_max": float(max(row["energy_cost"] for row in rows)),
        "interference_varies_within_group": float(np.mean(vary)) if vary else float("nan"),
        "ranking": ranking_summary(rows),
        "failure_counts": counts,
        "utility_config": BASELINE_CONFIG,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "utility_config.json").write_text(json.dumps(BASELINE_CONFIG, indent=2) + "\n", encoding="utf-8")
    _write_csv(OUTPUT / "trajectory_utility_dataset.csv", rows, DATASET_COLUMNS)
    failure_columns = [
        "scenario",
        "time_step",
        "uav",
        "candidate",
        "case",
        "payload_gain",
        "utility_score",
        "utility_rank",
        "assignment_score",
        "assignment_rank",
        "payload_best_candidates",
    ]
    _write_csv(OUTPUT / "utility_failure_cases.csv", failures, failure_columns)
    (OUTPUT / "phase5e_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(_report(summary), encoding="utf-8")
    print(json.dumps({"n_rows": summary["n_rows"], "failures": counts, "ranking": summary["ranking"]}, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    run()
