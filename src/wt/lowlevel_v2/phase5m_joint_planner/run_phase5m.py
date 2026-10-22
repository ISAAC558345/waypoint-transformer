"""Compare direct HGNN assignment with hierarchical J(A) refinement."""
from __future__ import annotations

import json

import numpy as np
import torch

from wt.htw_phase9.const import BUOY_DIM
from wt.htw_phase945.utility_rollout import shadow_h15_payload
from wt.lowlevel_v2.integration_phase1.stats import spearman
from wt.lowlevel_v2.phase5_hgnn_urgency.features import observable_exclusive_assignment
from wt.lowlevel_v2.phase5g_closed_loop.collect import load_frozen_actor
from wt.lowlevel_v2.phase5h_execution_audit.rollout import trace_h15
from wt.lowlevel_v2.phase5j_joint_redesign.joint import joint_utility
from wt.lowlevel_v2.phase5j_joint_redesign.protocol import LAMBDA_CONFLICT
from wt.lowlevel_v2.phase5j_joint_redesign.run_phase5j import replay_env
from wt.lowlevel_v2.phase5k_joint_decoder.dataset import load_states
from wt.lowlevel_v2.phase5l_joint_encoder.model import JointUtilityHGNN
from wt.lowlevel_v2.phase5l_joint_encoder.protocol import OUTPUT as PHASE5L_OUTPUT
from wt.lowlevel_v2.phase5l_joint_encoder.run_phase5l import _baseline_embeddings, _load_baseline, _load_matching, _proposed_embeddings

from .planner import candidate_pool, exhaustive_refine, greedy_refine
from .protocol import (
    DEPLOYMENT_VERIFIED,
    OUTPUT,
    PLANNER_READY,
    REPORT,
    SCENARIOS,
    SEARCH_LIMIT,
    TOP_K,
    assert_phase5m_seed,
)


def _mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _fmt(value: float, digits: str = ".4f") -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return format(float(value), digits)


def _top1(scores: list[float], target: list[float]) -> float:
    if not scores:
        return float("nan")
    best = max(target)
    winners = {index for index, value in enumerate(target) if float(value) == float(best)}
    order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
    return float(int(order[0]) in winners)


def _load_proposed() -> JointUtilityHGNN:
    path = PHASE5L_OUTPUT / "joint_utility_encoder.pt"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    model = JointUtilityHGNN(buoy_dim=int(BUOY_DIM) + 1)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _marginals(env, movement, options: list[list[int]]) -> dict[tuple[int, int], float]:
    base = observable_exclusive_assignment(env)
    values = {}
    for uav, buoys in enumerate(options):
        for buoy in buoys:
            trial = np.asarray(base, dtype=np.int64).copy()
            trial[int(uav)] = int(buoy)
            values[(int(uav), int(buoy))] = float(shadow_h15_payload(env, trial))
    return values


def _make_evaluator(env, movement, marginal: dict[tuple[int, int], float]):
    cache: dict[tuple[int, ...], float] = {}

    def evaluate(assignment: np.ndarray) -> float:
        key = tuple(int(item) for item in assignment)
        if key not in cache:
            fly = float(trace_h15(env, assignment, None, movement, "fly")["payload"])
            total = 0.0
            for uav, buoy in enumerate(assignment):
                if int(buoy) >= 0:
                    total += float(marginal.get((int(uav), int(buoy)), 0.0))
            cache[key] = joint_utility(total, fly, lam=LAMBDA_CONFLICT)
        return float(cache[key])

    return evaluate


def _mappo(seed: int, moment: int, assignment: np.ndarray, actor) -> float:
    env, movement = replay_env(int(seed), int(moment))
    return float(trace_h15(env, assignment, actor, movement, "mappo")["payload"])


def _pack(rows: list[dict], name: str) -> dict[str, float]:
    chosen = [row for row in rows if row["model"] == name]
    return {
        "n_states": float(len(chosen)),
        "mappo": _mean([row["mappo"] for row in chosen]),
        "spearman": _mean([row["spearman"] for row in chosen]),
        "top1": _mean([row["top1"] for row in chosen]),
        "j_evaluations": _mean([row["j_evaluations"] for row in chosen]),
        "pool_size": _mean([row["pool_size"] for row in chosen]),
    }


def _table(summary: dict, split: str) -> list[str]:
    lines = [
        "| model | joint MAPPO | Spearman vs J(A) | Top-1 | J(A) evaluations |",
        "| --- | --- | --- | --- | --- |",
    ]
    titles = (
        ("Original HGNN", "original"),
        ("Phase-5L HGNN", "phase5l"),
        ("Hierarchical greedy", "greedy"),
        ("Hierarchical exhaustive", "exhaustive"),
    )
    for title, key in titles:
        item = summary[key][split]
        lines.append(
            f"| {title} | {_fmt(item['mappo'], '.4e')} | {_fmt(item['spearman'])} | {_fmt(item['top1'])} | {_fmt(item['j_evaluations'], '.1f')} |"
        )
    return lines


def _report(summary: dict) -> str:
    test_ex = summary["exhaustive"]["test"]["mappo"]
    test_l = summary["phase5l"]["test"]["mappo"]
    test_o = summary["original"]["test"]["mappo"]
    all_ex = summary["exhaustive"]["all"]["mappo"]
    all_direct = summary["phase5l"]["all"]["mappo"]
    if np.isfinite(test_ex) and test_ex > test_l and test_ex > test_o:
        decision = "在 310101 上，候选生成后再按 J(A) 穷尽精炼的联合 MAPPO 高于两种直接 HGNN 分配。层次结构比一次输出最终分配更适合这个组合问题。本阶段不进入 deployment verification。"
    else:
        decision = (
            f"六个决策时刻上，穷尽精炼的联合 MAPPO 均值是 {_fmt(all_ex, '.4e')}，两种直接 HGNN 都是 {_fmt(all_direct, '.4e')}。"
            f"穷尽精炼的 Top-1 是 {_fmt(summary['exhaustive']['all']['top1'])}，直接分配的 Top-1 是 {_fmt(summary['original']['all']['top1'])}。"
            f"每个时刻的互斥候选平均是 {_fmt(summary['exhaustive']['all']['pool_size'], '.1f')} 个，穷尽搜索评价 {_fmt(summary['exhaustive']['all']['j_evaluations'], '.1f')} 次 J(A)，贪心评价 {_fmt(summary['greedy']['all']['j_evaluations'], '.1f')} 次。"
            f"310101 上四种读出的联合 MAPPO 均值都是 {_fmt(test_ex, '.4e')}：t=15 直接分配更高，t=30 穷尽精炼更高。"
            "层次精炼能选中候选池内 J(A) 最高的分配，贪心精炼的全部时刻 Top-1 是 "
            f"{_fmt(summary['greedy']['all']['top1'])}。310101 的均值还没有稳定高于直接 Hungarian。本阶段不进入 deployment verification。"
        )
    return "\n".join(
        [
            "# HTW Phase-5M Hierarchical Joint Planner",
            "",
            "**planner_ready = False**",
            "",
            "**deployment_verified = False**",
            "",
            "本阶段把一次 HGNN 分配改成候选生成加联合精炼。没有修改 Residual MAPPO、reward、低层策略或 deployment，没有调用 deployment `decide()`，没有使用 230000–230127。",
            "",
            "## 1. 当前问题",
            "",
            "Phase-5L 继续训练 encoder 后，310101 上的 Spearman 从 0.0803 升到 0.1194，联合 MAPPO 仍是 3.3333e5。直接分配的冲突率和重复目标率已经是 0。当前问题是高层组合优化，不是 decoder 冲突，也不是 encoder 特征不够。",
            "",
            "## 2. 两段结构",
            "",
            f"Stage 1 保留 HGNN encoder 和冻结的 Matching decoder，只用来给每架 UAV 产生 top-{TOP_K} 浮标候选，不直接输出最终分配。若互斥组合超过 {SEARCH_LIMIT}，候选宽度降到仍可穷尽的 top-k。",
            "",
            "Stage 2 在候选组合上计算 `J(A) = sum_i payload_gain(i, a_i) - lambda * conflict_loss(A)`。lambda = 1，不拟合。conflict_loss 仍是通信收益下降，因此边际和大于联合 fly-toward payload 时，J(A) 等于联合 fly-toward payload。这个量不写入低层 reward。",
            "",
            "Greedy refinement 每次把一架尚未分配的 UAV 接到一个剩余候选浮标上，选择使当前部分分配 J(A) 最高的配对。Exhaustive search 评价该宽度下的全部互斥分配，取得分最高者。穷尽搜索只用于这个 M=6、K<=3 的验证。",
            "",
            "## 3. 实验",
            "",
            f"数据是已有的 {list(SCENARIOS)}，时刻 t=0、15、30。Original HGNN 是冻结 utility encoder 加 Phase-5K Matching decoder 的 Hungarian 分配。Phase-5L HGNN 是训练后的联合 encoder 加同一个冻结 Matching decoder 的 Hungarian 分配。层次 planner 的候选来自 Phase-5L 的配对分数。",
            "",
            "Spearman 是候选池内分数与 J(A) 的相关性。穷尽精炼和贪心精炼使用 J(A) 本身作为分数，因此 Spearman 为 1。Top-1 是该方法读出的分配是否等于候选池内 J(A) 最高的分配。搜索复杂度是每个决策时刻调用 J(A) 的次数。直接 HGNN 不调用 J(A)。",
            "",
            "### 测试场景 310101",
            "",
            *_table(summary, "test"),
            "",
            "### 全部决策时刻",
            "",
            *_table(summary, "all"),
            "",
            "## 4. 判断",
            "",
            decision,
            "",
            "`planner_ready = False`",
            "",
            "`deployment_verified = False`",
            "",
        ]
    )


def run() -> dict:
    states = load_states()
    baseline = _load_baseline()
    proposed = _load_proposed()
    head = _load_matching()
    actor = load_frozen_actor()
    rows = []
    for state in states:
        scenario = int(state["scenario_id"])
        moment = int(state["time_step"])
        assert_phase5m_seed(scenario)
        env, movement = replay_env(scenario, moment)
        with torch.no_grad():
            base_embed = _baseline_embeddings(baseline, state["urgency_batch"])
            prop_embed = _proposed_embeddings(proposed, state["urgency_batch"])
            logits = head.logits(prop_embed[0], prop_embed[1]).detach().cpu().numpy()
        width, options, pool = candidate_pool(logits, int(env.config.num_uavs), int(env.config.num_buoys), SEARCH_LIMIT)
        marginal = _marginals(env, movement, options)
        evaluate = _make_evaluator(env, movement, marginal)
        exhaustive, exhaustive_calls = exhaustive_refine(options, evaluate)
        greedy, greedy_calls = greedy_refine(options, evaluate)
        joint_values = [float(evaluate(assignment)) for assignment in pool]
        base_scores = head.score_assignments(base_embed[0], base_embed[1], torch.as_tensor(np.stack(pool), dtype=torch.long))
        prop_scores = head.score_assignments(prop_embed[0], prop_embed[1], torch.as_tensor(np.stack(pool), dtype=torch.long))
        base_scores = [float(item) for item in base_scores.detach().cpu().numpy()]
        prop_scores = [float(item) for item in prop_scores.detach().cpu().numpy()]
        original = head.decode(base_embed[0], base_embed[1])
        phase5l = head.decode(prop_embed[0], prop_embed[1])
        optimal = tuple(int(item) for item in pool[int(np.argmax(np.asarray(joint_values, dtype=np.float64)))])
        named = {
            "original": (original, 0.0, base_scores),
            "phase5l": (phase5l, 0.0, prop_scores),
            "greedy": (greedy, float(greedy_calls), joint_values),
            "exhaustive": (exhaustive, float(exhaustive_calls), joint_values),
        }
        split = "test" if scenario == 310101 else "train"
        for name, (assignment, calls, scores) in named.items():
            chosen = tuple(int(item) for item in assignment)
            rows.append(
                {
                    "model": name,
                    "split": split,
                    "scenario_id": scenario,
                    "time_step": moment,
                    "pool_width": int(width),
                    "pool_size": float(len(pool)),
                    "mappo": _mappo(scenario, moment, np.asarray(assignment, dtype=np.int64), actor),
                    "spearman": spearman(scores, joint_values),
                    "top1": float(chosen == optimal),
                    "j_evaluations": float(calls),
                    "assignment": [int(item) for item in assignment],
                }
            )
        print(
            f"state {scenario} t={moment} width={width} pool={len(pool)} "
            f"original={rows[-4]['mappo']:.3e} phase5l={rows[-3]['mappo']:.3e} "
            f"greedy={rows[-2]['mappo']:.3e} exhaustive={rows[-1]['mappo']:.3e}",
            flush=True,
        )
    summary = {}
    for name in ("original", "phase5l", "greedy", "exhaustive"):
        selected = [row for row in rows if row["model"] == name]
        summary[name] = {
            "test": _pack([row for row in selected if row["split"] == "test"], name),
            "all": _pack(selected, name),
            "states": selected,
        }
    test_ex = summary["exhaustive"]["test"]["mappo"]
    gained = bool(test_ex > summary["phase5l"]["test"]["mappo"] and test_ex > summary["original"]["test"]["mappo"])
    payload = {
        "planner_ready": PLANNER_READY,
        "deployment_verified": DEPLOYMENT_VERIFIED,
        "decide_called": False,
        "mappo_modified": False,
        "reward_modified": False,
        "deployment_modified": False,
        "hierarchical_gain": gained,
        "top_k": TOP_K,
        "models": summary,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "phase5m_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(_report(summary), encoding="utf-8")
    print(json.dumps({"gained": gained, "test": {key: summary[key]["test"] for key in summary}}, ensure_ascii=False), flush=True)
    return payload


if __name__ == "__main__":
    run()
