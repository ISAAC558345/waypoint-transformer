"""Offline joint-label audit on 310100 and 310101. No deployment training."""
from __future__ import annotations

import json

import numpy as np
import torch

from wt.htw_phase943.evaluation import decode_assignment
from wt.htw_phase945.utility_rollout import shadow_h15_payload
from wt.l3b.env import L3BEnv
from wt.l3b.policies import fly_toward
from wt.lowlevel_v2.integration_phase1.stats import kendall_tau, spearman
from wt.lowlevel_v2.phase5_hgnn_urgency.features import graph_batch, observable_exclusive_assignment
from wt.lowlevel_v2.phase5_hgnn_urgency.urgency import shared_task_urgency
from wt.lowlevel_v2.phase5e_utility_dataset.build import _legal_buoys
from wt.lowlevel_v2.phase5f_utility_rank.protocol import FORBIDDEN_INPUTS, MODEL_INPUT_KEYS
from wt.lowlevel_v2.phase5f_utility_rank.score import new_hgnn
from wt.lowlevel_v2.phase5g_closed_loop.collect import load_frozen_actor, mappo_h15
from wt.lowlevel_v2.phase5g_closed_loop.protocol import UTILITY_CKPT
from wt.lowlevel_v2.phase5g_closed_loop.run_phase5g import _greedy_assignment
from wt.lowlevel_v2.phase5h_execution_audit.rollout import trace_h15
from wt.lowlevel_v2.phase5i_joint_utility.search import largest_complete_pool, top_options

from .joint import conflict_loss, conflict_rate, duplicate_count, joint_utility
from .protocol import (
    DECISION_TIMES,
    DEPLOYMENT_VERIFIED,
    LAMBDA_CONFLICT,
    OUTPUT,
    PLANNER_READY,
    REPORT,
    SCENARIOS,
    SEARCH_LIMIT,
    assert_phase5j_seed,
)


def _mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _fmt(value: float) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value):.4f}"


def _assert_inputs(batch: dict) -> None:
    leaked = set(batch).intersection(FORBIDDEN_INPUTS)
    if leaked:
        raise RuntimeError(f"model input contains label fields: {sorted(leaked)}")


def _advance(env: L3BEnv, target: int, last_positions, last_arrival, last_movement):
    while int(env.t) < int(target):
        positions = np.asarray(env.uav_positions, dtype=np.float64).copy()
        assignment = observable_exclusive_assignment(env)
        movement = fly_toward(env, assignment)
        info = env.step(movement, assignment)
        last_positions = positions
        last_arrival = np.asarray(info["arrivals_blocks"], dtype=np.float32)
        last_movement = np.asarray(movement, dtype=np.float64)
    if int(env.t) != int(target):
        raise RuntimeError(f"phase-5j stopped at t={env.t}, expected {target}")
    return last_positions, last_arrival, last_movement


def _marginal(sample: dict, buoy: int, key: str) -> float:
    for candidate, value in zip(sample["candidate_ids"], sample[key]):
        if int(candidate) == int(buoy):
            return float(value)
    return 0.0


def _sum_over(group: list[dict], assignment: np.ndarray, key: str) -> float:
    by_uav = {int(sample["uav_id"]): sample for sample in group}
    total = 0.0
    for uav, buoy in enumerate(assignment):
        if int(buoy) < 0:
            continue
        total += _marginal(by_uav[int(uav)], int(buoy), key)
    return float(total)


def _rank_row(scores: list[float], target: list[float]) -> dict[str, float]:
    if len(scores) < 3:
        return {"spearman": float("nan"), "kendall": float("nan"), "top1": float("nan"), "top3": float("nan")}
    best = max(target)
    winners = {index for index, value in enumerate(target) if float(value) == float(best)}
    order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
    top = int(order[0])
    return {
        "spearman": spearman(scores, target),
        "kendall": kendall_tau(scores, target),
        "top1": float(top in winners),
        "top3": float(any(int(index) in winners for index in order[:3])),
    }


def _pack(rows: list[dict]) -> dict[str, float]:
    packed = {}
    for name in ("spearman", "kendall", "top1", "top3"):
        packed[name] = _mean([float(row[name]) for row in rows])
    packed["n_states"] = float(len(rows))
    return packed


def collect_marginals(seed: int) -> list[dict]:
    assert_phase5j_seed(int(seed))
    env = L3BEnv()
    env.reset(seed=int(seed))
    last_positions = None
    last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
    last_movement = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    rows = []
    for moment in DECISION_TIMES:
        last_positions, last_arrival, last_movement = _advance(env, int(moment), last_positions, last_arrival, last_movement)
        info = env.info()
        base = observable_exclusive_assignment(env)
        positions = np.asarray(env.uav_positions, dtype=np.float64).copy()
        graph_last = positions if last_positions is None else np.asarray(last_positions, dtype=np.float32)
        urgency = shared_task_urgency(env.queue)
        batch = graph_batch(info, env, last_arrival, graph_last, urgency=None)
        urgency_batch = graph_batch(info, env, last_arrival, graph_last, urgency=urgency)
        _assert_inputs(batch)
        _assert_inputs(urgency_batch)
        if not set(MODEL_INPUT_KEYS).issubset(batch):
            raise RuntimeError("graph batch is missing an encoder input")
        queue = np.asarray(env.queue, dtype=np.float64)
        distance = np.asarray(info["distance_uav_buoy"], dtype=np.float64)
        rate = np.asarray(info["rate_if_assigned"], dtype=np.float64)
        for focal in range(int(env.config.num_uavs)):
            legal = _legal_buoys(base, focal, int(env.config.num_buoys))
            utility = []
            similarity = []
            for buoy in legal:
                trial = np.asarray(base, dtype=np.int64).copy()
                trial[focal] = int(buoy)
                utility.append(float(shadow_h15_payload(env, trial)))
                similarity.append(float(queue[buoy]) * float(rate[focal, buoy]) / (float(distance[focal, buoy]) + 1.0))
            rows.append(
                {
                    "scenario_id": int(seed),
                    "time_step": int(moment),
                    "uav_id": int(focal),
                    "candidate_ids": [int(item) for item in legal],
                    "utility_payload": utility,
                    "assignment_score": similarity,
                    "urgency_batch": urgency_batch,
                    "context": np.asarray(base, dtype=np.int64).copy(),
                    "queue": queue.copy(),
                }
            )
        print(f"marginals {seed} t={moment}", flush=True)
    return rows


def _report(summary: dict) -> str:
    labels = (
        ("A assignment similarity", "similarity"),
        ("B individual trajectory utility", "individual"),
        ("C joint trajectory utility", "joint"),
    )
    table = [
        "| label | subset | Spearman | Kendall | Top-1 | Top-3 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for title, key in labels:
        for subset in ("all", "payload_separated"):
            item = summary["ranking"][key][subset]
            table.append(
                f"| {title} | {subset} | {_fmt(item['spearman'])} | {_fmt(item['kendall'])} | {_fmt(item['top1'])} | {_fmt(item['top3'])} |"
            )
    readouts = (
        ("OrderFreeSetDecoder", "decoder"),
        ("oracle global J(A)", "global_joint"),
        ("exclusive matching", "matching"),
        ("raw individual argmax", "raw"),
    )
    readout = [
        "| readout | joint fly | joint MAPPO | duplicate targets | conflict rate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for title, key in readouts:
        item = summary["readout"][key]
        readout.append(
            f"| {title} | {_fmt(item['fly'])} | {_fmt(item['mappo'])} | {_fmt(item['duplicate_targets'])} | {_fmt(item['conflict_rate'])} |"
        )
    enter = summary["enter_training"]
    enter_text = "在 payload 有差别的时刻，联合标签 Top-1 高于单个效用标签。这是以后用 J(A) 重新监督现有 HGNN 的依据。Top-1 仍低于 1，本阶段不开始这次训练。" if enter else "联合标签还没有在 Top-1 上高于单个效用标签。本阶段不开始 HGNN 重新训练。"
    return "\n".join(
        [
            "# HTW Phase-5J Joint Utility HGNN Redesign Audit",
            "",
            "**planner_ready = False**",
            "",
            "**deployment_verified = False**",
            "",
            "本阶段只核对联合分配标签和三种读出方式。没有训练 deployment 模型，没有修改 Residual MAPPO，没有修改 B2、τ、Gate-1、Candidate-C、planner 或 deployment `decide()`，没有使用 230000–230127。",
            "",
            "## 1. Individual utility 的局限",
            "",
            "单个效用最高的组合不是联合 MAPPO 最高的组合。不做排他约束时，各无人机的最高分会重复选中目标。",
            "",
            f"数据是已有重置 {list(SCENARIOS)}，决策时刻 t=0、15、30。没有新的 scenario。310100 曾用于 utility HGNN 训练，310101 是它的测试场景；这里只评标签，不更新检查点。",
            "",
            "## 2. Joint utility",
            "",
            "`J(A) = sum_i payload_gain(i, a_i) - lambda * conflict_loss(A)`。",
            "",
            f"lambda = {LAMBDA_CONFLICT}，不拟合。conflict_loss 的标量是通信收益下降：`max(0, sum_i payload_gain(i, a_i) - joint fly payload)`，单位是 bit。因此在边际和大于联合 payload 时，J(A) 等于联合 fly-toward payload。",
            "",
            "另外三项只作诊断，不换算后加进这个标量，避免和通信下降重复计算，也不写入低层 reward：同一目标的重复架数、`conflict_rate`、联合飞行时的干扰矩阵和。",
            "",
            *table,
            "",
            "Spearman、Kendall、Top-1、Top-3 都在每个决策时刻的分配池内部计算，再对时刻取平均。对照量是该分配的联合 MAPPO payload。payload_separated 只保留联合 MAPPO 极差至少一个服务块的时刻。",
            "",
            "## 3. Decoder",
            "",
            "HGNN encoder 保持冻结。三种读出都没有新的训练参数。",
            "",
            "OrderFreeSetDecoder 是检查点里的联合解码。oracle global J(A) 是在穷尽池里直接选取测量到的 J(A) 最大分配，用来表示 global assignment head 若完美预测 J(A) 能到达的离线上界，不是一个已经训练的头。exclusive matching 是在单个效用上做排他贪心，禁止两架飞机使用同一浮标。",
            "",
            *readout,
            "",
            summary["decoder_text"],
            "",
            "## 4. Global assignment head",
            "",
            summary["head_text"],
            "",
            "## 5. Conflict-aware assignment",
            "",
            f"原始最高分的冲突率是 {_fmt(summary['readout']['raw']['conflict_rate'])}，重复目标 {_fmt(summary['readout']['raw']['duplicate_targets'])}。排他匹配的冲突率是 {_fmt(summary['readout']['matching']['conflict_rate'])}。按 J(A) 选出的分配冲突率是 {_fmt(summary['readout']['global_joint']['conflict_rate'])}。",
            "",
            f"池内分配的平均通信下降是 {_fmt(summary['diagnostics']['communication_drop'])} bit，平均干扰是 {_fmt(summary['diagnostics']['interference'])}。",
            "",
            "## 6. 是否重新训练 HGNN",
            "",
            enter_text,
            "",
            "本阶段不进入 deployment verification。",
            "",
            "`planner_ready = False`",
            "",
            "`deployment_verified = False`",
            "",
        ]
    )


def replay_env(seed: int, moment: int) -> tuple[L3BEnv, np.ndarray]:
    assert_phase5j_seed(int(seed))
    env = L3BEnv()
    env.reset(seed=int(seed))
    last_positions = None
    last_arrival = np.zeros(env.config.num_buoys, dtype=np.float32)
    last_movement = np.zeros((env.config.num_uavs, 2), dtype=np.float64)
    _positions, _arrival, last_movement = _advance(env, int(moment), last_positions, last_arrival, last_movement)
    return env, last_movement


def _measure(group: list[dict], env: L3BEnv, assignment: np.ndarray, actor, movement: np.ndarray) -> dict:
    individual = _sum_over(group, assignment, "utility_payload")
    fly = trace_h15(env, assignment, actor, movement, "fly")
    mappo = float(mappo_h15(env, assignment, actor, movement))
    drop = max(0.0, individual - float(fly["payload"]))
    rate = conflict_rate(assignment, int(env.config.num_buoys))
    parts = conflict_loss(drop, duplicate_count(assignment), rate, float(fly["interference"]))
    return {
        "assignment": np.asarray(assignment, dtype=np.int64),
        "similarity": _sum_over(group, assignment, "assignment_score"),
        "individual": individual,
        "joint": joint_utility(individual, float(fly["payload"]), lam=LAMBDA_CONFLICT),
        "mappo": mappo,
        "fly": float(fly["payload"]),
        "duplicate_targets": parts["duplicate_targets"],
        "conflict_rate": rate,
        "communication_drop": parts["communication_drop"],
        "interference": parts["interference"],
    }


def run() -> dict:
    actor = load_frozen_actor()
    rows = []
    for seed in SCENARIOS:
        print(f"redesign scenario {seed}", flush=True)
        rows.extend(collect_marginals(int(seed)))
    model = new_hgnn()
    if not UTILITY_CKPT.is_file():
        raise FileNotFoundError(str(UTILITY_CKPT))
    model.load_state_dict(torch.load(UTILITY_CKPT, map_location="cpu"))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    grouped: dict[tuple[int, int], list[dict]] = {}
    for sample in rows:
        grouped.setdefault((int(sample["scenario_id"]), int(sample["time_step"])), []).append(sample)
    per_state = {"similarity": [], "individual": [], "joint": []}
    separated = {"similarity": [], "individual": [], "joint": []}
    readout_names = ("decoder", "global_joint", "matching", "raw")
    readout_values = {name: {"fly": [], "mappo": [], "duplicate_targets": [], "conflict_rate": []} for name in readout_names}
    diagnostics = {"communication_drop": [], "interference": [], "duplicate_targets": []}
    for (scenario, moment), group in grouped.items():
        group = sorted(group, key=lambda sample: int(sample["uav_id"]))
        pools = {
            width: [top_options(sample["candidate_ids"], sample["utility_payload"], width) for sample in group]
            for width in (1, 2, 3)
        }
        _width, assignments = largest_complete_pool(pools, SEARCH_LIMIT)
        if not assignments:
            raise RuntimeError(f"phase-5j found no exclusive assignment at {scenario} t={moment}")
        local_utility = {int(sample["uav_id"]): sample["utility_payload"] for sample in group}
        raw = np.full(len(group), -1, dtype=np.int64)
        for sample in group:
            raw[int(sample["uav_id"])] = int(sample["candidate_ids"][int(np.argmax(np.asarray(sample["utility_payload"], dtype=np.float64)))])
        matching = _greedy_assignment(group, local_utility)
        env, movement = replay_env(scenario, moment)
        decoder = np.asarray(decode_assignment(model, group[0]["urgency_batch"], env), dtype=np.int64)
        lookup: dict[tuple[int, ...], dict] = {}

        def consider(assignment: np.ndarray) -> dict:
            key = tuple(int(item) for item in assignment)
            if key not in lookup:
                item = _measure(group, env, assignment, actor, movement)
                lookup[key] = item
                diagnostics["communication_drop"].append(item["communication_drop"])
                diagnostics["interference"].append(item["interference"])
                diagnostics["duplicate_targets"].append(item["duplicate_targets"])
            return lookup[key]

        evaluated = [consider(assignment) for assignment in assignments]
        metrics = {
            "similarity": _rank_row([item["similarity"] for item in evaluated], [item["mappo"] for item in evaluated]),
            "individual": _rank_row([item["individual"] for item in evaluated], [item["mappo"] for item in evaluated]),
            "joint": _rank_row([item["joint"] for item in evaluated], [item["mappo"] for item in evaluated]),
        }
        for name, row in metrics.items():
            per_state[name].append(row)
        if max(item["mappo"] for item in evaluated) - min(item["mappo"] for item in evaluated) >= 1.0e6 - 1.0:
            for name, row in metrics.items():
                separated[name].append(row)
        named = {
            "decoder": consider(decoder),
            "global_joint": max(evaluated, key=lambda item: (item["joint"], item["mappo"])),
            "matching": consider(matching),
            "raw": consider(raw),
        }
        for name, item in named.items():
            readout_values[name]["fly"].append(item["fly"])
            readout_values[name]["mappo"].append(item["mappo"])
            readout_values[name]["duplicate_targets"].append(item["duplicate_targets"])
            readout_values[name]["conflict_rate"].append(item["conflict_rate"])
        print(
            f"state {scenario} t={moment} n={len(evaluated)} joint_top1={metrics['joint']['top1']} individual_top1={metrics['individual']['top1']}",
            flush=True,
        )
    ranking = {
        name: {"all": _pack(per_state[name]), "payload_separated": _pack(separated[name])}
        for name in per_state
    }
    readout = {name: {key: _mean(values) for key, values in bucket.items()} for name, bucket in readout_values.items()}
    joint_top = ranking["joint"]["payload_separated"]["top1"]
    individual_top = ranking["individual"]["payload_separated"]["top1"]
    enter = bool(np.isfinite(joint_top) and np.isfinite(individual_top) and joint_top > individual_top)
    global_mappo = readout["global_joint"]["mappo"]
    decoder_mappo = readout["decoder"]["mappo"]
    if global_mappo > decoder_mappo + 1.0:
        decoder_text = "冻结 OrderFreeSetDecoder 的联合 MAPPO payload 低于按 J(A) 选出的分配，也低于排他匹配。现有解码没有读出联合效用。"
        head_text = "不增加 global assignment head。按测得 J(A) 选取的离线上界高于排他匹配，差距来自联合标签。本阶段没有训练新的头。"
    else:
        decoder_text = "冻结 decoder 的联合 MAPPO payload 没有低于 J(A) 读出。本阶段不修改 decoder。"
        head_text = "不增加 global assignment head。"
    summary = {
        "planner_ready": PLANNER_READY,
        "deployment_verified": DEPLOYMENT_VERIFIED,
        "decide_called": False,
        "deployment_model_trained": False,
        "mappo_modified": False,
        "lambda_conflict": LAMBDA_CONFLICT,
        "ranking": ranking,
        "readout": readout,
        "diagnostics": {key: _mean(values) for key, values in diagnostics.items()},
        "enter_training": enter,
        "decoder_text": decoder_text,
        "head_text": head_text,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "phase5j_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(_report(summary), encoding="utf-8")
    print(json.dumps({"ranking": ranking, "readout": readout, "enter": enter}, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    run()
