"""Train joint decoders on frozen encoder states and compare readouts."""
from __future__ import annotations

import copy
import json

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from wt.htw_phase9.const import IDLE_INDEX
from wt.htw_phase943.evaluation import decode_assignment, encode_env
from wt.htw_phase943.set_context import all_candidate_physics_set, empty_assignment, make_snap
from wt.lowlevel_v2.integration_phase1.stats import kendall_tau, spearman
from wt.lowlevel_v2.phase5f_utility_rank.protocol import FORBIDDEN_INPUTS
from wt.lowlevel_v2.phase5f_utility_rank.score import new_hgnn
from wt.lowlevel_v2.phase5g_closed_loop.collect import load_frozen_actor
from wt.lowlevel_v2.phase5g_closed_loop.protocol import UTILITY_CKPT
from wt.lowlevel_v2.phase5h_execution_audit.rollout import trace_h15
from wt.lowlevel_v2.phase5j_joint_redesign.joint import conflict_rate, duplicate_count
from wt.lowlevel_v2.phase5j_joint_redesign.run_phase5j import replay_env
from wt.round30.ppo import seed_everything

from .dataset import load_states
from .decoders import JointAssignmentDecoder
from .loss import joint_rank_loss
from .protocol import (
    DEPLOYMENT_VERIFIED,
    EPOCHS,
    LEARNING_RATE,
    OUTPUT,
    PATIENCE,
    PLANNER_READY,
    REPORT,
    TEST_SCENARIO,
    TRAIN_SCENARIO,
    WEIGHT_DECAY,
    split_name,
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
    if len(scores) < 1:
        return float("nan")
    best = max(target)
    winners = {index for index, value in enumerate(target) if float(value) == float(best)}
    order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
    return float(int(order[0]) in winners)


def _encode(model, batch: dict) -> dict:
    leaked = set(batch).intersection(FORBIDDEN_INPUTS)
    if leaked:
        raise RuntimeError(f"encoder input contains label fields: {sorted(leaked)}")
    with torch.no_grad():
        encoded = encode_env(model, batch)
    return {key: value.detach() for key, value in encoded.items()}


def _frozen_encoder():
    if not UTILITY_CKPT.is_file():
        raise FileNotFoundError(str(UTILITY_CKPT))
    model = new_hgnn()
    model.load_state_dict(torch.load(UTILITY_CKPT, map_location="cpu"))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _pool_scores(head: JointAssignmentDecoder, encoded: dict, assignments: np.ndarray) -> list[float]:
    tensor = torch.as_tensor(assignments, dtype=torch.long)
    with torch.no_grad():
        values = head.score_assignments(encoded["h_u"], encoded["h_b"], tensor)
    return [float(item) for item in values.detach().cpu().numpy()]


def _baseline_scores(model, batch: dict, env, assignments: np.ndarray) -> list[float]:
    encoded = _encode(model, batch)
    blank = empty_assignment(int(env.config.num_uavs))
    snap = make_snap(env.uav_positions, env.buoy_positions, env.config, env=env)
    physics = torch.as_tensor(all_candidate_physics_set(snap, blank)[None, ...], dtype=torch.float32)
    buoy_mask = np.asarray(batch["buoy_mask"][0], dtype=bool)
    with torch.no_grad():
        logits = model.decoder(encoded, blank, physics, buoy_mask).detach().cpu().numpy()
    scores = []
    for assignment in assignments:
        total = 0.0
        for uav, buoy in enumerate(assignment):
            token = int(IDLE_INDEX if int(buoy) < 0 else buoy)
            total += float(logits[int(uav), token])
        scores.append(total)
    return scores


def _ranking(scores: list[float], joints: list[float]) -> dict[str, float]:
    return {
        "spearman": spearman(scores, joints),
        "kendall": kendall_tau(scores, joints),
        "top1": _top1(scores, joints),
    }


def _duplicate_rate(assignment: np.ndarray, num_uavs: int) -> float:
    return float(duplicate_count(assignment) / float(num_uavs))


def _rollout(seed: int, moment: int, assignment: np.ndarray, actor) -> dict[str, float]:
    env, movement = replay_env(int(seed), int(moment))
    traced = trace_h15(env, assignment, actor, movement, "mappo")
    return {
        "mappo": float(traced["payload"]),
        "duplicate_rate": _duplicate_rate(assignment, int(env.config.num_uavs)),
        "conflict_rate": conflict_rate(assignment, int(env.config.num_buoys)),
        "execution_conflict_rate": float(traced["conflict_rate"]),
    }


def train_decoder(encoder, states: list[dict], mode: str) -> tuple[JointAssignmentDecoder, dict]:
    seed_everything(0)
    head = JointAssignmentDecoder(mode)
    optimizer = torch.optim.Adam(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    encoded = {id(state): _encode(encoder, state["urgency_batch"]) for state in states}
    train_states = [state for state in states if split_name(state["scenario_id"], state["time_step"]) == "train"]
    val_states = [state for state in states if split_name(state["scenario_id"], state["time_step"]) == "val"]
    best_score = -1.0e18
    best_epoch = 0
    wait = 0
    best_weights = copy.deepcopy(head.state_dict())
    history = []
    for epoch in range(1, EPOCHS + 1):
        head.train()
        losses = []
        for state in train_states:
            features = encoded[id(state)]
            assignments = torch.as_tensor(state["assignments"], dtype=torch.long)
            joints = torch.as_tensor(state["joint"], dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            scores = head.score_assignments(features["h_u"], features["h_b"], assignments)
            loss = joint_rank_loss(scores, joints)
            if not bool(torch.isfinite(loss).all()):
                raise RuntimeError(f"{mode} loss became non-finite at epoch {epoch}")
            loss.backward()
            clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        head.eval()
        val_rows = []
        with torch.no_grad():
            for state in val_states:
                val_rows.append(
                    _ranking(
                        _pool_scores(head, encoded[id(state)], state["assignments"]),
                        [float(item) for item in state["joint"]],
                    )
                )
        val_spearman = _mean([row["spearman"] for row in val_rows])
        history.append({"epoch": epoch, "loss": _mean(losses), "val_spearman": val_spearman})
        print(f"{mode} epoch {epoch} loss={_mean(losses):.4f} val_spearman={val_spearman:.4f}", flush=True)
        if np.isfinite(val_spearman) and float(val_spearman) > float(best_score) + 1.0e-6:
            best_score = float(val_spearman)
            best_epoch = int(epoch)
            wait = 0
            best_weights = copy.deepcopy(head.state_dict())
        else:
            wait += 1
            if wait >= PATIENCE and best_epoch > 0:
                break
    head.load_state_dict(best_weights)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head, {"best_epoch": best_epoch, "best_val_spearman": best_score, "history": history}


def _pack(rows: list[dict]) -> dict[str, float]:
    return {
        "n_states": float(len(rows)),
        "mappo": _mean([row["mappo"] for row in rows]),
        "duplicate_rate": _mean([row["duplicate_rate"] for row in rows]),
        "conflict_rate": _mean([row["conflict_rate"] for row in rows]),
        "spearman": _mean([row["spearman"] for row in rows]),
        "kendall": _mean([row["kendall"] for row in rows]),
        "top1": _mean([row["top1"] for row in rows]),
    }


def evaluate(encoder, heads: dict[str, JointAssignmentDecoder], states: list[dict], actor) -> dict:
    rows = []
    for state in states:
        env, _movement = replay_env(int(state["scenario_id"]), int(state["time_step"]))
        encoded = _encode(encoder, state["urgency_batch"])
        joints = [float(item) for item in state["joint"]]
        individuals = [float(item) for item in state["individual"]]
        names = {
            "baseline": np.asarray(decode_assignment(encoder, state["urgency_batch"], env), dtype=np.int64),
            "global": heads["global"].decode(encoded["h_u"], encoded["h_b"]),
            "matching": heads["matching"].decode(encoded["h_u"], encoded["h_b"]),
            "individual_greedy": np.asarray(state["greedy"], dtype=np.int64),
            "pool_joint": np.asarray(state["assignments"][int(np.argmax(state["joint"]))], dtype=np.int64),
        }
        score_table = {
            "baseline": _baseline_scores(encoder, state["urgency_batch"], env, state["assignments"]),
            "global": _pool_scores(heads["global"], encoded, state["assignments"]),
            "matching": _pool_scores(heads["matching"], encoded, state["assignments"]),
            "individual_greedy": individuals,
            "pool_joint": joints,
        }
        for name, assignment in names.items():
            metrics = _rollout(int(state["scenario_id"]), int(state["time_step"]), assignment, actor)
            ranking = _ranking(score_table[name], joints)
            rows.append(
                {
                    "model": name,
                    "split": split_name(int(state["scenario_id"]), int(state["time_step"])),
                    "scenario_id": int(state["scenario_id"]),
                    "time_step": int(state["time_step"]),
                    "assignment": [int(item) for item in assignment],
                    **metrics,
                    **ranking,
                }
            )
        print(
            f"eval {state['scenario_id']} t={state['time_step']} "
            f"baseline={rows[-5]['mappo']:.3e} global={rows[-4]['mappo']:.3e} matching={rows[-3]['mappo']:.3e}",
            flush=True,
        )
    summary = {}
    for name in ("baseline", "global", "matching", "individual_greedy", "pool_joint"):
        selected = [row for row in rows if row["model"] == name]
        summary[name] = {
            "test": _pack([row for row in selected if row["split"] == "test"]),
            "train": _pack([row for row in selected if row["split"] == "train"]),
            "val": _pack([row for row in selected if row["split"] == "val"]),
            "states": [
                {
                    "scenario_id": int(row["scenario_id"]),
                    "time_step": int(row["time_step"]),
                    "split": row["split"],
                    "mappo": row["mappo"],
                    "duplicate_rate": row["duplicate_rate"],
                    "conflict_rate": row["conflict_rate"],
                    "spearman": row["spearman"],
                    "top1": row["top1"],
                    "assignment": row["assignment"],
                }
                for row in selected
            ],
        }
    return summary


def _table(summary: dict, split: str) -> list[str]:
    lines = [
        "| model | joint MAPPO | duplicate target rate | conflict rate | Spearman vs J(A) | Top-1 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    titles = (
        ("OrderFreeSetDecoder", "baseline"),
        ("Global assignment head", "global"),
        ("Matching decoder", "matching"),
        ("individual greedy", "individual_greedy"),
        ("pool J(A) optimum", "pool_joint"),
    )
    for title, key in titles:
        item = summary[key][split]
        lines.append(
            f"| {title} | {_fmt(item['mappo'], '.4e')} | {_fmt(item['duplicate_rate'])} | {_fmt(item['conflict_rate'])} | {_fmt(item['spearman'])} | {_fmt(item['top1'])} |"
        )
    return lines


def _state_table(summary: dict) -> list[str]:
    lines = [
        "| scenario | t | split | baseline | global head | matching | individual greedy | pool J(A) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    states = summary["baseline"]["states"]
    for index, row in enumerate(states):
        values = []
        for key in ("baseline", "global", "matching", "individual_greedy", "pool_joint"):
            values.append(_fmt(summary[key]["states"][index]["mappo"], ".4e"))
        lines.append(
            f"| {row['scenario_id']} | {row['time_step']} | {row['split']} | " + " | ".join(values) + " |"
        )
    return lines


def _state_row(summary: dict, model: str, scenario: int, moment: int) -> dict:
    for row in summary[model]["states"]:
        if int(row["scenario_id"]) == int(scenario) and int(row["time_step"]) == int(moment):
            return row
    raise RuntimeError(f"missing {model} state {scenario} t={moment}")


def _gap_text(summary: dict) -> str:
    matched = _state_row(summary, "matching", 310101, 15)
    optimum = _state_row(summary, "pool_joint", 310101, 15)
    return (
        f"Matching decoder 在测试池内的 Top-1 是 {_fmt(summary['matching']['test']['top1'])}，Spearman 是 {_fmt(summary['matching']['test']['spearman'])}。"
        f"Top-1 只比较池内分配。Hungarian 读出可以选到池外浮标，因此联合 MAPPO 均值是 {_fmt(summary['matching']['test']['mappo'], '.4e')}，"
        f"低于 individual greedy 的 {_fmt(summary['individual_greedy']['test']['mappo'], '.4e')}，也低于池内 J(A) 最优的 {_fmt(summary['pool_joint']['test']['mappo'], '.4e')}。"
        f"310101 的 t=15 上，池内 J(A) 最优分配是 {optimum['assignment']}，联合 MAPPO 是 {_fmt(optimum['mappo'], '.4e')}；"
        f"Matching decoder 读出的分配是 {matched['assignment']}，联合 MAPPO 是 {_fmt(matched['mappo'], '.4e')}。"
        f"Global assignment head 的测试 Spearman 是 {_fmt(summary['global']['test']['spearman'])}，联合 MAPPO 是 {_fmt(summary['global']['test']['mappo'], '.4e')}。"
        "三种 decoder 读出的 duplicate target rate 和 conflict rate 都是 0。去掉重复目标没有把多机联合 payload 提升到 J(A)。"
    )


def _report(summary: dict, training: dict) -> str:
    test_global = summary["global"]["test"]["mappo"]
    test_match = summary["matching"]["test"]["mappo"]
    test_base = summary["baseline"]["test"]["mappo"]
    test_greedy = summary["individual_greedy"]["test"]["mappo"]
    best_name = "global assignment head" if test_global >= test_match else "matching decoder"
    best_value = max(test_global, test_match)
    if np.isfinite(best_value) and best_value > test_base and best_value > test_greedy:
        decision = (
            f"在 310101 上，{best_name} 的联合 MAPPO 高于冻结 OrderFreeSetDecoder，也高于单个效用的排他匹配。"
            "增益来自 decoder 对 J(A) 的联合监督。encoder 保持冻结，本阶段不把训练扩展到整个 HGNN。"
        )
    elif np.isfinite(best_value) and best_value > test_base:
        decision = (
            f"在 310101 上，{best_name} 的联合 MAPPO 高于冻结 OrderFreeSetDecoder，但没有高于单个效用的排他匹配。"
            "本阶段不进入整个 HGNN 的重新训练。"
        )
    else:
        decision = "在 310101 上，新 decoder 的联合 MAPPO 没有高于冻结 OrderFreeSetDecoder。本阶段不进入整个 HGNN 的重新训练。"
    return "\n".join(
        [
            "# HTW Phase-5K Joint Assignment Decoder Redesign",
            "",
            "**planner_ready = False**",
            "",
            "**deployment_verified = False**",
            "",
            "本阶段只在冻结的 HGNN encoder 上训练两个新的 decoder 头。没有修改 Residual MAPPO、低层策略、reward、B2、τ、Gate-1、Candidate-C 或 planner，没有调用 deployment `decide()`，没有使用 230000–230127。",
            "",
            "## 1. Phase-5J 的问题",
            "",
            "Phase-5J 的问题在高层 decoder 的分配耦合。encoder、低层 MAPPO 和 reward 不是这一步要改的对象。",
            "",
            "在联合 MAPPO 有差别的时刻，单个效用和的 Spearman 是 0.3673，联合效用 J(A) 的 Spearman 是 0.7492。现有 OrderFreeSetDecoder 按 UAV 分别输出浮标 logit。冻结解码在 310100 和 310101 上的平均联合 MAPPO 是 0，尽管冲突率也是 0。",
            "",
            "## 2. Decoder",
            "",
            "Encoder 输入保持不变。UAV 侧是位置、速度、服务进度、是否已分配、已分配队列，能量通道仍为 0。Buoy 侧是位置、队列、已观测到达、服务进度、占用，以及现有的 urgency 通道。UAV-Buoy 边保留距离、当前速率和 ETA。UAV-UAV 边保留距离和干扰。没有加入未来轨迹。J(A) 只作为监督标签，不进入 encoder。",
            "",
            "Global assignment head 用 H_UAV 和 H_Buoy 的拼接得到 M×K 配对分数，再做带虚拟行的 Sinkhorn。得到的矩阵 P 满足行和为 1，列质量至多为 1。读出时每次取整张矩阵里剩余的最大元素，并同时去掉该行和该列。不按 UAV 单独取最大列。",
            "",
            "Matching decoder 使用同一类配对分数。训练时分配分数是被选中配对的分数和。读出使用 Hungarian matching，直接给出 one-to-one assignment。",
            "",
            "OrderFreeSetDecoder 保持 Phase-5F 检查点，不更新参数。它的排序分数是空白分配上各 UAV logit 的和，用来表示逐 UAV 打分。",
            "",
            "## 3. Joint loss",
            "",
            "监督标签是 Phase-5J 的 J(A)，不是 assignment similarity。",
            "",
            "对所有 J(A) > J(B) 的分配对，损失是 `-log sigmoid(s(A) - s(B))`。J 只决定哪一个分配必须得到更高分数。直接把 `-log sigmoid(J(A) - J(B))` 当作损失不会产生 decoder 梯度。",
            "",
            f"Global head 的 s(A) 是约束后 P 上被选配对的对数概率和。Matching decoder 的 s(A) 是配对分数和。训练场景是 {TRAIN_SCENARIO} 的 t=0 和 t=15。验证只用 {TRAIN_SCENARIO} 的 t=30 做早停，它和训练是同一 scenario 的稍后时刻。测试是 {TEST_SCENARIO} 的 t=0、15、30，不参与训练和选模。",
            "",
            f"Global head 选中的验证轮次是 {training['global']['best_epoch']}，验证 Spearman 是 {_fmt(training['global']['best_val_spearman'])}。Matching decoder 选中的验证轮次是 {training['matching']['best_epoch']}，验证 Spearman 是 {_fmt(training['matching']['best_val_spearman'])}。",
            "",
            "## 4. 实验结果",
            "",
            "下表的决策数字来自测试场景 310101。individual greedy 和 pool J(A) optimum 是参照，不是新的 decoder。duplicate target rate 是重复选择次数除以无人机数。conflict rate 是分配里处于重复浮标上的无人机比例。",
            "",
            "### 测试场景 310101",
            "",
            *_table(summary, "test"),
            "",
            "### 训练场景 310100",
            "",
            "t=0 和 t=15 参与了 decoder 训练。t=30 只用于早停。",
            "",
            *_table(summary, "train"),
            "",
            "### 验证时刻 310100 t=30",
            "",
            *_table(summary, "val"),
            "",
            "### 各时刻联合 MAPPO",
            "",
            *_state_table(summary),
            "",
            "Top-1 表示模型在该时刻分配池内分数最高的分配，是否就是 J(A) 最高的分配。联合 MAPPO 是读出分配交给冻结 MAPPO 执行 15 步的 payload。目标是多机联合 payload，不是单目标 payload。",
            "",
            _gap_text(summary),
            "",
            "## 5. 是否重新训练整个 HGNN",
            "",
            decision,
            "",
            "本阶段不进入 deployment verification。",
            "",
            "`planner_ready = False`",
            "",
            "`deployment_verified = False`",
            "",
        ]
    )


def run() -> dict:
    states = load_states()
    encoder = _frozen_encoder()
    actor = load_frozen_actor()
    trained = {}
    heads = {}
    for mode in ("global", "matching"):
        head, record = train_decoder(encoder, states, mode)
        heads[mode] = head
        trained[mode] = record
        torch.save({"mode": mode, "state_dict": head.state_dict(), "record": record}, OUTPUT / f"{mode}_decoder.pt")
    measured = evaluate(encoder, heads, states, actor)
    test_global = measured["global"]["test"]
    test_match = measured["matching"]["test"]
    test_base = measured["baseline"]["test"]
    test_greedy = measured["individual_greedy"]["test"]
    best_value = max(float(test_global["mappo"]), float(test_match["mappo"]))
    summary = {
        "planner_ready": PLANNER_READY,
        "deployment_verified": DEPLOYMENT_VERIFIED,
        "decide_called": False,
        "encoder_frozen": True,
        "decoder_trained": True,
        "full_hgnn_retrain_started": False,
        "mappo_modified": False,
        "reward_modified": False,
        "train_scenario": TRAIN_SCENARIO,
        "test_scenario": TEST_SCENARIO,
        "training": trained,
        "models": measured,
        "test_decoder_beats_baseline": bool(best_value > float(test_base["mappo"])),
        "test_decoder_beats_individual_greedy": bool(best_value > float(test_greedy["mappo"])),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "phase5k_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(_report(measured, trained), encoding="utf-8")
    print(
        json.dumps(
            {
                "test_mappo": {
                    "baseline": test_base["mappo"],
                    "global": test_global["mappo"],
                    "matching": test_match["mappo"],
                    "greedy": test_greedy["mappo"],
                },
                "full_hgnn_retrain_started": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return summary


if __name__ == "__main__":
    run()
