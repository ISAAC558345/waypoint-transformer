"""Train an assignment-level joint head inside the Phase-5M candidate pool."""
from __future__ import annotations

import copy
import json
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.nn.utils import clip_grad_norm_
from wt.lowlevel_v2.phase5j_joint_redesign.run_phase5j import replay_env
from wt.lowlevel_v2.phase5k_joint_decoder.dataset import load_states
from wt.lowlevel_v2.phase5k_joint_decoder.loss import joint_rank_loss
from wt.lowlevel_v2.phase5l_joint_encoder.model import batch_tensors
from wt.lowlevel_v2.phase5l_joint_encoder.run_phase5l import _baseline_embeddings, _load_baseline, _load_matching, _proposed_embeddings
from wt.lowlevel_v2.phase5m_joint_planner.planner import candidate_lists
from wt.lowlevel_v2.phase5m_joint_planner.protocol import TOP_K
from wt.lowlevel_v2.phase5m_joint_planner.run_phase5m import _load_proposed, _make_evaluator, _marginals, _mappo
from wt.round30.ppo import seed_everything

from .head import JointAssignmentHead, assignment_scores, global_embedding, hungarian_from_plan, masked_plan
from .matchings import maximal_assignments
from .protocol import (
    DEPLOYMENT_VERIFIED,
    EPOCHS,
    LABEL_VERSION,
    LABELS,
    LEARNING_RATE,
    OUTPUT,
    PATIENCE,
    PLANNER_READY,
    REPORT,
    TEST_SCENARIO,
    TRAIN_SCENARIO,
    WEIGHT_DECAY,
    assert_phase5n_seed,
)


def _mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _fmt(value: float, digits: str = ".4f") -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return format(float(value), digits)


def _candidate_mask(options: list[list[int]], num_uavs: int, num_buoys: int) -> np.ndarray:
    mask = np.zeros((int(num_uavs), int(num_buoys)), dtype=bool)
    for uav, buoys in enumerate(options):
        for buoy in buoys:
            mask[int(uav), int(buoy)] = True
    return mask


def _features(proposed, matching, batch: dict) -> dict[str, torch.Tensor | np.ndarray]:
    with torch.no_grad():
        h_u, h_b = _proposed_embeddings(proposed, batch)
        logits = matching.logits(h_u, h_b)
        tensors = batch_tensors(batch)
        z_global = global_embedding(h_u, h_b, tensors["uav_mask"], tensors["buoy_mask"])
    return {"h_u": h_u.detach(), "h_b": h_b.detach(), "logits": logits.detach(), "z_global": z_global.detach()}


def build_labels(proposed, matching) -> list[dict]:
    if LABELS.is_file():
        payload = torch.load(LABELS, map_location="cpu", weights_only=False)
        if int(payload.get("version", -1)) == int(LABEL_VERSION):
            return list(payload["states"])
    labeled = []
    for state in load_states():
        scenario = int(state["scenario_id"])
        moment = int(state["time_step"])
        assert_phase5n_seed(scenario)
        env, movement = replay_env(scenario, moment)
        features = _features(proposed, matching, state["urgency_batch"])
        options = candidate_lists(
            features["logits"].cpu().numpy(),
            TOP_K,
            int(env.config.num_uavs),
            int(env.config.num_buoys),
        )
        pool = maximal_assignments(options)
        marginal = _marginals(env, movement, options)
        evaluate = _make_evaluator(env, movement, marginal)
        started = time.perf_counter()
        joints = [float(evaluate(assignment)) for assignment in pool]
        elapsed = time.perf_counter() - started
        calls = len(pool)
        optimal = np.asarray(pool[int(np.argmax(np.asarray(joints, dtype=np.float64)))], dtype=np.int64)
        labeled.append(
            {
                "scenario_id": scenario,
                "time_step": moment,
                "urgency_batch": state["urgency_batch"],
                "assignments": np.stack([np.asarray(item, dtype=np.int64) for item in pool], axis=0),
                "joint": np.asarray(joints, dtype=np.float64),
                "optimal": np.asarray(optimal, dtype=np.int64),
                "candidate_mask": _candidate_mask(options, int(env.config.num_uavs), int(env.config.num_buoys)),
                "exhaustive_seconds": float(elapsed),
                "exhaustive_calls": float(calls),
            }
        )
        print(
            f"label {scenario} t={moment} pool={len(pool)} unique_j={len(np.unique(np.round(joints, 3)))} "
            f"search_s={elapsed:.4f}",
            flush=True,
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.save({"version": int(LABEL_VERSION), "states": labeled}, LABELS)
    return labeled


def _score_state(head: JointAssignmentHead, features: dict, state: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = head.pair_scores(features["h_u"], features["h_b"], features["z_global"])
    mask = torch.as_tensor(state["candidate_mask"], dtype=torch.bool)
    plan = masked_plan(logits, mask)
    scores = assignment_scores(plan, torch.as_tensor(state["assignments"], dtype=torch.long))
    return logits, plan, scores


def _ranking(scores: list[float], joints: list[float]) -> float:
    """Spearman with tie correction. A constant J(A) has no rank to learn."""
    left = np.asarray(scores, dtype=np.float64)
    right = np.asarray(joints, dtype=np.float64)
    if left.size < 3 or np.unique(right).size < 2 or np.unique(left).size < 2:
        return float("nan")
    result = spearmanr(left, right)
    value = getattr(result, "statistic", None)
    if value is None:
        value = result.correlation
    if not np.isfinite(float(value)):
        return float("nan")
    return float(value)


def train_head(head: JointAssignmentHead, proposed, matching, states: list[dict]) -> dict:
    train_states = [state for state in states if int(state["scenario_id"]) == int(TRAIN_SCENARIO)]
    if not train_states:
        raise RuntimeError("phase-5n has no training states")
    features = {id(state): _features(proposed, matching, state["urgency_batch"]) for state in states}
    optimizer = torch.optim.Adam(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_score = -1.0e18
    best_epoch = 0
    wait = 0
    best_weights = copy.deepcopy(head.state_dict())
    for epoch in range(1, EPOCHS + 1):
        head.train()
        losses = []
        measured = []
        for state in train_states:
            optimizer.zero_grad(set_to_none=True)
            _logits, _plan, scores = _score_state(head, features[id(state)], state)
            joints = torch.as_tensor(state["joint"], dtype=torch.float32)
            loss = joint_rank_loss(scores, joints)
            if not bool(torch.isfinite(loss).all()):
                raise RuntimeError(f"assignment loss became non-finite at epoch {epoch}")
            loss.backward()
            clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            head.eval()
            with torch.no_grad():
                _after_logits, _after_plan, after = _score_state(head, features[id(state)], state)
            head.train()
            measured.append(_ranking([float(item) for item in after.detach().cpu().numpy()], [float(item) for item in state["joint"]]))
        spearman_value = _mean(measured)
        print(f"head epoch {epoch} loss={_mean(losses):.4f} train_spearman={spearman_value:.4f}", flush=True)
        if np.isfinite(spearman_value) and float(spearman_value) > float(best_score) + 1.0e-6:
            best_score = float(spearman_value)
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
    return {"best_epoch": best_epoch, "best_train_spearman": best_score, "features": features}


def _top1_assignment(assignment: np.ndarray, pool: np.ndarray, joints: np.ndarray) -> float:
    """1 when the decoded assignment is one of the J(A) maximizers."""
    chosen = np.asarray(assignment, dtype=np.int64)
    values = np.asarray(joints, dtype=np.float64)
    best = float(np.max(values))
    for row, value in zip(pool, values):
        if abs(float(value) - best) <= 1.0 and np.array_equal(chosen, np.asarray(row, dtype=np.int64)):
            return 1.0
    return 0.0


def evaluate(head, baseline, proposed, matching, states: list[dict], features: dict, actor) -> dict:
    rows = []
    for state in states:
        scenario = int(state["scenario_id"])
        moment = int(state["time_step"])
        encoded = features[id(state)]
        split = "test" if scenario == int(TEST_SCENARIO) else "train"
        joints = [float(item) for item in state["joint"]]
        pool = torch.as_tensor(state["assignments"], dtype=torch.long)
        mask = torch.as_tensor(state["candidate_mask"], dtype=torch.bool)
        with torch.no_grad():
            base_h = _baseline_embeddings(baseline, state["urgency_batch"])
            started = time.perf_counter()
            original = matching.decode(base_h[0], base_h[1])
            original_time = time.perf_counter() - started
            original_scores = matching.score_assignments(base_h[0], base_h[1], pool)
            started = time.perf_counter()
            phase5l = matching.decode(encoded["h_u"], encoded["h_b"])
            phase5l_time = time.perf_counter() - started
            phase5l_scores = matching.score_assignments(encoded["h_u"], encoded["h_b"], pool)
            started = time.perf_counter()
            _logits, plan, learned_scores = _score_state(head, encoded, state)
            learned = hungarian_from_plan(plan, mask)
            learned_time = time.perf_counter() - started
        named = {
            "original": (original, [float(item) for item in original_scores.detach().cpu().numpy()], original_time),
            "phase5l": (phase5l, [float(item) for item in phase5l_scores.detach().cpu().numpy()], phase5l_time),
            "exhaustive": (state["optimal"], joints, float(state["exhaustive_seconds"])),
            "phase5n": (learned, [float(item) for item in learned_scores.detach().cpu().numpy()], learned_time),
        }
        for name, (assignment, scores, seconds) in named.items():
            rows.append(
                {
                    "model": name,
                    "split": split,
                    "scenario_id": scenario,
                    "time_step": moment,
                    "mappo": _mappo(scenario, moment, np.asarray(assignment, dtype=np.int64), actor),
                    "spearman": _ranking(scores, joints),
                    "top1": _top1_assignment(assignment, state["assignments"], state["joint"]),
                    "inference_ms": float(seconds) * 1.0e3,
                    "assignment": [int(item) for item in np.asarray(assignment, dtype=np.int64)],
                }
            )
        print(
            f"eval {scenario} t={moment} phase5n_top1={rows[-1]['top1']} "
            f"phase5n_mappo={rows[-1]['mappo']:.3e} exhaustive_mappo={rows[-2]['mappo']:.3e}",
            flush=True,
        )
    summary = {}
    for name in ("original", "phase5l", "exhaustive", "phase5n"):
        selected = [row for row in rows if row["model"] == name]
        summary[name] = {}
        for split, subset in (
            ("test", [row for row in selected if row["split"] == "test"]),
            ("all", selected),
        ):
            summary[name][split] = {
                "n_states": float(len(subset)),
                "mappo": _mean([row["mappo"] for row in subset]),
                "spearman": _mean([row["spearman"] for row in subset]),
                "top1": _mean([row["top1"] for row in subset]),
                "inference_ms": _mean([row["inference_ms"] for row in subset]),
            }
        summary[name]["states"] = selected
    return summary


def _table(summary: dict, split: str) -> list[str]:
    lines = [
        "| model | joint MAPPO | Spearman vs J(A) | Top-1 | inference ms |",
        "| --- | --- | --- | --- | --- |",
    ]
    titles = (
        ("Original HGNN", "original"),
        ("Phase-5L HGNN", "phase5l"),
        ("穷尽搜索（极大匹配）", "exhaustive"),
        ("Phase-5N Joint Planner", "phase5n"),
    )
    for title, key in titles:
        item = summary[key][split]
        lines.append(
            f"| {title} | {_fmt(item['mappo'], '.4e')} | {_fmt(item['spearman'])} | {_fmt(item['top1'])} | {_fmt(item['inference_ms'], '.4f')} |"
        )
    return lines


def _paired_states(summary: dict) -> list[str]:
    learned = {(int(row["scenario_id"]), int(row["time_step"])): row for row in summary["phase5n"]["states"]}
    lines = [
        "| scenario | t | J(A) 是否可排序 | exhaustive MAPPO | Phase-5N MAPPO | Phase-5N Spearman | Phase-5N Top-1 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["exhaustive"]["states"]:
        key = (int(row["scenario_id"]), int(row["time_step"]))
        other = learned[key]
        ranked = "是" if np.isfinite(float(row["spearman"])) else "否，120 个分配的 J(A) 相同"
        lines.append(
            f"| {key[0]} | {key[1]} | {ranked} | {_fmt(row['mappo'], '.4e')} | {_fmt(other['mappo'], '.4e')} | "
            f"{_fmt(other['spearman'])} | {_fmt(other['top1'])} |"
        )
    return lines


def _report(summary: dict, training: dict) -> str:
    learned = summary["phase5n"]["test"]
    search = summary["exhaustive"]["test"]
    direct = summary["phase5l"]["test"]
    ranked_keys = [
        (int(row["scenario_id"]), int(row["time_step"]))
        for row in summary["exhaustive"]["states"]
        if row["split"] == "test" and np.isfinite(float(row["spearman"]))
    ]
    learned_rows = {(int(row["scenario_id"]), int(row["time_step"])): row for row in summary["phase5n"]["states"]}
    search_rows = {(int(row["scenario_id"]), int(row["time_step"])): row for row in summary["exhaustive"]["states"]}
    details = []
    reproduced = True
    for key in ranked_keys:
        got = learned_rows[key]
        target = search_rows[key]
        reproduced = reproduced and float(got["top1"]) >= 1.0 and float(got["mappo"]) + 1.0 >= float(target["mappo"])
        details.append(
            f"{key[0]} t={key[1]} 的 Phase-5N Top-1 是 {_fmt(got['top1'])}，Spearman 是 {_fmt(got['spearman'])}，"
            f"联合 MAPPO 是 {_fmt(got['mappo'], '.4e')}，穷尽搜索是 {_fmt(target['mappo'], '.4e')}。"
        )
    if ranked_keys and reproduced and float(learned["spearman"]) > float(direct["spearman"]):
        decision = "在 J(A) 有差异的测试时刻上，学习到的分配级规划器复现了穷尽搜索的最优分配。推理不调用 J(A) 的飞行展开。本阶段不进入 deployment verification。"
    else:
        detail = "".join(details) if details else "测试场景里没有 J(A) 有差异的决策。"
        decision = (
            f"分配级排序没有复现穷尽搜索。{detail}"
            f"310101 的平均 Top-1 是 {_fmt(learned['top1'])}，平均联合 MAPPO 是 {_fmt(learned['mappo'], '.4e')}，"
            f"穷尽搜索的平均联合 MAPPO 是 {_fmt(search['mappo'], '.4e')}。"
            "平均 MAPPO 更高来自 J(A) 完全相同的时刻：A* 只是并列中的第一个，另一个并列分配可以被 MAPPO 执行出更高载荷，但这不是对 J(A) 的更优排序。"
            "本阶段不进入 deployment verification。"
        )
    return "\n".join(
        [
            "# HTW Phase-5N Learnable Joint Planner",
            "",
            "**planner_ready = False**",
            "",
            "**deployment_verified = False**",
            "",
            "本阶段用可学习的 Joint Assignment Head 替换 Phase-5M 的搜索式精炼。Phase-5L encoder 和 Matching decoder 保持冻结，没有修改 Residual MAPPO、reward、低层策略或 deployment，没有调用 deployment `decide()`，没有使用 230000–230127。",
            "",
            "## 1. 结构",
            "",
            "Stage 1 仍由冻结的 Phase-5L HGNN 给出每个 UAV 的 top-3 候选。这六个 UAV 的 top-3 在所测时刻都是同一组浮标。Stage 2 是新的 Joint Assignment Head。它的输入是 UAV embedding、Buoy embedding 和二者的 global pooled embedding，输出候选配对分数。",
            "",
            "配对分数先做 Sinkhorn，得到 soft assignment，再做 Hungarian 解码。非候选格子不参与匹配；候选列用尽后，其余 UAV 记为 idle，不会被配到掩码外的浮标。",
            "",
            "监督集合是该 top-3 掩码里的全部极大互斥分配：只要某个 UAV 还有未被占用的自身候选，它就不能闲置。按 UAV 顺序填满后再把后面的 UAV 全部闲置，只覆盖了这个集合的一小部分，Hungarian 可以选出集合里的其他分配。穷尽搜索在这个完整集合上计算 J(A)，A* 是 J(A) 最大的那些分配。",
            "",
            "## 2. 分配级损失",
            "",
            "标签是上述集合上的穷尽搜索结果。损失比较的是完整分配的分数 S(A)，不是单个 UAV-Buoy pair。J(A) 全部相等的决策没有正样本对，损失为 0。",
            "",
            "当 J(A) > J(B) 时，损失是 `-log sigmoid(S(A) - S(B))`。S(A) 是该分配在 Sinkhorn soft plan 上的概率和。Hungarian 最大化的也是这个分配级分数，不是逐个 UAV-Buoy pair 的名次。",
            "",
            f"梯度只使用 {TRAIN_SCENARIO} 的 t=0、15、30。{TEST_SCENARIO} 不进入梯度和选模。最佳轮次是 {training['best_epoch']}，训练 Spearman 是 {_fmt(training['best_train_spearman'])}。",
            "",
            "## 3. 结果",
            "",
            "每个决策的候选掩码含 120 个极大匹配。推理时间是每次决策生成分配的耗时，不含 MAPPO 执行。穷尽搜索的时间是这 120 次 J(A) fly-toward 展开。Phase-5M 报告里的穷尽搜索只包含按 UAV 顺序填满的 6 个前缀分配，本表的耗时和 MAPPO 属于扩大后的极大匹配集合。",
            "",
            "### 测试场景 310101",
            "",
            *_table(summary, "test"),
            "",
            "### 全部决策时刻",
            "",
            *_table(summary, "all"),
            "",
            "Top-1 表示读出的分配是否属于 J(A) 最大的那一组极大匹配。J(A) 全相同时，120 个分配都是最优，Top-1 为 1 只说明解码留在掩码内。Spearman 使用处理并列的相关系数，并且只在 J(A) 和分数都不是常数的决策上平均。这样的决策只有 310100 t=30 和 310101 t=30。穷尽搜索的分数就是 J(A)，所以在这两个时刻它的 Spearman 为 1。",
            "",
            *_paired_states(summary),
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
    baseline = _load_baseline()
    proposed = _load_proposed()
    matching = _load_matching()
    for parameter in list(proposed.parameters()) + list(matching.parameters()) + list(baseline.parameters()):
        parameter.requires_grad_(False)
    labels = build_labels(proposed, matching)
    seed_everything(0)
    head = JointAssignmentHead()
    trained = train_head(head, proposed, matching, labels)
    features = trained.pop("features")
    from wt.lowlevel_v2.phase5g_closed_loop.collect import load_frozen_actor

    actor = load_frozen_actor()
    measured = evaluate(head, baseline, proposed, matching, labels, features, actor)
    learned_rows = {(int(row["scenario_id"]), int(row["time_step"])): row for row in measured["phase5n"]["states"]}
    ranked = [
        row
        for row in measured["exhaustive"]["states"]
        if row["split"] == "test" and np.isfinite(float(row["spearman"]))
    ]
    learned_gain = bool(ranked) and all(
        float(learned_rows[(int(row["scenario_id"]), int(row["time_step"]))]["top1"]) >= 1.0
        and float(learned_rows[(int(row["scenario_id"]), int(row["time_step"]))]["mappo"]) + 1.0 >= float(row["mappo"])
        and float(learned_rows[(int(row["scenario_id"]), int(row["time_step"]))]["spearman"])
        > float(measured["phase5l"]["test"]["spearman"])
        for row in ranked
    )
    summary = {
        "planner_ready": PLANNER_READY,
        "deployment_verified": DEPLOYMENT_VERIFIED,
        "decide_called": False,
        "encoder_frozen": True,
        "matching_decoder_frozen": True,
        "head_trained": True,
        "mappo_modified": False,
        "reward_modified": False,
        "deployment_modified": False,
        "learned_search_gain": learned_gain,
        "training": trained,
        "models": measured,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "training": trained}, OUTPUT / "joint_assignment_head.pt")
    (OUTPUT / "phase5n_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(_report(measured, trained), encoding="utf-8")
    print(json.dumps({"gain": learned_gain, "test": {key: measured[key]["test"] for key in measured}}, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    run()
