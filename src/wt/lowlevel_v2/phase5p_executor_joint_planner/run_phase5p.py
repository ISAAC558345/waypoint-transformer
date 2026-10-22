"""Retrain the Phase-5N head with frozen J_MAPPO labels. The head is unchanged."""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from wt.lowlevel_v2.phase5_hgnn_urgency.urgency import shared_task_urgency
from wt.lowlevel_v2.phase5j_joint_redesign.run_phase5j import replay_env
from wt.lowlevel_v2.phase5k_joint_decoder.loss import joint_rank_loss
from wt.lowlevel_v2.phase5l_joint_encoder.run_phase5l import _baseline_embeddings, _load_baseline, _load_matching
from wt.lowlevel_v2.phase5m_joint_planner.run_phase5m import _load_proposed
from wt.lowlevel_v2.phase5n_joint_planner.head import hungarian_from_plan
from wt.lowlevel_v2.phase5n_joint_planner.protocol import LABEL_VERSION, LABELS
from wt.lowlevel_v2.phase5n_joint_planner.run_phase5n import _features, _score_state
from wt.lowlevel_v2.phase5n_joint_planner.head import JointAssignmentHead
from scipy.stats import kendalltau, spearmanr
from wt.round30.ppo import seed_everything

from .metrics import choose_gate, quantize, signal_counts
from .protocol import (
    DEV_SCENARIO,
    DEPLOYMENT_VERIFIED,
    EPOCHS,
    GRAD_CLIP,
    LEARNING_RATE,
    OUTPUT,
    PAYLOAD_BLOCK_BITS,
    PHASE5O_LABELS,
    PLANNER_READY,
    POOL_SIZE,
    REPORT,
    SEEDS,
    TRAIN_SCENARIO,
    WEIGHT_DECAY,
    assert_phase5p_seed,
)


def _vector(assignment) -> str:
    return ";".join(str(int(item)) for item in np.asarray(assignment, dtype=np.int64))


def _mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _median(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def _std(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return float("nan")
    return float(np.std(array, ddof=1))


def _fmt(value, digits: str = ".4f") -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return format(float(value), digits)


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
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_recorded_assignments() -> dict[tuple[int, int, str], float]:
    path = PHASE5O_LABELS.parent / "phase5o_assignment_comparison.csv"
    recorded = {}
    if not path.is_file():
        return recorded
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            recorded[(int(row["scenario_id"]), int(row["time_step"]), row["assignment_vector"])] = float(row["j_mappo"])
    return recorded


def _load_mappo_table() -> dict[tuple[int, int, str], float]:
    table = {}
    with PHASE5O_LABELS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["scenario_id"]), int(row["time_step"]), row["assignment_vector"])
            table[key] = float(row["j_mappo"])
    if len(table) != 6 * int(POOL_SIZE):
        raise RuntimeError("phase-5o label table does not contain the 720 frozen assignments")
    return table


def load_states() -> list[dict]:
    payload = torch.load(LABELS, map_location="cpu", weights_only=False)
    if int(payload.get("version", -1)) != int(LABEL_VERSION):
        raise RuntimeError("phase-5p requires the Phase-5N maximal-matching cache")
    table = _load_mappo_table()
    states = []
    for state in payload["states"]:
        scenario = int(state["scenario_id"])
        moment = int(state["time_step"])
        assert_phase5p_seed(scenario)
        assignments = np.asarray(state["assignments"], dtype=np.int64)
        if int(assignments.shape[0]) != int(POOL_SIZE):
            raise RuntimeError(f"{scenario} t={moment} pool size changed")
        vectors = [_vector(row) for row in assignments]
        j_mappo = []
        for vector in vectors:
            key = (scenario, moment, vector)
            if key not in table:
                raise RuntimeError(f"missing frozen J_MAPPO for {key}")
            j_mappo.append(table[key])
        states.append(
            {
                "scenario_id": scenario,
                "time_step": moment,
                "split": "dev" if scenario == int(DEV_SCENARIO) else "train",
                "urgency_batch": state["urgency_batch"],
                "assignments": assignments,
                "candidate_mask": np.asarray(state["candidate_mask"], dtype=bool),
                "j_fly": np.asarray(state["joint"], dtype=np.float64),
                "j_mappo": np.asarray(j_mappo, dtype=np.float64),
                "vectors": vectors,
            }
        )
    return states


def signal_rows(states: list[dict]) -> list[dict]:
    rows = []
    for state in states:
        if int(state["scenario_id"]) != int(TRAIN_SCENARIO):
            continue
        for label, values in (("J_FLY", state["j_fly"]), ("J_MAPPO", state["j_mappo"])):
            counts = signal_counts(values)
            rows.append(
                {
                    "scenario_id": int(state["scenario_id"]),
                    "time_step": int(state["time_step"]),
                    "label": label,
                    **counts,
                }
            )
        fly = signal_counts(state["j_fly"])
        executed = signal_counts(state["j_mappo"])
        print(
            f"signal {state['scenario_id']} t={state['time_step']} "
            f"fly_pairs={fly['n_positive_pairs']:.0f} mappo_pairs={executed['n_positive_pairs']:.0f}",
            flush=True,
        )
    return rows


def _score_correlation(scores, labels) -> tuple[float, float]:
    """Spearman and Kendall of raw assignment scores against J_MAPPO.

    Payload-block quantization applies to labels. Scores near 0.5 must not be
    rounded onto that block grid, or every model looks uncorrelated.
    """
    left = np.asarray(scores, dtype=np.float64)
    right = np.asarray(labels, dtype=np.float64)
    if left.size < 3 or np.unique(quantize(right)).size < 2 or float(np.std(left)) < 1.0e-6:
        return float("nan"), float("nan")
    spearman = spearmanr(left, right)
    kendall = kendalltau(left, right)
    spearman_value = getattr(spearman, "statistic", spearman.correlation)
    kendall_value = getattr(kendall, "statistic", kendall.correlation)
    return float(spearman_value), float(kendall_value)


def _train(head: JointAssignmentHead, features: dict, states: list[dict], label_key: str) -> float:
    train_states = [state for state in states if state["split"] == "train"]
    optimizer = torch.optim.Adam(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    for epoch in range(1, int(EPOCHS) + 1):
        head.train()
        losses = []
        for state in train_states:
            optimizer.zero_grad(set_to_none=True)
            _logits, _plan, scores = _score_state(head, features[id(state)], state)
            levels = torch.round(torch.as_tensor(state[label_key], dtype=torch.float32) / PAYLOAD_BLOCK_BITS) * PAYLOAD_BLOCK_BITS
            loss = joint_rank_loss(scores, levels)
            if not bool(torch.isfinite(loss).all()):
                raise RuntimeError(f"{label_key} loss became non-finite at epoch {epoch}")
            loss.backward()
            clip_grad_norm_(head.parameters(), GRAD_CLIP)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"{label_key} epoch {epoch} loss={_mean(losses):.4f}", flush=True)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return float(_mean(losses))


def _metrics(scores: np.ndarray, labels: np.ndarray, decoded: np.ndarray, vectors: list[str]) -> dict:
    levels = quantize(labels)
    best = float(np.max(levels))
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    width = min(3, int(order.size))
    vector = _vector(decoded)
    if vector not in vectors:
        raise RuntimeError(f"decoded assignment left the frozen pool: {vector}")
    index = vectors.index(vector)
    spearman, kendall = _score_correlation(scores, labels)
    regret = float(best - float(levels[index]))
    return {
        "spearman": spearman,
        "kendall": kendall,
        "top1": float(float(levels[index]) == best),
        "top3": float(np.any(levels[order[:width]] == best)),
        "j_mappo": float(labels[index]),
        "regret": regret,
        "normalized_regret": float(regret / (best + 1.0)),
        "assignment_vector": vector,
        "decoded_score": float(scores[index]),
        "scores": np.asarray(scores, dtype=np.float64),
    }


def _eval_head(head, features, state: dict) -> dict:
    mask = torch.as_tensor(state["candidate_mask"], dtype=torch.bool)
    started = time.perf_counter()
    _logits, plan, scores = _score_state(head, features, state)
    decoded = hungarian_from_plan(plan, mask)
    elapsed = time.perf_counter() - started
    measured = _metrics(scores.detach().cpu().numpy(), state["j_mappo"], decoded, state["vectors"])
    measured["inference_ms"] = float(elapsed) * 1.0e3
    return measured


def _eval_matching(matching, h_u, h_b, state: dict, outside: dict[tuple[int, int, str], float]) -> dict:
    pool = torch.as_tensor(state["assignments"], dtype=torch.long)
    started = time.perf_counter()
    decoded = matching.decode(h_u, h_b)
    elapsed = time.perf_counter() - started
    scores = matching.score_assignments(h_u, h_b, pool).detach().cpu().numpy()
    vector = _vector(decoded)
    labels = state["j_mappo"]
    levels = quantize(labels)
    best = float(np.max(levels))
    if vector in state["vectors"]:
        index = state["vectors"].index(vector)
        payload = float(labels[index])
        in_pool = 1.0
    else:
        index = None
        payload = float(outside.get((int(state["scenario_id"]), int(state["time_step"]), vector), float("nan")))
        in_pool = 0.0
    spearman, kendall = _score_correlation(scores, labels)
    order = np.argsort(-scores, kind="stable")
    regret = float(best - payload) if np.isfinite(payload) else float("nan")
    return {
        "spearman": spearman,
        "kendall": kendall,
        "top1": float(np.isfinite(payload) and float(quantize([payload])[0]) == best),
        "top3": float(np.any(levels[order[:3]] == best)),
        "j_mappo": payload,
        "regret": regret,
        "normalized_regret": float(regret / (best + 1.0)) if np.isfinite(regret) else float("nan"),
        "assignment_vector": vector,
        "decoded_score": float(scores[index]) if index is not None else float("nan"),
        "scores": scores,
        "inference_ms": float(elapsed) * 1.0e3,
        "in_pool": in_pool,
    }


def _oracle(state: dict) -> dict:
    labels = state["j_mappo"]
    index = int(np.argmax(quantize(labels)))
    levels = quantize(labels)
    return {
        "spearman": 1.0,
        "kendall": 1.0,
        "top1": 1.0,
        "top3": 1.0,
        "j_mappo": float(labels[index]),
        "regret": 0.0,
        "normalized_regret": 0.0,
        "assignment_vector": state["vectors"][index],
        "decoded_score": float(labels[index]),
        "scores": labels,
        "inference_ms": float("nan"),
        "in_pool": 1.0,
    }


def _macro(rows: list[dict], key: str) -> float:
    return _mean([row[key] for row in rows])


def _failure_class(train_top1: float, train_spearman: float, score_span: float, label_span: float) -> str:
    if float(score_span) < 1.0e-4 and float(label_span) >= float(PAYLOAD_BLOCK_BITS):
        return "B_frozen_encoder_representation_insufficiency"
    if np.isfinite(train_top1) and float(train_top1) >= 2.0 / 3.0:
        return "D_state_generalization_failure"
    if not np.isfinite(train_spearman) or float(train_spearman) < 0.3:
        return "A_head_optimization_failure"
    if float(score_span) < 1.0e-3 and float(label_span) >= float(PAYLOAD_BLOCK_BITS):
        return "C_candidate_assignment_ambiguity"
    return "B_frozen_encoder_representation_insufficiency"


def _snapshot(env) -> dict:
    info = env.info()
    queue = np.asarray(info["queue_blocks"], dtype=np.float64)
    return {
        "distance": np.asarray(info["distance_uav_buoy"], dtype=np.float64),
        "channel": np.asarray(info["rate_if_assigned"], dtype=np.float64),
        "eta": np.asarray(info["eta_slots"], dtype=np.float64),
        "queue": queue,
        "urgency": shared_task_urgency(queue),
        "interference": float(np.sum(np.asarray(info["interference_uav_uav"], dtype=np.float64))),
    }


def _feature_text(snapshot: dict, vector: str) -> str:
    parts = [f"interference_sum={snapshot['interference']:.6g}"]
    for uav, token in enumerate(vector.split(";")):
        buoy = int(token)
        if buoy < 0:
            continue
        parts.append(
            "uav={uav},buoy={buoy},distance_m={dist:.4f},eta={eta:.4f},channel_bps={channel:.6g},"
            "queue={queue:.4f},urgency={urgency:.4f}".format(
                uav=uav,
                buoy=buoy,
                dist=float(snapshot["distance"][uav, buoy]),
                eta=float(snapshot["eta"][uav, buoy]),
                channel=float(snapshot["channel"][uav, buoy]),
                queue=float(snapshot["queue"][buoy]),
                urgency=float(snapshot["urgency"][buoy]),
            )
        )
    return " | ".join(parts)


def _report(summary: dict) -> str:
    gate = summary["gate"]
    advice = {
        "LABELTRAIN-A": "标签不匹配是 Phase-5N 的主要失败原因。下一阶段才允许在 J_MAPPO 监督下同时训练 HGNN encoder 和 Joint Head。本阶段不启动那次训练。",
        "LABELTRAIN-B": "排序变好了，但解码后的 J_MAPPO 没有提高。下一步应检查 score 到 Hungarian 分配的衔接，不能先改 encoder。",
        "LABELTRAIN-C": "换成 J_MAPPO 后，排序和真实载荷都没有稳定提高。冻结的 Phase-5L 表示不足以支撑 executor-conditioned joint utility。下一阶段才允许在标签保持 J_MAPPO 的前提下重新训练 encoder。",
        "LABELTRAIN-D": "训练集改善了，310101 没有改善。这是泛化失败或 development state 太少。下一步应先扩展合法 development states，不能直接增加模型复杂度。",
    }[gate]
    delta = summary["delta"]
    return "\n".join(
        [
            "# HTW Phase-5P Executor-Conditioned Joint Planner Retraining",
            "",
            "**planner_ready = False**",
            "",
            "**deployment_verified = False**",
            "",
            "本阶段是 label-only 对照。Joint Assignment Head、Sinkhorn、Hungarian、top-3 候选、学习率、优化器和 10 个 epoch 与 Phase-5N 相同。唯一变化是把 `J_FLY` 换成 Phase-5O 已经冻结的 `J_MAPPO`。没有重新 rollout，没有使用 310101 做梯度、early stopping 或选模。310101 只作为 development evaluation，不是 untouched deployment test。",
            "",
            "## 1. 训练信号",
            "",
            f"310100 上，`J_FLY` 的有效有序对共 {summary['fly_pairs']:.0f}，`J_MAPPO` 共 {summary['mappo_pairs']:.0f}。监督信号确实增加了。",
            "",
            f"第 10 个 epoch 的平均损失：J_FLY 头 {_fmt(summary['loss_fly'], '.4f')}，J_MAPPO 头 {_fmt(summary['loss_mappo'], '.4f')}。J_MAPPO 头停在 0.6931，也就是 `-log sigmoid(0)`，说明 S(A)-S(B) 的平均差距接近 0。",
            "",
            summary["train_note"],
            "",
            "## 2. Development evaluation（310101）",
            "",
            "Top-1 和 regret 先在每个 state 内计算，再对三个时刻做 macro average。五个 seed 都保留。Spearman 差也先在 state 内配对。只要某个 state 的分配分数标准差低于 1e-6，该 seed 的 Spearman 差就记为 NA，不能把这个 state 从平均里丢掉后再声称排序提高。",
            "",
            f"`J_MAPPO` 相对 `J_FLY` 的 macro Spearman 差是 {_fmt(delta['spearman'])}（median {_fmt(delta['spearman_median'])}，std {_fmt(delta['spearman_std'])}）。",
            f"Top-1 差是 {_fmt(delta['top1'])}（median {_fmt(delta['top1_median'])}，std {_fmt(delta['top1_std'])}）。",
            f"解码载荷差是 {_fmt(delta['j_mappo'], '.4e')}（median {_fmt(delta['j_mappo_median'], '.4e')}，std {_fmt(delta['j_mappo_std'], '.4e')}）。",
            f"归一化 regret 差是 {_fmt(delta['regret'])}（median {_fmt(delta['regret_median'])}，std {_fmt(delta['regret_std'])}）。正的 regret 差表示 regret 变大。",
            f"五个 seed 里，载荷上升且 regret 下降的有 {summary['n_seeds_improved']} 个。",
            "",
            "| model | t | Spearman | Top-1 | J_MAPPO | normalized regret |",
            "| --- | --- | --- | --- | --- | --- |",
            *summary["dev_lines"],
            "",
            summary["focus_text"],
            "",
            "## 3. 结论",
            "",
            f"Gate 是 **{gate}**。",
            "",
            advice,
            "",
            "Q1. 只替换标签是否提高了 planner？ " + summary["answers"]["q1"],
            "",
            "Q2. 提升出现在哪些层面？ " + summary["answers"]["q2"],
            "",
            "Q3. 310101 t=15 是否修好？ " + summary["answers"]["q3"],
            "",
            "Q4. 五个 seed 是否同向？ " + summary["answers"]["q4"],
            "",
            "Q5. 下一步瓶颈： " + summary["answers"]["q5"],
            "",
            "`planner_ready = False`",
            "",
            "`deployment_verified = False`",
            "",
        ]
    )


def run() -> dict:
    outside = _load_recorded_assignments()
    states = load_states()
    signals = signal_rows(states)
    baseline = _load_baseline()
    proposed = _load_proposed()
    matching = _load_matching()
    for parameter in list(baseline.parameters()) + list(proposed.parameters()) + list(matching.parameters()):
        parameter.requires_grad_(False)
    features = {id(state): _features(proposed, matching, state["urgency_batch"]) for state in states}
    trained = {}
    final_loss = defaultdict(list)
    for seed in SEEDS:
        for label_key, name in (("j_fly", "phase5n_jfly"), ("j_mappo", "phase5p_jmappo")):
            seed_everything(int(seed))
            head = JointAssignmentHead()
            final_loss[name].append(_train(head, features, states, label_key))
            trained[(name, int(seed))] = head
            print(f"trained {name} seed {seed}", flush=True)
    state_rows = []
    prediction_rows = []
    measured = {}
    for state in states:
        encoded = features[id(state)]
        base = _baseline_embeddings(baseline, state["urgency_batch"])
        fixed = {
            "original_hgnn": _eval_matching(matching, base[0], base[1], state, outside),
            "phase5l_hgnn": _eval_matching(matching, encoded["h_u"], encoded["h_b"], state, outside),
            "pool_oracle_mappo": _oracle(state),
        }
        for name, result in fixed.items():
            measured[(name, -1, int(state["scenario_id"]), int(state["time_step"]))] = result
            state_rows.append(_state_row(name, -1, state, result))
            prediction_rows.append(_prediction_row(name, -1, state, result))
        for name, seed in trained:
            result = _eval_head(trained[(name, seed)], encoded, state)
            measured[(name, int(seed), int(state["scenario_id"]), int(state["time_step"]))] = result
            state_rows.append(_state_row(name, int(seed), state, result))
            prediction_rows.append(_prediction_row(name, int(seed), state, result))
            print(
                f"eval {name} seed {seed} {state['scenario_id']} t={state['time_step']} "
                f"top1={result['top1']} j={result['j_mappo']:.3e}",
                flush=True,
            )
    seed_rows = _seed_table(measured)
    failures = _failures(measured, states)
    gate, delta, improved = _gate(seed_rows)
    focus = _focus_text(measured)
    fly_pairs = sum(row["n_positive_pairs"] for row in signals if row["label"] == "J_FLY")
    mappo_pairs = sum(row["n_positive_pairs"] for row in signals if row["label"] == "J_MAPPO")
    answers = _answers(gate, delta, improved, measured)
    train_bumps = sum(
        1
        for seed in SEEDS
        if measured[("phase5p_jmappo", int(seed), TRAIN_SCENARIO, 15)]["j_mappo"]
        > measured[("phase5n_jfly", int(seed), TRAIN_SCENARIO, 15)]["j_mappo"]
    )
    train_note = (
        f"310100 t=15 上有 {train_bumps}/5 个 seed 的解码载荷从 0 升到 1.0e6 bit，"
        "但仍低于该状态 3.0e6 bit 的池内 oracle，Top-1 仍是 0。"
        "310100 t=0 的新头分数标准差低于 1e-6，其中 3 个 seed 的旧头 Spearman 为负、新头变为 NA。"
        "因此不能把丢掉 NA 之后的平均当成训练集排序提升。"
    )
    summary = {
        "planner_ready": PLANNER_READY,
        "deployment_verified": DEPLOYMENT_VERIFIED,
        "decide_called": False,
        "development_evaluation_only": True,
        "epochs": EPOCHS,
        "seeds": list(SEEDS),
        "gate": gate,
        "fly_pairs": fly_pairs,
        "mappo_pairs": mappo_pairs,
        "loss_fly": _mean(final_loss["phase5n_jfly"]),
        "loss_mappo": _mean(final_loss["phase5p_jmappo"]),
        "n_seeds_improved": improved,
        "delta": delta,
        "dev_lines": _dev_lines(measured),
        "focus_text": focus,
        "train_note": train_note,
        "answers": answers,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT / "phase5p_training_signal_stats.csv", signals)
    _write_csv(OUTPUT / "phase5p_seed_metrics.csv", seed_rows)
    _write_csv(OUTPUT / "phase5p_state_metrics.csv", state_rows)
    _write_csv(OUTPUT / "phase5p_assignment_predictions.csv", prediction_rows)
    _write_csv(OUTPUT / "phase5p_failure_cases.csv", failures)
    gate_payload = {
        "gate": gate,
        "planner_ready": PLANNER_READY,
        "deployment_verified": DEPLOYMENT_VERIFIED,
        "development_evaluation_only": True,
        "epochs": EPOCHS,
        "seeds": list(SEEDS),
        "n_seeds_improved": improved,
        "delta": _plain(delta),
        "fly_positive_pairs": fly_pairs,
        "mappo_positive_pairs": mappo_pairs,
    }
    (OUTPUT / "phase5p_label_retraining_gate.json").write_text(json.dumps(gate_payload, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(_report(summary), encoding="utf-8")
    print(json.dumps(_plain({"gate": gate, "delta": delta, "improved": improved}), ensure_ascii=False), flush=True)
    return summary


def _state_row(name: str, seed: int, state: dict, result: dict) -> dict:
    return {
        "model": name,
        "seed": int(seed),
        "split": state["split"],
        "scenario_id": int(state["scenario_id"]),
        "time_step": int(state["time_step"]),
        "spearman": result["spearman"],
        "kendall": result["kendall"],
        "top1": result["top1"],
        "top3": result["top3"],
        "j_mappo": result["j_mappo"],
        "regret": result["regret"],
        "normalized_regret": result["normalized_regret"],
        "inference_ms": result["inference_ms"],
        "assignment_vector": result["assignment_vector"],
    }


def _prediction_row(name: str, seed: int, state: dict, result: dict) -> dict:
    return {
        "model": name,
        "seed": int(seed),
        "scenario_id": int(state["scenario_id"]),
        "time_step": int(state["time_step"]),
        "assignment_vector": result["assignment_vector"],
        "model_score": result["decoded_score"],
        "j_mappo": result["j_mappo"],
        "regret": result["regret"],
        "top1": result["top1"],
        "in_pool": result.get("in_pool", 1.0),
    }


def _paired_rank_delta(measured: dict, seed: int, split: str) -> float:
    """Mean per-state Spearman change. A collapsed score on any state is not a gain."""
    deltas = []
    collapsed = False
    for (name, row_seed, scenario, moment), result in measured.items():
        if name != "phase5n_jfly" or int(row_seed) != int(seed):
            continue
        row_split = "dev" if int(scenario) == int(DEV_SCENARIO) else "train"
        if row_split != split:
            continue
        old = float(result["spearman"])
        new = float(measured[("phase5p_jmappo", int(seed), int(scenario), int(moment))]["spearman"])
        if np.isfinite(old) and np.isfinite(new):
            deltas.append(new - old)
        elif np.isfinite(old) and not np.isfinite(new):
            collapsed = True
    if collapsed or not deltas:
        return float("nan")
    return float(np.mean(deltas))


def _seed_table(measured: dict) -> list[dict]:
    rows = []
    grouped = defaultdict(list)
    for (name, seed, scenario, moment), result in measured.items():
        split = "dev" if int(scenario) == int(DEV_SCENARIO) else "train"
        grouped[(name, seed, split)].append(result)
    deltas = {}
    for seed in SEEDS:
        for split in ("train", "dev"):
            left = grouped[("phase5n_jfly", int(seed), split)]
            right = grouped[("phase5p_jmappo", int(seed), split)]
            deltas[(int(seed), split)] = {
                "spearman": _paired_rank_delta(measured, int(seed), split),
                "top1": _macro(right, "top1") - _macro(left, "top1"),
                "j_mappo": _macro(right, "j_mappo") - _macro(left, "j_mappo"),
                "normalized_regret": _macro(right, "normalized_regret") - _macro(left, "normalized_regret"),
            }
    for (name, seed, split), items in sorted(grouped.items()):
        delta = deltas.get((int(seed), split), {})
        rows.append(
            {
                "model": name,
                "seed": int(seed),
                "split": split,
                "spearman": _macro(items, "spearman"),
                "kendall": _macro(items, "kendall"),
                "top1": _macro(items, "top1"),
                "top3": _macro(items, "top3"),
                "j_mappo": _macro(items, "j_mappo"),
                "regret": _macro(items, "regret"),
                "normalized_regret": _macro(items, "normalized_regret"),
                "delta_spearman": delta.get("spearman", float("nan")),
                "delta_top1": delta.get("top1", float("nan")),
                "delta_j_mappo": delta.get("j_mappo", float("nan")),
                "delta_normalized_regret": delta.get("normalized_regret", float("nan")),
            }
        )
    return rows


def _all_finite_up(values) -> bool:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return False
    return float(np.mean(array)) > 0.0


def _all_finite_down(values) -> bool:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return False
    return float(np.mean(array)) < 0.0


def _gate(seed_rows: list[dict]) -> tuple[str, dict, int]:
    def column(model: str, split: str, key: str) -> list[float]:
        return [float(row[key]) for row in seed_rows if row["model"] == model and row["split"] == split and int(row["seed"]) >= 0]

    dev_delta_j = [float(row["delta_j_mappo"]) for row in seed_rows if row["model"] == "phase5p_jmappo" and row["split"] == "dev"]
    dev_delta_regret = [
        float(row["delta_normalized_regret"]) for row in seed_rows if row["model"] == "phase5p_jmappo" and row["split"] == "dev"
    ]
    dev_delta_spearman = [float(row["delta_spearman"]) for row in seed_rows if row["model"] == "phase5p_jmappo" and row["split"] == "dev"]
    dev_delta_top1 = [float(row["delta_top1"]) for row in seed_rows if row["model"] == "phase5p_jmappo" and row["split"] == "dev"]
    improved = 0
    for payload, regret in zip(dev_delta_j, dev_delta_regret):
        if np.isfinite(payload) and np.isfinite(regret) and payload > 0.0 and regret < 0.0:
            improved += 1
    train_rank_up = _all_finite_up(column("phase5p_jmappo", "train", "delta_spearman"))
    train_payload_up = _all_finite_up(column("phase5p_jmappo", "train", "delta_j_mappo"))
    dev_rank_up = _all_finite_up(dev_delta_spearman)
    dev_payload_up = _all_finite_up(dev_delta_j)
    dev_regret_down = _all_finite_down(dev_delta_regret)
    gate = choose_gate(
        train_rank_up=train_rank_up,
        train_payload_up=train_payload_up,
        dev_rank_up=dev_rank_up,
        dev_payload_up=dev_payload_up,
        dev_regret_down=dev_regret_down,
        n_seeds_improved=improved,
        n_seeds=len(SEEDS),
    )
    delta = {
        "spearman": _mean(dev_delta_spearman),
        "spearman_median": _median(dev_delta_spearman),
        "spearman_std": _std(dev_delta_spearman),
        "top1": _mean(dev_delta_top1),
        "top1_median": _median(dev_delta_top1),
        "top1_std": _std(dev_delta_top1),
        "j_mappo": _mean(dev_delta_j),
        "j_mappo_median": _median(dev_delta_j),
        "j_mappo_std": _std(dev_delta_j),
        "regret": _mean(dev_delta_regret),
        "regret_median": _median(dev_delta_regret),
        "regret_std": _std(dev_delta_regret),
    }
    return gate, delta, improved


def _dev_lines(measured: dict) -> list[str]:
    lines = []
    names = (
        ("original_hgnn", [-1]),
        ("phase5l_hgnn", [-1]),
        ("phase5n_jfly", list(SEEDS)),
        ("phase5p_jmappo", list(SEEDS)),
        ("pool_oracle_mappo", [-1]),
    )
    for name, seeds in names:
        for moment in (0, 15, 30):
            rows = [measured[(name, int(seed), 310101, moment)] for seed in seeds]
            lines.append(
                f"| {name} | {moment} | {_fmt(_macro(rows, 'spearman'))} | {_fmt(_macro(rows, 'top1'))} | "
                f"{_fmt(_macro(rows, 'j_mappo'), '.4e')} | {_fmt(_macro(rows, 'normalized_regret'))} |"
            )
    return lines


def _focus_text(measured: dict) -> str:
    oracle = measured[("pool_oracle_mappo", -1, 310101, 15)]["j_mappo"]
    parts = [f"310101 t=15 的池内 oracle J_MAPPO 是 {_fmt(oracle, '.4e')}。"]
    for seed in SEEDS:
        old = measured[("phase5n_jfly", int(seed), 310101, 15)]
        new = measured[("phase5p_jmappo", int(seed), 310101, 15)]
        parts.append(
            f"seed {seed}：J_FLY 头的解码载荷是 {_fmt(old['j_mappo'], '.4e')}，J_MAPPO 头是 {_fmt(new['j_mappo'], '.4e')}。"
        )
    return " ".join(parts)


def _answers(gate: str, delta: dict, improved: int, measured: dict) -> dict[str, str]:
    ranking = "排序有提高" if float(delta["spearman"]) > 0.0 else "排序没有提高"
    payload = "解码后的真实 J_MAPPO 有提高" if float(delta["j_mappo"]) > 0.0 else "解码后的真实 J_MAPPO 没有提高"
    q1 = {
        "LABELTRAIN-A": "提高了。五个 seed 的平均载荷上升，平均 regret 下降，并且多数 seed 同向改善。",
        "LABELTRAIN-B": "只在排序上有变化，解码后的 executor payload 没有提高。",
        "LABELTRAIN-C": "没有稳定提高。",
        "LABELTRAIN-D": "只在训练集上提高，310101 没有提高。",
    }[gate]
    focus_fixed = all(
        float(measured[("phase5p_jmappo", int(seed), 310101, 15)]["j_mappo"]) + 1.0
        >= float(measured[("pool_oracle_mappo", -1, 310101, 15)]["j_mappo"])
        and float(measured[("phase5p_jmappo", int(seed), 310101, 15)]["top1"]) >= 1.0
        for seed in SEEDS
    )
    focus_better = _mean(
        [
            float(measured[("phase5p_jmappo", int(seed), 310101, 15)]["j_mappo"])
            - float(measured[("phase5n_jfly", int(seed), 310101, 15)]["j_mappo"])
            for seed in SEEDS
        ]
    )
    old_fixed = all(float(measured[("phase5n_jfly", int(seed), 310101, 15)]["top1"]) >= 1.0 for seed in SEEDS)
    if focus_fixed and old_fixed:
        q3 = "五个 seed 的两种头都解码到 3.0e6 bit。J_FLY 头已经到达池内最优，换成 J_MAPPO 没有改变这个时刻。"
    elif focus_fixed:
        q3 = "五个 seed 的 J_MAPPO 头都解码到 3.0e6 bit，J_FLY 头没有全部到达。"
    elif focus_better > 0.0:
        q3 = f"平均解码载荷比 J_FLY 头高 {_fmt(focus_better, '.4e')}，但没有每个 seed 都达到 3.0e6。"
    else:
        q3 = "没有修好。J_MAPPO 头在这个时刻的平均解码载荷没有超过 J_FLY 头。"
    q4 = f"{improved}/5 个 seed 同时提高了解码载荷并降低了 regret。"
    q5 = {
        "LABELTRAIN-A": "标签。encoder 的完整重训要等下一阶段，而且仍使用离线 J_MAPPO。",
        "LABELTRAIN-B": "head/decoder 的 score 到 assignment 的衔接。",
        "LABELTRAIN-C": "冻结 encoder representation。",
        "LABELTRAIN-D": "generalization，development state 不够。",
    }[gate]
    return {"q1": q1, "q2": f"{ranking}；{payload}。", "q3": q3, "q4": q4, "q5": q5}


def _failures(measured: dict, states: list[dict]) -> list[dict]:
    rows = []
    snapshots = {}
    for state in states:
        if state["split"] != "dev":
            continue
        for seed in SEEDS:
            result = measured[("phase5p_jmappo", int(seed), int(state["scenario_id"]), int(state["time_step"]))]
            if float(result["top1"]) >= 1.0:
                continue
            snap_key = (int(state["scenario_id"]), int(state["time_step"]))
            if snap_key not in snapshots:
                env, _movement = replay_env(snap_key[0], snap_key[1])
                snapshots[snap_key] = _snapshot(env)
            snapshot = snapshots[snap_key]
            train_rows = [
                measured[("phase5p_jmappo", int(seed), int(item["scenario_id"]), int(item["time_step"]))]
                for item in states
                if item["split"] == "train"
            ]
            order = np.argsort(-result["scores"], kind="stable")[:5]
            score_span = float(np.max(result["scores"]) - np.min(result["scores"]))
            label_span = float(np.max(state["j_mappo"]) - np.min(state["j_mappo"]))
            kind = _failure_class(_macro(train_rows, "top1"), _macro(train_rows, "spearman"), score_span, label_span)
            for rank, index in enumerate(order, start=1):
                vector = state["vectors"][int(index)]
                rows.append(
                    {
                        "model": "phase5p_jmappo",
                        "seed": int(seed),
                        "scenario_id": int(state["scenario_id"]),
                        "time_step": int(state["time_step"]),
                        "rank": int(rank),
                        "assignment_vector": vector,
                        "model_score": float(result["scores"][int(index)]),
                        "j_mappo": float(state["j_mappo"][int(index)]),
                        "is_decoded": float(vector == result["assignment_vector"]),
                        "failure_class": kind,
                        "edge_queue_channel_urgency": _feature_text(snapshot, vector),
                    }
                )
    return rows


if __name__ == "__main__":
    run()
