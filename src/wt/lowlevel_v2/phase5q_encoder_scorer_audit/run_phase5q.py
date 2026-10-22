"""Train either the existing encoder or a non-additive scorer, never both."""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from wt.htw_phase9.const import BUOY_DIM
from wt.lowlevel_v2.phase5f_utility_rank.protocol import FORBIDDEN_INPUTS
from wt.lowlevel_v2.phase5k_joint_decoder.loss import joint_rank_loss
from wt.lowlevel_v2.phase5l_joint_encoder.model import JointUtilityHGNN, append_queue_edge, batch_tensors
from wt.lowlevel_v2.phase5l_joint_encoder.run_phase5l import _load_matching
from wt.lowlevel_v2.phase5m_joint_planner.run_phase5m import _load_proposed
from wt.lowlevel_v2.phase5n_joint_planner.head import JointAssignmentHead, global_embedding, hungarian_from_plan
from wt.lowlevel_v2.phase5n_joint_planner.run_phase5n import _features, _score_state
from wt.lowlevel_v2.phase5p_executor_joint_planner.run_phase5p import _eval_head, _train, _vector, load_states
from wt.round30.ppo import seed_everything

from .metrics import (
    _targets,
    choose_gate,
    design_matrix,
    fit_linear,
    metrics_at,
    pairwise_beats_additive,
    quantize,
    regression_metrics,
    stable_improvement,
    train_fit,
)
from .protocol import (
    DEV_SCENARIO,
    DEPLOYMENT_VERIFIED,
    EPOCHS,
    GRAD_CLIP,
    LEARNING_RATE,
    MODEL_A,
    MODEL_B,
    MODEL_C,
    OUTPUT,
    PAYLOAD_BLOCK_BITS,
    PHASE5P_PREDICTIONS,
    PHASE5P_RECORDED_LOSS,
    PHASE5P_STATE,
    PLANNER_READY,
    POOL_SIZE,
    REPORT,
    SEEDS,
    TRAIN_SCENARIO,
    WEIGHT_DECAY,
    assert_phase5q_seed,
)
from .scorer import AssignmentConditionedSetScorer


ADVICE = {
    "Q-A": "Model-B 在 310101 上稳定优于 Model-A，Model-C 没有稳定改善。瓶颈在 encoder representation。下一阶段可以保持现有 Joint Head，研究 J_MAPPO 监督下的 HGNN 表示。",
    "Q-B": "Model-C 稳定优于 Model-A，Model-B 没有稳定改善，而且 pairwise audit 显著优于 additive。瓶颈在 assignment scorer 的表达能力。下一阶段应保留冻结 encoder，开发可扩展的非加性 joint scorer。",
    "Q-C": "Model-B 和 Model-C 都有稳定的独立改善。下一阶段才允许同时训练 HGNN 和非加性 scorer。本阶段没有做这个组合。",
    "Q-D": "模型能在 310100 上形成训练信号，但 310101 没有稳定改善。瓶颈是 development state 太少。下一步应扩展合法 development states，不能继续增加网络复杂度。",
    "Q-E": "Model-B 和 Model-C 都没有在 310100 上同时降低 ranking loss 并解除 score collapse，development evaluation 也没有达到稳定改善线。下一步应检查标签、assignment 表示和优化实现，不能继续堆模型。",
}


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
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
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


def _check_batch(batch: dict) -> None:
    leaked = set(batch).intersection(FORBIDDEN_INPUTS)
    if leaked or "j_mappo" in batch or "joint" in batch:
        raise RuntimeError(f"model input contains a label field: {sorted(leaked)}")


def _levels(state: dict) -> torch.Tensor:
    values = torch.as_tensor(state["j_mappo"], dtype=torch.float32)
    return torch.round(values / PAYLOAD_BLOCK_BITS) * PAYLOAD_BLOCK_BITS


def _edge_features(model: JointUtilityHGNN, batch: dict) -> dict[str, torch.Tensor]:
    _check_batch(batch)
    model.eval()
    with torch.no_grad():
        conditioned = append_queue_edge(batch)
        tensors = batch_tensors(conditioned)
        h_u, h_b = model.condition(tensors)
        return {
            "h_u": h_u.detach(),
            "h_b": h_b.detach(),
            "z_global": global_embedding(h_u, h_b, tensors["uav_mask"], tensors["buoy_mask"]).detach(),
            "h_edge": model.encoder.edge_enc(tensors["edge"]).detach(),
            "h_uu": model.encoder.uu_enc(tensors["edge_uu"]).detach(),
        }


def _live_plan(encoder, head, batch: dict, state: dict):
    _check_batch(batch)
    conditioned = append_queue_edge(batch)
    tensors = batch_tensors(conditioned)
    h_u, h_b = encoder.condition(tensors)
    z_global = global_embedding(h_u, h_b, tensors["uav_mask"], tensors["buoy_mask"])
    _logits, plan, scores = _score_state(head, {"h_u": h_u, "h_b": h_b, "z_global": z_global}, state)
    return plan, scores


def _load_model_a() -> dict[tuple[int, int, int], dict]:
    if not PHASE5P_STATE.is_file():
        raise FileNotFoundError(str(PHASE5P_STATE))
    recorded = {}
    with PHASE5P_STATE.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] != "phase5p_jmappo":
                continue
            scenario = int(row["scenario_id"])
            assert_phase5q_seed(scenario)
            key = (int(row["seed"]), scenario, int(row["time_step"]))
            recorded[key] = row
    if len(recorded) != len(SEEDS) * 6:
        raise RuntimeError("Phase-5P J_MAPPO rows are not the frozen 5-seed record")
    return recorded


def _curve_from_training(model: str, seed: int, epoch: int, state: dict, loss: float, measured: dict) -> dict:
    return {
        "model": model,
        "seed": int(seed),
        "epoch": int(epoch),
        "scenario_id": int(state["scenario_id"]),
        "time_step": int(state["time_step"]),
        "loss": loss,
        "score_std": measured["score_std"],
        "score_span": measured["score_span"],
        "spearman": measured["spearman"],
        "kendall": measured["kendall"],
        "top1": measured["top1"],
        "normalized_regret": measured["normalized_regret"],
        "collapsed": measured["collapsed"],
        "source": "phase5q_measured",
    }


def _train_additive(encoder, head, states: list[dict], model: str, seed: int) -> tuple[list[dict], float, float, bool]:
    train_states = [state for state in states if state["split"] == "train"]
    parameters = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    curves = []
    first_loss = float("nan")
    final_loss = float("nan")
    diverged = False
    for epoch in range(1, int(EPOCHS) + 1):
        encoder.train()
        head.train()
        losses = []
        for state in train_states:
            if int(state["scenario_id"]) != int(TRAIN_SCENARIO):
                raise RuntimeError("development state entered the Model-B gradient")
            optimizer.zero_grad(set_to_none=True)
            _plan, scores = _live_plan(encoder, head, state["urgency_batch"], state)
            loss = joint_rank_loss(scores, _levels(state))
            if not bool(torch.isfinite(loss).all()):
                diverged = True
                losses.append(float("nan"))
                break
            loss.backward()
            clip_grad_norm_(parameters, GRAD_CLIP)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        encoder.eval()
        head.eval()
        with torch.no_grad():
            for state_index, state in enumerate(train_states):
                plan, scores = _live_plan(encoder, head, state["urgency_batch"], state)
                decoded = hungarian_from_plan(plan, torch.as_tensor(state["candidate_mask"], dtype=torch.bool))
                vector = _vector(decoded)
                if vector not in state["vectors"]:
                    raise RuntimeError(f"Model-B left the frozen pool: {vector}")
                measured = metrics_at(scores.detach().cpu().numpy(), state["j_mappo"], state["vectors"].index(vector))
                state_loss = losses[state_index] if state_index < len(losses) else float("nan")
                curves.append(_curve_from_training(model, seed, epoch, state, state_loss, measured))
        epoch_loss = _mean(losses)
        if epoch == 1:
            first_loss = epoch_loss
        final_loss = epoch_loss
        print(f"{model} seed {seed} epoch {epoch} loss={epoch_loss:.4f}", flush=True)
        if diverged:
            break
    encoder.eval()
    head.eval()
    for parameter in parameters:
        parameter.requires_grad_(False)
    return curves, first_loss, final_loss, diverged


def _train_scorer(scorer, features: dict, states: list[dict], seed: int) -> tuple[list[dict], float, float, bool]:
    train_states = [state for state in states if state["split"] == "train"]
    optimizer = torch.optim.Adam(scorer.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    curves = []
    first_loss = float("nan")
    final_loss = float("nan")
    diverged = False
    for epoch in range(1, int(EPOCHS) + 1):
        scorer.train()
        losses = []
        for state in train_states:
            if int(state["scenario_id"]) != int(TRAIN_SCENARIO):
                raise RuntimeError("development state entered the Model-C gradient")
            optimizer.zero_grad(set_to_none=True)
            assignments = torch.as_tensor(state["assignments"], dtype=torch.long)
            scores = scorer(features[id(state)], assignments)
            loss = joint_rank_loss(scores, _levels(state))
            if not bool(torch.isfinite(loss).all()):
                diverged = True
                losses.append(float("nan"))
                break
            loss.backward()
            clip_grad_norm_(scorer.parameters(), GRAD_CLIP)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scorer.eval()
        with torch.no_grad():
            for state_index, state in enumerate(train_states):
                scores = scorer(features[id(state)], torch.as_tensor(state["assignments"], dtype=torch.long))
                values = scores.detach().cpu().numpy()
                chosen = int(np.argsort(-values, kind="stable")[0])
                measured = metrics_at(values, state["j_mappo"], chosen)
                state_loss = losses[state_index] if state_index < len(losses) else float("nan")
                curves.append(_curve_from_training(MODEL_C, seed, epoch, state, state_loss, measured))
        epoch_loss = _mean(losses)
        if epoch == 1:
            first_loss = epoch_loss
        final_loss = epoch_loss
        print(f"{MODEL_C} seed {seed} epoch {epoch} loss={epoch_loss:.4f}", flush=True)
        if diverged:
            break
    scorer.eval()
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    return curves, first_loss, final_loss, diverged


def _eval_additive(encoder, head, state: dict) -> dict:
    encoder.eval()
    head.eval()
    with torch.no_grad():
        plan, scores = _live_plan(encoder, head, state["urgency_batch"], state)
        decoded = hungarian_from_plan(plan, torch.as_tensor(state["candidate_mask"], dtype=torch.bool))
    vector = _vector(decoded)
    if vector not in state["vectors"]:
        raise RuntimeError(f"decoded assignment left the frozen pool: {vector}")
    measured = metrics_at(scores.detach().cpu().numpy(), state["j_mappo"], state["vectors"].index(vector))
    measured["assignment_vector"] = vector
    measured["scores"] = scores.detach().cpu().numpy()
    return measured


def _eval_scorer(scorer, features: dict, state: dict) -> dict:
    scorer.eval()
    with torch.no_grad():
        scores = scorer(features, torch.as_tensor(state["assignments"], dtype=torch.long))
    values = scores.detach().cpu().numpy()
    chosen = int(np.argsort(-values, kind="stable")[0])
    measured = metrics_at(values, state["j_mappo"], chosen)
    measured["assignment_vector"] = state["vectors"][chosen]
    measured["scores"] = values
    return measured


def _model_a_curves(recorded: dict) -> list[dict]:
    rows = []
    for (seed, scenario, moment), row in sorted(recorded.items()):
        collapsed = float(not np.isfinite(float(row["spearman"])))
        for epoch in range(1, int(EPOCHS) + 1):
            final = epoch == int(EPOCHS)
            rows.append(
                {
                    "model": MODEL_A,
                    "seed": int(seed),
                    "epoch": int(epoch),
                    "scenario_id": int(scenario),
                    "time_step": int(moment),
                    "loss": PHASE5P_RECORDED_LOSS,
                    "score_std": "",
                    "score_span": "",
                    "spearman": row["spearman"] if final else "",
                    "kendall": row["kendall"] if final else "",
                    "top1": row["top1"] if final else "",
                    "normalized_regret": row["normalized_regret"] if final else "",
                    "collapsed": collapsed if final else "",
                    "source": "phase5p_recorded",
                }
            )
    return rows


def _state_row(model: str, seed: int, state: dict, measured: dict) -> dict:
    return {
        "model": model,
        "seed": int(seed),
        "split": state["split"],
        "scenario_id": int(state["scenario_id"]),
        "time_step": int(state["time_step"]),
        "spearman": measured["spearman"],
        "kendall": measured["kendall"],
        "top1": measured["top1"],
        "top3": measured["top3"],
        "j_mappo": measured["j_mappo"],
        "oracle_j_mappo": measured["oracle_j_mappo"],
        "regret": measured["regret"],
        "normalized_regret": measured["normalized_regret"],
        "score_std": measured["score_std"],
        "score_span": measured["score_span"],
        "collapsed": measured["collapsed"],
        "assignment_vector": measured["assignment_vector"],
    }


def _prediction_rows(model: str, seed: int, state: dict, measured: dict) -> list[dict]:
    levels = quantize(state["j_mappo"])
    best = float(np.max(levels))
    chosen = state["vectors"].index(measured["assignment_vector"])
    rows = []
    for index, vector in enumerate(state["vectors"]):
        payload = float(state["j_mappo"][index])
        rows.append(
            {
                "model": model,
                "seed": int(seed),
                "scenario_id": int(state["scenario_id"]),
                "time_step": int(state["time_step"]),
                "assignment_vector": vector,
                "score": float(measured["scores"][index]),
                "j_mappo": payload,
                "regret": float(best - float(levels[index])),
                "selected": float(index == chosen),
            }
        )
    return rows


def _top_rows(model: str, seed: int, state: dict, measured: dict, limit: int = 5) -> list[dict]:
    order = np.argsort(-np.asarray(measured["scores"], dtype=np.float64), kind="stable")
    levels = quantize(state["j_mappo"])
    best = float(np.max(levels))
    rows = []
    for rank, index in enumerate(order[:limit], start=1):
        rows.append(
            {
                "model": model,
                "seed": int(seed),
                "scenario_id": int(state["scenario_id"]),
                "time_step": int(state["time_step"]),
                "rank": int(rank),
                "assignment_vector": state["vectors"][int(index)],
                "score": float(measured["scores"][int(index)]),
                "j_mappo": float(state["j_mappo"][int(index)]),
                "regret": float(best - float(levels[int(index)])),
            }
        )
    return rows


def _csv_metric(row: dict) -> dict:
    spearman = float(row["spearman"])
    collapsed = float(not np.isfinite(spearman))
    return {
        "spearman": spearman,
        "kendall": float(row["kendall"]),
        "top1": float(row["top1"]),
        "top3": float(row["top3"]),
        "j_mappo": float(row["j_mappo"]),
        "oracle_j_mappo": float(row["j_mappo"]) + float(row["regret"]),
        "regret": float(row["regret"]),
        "normalized_regret": float(row["normalized_regret"]),
        "score_std": float("nan"),
        "score_span": float("nan"),
        "collapsed": collapsed,
        "assignment_vector": row["assignment_vector"],
    }


def _seed_macro(measured: dict, model: str, seed: int, split: str, field: str) -> float:
    values = [
        float(item[field])
        for (name, row_seed, row_split, _scenario, _moment), item in measured.items()
        if name == model and int(row_seed) == int(seed) and row_split == split
    ]
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return float("nan")
    return float(np.mean(array))


def _audit(states: list[dict]) -> tuple[list[dict], dict[str, dict[str, float]]]:
    train_states = [state for state in states if state["split"] == "train"]
    dev_states = [state for state in states if state["split"] == "dev"]
    targets = _targets(np.concatenate([state["assignments"] for state in train_states], axis=0))
    dev_targets = set(_targets(np.concatenate([state["assignments"] for state in dev_states], axis=0)))
    if not dev_targets.issubset(targets):
        raise RuntimeError("development assignments use a buoy absent from the training fit")
    summaries = {}
    rows = []
    y_train = np.concatenate([state["j_mappo"] for state in train_states], axis=0)
    y_dev = np.concatenate([state["j_mappo"] for state in dev_states], axis=0)
    for name, pairwise in (("additive", False), ("pairwise", True)):
        x_train = np.concatenate([design_matrix(state["assignments"], targets, pairwise=pairwise) for state in train_states], axis=0)
        x_dev = np.concatenate([design_matrix(state["assignments"], targets, pairwise=pairwise) for state in dev_states], axis=0)
        dev_pred, train_pred, n_features = fit_linear(x_train, y_train, x_dev)
        chunks = (
            ("train", train_states, train_pred, y_train),
            ("dev", dev_states, dev_pred, y_dev),
        )
        for split, group, prediction, truth in chunks:
            offset = 0
            state_metrics = []
            for state in group:
                width = int(state["assignments"].shape[0])
                part = prediction[offset : offset + width]
                metrics = regression_metrics(part, truth[offset : offset + width])
                offset += width
                state_metrics.append(metrics)
                rows.append(
                    {
                        "model": name,
                        "split": split,
                        "scenario_id": int(state["scenario_id"]),
                        "time_step": int(state["time_step"]),
                        "r2": metrics["r2"],
                        "mae": metrics["mae"],
                        "spearman": metrics["spearman"],
                        "kendall": metrics["kendall"],
                        "top1": metrics["top1"],
                        "normalized_regret": metrics["normalized_regret"],
                        "n_features": int(n_features),
                        "pooled_r2": "",
                        "pooled_mae": "",
                    }
                )
            pooled = regression_metrics(prediction, truth)
            macro = {key: _mean([item[key] for item in state_metrics]) for key in ("r2", "mae", "spearman", "kendall", "top1", "normalized_regret")}
            rows.append(
                {
                    "model": name,
                    "split": split,
                    "scenario_id": -1,
                    "time_step": -1,
                    "r2": macro["r2"],
                    "mae": macro["mae"],
                    "spearman": macro["spearman"],
                    "kendall": macro["kendall"],
                    "top1": macro["top1"],
                    "normalized_regret": macro["normalized_regret"],
                    "n_features": int(n_features),
                    "pooled_r2": pooled["r2"],
                    "pooled_mae": pooled["mae"],
                }
            )
            if split == "dev":
                summaries[name] = macro
    return rows, summaries


def _fit_flag(curves: list[dict], seed_losses: dict[int, tuple[float, float]]) -> bool:
    fits = []
    for seed, (first_loss, final_loss) in seed_losses.items():
        final_rows = [row for row in curves if int(row["seed"]) == int(seed) and int(row["epoch"]) == int(EPOCHS) and row["source"] == "phase5q_measured"]
        collapsed = _mean([float(row["collapsed"]) for row in final_rows]) if final_rows else float("nan")
        fits.append(train_fit(first_loss=first_loss, final_loss=final_loss, collapsed_fraction=collapsed))
    return sum(bool(item) for item in fits) * 2 > len(SEEDS)


def _deltas(measured: dict, model: str) -> dict[str, list[float]]:
    payload = []
    regret = []
    spearman = []
    for seed in SEEDS:
        payload.append(_seed_macro(measured, model, seed, "dev", "j_mappo") - _seed_macro(measured, MODEL_A, seed, "dev", "j_mappo"))
        regret.append(
            _seed_macro(measured, model, seed, "dev", "normalized_regret")
            - _seed_macro(measured, MODEL_A, seed, "dev", "normalized_regret")
        )
        spearman.append(_seed_macro(measured, model, seed, "dev", "spearman") - _seed_macro(measured, MODEL_A, seed, "dev", "spearman"))
    return {"j_mappo": payload, "regret": regret, "spearman": spearman}


def _collapse_rate(measured: dict, model: str, split: str) -> float:
    values = [
        float(item["collapsed"])
        for (name, _seed, row_split, _scenario, _moment), item in measured.items()
        if name == model and row_split == split
    ]
    return _mean(values)


def _moment_mean(measured: dict, model: str, moment: int, field: str) -> float:
    values = [
        float(item[field])
        for (name, _seed, split, _scenario, row_moment), item in measured.items()
        if name == model and split == "dev" and int(row_moment) == int(moment)
    ]
    return _mean(values)


def _write_report(summary: dict) -> None:
    gate = summary["gate"]
    advice = ADVICE[gate]
    if gate == "Q-D" and not (summary["b_train_fit"] and summary["c_train_fit"]):
        advice = (
            "只有一侧在 310100 上同时降低了 ranking loss 并解除了 score collapse，310101 没有达到稳定改善线。"
            "下一步先扩展合法 development states，不继续增加没有形成训练信号那一侧的模型复杂度。"
        )
    if gate == "Q-E" and summary["c_dev_improved"] and not summary["pairwise_better"]:
        advice = (
            "五个预先条件没有一条被完整满足。Model-C 在 310101 上达到了载荷改善线，Model-B 没有，所以不是 Q-A、Q-C 或 Q-D。"
            "状态无关的 pairwise 回归没有优于 additive，所以也不满足 Q-B 的 audit 条款。"
            "Q-E 的原始定义是两边都无法在 310100 上学习；这个定义与 Model-C 的训练曲线不符。文件中的 Q-E 只是这个未覆盖组合的回退标签。"
            "因果对照本身指向 assignment scorer：下一步若继续，应保留冻结 encoder，检查非加性 scorer，而不是先把 HGNN 加深。"
        )
    delta_lines = []
    for model in (MODEL_B, MODEL_C):
        for seed, payload, regret, spearman in zip(
            SEEDS,
            summary["deltas"][model]["j_mappo"],
            summary["deltas"][model]["regret"],
            summary["deltas"][model]["spearman"],
        ):
            delta_lines.append(
                f"{model} seed {seed}：ΔJ_MAPPO {_fmt(payload, '.4e')}，Δnormalized regret {_fmt(regret)}，ΔSpearman {_fmt(spearman)}。"
            )
    lines = [
        "# HTW Phase-5Q Encoder vs Assignment Scorer Audit",
        "",
        "**planner_ready = False**",
        "",
        "**deployment_verified = False**",
        "",
        "本阶段只分开两个因素。Model-A 直接使用 Phase-5P 已冻结的 J_MAPPO 头结果。Model-B 只把同一个 Phase-5L encoder 从冻结改为可训练，Joint Head、Sinkhorn、Hungarian 和候选池不变。Model-C 保持 encoder 冻结，用 AssignmentConditionedSetScorer 直接给 120 个合法 assignment 打分，不再把分数拆回 pair score 后做 Hungarian。两组都使用相同的 J_MAPPO ranking pairs、10 个 epoch 和 seed 0 到 4。310101 不进入梯度、early stopping 或选模，只作为 development evaluation。",
        "",
        "## 1. 训练信号与 score collapse",
        "",
        f"Model-A 的记录损失是 {_fmt(PHASE5P_RECORDED_LOSS)}。它在训练集上的 collapse 比例是 {_fmt(summary['collapse']['model_a_train'])}，在 310101 上是 {_fmt(summary['collapse']['model_a_dev'])}。",
        f"Model-B 第 1 个 epoch 的平均损失是 {_fmt(summary['loss']['model_b_first'])}，第 10 个 epoch 是 {_fmt(summary['loss']['model_b_final'])}。训练 collapse 比例 {_fmt(summary['collapse']['model_b_train'])}，310101 collapse 比例 {_fmt(summary['collapse']['model_b_dev'])}。训练拟合是否成立：{summary['b_train_fit']}。",
        f"Model-C 第 1 个 epoch 的平均损失是 {_fmt(summary['loss']['model_c_first'])}，第 10 个 epoch 是 {_fmt(summary['loss']['model_c_final'])}。训练 collapse 比例 {_fmt(summary['collapse']['model_c_train'])}，310101 collapse 比例 {_fmt(summary['collapse']['model_c_dev'])}。训练拟合是否成立：{summary['c_train_fit']}。",
        "",
        "score collapse 的定义是 state 内 score 标准差或极差低于 1e-6。Model-A 没有保存每个 epoch 的分数，表里 1 到 9 epoch 只有当时记录的损失；第 10 epoch 的排序指标来自 Phase-5P。",
        "",
        "## 2. Development evaluation（310101）",
        "",
        "Top-1 和 regret 先在每个 state 内计算，再对 seed 和时刻做平均。稳定改善的预先标准是：相对 Model-A，五个 seed 的 macro J_MAPPO 平均至少增加 1 个 payload block（1.0e6 bit），macro normalized regret 下降，并且至少 3 个 seed 同时满足载荷上升和 regret 下降。",
        "",
        f"Model-B 相对 Model-A：mean ΔJ_MAPPO {_fmt(_mean(summary['deltas']['model_b']['j_mappo']), '.4e')}，median {_fmt(_median(summary['deltas']['model_b']['j_mappo']), '.4e')}，std {_fmt(_std(summary['deltas']['model_b']['j_mappo']), '.4e')}。",
        f"mean Δnormalized regret {_fmt(_mean(summary['deltas']['model_b']['regret']))}，median {_fmt(_median(summary['deltas']['model_b']['regret']))}，std {_fmt(_std(summary['deltas']['model_b']['regret']))}。",
        f"Model-C 相对 Model-A：mean ΔJ_MAPPO {_fmt(_mean(summary['deltas']['model_c']['j_mappo']), '.4e')}，median {_fmt(_median(summary['deltas']['model_c']['j_mappo']), '.4e')}，std {_fmt(_std(summary['deltas']['model_c']['j_mappo']), '.4e')}。",
        f"mean Δnormalized regret {_fmt(_mean(summary['deltas']['model_c']['regret']))}，median {_fmt(_median(summary['deltas']['model_c']['regret']))}，std {_fmt(_std(summary['deltas']['model_c']['regret']))}。",
        "",
        *delta_lines,
        "",
        "| model | t | Spearman | Top-1 | J_MAPPO | normalized regret | collapsed |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *summary["dev_lines"],
        "",
        summary["focus_text"],
        "",
        "## 3. Additivity audit",
        "",
        "additive 模型和 pairwise 模型只在 310100 上做无惩罚最小二乘，再在 310101 上评价。它们是离线诊断，不是可部署模型。",
        "",
        f"310101 macro：additive R2 {_fmt(summary['audit']['additive']['r2'])}，Spearman {_fmt(summary['audit']['additive']['spearman'])}，Top-1 {_fmt(summary['audit']['additive']['top1'])}，normalized regret {_fmt(summary['audit']['additive']['normalized_regret'])}，MAE {_fmt(summary['audit']['additive']['mae'], '.4e')}。",
        f"pairwise R2 {_fmt(summary['audit']['pairwise']['r2'])}，Spearman {_fmt(summary['audit']['pairwise']['spearman'])}，Top-1 {_fmt(summary['audit']['pairwise']['top1'])}，normalized regret {_fmt(summary['audit']['pairwise']['normalized_regret'])}，MAE {_fmt(summary['audit']['pairwise']['mae'], '.4e')}。",
        f"pairwise 是否按预先标准显著优于 additive：{summary['pairwise_better']}。",
        "",
        "## 4. 结论",
        "",
        f"Gate 是 **{gate}**。",
        "",
        advice,
        "",
        "Q1. score collapse 在哪里解除？ " + summary["answers"]["q1"],
        "",
        "Q2. J_MAPPO 能否被单边效用的和充分表示？ " + summary["answers"]["q2"],
        "",
        "Q3. UAV-UAV / assignment pair interaction 是否必要？ " + summary["answers"]["q3"],
        "",
        "Q4. 310101 t=30 哪个模型更接近 pool oracle？ " + summary["answers"]["q4"],
        "",
        "Q5. 下一步应该优化什么？ " + summary["answers"]["q5"],
        "",
        "`planner_ready = False`",
        "",
        "`deployment_verified = False`",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def _collapse_sentence(action: str, base_train: float, new_train: float, base_dev: float, new_dev: float) -> str:
    train_down = np.isfinite(new_train) and np.isfinite(base_train) and float(new_train) < float(base_train) - 1.0e-6
    dev_down = np.isfinite(new_dev) and np.isfinite(base_dev) and float(new_dev) < float(base_dev) - 1.0e-6
    if train_down or dev_down:
        return (
            f"{action} 后 collapse 比例下降：训练集 {_fmt(base_train)} -> {_fmt(new_train)}，"
            f"310101 {_fmt(base_dev)} -> {_fmt(new_dev)}。"
        )
    return (
        f"{action} 后 collapse 比例没有下降：训练集 {_fmt(base_train)} -> {_fmt(new_train)}，"
        f"310101 {_fmt(base_dev)} -> {_fmt(new_dev)}。"
    )


def _answers(summary: dict) -> dict[str, str]:
    encoder_text = _collapse_sentence("重新训练 encoder", summary["collapse"]["model_a_train"], summary["collapse"]["model_b_train"], summary["collapse"]["model_a_dev"], summary["collapse"]["model_b_dev"])
    scorer_text = _collapse_sentence("换成 non-additive scorer", summary["collapse"]["model_a_train"], summary["collapse"]["model_c_train"], summary["collapse"]["model_a_dev"], summary["collapse"]["model_c_dev"])
    additive = summary["audit"]["additive"]
    if np.isfinite(additive["r2"]) and float(additive["r2"]) >= 0.8 and float(additive["normalized_regret"]) <= 0.25:
        q2 = f"可以。310101 上 additive macro R2 是 {_fmt(additive['r2'])}，normalized regret 是 {_fmt(additive['normalized_regret'])}。"
    else:
        q2 = (
            f"不能。310101 上 additive macro R2 是 {_fmt(additive['r2'])}，Spearman 是 {_fmt(additive['spearman'])}，"
            f"Top-1 是 {_fmt(additive['top1'])}，normalized regret 是 {_fmt(additive['normalized_regret'])}。"
        )
    q3 = "必要。pairwise 诊断在 310101 上按预先标准优于 additive。" if summary["pairwise_better"] else "现有 audit 没有证明它是必要项。pairwise 诊断没有按预先标准优于 additive。"
    q4 = summary["t30_text"]
    q5 = {
        "Q-A": "encoder representation。",
        "Q-B": "assignment scorer。",
        "Q-C": "encoder 和 assignment scorer 都要，但要到下一阶段才能联合训练。",
        "Q-D": "development-state diversity。",
        "Q-E": "先检查标签、assignment 表示和优化，不继续增加模型。",
    }[summary["gate"]]
    if summary["gate"] == "Q-E" and summary["c_dev_improved"] and not summary["b_dev_improved"]:
        q5 = (
            "assignment scorer。Model-B 没有改善，Model-C 解除了 score collapse，并在五个 seed 上提高了 310101 的真实载荷。"
            "状态无关的 pairwise 回归没有优于 additive，因此这还不是 Q-B 的完整授权，也不能据此把 encoder 加深。"
        )
    return {"q1": encoder_text + scorer_text, "q2": q2, "q3": q3, "q4": q4, "q5": q5}


def run() -> dict:
    recorded = _load_model_a()
    states = load_states()
    for state in states:
        assert_phase5q_seed(int(state["scenario_id"]))
        if int(state["assignments"].shape[0]) != int(POOL_SIZE):
            raise RuntimeError("candidate pool changed")
    frozen = _load_proposed()
    matching = _load_matching()
    for parameter in list(frozen.parameters()) + list(matching.parameters()):
        parameter.requires_grad_(False)
    cached = {id(state): _features(frozen, matching, state["urgency_batch"]) for state in states}
    embedded = {id(state): _edge_features(frozen, state["urgency_batch"]) for state in states}
    curves = _model_a_curves(recorded)
    measured = {}
    predictions = []
    failures = []
    state_rows = []
    losses = {MODEL_B: {}, MODEL_C: {}}
    diverged = {MODEL_B: [], MODEL_C: []}
    for seed in SEEDS:
        seed_everything(int(seed))
        reference = JointAssignmentHead()
        _train(reference, cached, states, "j_mappo")
        reproduced_rows = {}
        seed_match = True
        for state in states:
            reproduced = _eval_head(reference, cached[id(state)], state)
            official = recorded[(int(seed), int(state["scenario_id"]), int(state["time_step"]))]
            reproduced_rows[(int(state["scenario_id"]), int(state["time_step"]))] = reproduced
            if reproduced["assignment_vector"] != official["assignment_vector"]:
                seed_match = False
        print(f"model_a seed {seed} match={seed_match}", flush=True)
        seed_everything(int(seed))
        head = JointAssignmentHead()
        encoder = JointUtilityHGNN(buoy_dim=int(BUOY_DIM) + 1)
        encoder.load_state_dict(frozen.state_dict())
        trained_curves, first_loss, final_loss, failed = _train_additive(encoder, head, states, MODEL_B, int(seed))
        curves.extend(trained_curves)
        losses[MODEL_B][int(seed)] = (first_loss, final_loss)
        if failed:
            diverged[MODEL_B].append(int(seed))
        seed_everything(int(seed))
        scorer = AssignmentConditionedSetScorer()
        scorer_curves, first_loss, final_loss, failed = _train_scorer(scorer, embedded, states, int(seed))
        curves.extend(scorer_curves)
        losses[MODEL_C][int(seed)] = (first_loss, final_loss)
        if failed:
            diverged[MODEL_C].append(int(seed))
        for state in states:
            additive = _eval_additive(encoder, head, state)
            scored = _eval_scorer(scorer, embedded[id(state)], state)
            official = _csv_metric(recorded[(int(seed), int(state["scenario_id"]), int(state["time_step"]))])
            for model, result in ((MODEL_A, official), (MODEL_B, additive), (MODEL_C, scored)):
                key = (model, int(seed), state["split"], int(state["scenario_id"]), int(state["time_step"]))
                measured[key] = result
                state_rows.append(_state_row(model, int(seed), state, result))
            predictions.extend(_prediction_rows(MODEL_B, int(seed), state, additive))
            predictions.extend(_prediction_rows(MODEL_C, int(seed), state, scored))
            if int(state["scenario_id"]) == int(DEV_SCENARIO) and int(state["time_step"]) == 30:
                failures.extend(_top_rows(MODEL_B, int(seed), state, additive))
                failures.extend(_top_rows(MODEL_C, int(seed), state, scored))
                if seed_match:
                    failures.extend(_top_rows(MODEL_A, int(seed), state, reproduced_rows[(int(state["scenario_id"]), int(state["time_step"]))]))
                else:
                    failures.append(
                        {
                            "model": MODEL_A,
                            "seed": int(seed),
                            "scenario_id": int(state["scenario_id"]),
                            "time_step": int(state["time_step"]),
                            "rank": 1,
                            "assignment_vector": official["assignment_vector"],
                            "score": "",
                            "j_mappo": float(official["j_mappo"]),
                            "regret": float(official["regret"]),
                        }
                    )
        print(f"finished seed {seed}", flush=True)
    audit_rows, audit = _audit(states)
    b_fit = _fit_flag([row for row in curves if row["model"] == MODEL_B], losses[MODEL_B])
    c_fit = _fit_flag([row for row in curves if row["model"] == MODEL_C], losses[MODEL_C])
    deltas = {MODEL_B: _deltas(measured, MODEL_B), MODEL_C: _deltas(measured, MODEL_C)}
    b_dev = stable_improvement(deltas[MODEL_B]["j_mappo"], deltas[MODEL_B]["regret"])
    c_dev = stable_improvement(deltas[MODEL_C]["j_mappo"], deltas[MODEL_C]["regret"])
    pairwise_better = pairwise_beats_additive(audit["additive"], audit["pairwise"])
    gate = choose_gate(
        b_dev_improved=b_dev,
        c_dev_improved=c_dev,
        b_train_fit=b_fit,
        c_train_fit=c_fit,
        pairwise_better=pairwise_better,
    )
    dev_lines = []
    for model in (MODEL_A, MODEL_B, MODEL_C):
        for moment in (0, 15, 30):
            dev_lines.append(
                "| "
                + " | ".join(
                    (
                        model,
                        str(moment),
                        _fmt(_moment_mean(measured, model, moment, "spearman")),
                        _fmt(_moment_mean(measured, model, moment, "top1")),
                        _fmt(_moment_mean(measured, model, moment, "j_mappo"), ".4e"),
                        _fmt(_moment_mean(measured, model, moment, "normalized_regret")),
                        _fmt(_moment_mean(measured, model, moment, "collapsed")),
                    )
                )
                + " |"
            )
    t30 = {
        model: _moment_mean(measured, model, 30, "j_mappo")
        for model in (MODEL_A, MODEL_B, MODEL_C)
    }
    oracle_t30 = _moment_mean(measured, MODEL_A, 30, "oracle_j_mappo")
    best_value = max(t30.values())
    best_names = [name for name, value in t30.items() if np.isfinite(value) and abs(float(value) - float(best_value)) <= 1.0]
    best_name = "、".join(best_names)
    focus = (
        f"310101 t=15 的平均解码 J_MAPPO：Model-A {_fmt(_moment_mean(measured, MODEL_A, 15, 'j_mappo'), '.4e')}，"
        f"Model-B {_fmt(_moment_mean(measured, MODEL_B, 15, 'j_mappo'), '.4e')}，"
        f"Model-C {_fmt(_moment_mean(measured, MODEL_C, 15, 'j_mappo'), '.4e')}。"
        "这个时刻不能单独作为成功标准。"
        f"310101 t=30 的平均解码 J_MAPPO：Model-A {_fmt(t30[MODEL_A], '.4e')}，"
        f"Model-B {_fmt(t30[MODEL_B], '.4e')}，Model-C {_fmt(t30[MODEL_C], '.4e')}，"
        f"pool oracle {_fmt(oracle_t30, '.4e')}。平均载荷最高的是 {best_name}。"
    )
    collapse = {
        "model_a_train": _collapse_rate(measured, MODEL_A, "train"),
        "model_a_dev": _collapse_rate(measured, MODEL_A, "dev"),
        "model_b_train": _collapse_rate(measured, MODEL_B, "train"),
        "model_b_dev": _collapse_rate(measured, MODEL_B, "dev"),
        "model_c_train": _collapse_rate(measured, MODEL_C, "train"),
        "model_c_dev": _collapse_rate(measured, MODEL_C, "dev"),
    }
    summary = {
        "planner_ready": PLANNER_READY,
        "deployment_verified": DEPLOYMENT_VERIFIED,
        "decide_called": False,
        "development_evaluation_only": True,
        "gate": gate,
        "b_train_fit": b_fit,
        "c_train_fit": c_fit,
        "b_dev_improved": b_dev,
        "c_dev_improved": c_dev,
        "pairwise_better": pairwise_better,
        "diverged": diverged,
        "loss": {
            "model_b_first": _mean([item[0] for item in losses[MODEL_B].values()]),
            "model_b_final": _mean([item[1] for item in losses[MODEL_B].values()]),
            "model_c_first": _mean([item[0] for item in losses[MODEL_C].values()]),
            "model_c_final": _mean([item[1] for item in losses[MODEL_C].values()]),
        },
        "collapse": collapse,
        "deltas": deltas,
        "audit": audit,
        "dev_lines": dev_lines,
        "focus_text": focus,
        "t30_text": (
            f"pool oracle 是 {_fmt(oracle_t30, '.4e')}。Model-A {_fmt(t30[MODEL_A], '.4e')}，"
            f"Model-B {_fmt(t30[MODEL_B], '.4e')}，Model-C {_fmt(t30[MODEL_C], '.4e')}。"
            f"平均载荷最高的是 {best_name}。"
        ),
    }
    summary["answers"] = _answers(summary)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT / "phase5q_train_curves.csv", curves)
    _write_csv(OUTPUT / "phase5q_seed_state_metrics.csv", state_rows)
    _write_csv(OUTPUT / "phase5q_assignment_predictions.csv", predictions)
    _write_csv(OUTPUT / "phase5q_additivity_audit.csv", audit_rows)
    _write_csv(OUTPUT / "phase5q_t30_failure_analysis.csv", failures)
    payload = {
        "gate": gate,
        "gate_condition_met": bool(gate != "Q-E" or (not c_dev and not b_dev and not b_fit and not c_fit)),
        "planner_ready": PLANNER_READY,
        "deployment_verified": DEPLOYMENT_VERIFIED,
        "development_evaluation_only": True,
        "b_train_fit": b_fit,
        "c_train_fit": c_fit,
        "b_dev_improved": b_dev,
        "c_dev_improved": c_dev,
        "pairwise_better": pairwise_better,
        "diverged": diverged,
        "collapse": collapse,
        "loss": summary["loss"],
        "deltas": deltas,
        "audit_dev": audit,
        "clear_payload_bits": PAYLOAD_BLOCK_BITS,
    }
    (OUTPUT / "phase5q_bottleneck_gate.json").write_text(json.dumps(_plain(payload), indent=2), encoding="utf-8")
    _write_report(summary)
    print(json.dumps(_plain({"gate": gate, "b_dev": b_dev, "c_dev": c_dev, "b_fit": b_fit, "c_fit": c_fit, "pairwise": pairwise_better})), flush=True)
    return summary


if __name__ == "__main__":
    run()
