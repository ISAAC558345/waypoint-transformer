"""Train the matched H=1 response value. Confirmation is read after epoch 10."""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from wt.l3b.config import L3BConfig
from wt.lowlevel_v2.phase5g_closed_loop.collect import load_frozen_actor
from wt.lowlevel_v2.phase5l_joint_encoder.run_phase5l import _load_matching
from wt.lowlevel_v2.phase5m_joint_planner.run_phase5m import _load_proposed
from wt.lowlevel_v2.phase5n_joint_planner.run_phase5n import _features
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.metrics import is_collapsed, metrics_at, quantize
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.run_phase5q import _edge_features
from wt.lowlevel_v2.phase5r_nonadditive_generalization.metrics import cluster_bootstrap
from wt.lowlevel_v2.phase5r_nonadditive_generalization.world import candidate_pool
from wt.lowlevel_v2.phase5s_multiscenario_nonadditive.protocol import MODEL_POOL_WIDTH, PHASE5O_LABELS, PHASE5R_LABELS
from wt.lowlevel_v2.phase5t_scorer_failure_audit.protocol import PHASE5S_LABELS
from wt.lowlevel_v2.phase5u_representation_repair.features import relational_matrix
from wt.lowlevel_v2.phase5u_representation_repair.run_phase5u import _replay, _snapshot
from wt.lowlevel_v2.phase5v_temporal_representation.run_phase5v import _combine, _history_snapshot, _load_v1, _scale, _scores, _vector
from wt.lowlevel_v2.phase5v_temporal_representation.temporal import temporal_matrix
from wt.lowlevel_v2.phase5y_executor_value_distillation.protocol import CHECKPOINTS as PHASE5Y_CHECKPOINTS
from wt.lowlevel_v2.phase5y_executor_value_distillation.value import ExecutorConditionedJointValue
from wt.round30.ppo import seed_everything

from .metrics import choose_gate, regression_block
from .protocol import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CASE_OPTIMAL,
    CASE_PREDICTED,
    CASE_SCENARIO,
    CASE_TIME,
    CASE_TOP_RANK,
    CHECKPOINTS,
    CONFIRM_SCENARIOS,
    DISAGREEMENT_BAR,
    EPOCHS,
    FORBIDDEN_INPUTS,
    GRAD_CLIP,
    INTEGRITY_SAMPLES,
    INTEGRITY_SEED,
    J15_DROP_BITS,
    LEARNING_RATE,
    MIN_SCENARIO_UP,
    MIN_SEED_UP,
    OUTPUT,
    PAYLOAD_BLOCK_BITS,
    PAYLOAD_MATCH_ABS,
    PLANNER_READY,
    PROGRESS_MATCH_ABS,
    REGRET_BAR,
    REPORT,
    RESPONSE_FEATURES,
    RESPONSE_GROUPS,
    ROOT,
    SEEDS,
    SINR_MATCH_ABS,
    SPEARMAN_BAR,
    TIMES,
    TOP3_BAR,
    TRAIN_SCENARIOS,
    WEIGHT_DECAY,
    assert_consumed,
)
from .response import one_step
from .scorer import ExecutorConditionedJointScorer, state_huber

PHASE5X_ROLLOUTS = ROOT / "results/phase5x_executor_lookahead" / "phase5x_executor_rollouts.csv"
PHASE5W_RESIDUAL = ROOT / "results/phase5w_residual_mechanism_audit" / "phase5w_residual_case_hybrids.csv"


def _plain(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _write_csv(name: str, rows: list[dict]) -> None:
    path = OUTPUT / name
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _fmt(value) -> str:
    number = float(value) if value is not None else float("nan")
    return "nan" if not np.isfinite(number) else f"{number:.4f}"


def _labels(split: str) -> dict[tuple[int, int, str], float]:
    table = {}
    if split == "train":
        sources = ((PHASE5O_LABELS, TRAIN_SCENARIOS), (PHASE5R_LABELS, TRAIN_SCENARIOS))
    elif split == "confirm":
        sources = ((PHASE5S_LABELS, CONFIRM_SCENARIOS),)
    else:
        raise RuntimeError(f"unknown split {split}")
    for path, allowed in sources:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                scenario = int(row["scenario_id"])
                if scenario not in allowed:
                    continue
                if float(row.get("in_model_pool", 1.0)) != 1.0:
                    continue
                assert_consumed(scenario)
                table[(scenario, int(row["time_step"]), row["assignment_vector"])] = float(row["j_mappo"])
    return table


def _h1_reference(allowed: tuple[int, ...]) -> dict[tuple[int, int, str], dict[str, float]]:
    table = {}
    with PHASE5X_ROLLOUTS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scenario = int(row["scenario_id"])
            if scenario not in allowed:
                continue
            if int(row["horizon"]) != 1:
                continue
            assert_consumed(scenario)
            table[(scenario, int(row["time_step"]), row["assignment_vector"])] = {
                "mean_sinr": float(row["mean_sinr_h"]),
                "progress_sum": float(row["progress_end_sum"]),
                "payload": float(row["payload_h"]),
                "q_exec": float(row["q_exec_h"]),
                "elapsed_ms": float(row["elapsed_ms"]),
            }
    return table


def _named_rank(scores: np.ndarray, vectors: list[str], target: str) -> int:
    index = list(vectors).index(str(target))
    values = np.asarray(scores, dtype=np.float64)
    return int(1 + np.sum(values > values[index]))


def _top5(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")[:5]
    levels = quantize(labels)
    return float(np.any(levels[order] == np.max(levels)))


def _fit_norm(states: list[dict]) -> dict[str, np.ndarray]:
    stacked = np.concatenate([state["response_raw"].reshape(-1, len(RESPONSE_FEATURES)) for state in states], axis=0)
    std = np.std(stacked, axis=0)
    std = np.maximum(std, 1.0e-6)
    return {
        "mean": np.mean(stacked, axis=0),
        "std": std,
        "min": np.min(stacked, axis=0),
        "max": np.max(stacked, axis=0),
        "count": np.asarray([stacked.shape[0]], dtype=np.float64),
    }


def _apply_norm(raw: np.ndarray, norm: dict[str, np.ndarray]) -> np.ndarray:
    return ((raw - norm["mean"]) / norm["std"]).astype(np.float32)


def _batch(state: dict, response: np.ndarray, relational: np.ndarray) -> tuple[dict, torch.Tensor]:
    features = {}
    for key, value in state["embedded"].items():
        features[key] = value.detach() if torch.is_tensor(value) else torch.as_tensor(value)
    features["z_rel"] = torch.as_tensor(relational, dtype=torch.float32)
    features["response"] = torch.as_tensor(response, dtype=torch.float32)
    return features, torch.as_tensor(state["assignments"], dtype=torch.long)


def _relational(state: dict, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return _combine(state["z_rel"], _scale(state["z_temp_raw"], center, scale))


def _grad_norm(module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum())
    return float(np.sqrt(total))


def _param_norm(module) -> float:
    total = 0.0
    for parameter in module.parameters():
        total += float(parameter.detach().pow(2).sum())
    return float(np.sqrt(total))


def _prepare(scenarios, labels, reference, frozen, matching, actor, config, samples: set[tuple[int, int, int]]) -> tuple[list[dict], list[dict], dict]:
    states = []
    integrity = []
    largest = 0.0
    sinr_gap = 0.0
    progress_gap = 0.0
    payload_gap = 0.0
    timing = defaultdict(float)
    counted = 0
    for scenario in scenarios:
        for moment in TIMES:
            assert_consumed(int(scenario))
            started = time.perf_counter()
            env, movement, batch, diagnosis = _replay(int(scenario), int(moment))
            embedded = {key: value.detach() if torch.is_tensor(value) else value for key, value in _edge_features(frozen, batch).items()}
            additive = _features(frozen, matching, batch)
            logits = additive["logits"].detach().cpu().numpy()
            pool = candidate_pool(logits, MODEL_POOL_WIDTH, int(logits.shape[0]), int(logits.shape[1]))
            vectors = [_vector(row) for row in pool["assignments"]]
            recorded = {key[2] for key in labels if key[0] == int(scenario) and key[1] == int(moment)}
            if set(vectors) != recorded:
                raise RuntimeError(f"pool does not match stored labels at {scenario} t={moment}")
            relational = relational_matrix(_snapshot(env, diagnosis), pool["assignments"], config)
            temporal = temporal_matrix(_history_snapshot(env, batch), pool["assignments"], config)
            timing["encoder"] += (time.perf_counter() - started) * 1000.0
            count = int(env.config.num_uavs)
            velocity_before = np.array(batch["uav"][0, :count, 2:4], copy=True)
            responses = []
            audits = []
            for index, assignment in enumerate(pool["assignments"]):
                result = one_step(env, assignment, actor, movement)
                responses.append(result["response"])
                audits.append(result)
                largest = max(largest, float(result["integrity"]))
                vector = vectors[index]
                ref = reference[(int(scenario), int(moment), vector)]
                sinr_gap = max(sinr_gap, abs(float(result["mean_sinr"]) - float(ref["mean_sinr"])))
                progress_gap = max(progress_gap, abs(float(result["progress_sum"]) - float(ref["progress_sum"])))
                payload_gap = max(payload_gap, abs(float(result["payload"]) - float(ref["payload"])))
                timing["action"] += float(result["action_ms"])
                timing["env"] += float(result["env_ms"])
                timing["extract"] += float(result["extract_ms"])
                counted += 1
                if (int(scenario), int(moment), int(index)) in samples:
                    integrity.append(
                        {
                            "scenario_id": int(scenario),
                            "time_step": int(moment),
                            "assignment_index": int(index),
                            "assignment_vector": vector,
                            "max_abs": float(result["integrity"]),
                        }
                    )
            velocity_gap = float(np.max(np.abs(velocity_before - np.asarray(batch["uav"][0, :count, 2:4]))))
            largest = max(largest, velocity_gap)
            states.append(
                {
                    "scenario_id": int(scenario),
                    "time_step": int(moment),
                    "split": "train" if int(scenario) in TRAIN_SCENARIOS else "confirmation",
                    "assignments": np.asarray(pool["assignments"], dtype=np.int64),
                    "vectors": vectors,
                    "j_mappo": np.asarray([labels[(int(scenario), int(moment), vector)] for vector in vectors], dtype=np.float64),
                    "embedded": embedded,
                    "z_rel": relational,
                    "z_temp_raw": temporal,
                    "response_raw": np.stack(responses, axis=0).astype(np.float64),
                    "q_exec": np.asarray([reference[(int(scenario), int(moment), vector)]["q_exec"] for vector in vectors], dtype=np.float64),
                    "elapsed_ms": np.asarray([reference[(int(scenario), int(moment), vector)]["elapsed_ms"] for vector in vectors], dtype=np.float64),
                }
            )
            print(f"state {scenario} t={moment}", flush=True)
    integrity.append(
        {
            "scenario_id": "",
            "time_step": "",
            "assignment_index": "",
            "assignment_vector": "",
            "max_abs": largest,
            "check": "caller_fingerprint",
            "sinr_gap": sinr_gap,
            "progress_gap": progress_gap,
            "payload_gap": payload_gap,
            "n_assignments": counted,
        }
    )
    timing["n_assignments"] = float(counted)
    timing["n_states"] = float(len(states))
    return states, integrity, {"largest": largest, "sinr_gap": sinr_gap, "progress_gap": progress_gap, "payload_gap": payload_gap, "timing": timing}


def _predict(scorer, state: dict, response: np.ndarray, relational: np.ndarray) -> np.ndarray:
    features, assignments = _batch(state, response, relational)
    with torch.no_grad():
        started = time.perf_counter()
        values = scorer(features, assignments).detach().cpu().numpy()
        elapsed = (time.perf_counter() - started) * 1000.0
    return values, elapsed


def _train_one(kind: str, seed: int, states: list[dict], norm: dict, center: np.ndarray, scale: np.ndarray) -> tuple[ExecutorConditionedJointScorer, list[dict]]:
    seed_everything(int(seed))
    scorer = ExecutorConditionedJointScorer()
    optimizer = torch.optim.Adam(scorer.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    curves = []
    for epoch in range(1, int(EPOCHS) + 1):
        scorer.train()
        losses = []
        grads = []
        for state in states:
            if int(state["scenario_id"]) not in TRAIN_SCENARIOS:
                raise RuntimeError("confirmation state entered the value gradient")
            optimizer.zero_grad(set_to_none=True)
            raw = np.zeros_like(state["response_raw"]) if kind == "static" else state["response_raw"]
            prediction = scorer(*_batch(state, _apply_norm(raw, norm), _relational(state, center, scale)))
            target = torch.as_tensor(state["j_mappo"] / PAYLOAD_BLOCK_BITS, dtype=torch.float32)
            loss = state_huber(prediction, target)
            if not bool(torch.isfinite(loss).all()):
                raise RuntimeError(f"{kind} loss became non-finite at seed {seed} epoch {epoch}")
            loss.backward()
            grads.append(_grad_norm(scorer))
            clip_grad_norm_(scorer.parameters(), GRAD_CLIP)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scorer.eval()
        predictions = []
        targets = []
        stds = []
        with torch.no_grad():
            for state in states:
                raw = np.zeros_like(state["response_raw"]) if kind == "static" else state["response_raw"]
                values, _elapsed = _predict(scorer, state, _apply_norm(raw, norm), _relational(state, center, scale))
                predictions.append(values)
                targets.append(state["j_mappo"] / PAYLOAD_BLOCK_BITS)
                stds.append(float(np.std(values)))
        block = regression_block(np.concatenate(predictions), np.concatenate(targets))
        curves.append(
            {
                "model": kind,
                "seed": int(seed),
                "epoch": int(epoch),
                "huber": _mean(losses),
                "mae": block["mae"],
                "rmse": block["rmse"],
                "r2": block["r2"],
                "score_std": _mean(stds),
                "grad_norm": _mean(grads),
                "param_norm": _param_norm(scorer),
                "collapse_fraction": _mean(value < 1.0e-6 for value in stds),
            }
        )
        print(f"{kind} seed {seed} epoch {epoch} huber={_mean(losses):.4f}", flush=True)
    scorer.eval()
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    name = "static" if kind == "static" else "response"
    torch.save(
        {
            "seed": int(seed),
            "epoch": int(EPOCHS),
            "model": name,
            "state_dict": scorer.state_dict(),
            "response_mean": norm["mean"],
            "response_std": norm["std"],
        },
        CHECKPOINTS / f"phase5zd_{name}_seed{int(seed)}.pt",
    )
    return scorer, curves


def _rank_row(model: str, seed: int, state: dict, scores: np.ndarray) -> dict:
    chosen = int(np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")[0])
    measured = metrics_at(scores, state["j_mappo"], chosen)
    return {
        "model": model,
        "seed": int(seed),
        "scenario_id": int(state["scenario_id"]),
        "time_step": int(state["time_step"]),
        "spearman": measured["spearman"],
        "kendall": measured["kendall"],
        "top1": measured["top1"],
        "top3": measured["top3"],
        "top5": _top5(scores, state["j_mappo"]),
        "normalized_regret": measured["normalized_regret"],
        "selected_j15": measured["j_mappo"],
        "optimum_rank": _named_rank(scores, state["vectors"], state["vectors"][int(np.argmax(quantize(state["j_mappo"])))]),
        "named_opt_rank": _named_rank(scores, state["vectors"], CASE_OPTIMAL) if int(state["scenario_id"]) == CASE_SCENARIO and int(state["time_step"]) == CASE_TIME else "",
        "collapsed": float(is_collapsed(scores)),
        "assignment_vector": state["vectors"][chosen],
    }


def _direction(value: float) -> str:
    if abs(float(value)) <= 1.0:
        return "tie"
    return "positive" if float(value) > 0.0 else "negative"


def _write_gate(payload: dict) -> None:
    (OUTPUT / "phase5zd_executor_response_value_gate.json").write_text(json.dumps(_plain(payload), indent=2), encoding="utf-8")


def _finish_early(gate: str, reason: str, integrity: list[dict]) -> dict:
    _write_csv("phase5zd_shadow_integrity.csv", integrity)
    payload = {
        "gate": gate,
        "reason": reason,
        "planner_ready": False,
        "deployment_verified": False,
        "trained": False,
        "opened_310118": False,
        "decide_called": False,
        "frozen_for_response_distillation": False,
    }
    _write_gate(payload)
    REPORT.write_text(
        "\n".join(
            [
                "# Phase-5ZD：一步执行响应条件下的联合价值",
                "",
                f"Gate = `{gate}`。",
                "",
                reason,
                "",
                "`planner_ready = False`",
                "",
                "`deployment_verified = False`",
                "",
                "没有打开 310118–310125，没有调用 deployment 决策接口。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(_plain(payload), indent=2), flush=True)
    return payload


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "phase5zd_protocol.json").write_text(
        json.dumps(
            {
                "training": True,
                "encoder_frozen": True,
                "mappo_frozen": True,
                "loss": "state-balanced Huber delta=1 on J15/1e6",
                "optimizer": "Adam",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "grad_clip": GRAD_CLIP,
                "epochs": EPOCHS,
                "checkpoint": "epoch_10",
                "seeds": list(SEEDS),
                "train_scenarios": list(TRAIN_SCENARIOS),
                "confirmation_read_before_checkpoints": False,
                "opened_310118": False,
                "planner_ready": PLANNER_READY,
                "deployment_verified": False,
                "huber_delta": 1.0,
                "top3_bar": TOP3_BAR,
                "regret_bar": REGRET_BAR,
                "spearman_bar": SPEARMAN_BAR,
                "min_scenario_up": MIN_SCENARIO_UP,
                "min_seed_up": MIN_SEED_UP,
                "case_top_rank": CASE_TOP_RANK,
                "disagreement_bar": DISAGREEMENT_BAR,
                "j15_drop_bits": J15_DROP_BITS,
                "response_features": list(RESPONSE_FEATURES),
                "forbidden_inputs": list(FORBIDDEN_INPUTS),
                "model_s": "same scorer with a zero response block",
                "model_r1": "same scorer with the H=1 response block",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUTPUT / "phase5zd_response_feature_spec.json").write_text(
        json.dumps(
            {
                "features": list(RESPONSE_FEATURES),
                "groups": {name: list(indexes) for name, indexes in RESPONSE_GROUPS.items()},
                "horizon": 1,
                "payload_is_input": False,
                "future_horizons_are_input": False,
                "idle_distance_and_motion": 0,
                "pooling": "selected UAV tokens only, same mask as AssignmentConditionedSetScorer",
                "delta_definition": "value_after_step - value_before_step",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if any(name in FORBIDDEN_INPUTS for name in RESPONSE_FEATURES):
        raise RuntimeError("a forbidden quantity was listed as a response feature")
    config = L3BConfig()
    frozen = _load_proposed()
    matching = _load_matching()
    for parameter in list(frozen.parameters()) + list(matching.parameters()):
        parameter.requires_grad_(False)
    frozen.eval()
    matching.eval()
    actor = load_frozen_actor()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    actor.eval()
    train_labels = _labels("train")
    train_reference = _h1_reference(TRAIN_SCENARIOS)
    pairs = [(int(scenario), int(moment), int(index)) for scenario in TRAIN_SCENARIOS for moment in TIMES for index in range(120)]
    picked = np.random.default_rng(INTEGRITY_SEED).choice(len(pairs), size=INTEGRITY_SAMPLES, replace=False)
    samples = {pairs[int(index)] for index in picked}
    train_states, integrity, audit = _prepare(TRAIN_SCENARIOS, train_labels, train_reference, frozen, matching, actor, config, samples)
    if float(audit["largest"]) > 0.0:
        return _finish_early("RESP-STATE-FAIL", f"主环境最大绝对变化是 {audit['largest']}，训练没有开始。", integrity)
    if len([row for row in integrity if row.get("assignment_vector")]) < INTEGRITY_SAMPLES:
        return _finish_early("RESP-STATE-FAIL", "影子抽查不足 100 条，训练没有开始。", integrity)
    aligned = (
        float(audit["sinr_gap"]) <= SINR_MATCH_ABS
        and float(audit["progress_gap"]) <= PROGRESS_MATCH_ABS
        and float(audit["payload_gap"]) <= PAYLOAD_MATCH_ABS
    )
    if not aligned:
        return _finish_early(
            "RESP-D",
            f"训练集 H=1 与 Phase-5X 不一致：SINR {audit['sinr_gap']}，progress {audit['progress_gap']}，payload {audit['payload_gap']}。训练没有开始。",
            integrity,
        )
    _v1, _curves, center, scale = _load_v1()
    norm = _fit_norm(train_states)
    (OUTPUT / "phase5zd_response_normalization.json").write_text(
        json.dumps(
            {
                "source": "310100-310109 only",
                "features": list(RESPONSE_FEATURES),
                "mean": [float(value) for value in norm["mean"]],
                "std": [float(value) for value in norm["std"]],
                "min": [float(value) for value in norm["min"]],
                "max": [float(value) for value in norm["max"]],
                "count": int(norm["count"][0]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    probe = ExecutorConditionedJointScorer()
    probe_state = train_states[0]
    prediction = probe(*_batch(probe_state, _apply_norm(probe_state["response_raw"], norm), _relational(probe_state, center, scale)))
    state_huber(prediction, torch.as_tensor(probe_state["j_mappo"] / PAYLOAD_BLOCK_BITS, dtype=torch.float32)).backward()
    encoder_grad = 0.0
    for parameter in list(frozen.parameters()) + list(matching.parameters()):
        if parameter.grad is not None:
            encoder_grad += float(parameter.grad.abs().sum())
    actor_grad = 0.0
    for parameter in actor.parameters():
        if parameter.grad is not None:
            actor_grad += float(parameter.grad.abs().sum())
    manifest_rows = [{"name": name, "numel": int(parameter.numel()), "requires_grad": bool(parameter.requires_grad)} for name, parameter in probe.named_parameters()]
    parameter_count = int(sum(row["numel"] for row in manifest_rows))
    (OUTPUT / "phase5zd_trainable_parameter_manifest.json").write_text(
        json.dumps(
            {
                "trainable_module": "ExecutorConditionedJointScorer",
                "parameter_count": parameter_count,
                "static_and_response_use_the_same_module": True,
                "encoder_requires_grad": bool(any(parameter.requires_grad for parameter in frozen.parameters())),
                "matching_requires_grad": bool(any(parameter.requires_grad for parameter in matching.parameters())),
                "mappo_requires_grad": bool(any(parameter.requires_grad for parameter in actor.parameters())),
                "encoder_grad_abs_sum": encoder_grad,
                "mappo_grad_abs_sum": actor_grad,
                "parameters": manifest_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if encoder_grad != 0.0 or actor_grad != 0.0:
        return _finish_early("RESP-STATE-FAIL", "HGNN 或 MAPPO 在反向传播中收到了梯度。训练没有继续。", integrity)
    del probe
    curves = []
    static_models = {}
    response_models = {}
    for seed in SEEDS:
        static_models[int(seed)], static_curves = _train_one("static", int(seed), train_states, norm, center, scale)
        response_models[int(seed)], response_curves = _train_one("response", int(seed), train_states, norm, center, scale)
        curves.extend(static_curves)
        curves.extend(response_curves)
    _write_csv("phase5zd_training_curves.csv", curves)
    missing = [path.name for path in (CHECKPOINTS / f"phase5zd_{kind}_seed{seed}.pt" for kind in ("static", "response") for seed in SEEDS) if not path.is_file()]
    if missing:
        return _finish_early("RESP-D", f"epoch-10 checkpoint 缺失：{missing}", integrity)
    confirm_labels = _labels("confirm")
    confirm_reference = _h1_reference(CONFIRM_SCENARIOS)
    confirm_states, confirm_integrity, confirm_audit = _prepare(CONFIRM_SCENARIOS, confirm_labels, confirm_reference, frozen, matching, actor, config, set())
    integrity.extend(confirm_integrity)
    _write_csv("phase5zd_shadow_integrity.csv", integrity)
    if float(confirm_audit["largest"]) > 0.0:
        return _finish_early("RESP-STATE-FAIL", f"确认集影子回放改变了主环境，最大绝对变化 {confirm_audit['largest']}。", integrity)
    ref1_aligned = (
        float(confirm_audit["sinr_gap"]) <= SINR_MATCH_ABS
        and float(confirm_audit["progress_gap"]) <= PROGRESS_MATCH_ABS
        and float(confirm_audit["payload_gap"]) <= PAYLOAD_MATCH_ABS
    )
    phase5v, _unused, _center, _scale = _load_v1()
    phase5y = {}
    for seed in SEEDS:
        module = ExecutorConditionedJointValue()
        module.load_state_dict(torch.load(PHASE5Y_CHECKPOINTS / f"phase5y_seed{int(seed)}.pt", map_location="cpu", weights_only=False)["state_dict"])
        module.eval()
        phase5y[int(seed)] = module
    rank_rows = []
    regression_rows = []
    prediction_static = []
    prediction_response = []
    value_ms = []
    scorebook = defaultdict(dict)
    for state in confirm_states:
        relational = _relational(state, center, scale)
        for seed in SEEDS:
            static_scores, _elapsed = _predict(static_models[int(seed)], state, _apply_norm(np.zeros_like(state["response_raw"]), norm), relational)
            started = time.perf_counter()
            response_scores, elapsed = _predict(response_models[int(seed)], state, _apply_norm(state["response_raw"], norm), relational)
            value_ms.append(elapsed)
            phase5v_scores = _scores(phase5v[int(seed)], state, relational)
            phase5y_scores = _scores(phase5y[int(seed)], state, relational)
            bundled = {
                "S": static_scores,
                "R1": response_scores,
                "B0": phase5v_scores,
                "B1": phase5y_scores,
                "REF1": state["q_exec"],
                "O": state["j_mappo"],
            }
            for model, scores in bundled.items():
                rank_rows.append(_rank_row(model, int(seed), state, scores))
                scorebook[(model, int(seed), int(state["scenario_id"]), int(state["time_step"]))] = scores
            for vector, static_value, response_value, label in zip(state["vectors"], static_scores, response_scores, state["j_mappo"]):
                base = {
                    "scenario_id": int(state["scenario_id"]),
                    "time_step": int(state["time_step"]),
                    "seed": int(seed),
                    "assignment_vector": vector,
                    "j15": float(label),
                    "y": float(label) / PAYLOAD_BLOCK_BITS,
                }
                prediction_static.append({**base, "v_static": float(static_value)})
                prediction_response.append({**base, "v_response": float(response_value)})
            for model, scores in (("S", static_scores), ("R1", response_scores)):
                block = regression_block(scores, state["j_mappo"] / PAYLOAD_BLOCK_BITS)
                regression_rows.append({"model": model, "seed": int(seed), "scenario_id": int(state["scenario_id"]), "time_step": int(state["time_step"]), **block})
    ref8 = _h1_reference_horizon(CONFIRM_SCENARIOS, 8)
    for state in confirm_states:
        scores = np.asarray([ref8[(int(state["scenario_id"]), int(state["time_step"]), vector)] for vector in state["vectors"]], dtype=np.float64)
        for seed in SEEDS:
            rank_rows.append(_rank_row("REF8", int(seed), state, scores))
            scorebook[("REF8", int(seed), int(state["scenario_id"]), int(state["time_step"]))] = scores
    _write_csv("phase5zd_static_control_predictions.csv", prediction_static)
    _write_csv("phase5zd_response_value_predictions.csv", prediction_response)
    _write_csv("phase5zd_regression_metrics.csv", regression_rows)
    _write_csv("phase5zd_ranking_metrics.csv", rank_rows)
    summary_rows = []
    for model in ("B0", "B1", "S", "R1", "REF1", "REF8", "O"):
        chosen = [row for row in rank_rows if row["model"] == model]
        summary_rows.append(
            {
                "model": model,
                "selected_j15_mean": _mean(row["selected_j15"] for row in chosen),
                "selected_j15_median": float(np.median([float(row["selected_j15"]) for row in chosen])),
                "normalized_regret": _mean(row["normalized_regret"] for row in chosen),
                "top1": _mean(row["top1"] for row in chosen),
                "top3": _mean(row["top3"] for row in chosen),
                "top5": _mean(row["top5"] for row in chosen),
                "spearman": _mean(row["spearman"] for row in chosen),
                "kendall": _mean(row["kendall"] for row in chosen),
            }
        )
    _write_csv("phase5zd_selected_j15.csv", summary_rows)
    ref1_rows = [row for row in rank_rows if row["model"] == "REF1"]
    ref1_ok = (
        ref1_aligned
        and _mean(row["spearman"] for row in ref1_rows) >= SPEARMAN_BAR
        and _mean(row["top3"] for row in ref1_rows) >= TOP3_BAR
        and _mean(row["normalized_regret"] for row in ref1_rows) <= REGRET_BAR
    )
    deltas = []
    scenario_rows = []
    for scenario in CONFIRM_SCENARIOS:
        for left, right, name in (("R1", "S", "R1-S"), ("R1", "B0", "R1-B0"), ("R1", "B1", "R1-B1")):
            values = []
            for row in rank_rows:
                if row["model"] != left or int(row["scenario_id"]) != int(scenario):
                    continue
                partner = next(
                    item["selected_j15"]
                    for item in rank_rows
                    if item["model"] == right and item["seed"] == row["seed"] and item["scenario_id"] == row["scenario_id"] and item["time_step"] == row["time_step"]
                )
                delta = float(row["selected_j15"]) - float(partner)
                values.append(delta)
                deltas.append({"comparison": name, "scenario_id": int(scenario), "time_step": int(row["time_step"]), "seed": int(row["seed"]), "delta_j15": delta})
            scenario_rows.append({"comparison": name, "scenario_id": int(scenario), "delta_j15": _mean(values), "direction": _direction(_mean(values))})
    _write_csv("phase5zd_scenario_comparison.csv", scenario_rows)
    seed_rows = []
    for seed in SEEDS:
        for model in ("S", "R1", "B0", "B1", "REF1", "REF8"):
            chosen = [row for row in rank_rows if row["model"] == model and int(row["seed"]) == int(seed)]
            reg = [row for row in regression_rows if row["model"] == model and int(row["seed"]) == int(seed)]
            seed_rows.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "selected_j15": _mean(row["selected_j15"] for row in chosen),
                    "normalized_regret": _mean(row["normalized_regret"] for row in chosen),
                    "top1": _mean(row["top1"] for row in chosen),
                    "top3": _mean(row["top3"] for row in chosen),
                    "top5": _mean(row["top5"] for row in chosen),
                    "spearman": _mean(row["spearman"] for row in chosen),
                    "mae": _mean(row["mae"] for row in reg) if reg else "",
                    "r2": _mean(row["r2"] for row in reg) if reg else "",
                    "collapse_rate": _mean(row["collapsed"] for row in chosen),
                }
            )
    _write_csv("phase5zd_seed_stability.csv", seed_rows)
    ablation_rows = []
    for group, indexes in RESPONSE_GROUPS.items():
        disagreements = []
        drops = []
        regrets = []
        for state in confirm_states:
            raw = np.array(state["response_raw"], copy=True)
            raw[..., list(indexes)] = 0.0
            normalized = _apply_norm(raw, norm)
            relational = _relational(state, center, scale)
            for seed in SEEDS:
                full = scorebook[("R1", int(seed), int(state["scenario_id"]), int(state["time_step"]))]
                ablated, _elapsed = _predict(response_models[int(seed)], state, normalized, relational)
                full_choice = int(np.argsort(-full, kind="stable")[0])
                ablated_choice = int(np.argsort(-ablated, kind="stable")[0])
                disagreements.append(float(full_choice != ablated_choice))
                drops.append(float(state["j_mappo"][full_choice] - state["j_mappo"][ablated_choice]))
                regrets.append(float(metrics_at(ablated, state["j_mappo"], ablated_choice)["normalized_regret"]))
        ablation_rows.append(
            {
                "group": group,
                "disagreement": _mean(disagreements),
                "selected_j15_drop": _mean(drops),
                "ablated_normalized_regret": _mean(regrets),
                "noticeable": bool(_mean(disagreements) >= DISAGREEMENT_BAR or _mean(drops) >= J15_DROP_BITS),
            }
        )
    _write_csv("phase5zd_signal_ablation.csv", ablation_rows)
    case_rows = []
    case_state = next(state for state in confirm_states if int(state["scenario_id"]) == CASE_SCENARIO and int(state["time_step"]) == CASE_TIME)
    for seed in SEEDS:
        for model in ("S", "R1", "B0", "B1", "REF1"):
            scores = scorebook[(model, int(seed), CASE_SCENARIO, CASE_TIME)]
            pred_index = case_state["vectors"].index(CASE_PREDICTED)
            opt_index = case_state["vectors"].index(CASE_OPTIMAL)
            case_rows.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "v_pred": float(scores[pred_index]),
                    "v_opt": float(scores[opt_index]),
                    "opt_rank": _named_rank(scores, case_state["vectors"], CASE_OPTIMAL),
                    "pred_j15": 4.0e6,
                    "opt_j15": 5.0e6,
                }
            )
    _write_csv("phase5zd_310114_t15.csv", case_rows)
    residual_rows = []
    with PHASE5W_RESIDUAL.open(encoding="utf-8", newline="") as handle:
        residual_cases = list(csv.DictReader(handle))
    for item in residual_cases:
        scenario = int(item["scenario_id"])
        moment = int(item["time_step"])
        assert_consumed(scenario)
        for seed in SEEDS:
            selected = {}
            for model in ("B0", "S", "R1", "REF1", "REF8"):
                match = next(row for row in rank_rows if row["model"] == model and int(row["seed"]) == int(seed) and int(row["scenario_id"]) == scenario and int(row["time_step"]) == moment)
                selected[model] = float(match["selected_j15"])
            oracle = float(next(row["selected_j15"] for row in rank_rows if row["model"] == "O" and int(row["seed"]) == int(seed) and int(row["scenario_id"]) == scenario and int(row["time_step"]) == moment))
            residual_rows.append(
                {
                    "scenario_id": scenario,
                    "time_step": moment,
                    "seed": int(seed),
                    "predicted_assignment": item["predicted_assignment"],
                    "optimal_assignment": item["optimal_assignment"],
                    "phase5v_j15": selected["B0"],
                    "static_j15": selected["S"],
                    "response_j15": selected["R1"],
                    "ref1_j15": selected["REF1"],
                    "ref8_j15": selected["REF8"],
                    "oracle_j15": oracle,
                    "direction_vs_phase5v": _direction(selected["R1"] - selected["B0"]),
                }
            )
    _write_csv("phase5zd_residual_cases.csv", residual_rows)
    per_state = (timing_per_state(train_states, audit["timing"]), timing_per_state(confirm_states, confirm_audit["timing"]))
    runtime_rows = [
        {"split": "train", **per_state[0], "value_forward_ms": ""},
        {
            "split": "confirmation",
            **per_state[1],
            "value_forward_ms": _mean(value_ms),
            "phase5x_h1_ms": _mean(float(np.mean(state["elapsed_ms"])) for state in confirm_states),
            "phase5x_h5_ms": _horizon_elapsed(CONFIRM_SCENARIOS, 5),
            "phase5x_h8_ms": _horizon_elapsed(CONFIRM_SCENARIOS, 8),
        },
    ]
    runtime_rows[1]["total_ms"] = float(per_state[1]["encoder_ms"]) + float(per_state[1]["mappo_action_ms"]) + float(per_state[1]["env_h1_ms"]) + float(per_state[1]["response_extract_ms"]) + float(runtime_rows[1]["value_forward_ms"])
    _write_csv("phase5zd_runtime.csv", runtime_rows)
    comparison = {}
    for name in ("R1-S", "R1-B0", "R1-B1"):
        comparison[name] = cluster_bootstrap([row for row in deltas if row["comparison"] == name], "delta_j15", draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    scenario_counts = {
        name: {
            "positive": int(sum(row["direction"] == "positive" for row in scenario_rows if row["comparison"] == name)),
            "tie": int(sum(row["direction"] == "tie" for row in scenario_rows if row["comparison"] == name)),
            "negative": int(sum(row["direction"] == "negative" for row in scenario_rows if row["comparison"] == name)),
        }
        for name in ("R1-S", "R1-B0", "R1-B1")
    }
    seed_delta = []
    for seed in SEEDS:
        r1 = _mean(row["selected_j15"] for row in seed_rows if row["model"] == "R1" and int(row["seed"]) == int(seed))
        static = _mean(row["selected_j15"] for row in seed_rows if row["model"] == "S" and int(row["seed"]) == int(seed))
        seed_delta.append(_direction(r1 - static))
    r1_summary = next(row for row in summary_rows if row["model"] == "R1")
    case_ranks = [int(row["opt_rank"]) for row in case_rows if row["model"] == "R1"]
    case_median = float(np.median(case_ranks))
    noticeable = {row["group"]: bool(row["noticeable"]) for row in ablation_rows}
    signal_used = bool(noticeable["sinr"] or noticeable["progress"])
    mean_delta = float(comparison["R1-S"]["mean"])
    confirmed = bool(
        mean_delta > 0.0
        and float(comparison["R1-S"]["low"]) > 0.0
        and float(comparison["R1-B0"]["mean"]) > 0.0
        and scenario_counts["R1-S"]["positive"] >= MIN_SCENARIO_UP
        and seed_delta.count("positive") >= MIN_SEED_UP
        and float(r1_summary["top3"]) >= TOP3_BAR
        and float(r1_summary["normalized_regret"]) <= REGRET_BAR
        and case_median <= CASE_TOP_RANK
        and signal_used
    )
    gate = choose_gate(integrity_ok=True, ref1_reproduced=ref1_ok, confirmed=confirmed, mean_improved=mean_delta > 0.0)
    grouped_cases = defaultdict(list)
    for row in residual_rows:
        grouped_cases[(row["scenario_id"], row["time_step"], row["predicted_assignment"])].append(row)
    repaired = unchanged = worsened = 0
    for rows in grouped_cases.values():
        direction = _direction(_mean(float(row["response_j15"]) - float(row["phase5v_j15"]) for row in rows))
        repaired += int(direction == "positive")
        unchanged += int(direction == "tie")
        worsened += int(direction == "negative")
    payload = {
        "gate": gate,
        "planner_ready": False,
        "deployment_verified": False,
        "trained": True,
        "opened_310118": False,
        "decide_called": False,
        "frozen_for_response_distillation": gate == "RESP-A",
        "parameter_count": parameter_count,
        "encoder_grad_abs_sum": encoder_grad,
        "mappo_grad_abs_sum": actor_grad,
        "ref1_reproduced": ref1_ok,
        "confirm_sinr_gap": confirm_audit["sinr_gap"],
        "confirm_progress_gap": confirm_audit["progress_gap"],
        "confirm_payload_gap": confirm_audit["payload_gap"],
        "comparison": comparison,
        "scenario_counts": scenario_counts,
        "seed_directions": seed_delta,
        "r1_top3": r1_summary["top3"],
        "r1_normalized_regret": r1_summary["normalized_regret"],
        "case_opt_rank_median": case_median,
        "case_opt_ranks": case_ranks,
        "signal_noticeable": noticeable,
        "residual_seed0": {"repaired": repaired, "unchanged": unchanged, "worsened": worsened},
        "runtime_confirmation_ms": runtime_rows[1]["total_ms"],
        "selected_j15": summary_rows,
    }
    _write_gate(payload)
    _report(payload, summary_rows, ablation_rows, case_rows, seed_rows, comparison, scenario_counts, seed_delta, runtime_rows[1])
    print(json.dumps(_plain({"gate": gate, "frozen_for_response_distillation": gate == "RESP-A"}), indent=2), flush=True)
    return payload


def timing_per_state(states: list[dict], timing: dict) -> dict[str, float]:
    width = max(len(states), 1)
    return {
        "encoder_ms": float(timing["encoder"]) / width,
        "mappo_action_ms": float(timing["action"]) / width,
        "env_h1_ms": float(timing["env"]) / width,
        "response_extract_ms": float(timing["extract"]) / width,
    }


def _h1_reference_horizon(allowed: tuple[int, ...], horizon: int) -> dict[tuple[int, int, str], float]:
    table = {}
    with PHASE5X_ROLLOUTS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scenario = int(row["scenario_id"])
            if scenario not in allowed or int(row["horizon"]) != int(horizon):
                continue
            assert_consumed(scenario)
            table[(scenario, int(row["time_step"]), row["assignment_vector"])] = float(row["q_exec_h"])
    return table


def _horizon_elapsed(allowed: tuple[int, ...], horizon: int) -> float:
    values = []
    with PHASE5X_ROLLOUTS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scenario = int(row["scenario_id"])
            if scenario not in allowed or int(row["horizon"]) != int(horizon):
                continue
            values.append(float(row["elapsed_ms"]))
    return _mean(values)


def _report(payload, summary_rows, ablation_rows, case_rows, seed_rows, comparison, scenario_counts, seed_delta, runtime) -> None:
    by_model = {row["model"]: row for row in summary_rows}
    lines = [
        "# Phase-5ZD：一步执行响应条件下的联合价值",
        "",
        "本阶段训练了参数量完全相同的 Model-S 和 Model-R1。HGNN、REP、TEMP、候选池和 MAPPO 都冻结。H=1 payload 不是输入。",
        "",
        f"Gate = `{payload['gate']}`。`FROZEN_FOR_RESPONSE_DISTILLATION = {payload['frozen_for_response_distillation']}`。",
        "",
        "`planner_ready = False`",
        "",
        "`deployment_verified = False`",
        "",
        "没有打开 310118–310125，没有调用 deployment 决策接口。310110–310117 在 5 个 seed 的 epoch-10 checkpoint 写盘之后才读取。",
        "",
        "## 最终选中的 J15",
        "",
        "| 模型 | mean J15 | median J15 | normalized regret | Top-1 | Top-3 | Top-5 | Spearman |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for model in ("B0", "B1", "S", "R1", "REF1", "REF8", "O"):
        row = by_model[model]
        lines.append(
            f"| {model} | {row['selected_j15_mean']:.0f} | {row['selected_j15_median']:.0f} | {_fmt(row['normalized_regret'])} | {_fmt(row['top1'])} | {_fmt(row['top3'])} | {_fmt(row['top5'])} | {_fmt(row['spearman'])} |"
        )
    gap = comparison["R1-S"]
    lines.extend(
        [
            "",
            "## Q1–Q3. H1 response 是否优于匹配的静态对照？",
            "",
            f"R1 相对 S 的 mean selected J15 差是 {gap['mean']:.0f} bit，scenario-cluster 95% CI 是 [{gap['low']:.0f}, {gap['high']:.0f}]。",
            f"8 个 development scenario 中，正向 {scenario_counts['R1-S']['positive']}，持平 {scenario_counts['R1-S']['tie']}，负向 {scenario_counts['R1-S']['negative']}。",
            f"5 个 seed 的方向是 {', '.join(seed_delta)}。",
            "",
            "## Q4–Q5. SINR 和 progress 有没有被用到？",
            "",
        ]
    )
    for row in ablation_rows:
        lines.append(
            f"- 置零 {row['group']}：argmax disagreement {_fmt(row['disagreement'])}，selected J15 下降 {row['selected_j15_drop']:.0f} bit，noticeable = {row['noticeable']}。"
        )
    lines.extend(["", "## Q6. 310114 t=15", ""])
    for model in ("S", "R1"):
        ranks = [str(int(row["opt_rank"])) for row in case_rows if row["model"] == model]
        values = [row for row in case_rows if row["model"] == model]
        lines.append(
            f"- Model-{model if model != 'S' else 'S'}：5 个 seed 的 A_opt rank 是 {', '.join(ranks)}。seed 0 的 V(A_pred)={values[0]['v_pred']:.4f}，V(A_opt)={values[0]['v_opt']:.4f}。"
        )
    lines.extend(
        [
            "",
            "## Q7. residual cases",
            "",
            f"10 个 residual cases 上，按 5 个 seed 的平均 selected J15，R1 相对 Phase-5V：修复 {payload['residual_seed0']['repaired']}，不变 {payload['residual_seed0']['unchanged']}，恶化 {payload['residual_seed0']['worsened']}。",
            "",
            "## Q8. 与 H1 / H8 Score-B 的差距",
            "",
            f"R1 mean selected J15 是 {by_model['R1']['selected_j15_mean']:.0f}，REF1 是 {by_model['REF1']['selected_j15_mean']:.0f}，REF8 是 {by_model['REF8']['selected_j15_mean']:.0f}。",
            f"R1 Spearman {_fmt(by_model['R1']['spearman'])}，REF1 {_fmt(by_model['REF1']['spearman'])}，REF8 {_fmt(by_model['REF8']['spearman'])}。",
            f"R1 normalized regret {_fmt(by_model['R1']['normalized_regret'])}，REF1 {_fmt(by_model['REF1']['normalized_regret'])}，REF8 {_fmt(by_model['REF8']['normalized_regret'])}。",
            "",
            "## Q9–Q12",
            "",
            "低层执行信息进入的是高层价值，不是第二个目标选择器。",
            f"一个 state 的 120 个候选，确认集平均耗时：encoder {runtime['encoder_ms']:.1f} ms，MAPPO action {runtime['mappo_action_ms']:.1f} ms，H=1 env {runtime['env_h1_ms']:.1f} ms，response extract {runtime['response_extract_ms']:.1f} ms，value forward {float(runtime['value_forward_ms']):.1f} ms，合计 {float(runtime['total_ms']):.1f} ms。",
            f"Phase-5X 单个候选的累计 shadow 时间：H=1 {runtime['phase5x_h1_ms']:.2f} ms，H=5 {runtime['phase5x_h5_ms']:.2f} ms，H=8 {runtime['phase5x_h8_ms']:.2f} ms。",
            f"当前 Gate 是 {payload['gate']}。只有 RESP-A 才进入 H1 response distillation。无论结果如何，都不把 120 次在线 H=1 rollout 放进正式 planner，也不增加 HGNN，也不修改 MAPPO。",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
