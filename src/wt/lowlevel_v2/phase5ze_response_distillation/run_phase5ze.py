"""Train the H=1 response predictor. The Phase-5ZD value models stay frozen."""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from wt.htw_phase100.contract import TaskOption
from wt.l3b.config import L3BConfig
from wt.lowlevel_v2.contracts import refuse_rematch
from wt.lowlevel_v2.phase5b_alignment.diagnose import LockedExecutor
from wt.lowlevel_v2.phase5g_closed_loop.collect import load_frozen_actor
from wt.lowlevel_v2.integration_phase1.stats import spearman
from wt.lowlevel_v2.phase5l_joint_encoder.run_phase5l import _load_matching
from wt.lowlevel_v2.phase5m_joint_planner.run_phase5m import _load_proposed
from wt.lowlevel_v2.phase5n_joint_planner.run_phase5n import _features
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.metrics import metrics_at, quantize
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.run_phase5q import _edge_features
from wt.lowlevel_v2.phase5r_nonadditive_generalization.metrics import cluster_bootstrap
from wt.lowlevel_v2.phase5r_nonadditive_generalization.world import candidate_pool
from wt.lowlevel_v2.phase5s_multiscenario_nonadditive.protocol import MODEL_POOL_WIDTH, PHASE5O_LABELS, PHASE5R_LABELS
from wt.lowlevel_v2.phase5t_scorer_failure_audit.protocol import PHASE5S_LABELS
from wt.lowlevel_v2.phase5u_representation_repair.features import relational_matrix
from wt.lowlevel_v2.phase5u_representation_repair.protocol import EDGE_FEATURES as REP_EDGE_FEATURES
from wt.lowlevel_v2.phase5u_representation_repair.run_phase5u import _replay, _snapshot
from wt.lowlevel_v2.phase5v_temporal_representation.protocol import EDGE_FEATURES as TEMP_EDGE_FEATURES
from wt.lowlevel_v2.phase5v_temporal_representation.run_phase5v import _checkpoint as _temporal_checkpoint
from wt.lowlevel_v2.phase5v_temporal_representation.run_phase5v import _combine, _history_snapshot, _scale, _vector
from wt.lowlevel_v2.phase5v_temporal_representation.temporal import temporal_matrix
from wt.lowlevel_v2.phase5x_executor_lookahead.shadow import fingerprint, max_abs_diff
from wt.lowlevel_v2.phase5zd_executor_response_value.protocol import (
    CHECKPOINTS as PHASE5ZD_CHECKPOINTS,
)
from wt.lowlevel_v2.phase5zd_executor_response_value.protocol import (
    OUTPUT as PHASE5ZD_OUTPUT,
)
from wt.lowlevel_v2.phase5zd_executor_response_value.protocol import (
    PAYLOAD_MATCH_ABS,
    PROGRESS_MATCH_ABS,
    SINR_MATCH_ABS,
)
from wt.lowlevel_v2.phase5zd_executor_response_value.response import one_step
from wt.lowlevel_v2.phase5zd_executor_response_value.scorer import ExecutorConditionedJointScorer
from wt.round30.ppo import seed_everything

from .metrics import choose_gate, retention_ratio
from .predictor import BINARY_INDEX, CONTINUOUS_INDEX, OneStepExecutorResponsePredictor, response_loss
from .protocol import (
    BINARY_FEATURES,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CASE_OPTIMAL,
    CASE_PREDICTED,
    CASE_SCENARIO,
    CASE_TIME,
    CHECKPOINTS,
    CONFIRM_SCENARIOS,
    CONTINUOUS_FEATURES,
    EDGE_MATCH_ABS,
    EPOCHS,
    GRAD_CLIP,
    HIDDEN,
    LABEL_MATCH_ABS,
    LEARNING_RATE,
    MAX_J15_DROP_VS_TRUE,
    MAX_SCENARIO_DOWN,
    MIN_SCENARIO_GE,
    MIN_SEED_UP,
    OUTPUT,
    PAYLOAD_BLOCK_BITS,
    PERMUTATION_ABS,
    PLANNER_READY,
    REGRET_BAR,
    REPORT,
    RESPONSE_FEATURES,
    RESPONSE_GROUPS,
    RETENTION_BAR,
    ROOT,
    RUNTIME_REPEATS,
    RUNTIME_WARMUP,
    SEEDS,
    SPEEDUP_BAR,
    TIE_BITS,
    TIMES,
    TOP3_BAR,
    TRAIN_SCENARIOS,
    WEIGHT_DECAY,
    assert_consumed,
)
from .tokens import TOKEN_BLOCKS, build_tokens, rep_temp_edges, token_dim
from .transition import frozen_h1

PHASE5X_ROLLOUTS = ROOT / "results/phase5x_executor_lookahead" / "phase5x_executor_rollouts.csv"
PHASE5W_RESIDUAL = ROOT / "results/phase5w_residual_mechanism_audit" / "phase5w_residual_case_hybrids.csv"
PHASE5ZD_NORM = PHASE5ZD_OUTPUT / "phase5zd_response_normalization.json"
PHASE5ZD_SPEC = PHASE5ZD_OUTPUT / "phase5zd_response_feature_spec.json"


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


def _direction(value: float) -> str:
    if abs(float(value)) <= float(TIE_BITS):
        return "tie"
    return "positive" if float(value) > 0.0 else "negative"


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


def _horizon_table(allowed: tuple[int, ...], horizon: int) -> dict[tuple[int, int, str], dict[str, float]]:
    table = {}
    with PHASE5X_ROLLOUTS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scenario = int(row["scenario_id"])
            if scenario not in allowed or int(row["horizon"]) != int(horizon):
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


def _drop_batch(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim >= 3 and int(values.shape[0]) == 1:
        return values[0]
    return values


def _block_stats(rows: list[np.ndarray], width: int) -> np.ndarray:
    if not rows:
        return np.zeros(4 * width, dtype=np.float32)
    stacked = np.stack(rows, axis=0)
    parts = []
    for feature in range(width):
        column = stacked[:, feature]
        parts.extend((float(np.mean(column)), float(np.max(column)), float(np.min(column)), float(np.std(column))))
    return np.asarray(parts, dtype=np.float32)


def _named_rank(scores: np.ndarray, vectors: list[str], target: str) -> int:
    index = list(vectors).index(str(target))
    values = np.asarray(scores, dtype=np.float64)
    return int(1 + np.sum(values > values[index]))


def _top5(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")[:5]
    levels = quantize(labels)
    return float(np.any(levels[order] == np.max(levels)))


def _apply_affine(raw: np.ndarray, norm: dict[str, np.ndarray]) -> np.ndarray:
    return ((raw - norm["mean"]) / norm["std"]).astype(np.float32)


def _fit_input(states: list[dict]) -> dict[str, np.ndarray]:
    stacked = np.concatenate([state["tokens"].reshape(-1, token_dim()) for state in states], axis=0)
    std = np.maximum(np.std(stacked, axis=0), 1.0e-6)
    return {"mean": np.mean(stacked, axis=0).astype(np.float32), "std": std.astype(np.float32)}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _to_raw(prediction: np.ndarray, response_norm: dict[str, np.ndarray]) -> np.ndarray:
    raw = np.zeros(prediction.shape, dtype=np.float64)
    continuous = list(CONTINUOUS_INDEX)
    binary = list(BINARY_INDEX)
    raw[..., continuous] = prediction[..., continuous] * response_norm["std"][continuous] + response_norm["mean"][continuous]
    raw[..., binary] = _sigmoid(prediction[..., binary])
    return raw


def _to_value(prediction: np.ndarray, response_norm: dict[str, np.ndarray]) -> np.ndarray:
    """Continuous channels are already normalized. Binary probabilities use the frozen normalizer."""
    value = np.zeros(prediction.shape, dtype=np.float32)
    continuous = list(CONTINUOUS_INDEX)
    binary = list(BINARY_INDEX)
    value[..., continuous] = prediction[..., continuous]
    probability = _sigmoid(prediction[..., binary])
    value[..., binary] = (probability - response_norm["mean"][binary]) / response_norm["std"][binary]
    return value


def _batch(state: dict, response: np.ndarray, relational: np.ndarray):
    features = {}
    for key, value in state["embedded"].items():
        features[key] = value.detach() if torch.is_tensor(value) else torch.as_tensor(value)
    features["z_rel"] = torch.as_tensor(relational, dtype=torch.float32)
    features["response"] = torch.as_tensor(response, dtype=torch.float32)
    return features, torch.as_tensor(state["assignments"], dtype=torch.long)


def _relational(state: dict, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return _combine(state["z_rel"], _scale(state["z_temp_raw"], center, scale))


def _predict_value(scorer, state: dict, response: np.ndarray, relational: np.ndarray) -> tuple[np.ndarray, float]:
    features, assignments = _batch(state, response, relational)
    with torch.no_grad():
        started = time.perf_counter()
        values = scorer(features, assignments).detach().cpu().numpy()
        elapsed = (time.perf_counter() - started) * 1000.0
    return values, elapsed


def _predict_response(model, tokens: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.as_tensor(tokens, dtype=torch.float32)).detach().cpu().numpy()


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


def _feature_block(prediction: np.ndarray, truth: np.ndarray) -> list[dict]:
    rows = []
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1, len(RESPONSE_FEATURES))
    true = np.asarray(truth, dtype=np.float64).reshape(-1, len(RESPONSE_FEATURES))
    groups = {index: name for name, indexes in RESPONSE_GROUPS.items() for index in indexes}
    for index, name in enumerate(RESPONSE_FEATURES):
        left = pred[:, index]
        right = true[:, index]
        residual = left - right
        center = float(np.mean(right))
        square = float(np.sum((right - center) ** 2))
        pearson = float("nan")
        if float(np.std(left)) > 1.0e-12 and float(np.std(right)) > 1.0e-12:
            pearson = float(np.corrcoef(left, right)[0, 1])
        rows.append(
            {
                "feature": name,
                "group": groups.get(index, "other"),
                "mae": float(np.mean(np.abs(residual))),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "r2": float(1.0 - np.sum(residual**2) / square) if square > 1.0e-12 else float("nan"),
                "bias": float(np.mean(residual)),
                "pearson": pearson,
                "spearman": spearman(left, right),
                "n": int(left.size),
            }
        )
    return rows


def _option(env, assignment: np.ndarray):
    chosen = np.asarray(assignment, dtype=np.int64).copy()
    option = TaskOption(
        targets=chosen.copy(),
        decision_time=int(env.t),
        remaining_hold=15,
        uav_mask=np.ones(env.config.num_uavs, dtype=bool),
        buoy_mask=np.ones(env.config.num_buoys, dtype=bool),
        association=chosen.copy(),
        urgency=None,
    )
    refuse_rematch(option)
    return option


def _prepare(scenarios, labels, reference, frozen, matching, actor, config, center_ready: bool) -> tuple[list[dict], dict]:
    del center_ready
    states = []
    largest = 0.0
    sinr_gap = 0.0
    progress_gap = 0.0
    payload_gap = 0.0
    edge_gap = 0.0
    parity_gap = 0.0
    for scenario in scenarios:
        for moment in TIMES:
            assert_consumed(int(scenario))
            env, movement, batch, diagnosis = _replay(int(scenario), int(moment))
            embedded = {key: value.detach() if torch.is_tensor(value) else value for key, value in _edge_features(frozen, batch).items()}
            additive = _features(frozen, matching, batch)
            logits = additive["logits"].detach().cpu().numpy()
            pool = candidate_pool(logits, MODEL_POOL_WIDTH, int(logits.shape[0]), int(logits.shape[1]))
            vectors = [_vector(row) for row in pool["assignments"]]
            recorded = {key[2] for key in labels if key[0] == int(scenario) and key[1] == int(moment)}
            if set(vectors) != recorded:
                raise RuntimeError(f"pool does not match stored labels at {scenario} t={moment}")
            if len(vectors) != 120:
                raise RuntimeError(f"candidate pool width changed at {scenario} t={moment}")
            relational = _snapshot(env, diagnosis)
            history = _history_snapshot(env, batch)
            z_rel = relational_matrix(relational, pool["assignments"], config)
            z_temp = temporal_matrix(history, pool["assignments"], config)
            h_u = _drop_batch(embedded["h_u"].detach().cpu().numpy())
            h_b = _drop_batch(embedded["h_b"].detach().cpu().numpy())
            h_edge = _drop_batch(embedded["h_edge"].detach().cpu().numpy())
            responses = []
            actions = []
            reps = []
            temps = []
            for index, assignment in enumerate(pool["assignments"]):
                result = frozen_h1(env, assignment, actor, movement)
                responses.append(result["response"])
                actions.append(result["action"])
                rep, temp = rep_temp_edges(relational, history, assignment, config)
                reps.append(rep)
                temps.append(temp)
                largest = max(largest, float(result["integrity"]))
                ref = reference[(int(scenario), int(moment), vectors[index])]
                sinr_gap = max(sinr_gap, abs(float(result["mean_sinr"]) - float(ref["mean_sinr"])))
                progress_gap = max(progress_gap, abs(float(result["progress_sum"]) - float(ref["progress_sum"])))
                payload_gap = max(payload_gap, abs(float(result["payload"]) - float(ref["payload"])))
                if index == 0:
                    selected = [rep[uav] for uav, target in enumerate(assignment) if int(target) >= 0]
                    reference_rep = relational_matrix(relational, np.asarray(assignment, dtype=np.int64).reshape(1, -1), config)[0, : 4 * len(REP_EDGE_FEATURES)]
                    reference_temp = temporal_matrix(history, np.asarray(assignment, dtype=np.int64).reshape(1, -1), config)[0, : 4 * len(TEMP_EDGE_FEATURES)]
                    edge_gap = max(edge_gap, float(np.max(np.abs(_block_stats(selected, len(REP_EDGE_FEATURES)) - reference_rep))))
                    selected_temp = [temp[uav] for uav, target in enumerate(assignment) if int(target) >= 0]
                    edge_gap = max(edge_gap, float(np.max(np.abs(_block_stats(selected_temp, len(TEMP_EDGE_FEATURES)) - reference_temp))))
                if int(scenario) == int(scenarios[0]) and int(moment) == int(TIMES[0]) and index < 3:
                    parity_gap = max(parity_gap, float(np.max(np.abs(result["response"] - one_step(env, assignment, actor, movement)["response"]))))
            tokens = build_tokens(h_u, h_b, h_edge, np.stack(reps), np.stack(temps), np.asarray(pool["assignments"]), np.stack(actions))
            state = {
                "scenario_id": int(scenario),
                "time_step": int(moment),
                "assignments": np.asarray(pool["assignments"], dtype=np.int64),
                "vectors": vectors,
                "j_mappo": np.asarray([labels[(int(scenario), int(moment), vector)] for vector in vectors], dtype=np.float64),
                "embedded": embedded,
                "z_rel": z_rel,
                "z_temp_raw": z_temp,
                "response_raw": np.stack(responses).astype(np.float64),
                "tokens": tokens,
                "actions": np.stack(actions).astype(np.float64),
                "q_exec8": np.asarray(
                    [_horizon_value(reference, int(scenario), int(moment), vector) for vector in vectors],
                    dtype=np.float64,
                ),
            }
            if not states:
                state["runtime_env"] = env
                state["runtime_movement"] = np.array(movement, copy=True)
                state["runtime_batch"] = batch
                state["runtime_h"] = (h_u, h_b, h_edge)
            states.append(state)
            print(f"state {scenario} t={moment}", flush=True)
    stacked = np.concatenate([state["response_raw"].reshape(-1, len(RESPONSE_FEATURES)) for state in states], axis=0)
    return states, {
        "largest": largest,
        "sinr_gap": sinr_gap,
        "progress_gap": progress_gap,
        "payload_gap": payload_gap,
        "edge_gap": edge_gap,
        "parity_gap": parity_gap,
        "response_mean": np.mean(stacked, axis=0),
        "n": int(stacked.shape[0]),
    }


def _horizon_value(reference, scenario: int, moment: int, vector: str) -> float:
    row = reference.get((scenario, moment, vector))
    if row is None:
        return float("nan")
    return float(row["q_exec"])


def _load_response_norm() -> dict[str, np.ndarray]:
    payload = json.loads(PHASE5ZD_NORM.read_text(encoding="utf-8"))
    if list(payload["features"]) != list(RESPONSE_FEATURES):
        raise RuntimeError("Phase-5ZD normalization schema drifted")
    return {
        "mean": np.asarray(payload["mean"], dtype=np.float64),
        "std": np.asarray(payload["std"], dtype=np.float64),
    }


def _load_value(kind: str, seed: int) -> ExecutorConditionedJointScorer:
    module = ExecutorConditionedJointScorer()
    payload = torch.load(PHASE5ZD_CHECKPOINTS / f"phase5zd_{kind}_seed{int(seed)}.pt", map_location="cpu", weights_only=False)
    module.load_state_dict(payload["state_dict"])
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def _load_predictor(seed: int) -> OneStepExecutorResponsePredictor:
    payload = torch.load(CHECKPOINTS / f"phase5ze_response_seed{int(seed)}.pt", map_location="cpu", weights_only=False)
    if int(payload["epoch"]) != int(EPOCHS):
        raise RuntimeError("response predictor checkpoint is not epoch 10")
    model = OneStepExecutorResponsePredictor(int(payload["input_dim"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _train_one(seed: int, states: list[dict], input_norm: dict[str, np.ndarray], response_norm: dict[str, np.ndarray]):
    seed_everything(int(seed))
    model = OneStepExecutorResponsePredictor(token_dim())
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    curves = []
    binary_index = list(BINARY_INDEX)
    for epoch in range(1, int(EPOCHS) + 1):
        model.train()
        totals, hubers, bces, grads = [], [], [], []
        for state in states:
            if int(state["scenario_id"]) not in TRAIN_SCENARIOS:
                raise RuntimeError("confirmation state entered the response gradient")
            optimizer.zero_grad(set_to_none=True)
            prediction = model(torch.as_tensor(_apply_affine(state["tokens"], input_norm), dtype=torch.float32))
            raw = state["response_raw"]
            normalized = _apply_affine(raw, response_norm)
            binary = torch.as_tensor(raw[..., binary_index], dtype=torch.float32)
            loss, huber, bce = response_loss(prediction, torch.as_tensor(normalized, dtype=torch.float32), binary)
            if not bool(torch.isfinite(loss).all()):
                raise RuntimeError(f"response loss became non-finite at seed {seed} epoch {epoch}")
            loss.backward()
            grads.append(_grad_norm(model))
            clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            totals.append(float(loss.detach()))
            hubers.append(float(huber.detach()))
            bces.append(float(bce.detach()))
        model.eval()
        predicted = []
        truth = []
        with torch.no_grad():
            for state in states:
                head = _predict_response(model, _apply_affine(state["tokens"], input_norm))
                predicted.append(_to_raw(head, response_norm))
                truth.append(state["response_raw"])
        block = _feature_block(np.concatenate(predicted), np.concatenate(truth))
        row = {
            "seed": int(seed),
            "epoch": int(epoch),
            "total_loss": _mean(totals),
            "continuous_huber": _mean(hubers),
            "binary_bce": _mean(bces),
            "grad_norm": _mean(grads),
            "param_norm": _param_norm(model),
            "constant_features": int(sum(float(np.std(np.concatenate(predicted)[..., index])) < 1.0e-6 for index in range(len(RESPONSE_FEATURES)))),
        }
        for item in block:
            row[f"mae_{item['feature']}"] = item["mae"]
            row[f"rmse_{item['feature']}"] = item["rmse"]
            row[f"r2_{item['feature']}"] = item["r2"]
        curves.append(row)
        print(f"predictor seed {seed} epoch {epoch} loss={row['total_loss']:.4f}", flush=True)
    model.eval()
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "seed": int(seed),
            "epoch": int(EPOCHS),
            "model": "response_predictor",
            "input_dim": token_dim(),
            "state_dict": model.state_dict(),
            "input_mean": input_norm["mean"],
            "input_std": input_norm["std"],
        },
        CHECKPOINTS / f"phase5ze_response_seed{int(seed)}.pt",
    )
    return curves


def _rank_row(model: str, seed: int, state: dict, scores: np.ndarray) -> dict:
    chosen = int(np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")[0])
    measured = metrics_at(scores, state["j_mappo"], chosen)
    optimum = state["vectors"][int(np.argmax(quantize(state["j_mappo"])))]
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
        "optimum_rank": _named_rank(scores, state["vectors"], optimum),
        "assignment_vector": state["vectors"][chosen],
        "agreement_with_r1": "",
    }


def _permute_audit(model, state: dict, input_norm: dict[str, np.ndarray]) -> float:
    perm = np.asarray([2, 0, 4, 1, 5, 3], dtype=np.int64)
    h_u, h_b, h_edge = state["runtime_h"]
    original = _apply_affine(state["tokens"], input_norm)
    rebuilt = build_tokens(
        h_u[perm],
        h_b,
        h_edge[perm],
        state["tokens"][:, :, TOKEN_BLOCKS[0][1] * 3 : TOKEN_BLOCKS[0][1] * 3 + len(REP_EDGE_FEATURES)][:, perm, :],
        state["tokens"][:, :, TOKEN_BLOCKS[0][1] * 3 + len(REP_EDGE_FEATURES) : TOKEN_BLOCKS[0][1] * 3 + len(REP_EDGE_FEATURES) + len(TEMP_EDGE_FEATURES)][:, perm, :],
        state["assignments"][:, perm],
        state["actions"][:, perm],
    )
    with torch.no_grad():
        left = model(torch.as_tensor(original, dtype=torch.float32)).detach().cpu().numpy()
        right = model(torch.as_tensor(_apply_affine(rebuilt, input_norm), dtype=torch.float32)).detach().cpu().numpy()
    return float(np.max(np.abs(right - left[:, perm, :])))


def _benchmark(state: dict, actor, frozen, static_scorer, response_scorer, predictor, input_norm, response_norm, center, scale) -> dict:
    env = state["runtime_env"]
    movement = state["runtime_movement"]
    batch = state["runtime_batch"]
    executor = LockedExecutor(actor)
    relational = _relational(state, center, scale)
    true_response = _apply_affine(state["response_raw"], response_norm)
    zero_response = _apply_affine(np.zeros_like(state["response_raw"]), response_norm)

    def _true_once() -> dict[str, float]:
        started = time.perf_counter()
        _edge_features(frozen, batch)
        encoder_ms = (time.perf_counter() - started) * 1000.0
        clone_ms = action_ms = step_ms = extract_ms = 0.0
        for assignment in state["assignments"]:
            result = frozen_h1(env, assignment, actor, movement)
            clone_ms += result["clone_ms"]
            action_ms += result["action_ms"]
            step_ms += result["step_ms"]
            extract_ms += result["extract_ms"]
        _values, value_ms = _predict_value(response_scorer, state, true_response, relational)
        return {
            "encoder_ms": encoder_ms,
            "mappo_action_ms": action_ms,
            "env_clone_ms": clone_ms,
            "env_step_ms": step_ms,
            "response_extract_ms": extract_ms,
            "response_predictor_ms": 0.0,
            "value_ms": value_ms,
            "total_ms": encoder_ms + action_ms + clone_ms + step_ms + extract_ms + value_ms,
        }

    def _predicted_once() -> dict[str, float]:
        before = fingerprint(env)
        started = time.perf_counter()
        _edge_features(frozen, batch)
        encoder_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        fresh_actions = []
        for assignment in state["assignments"]:
            executed, _desired, _delta = executor.act(env, _option(env, assignment), np.array(movement, copy=True))
            fresh_actions.append(np.asarray(executed, dtype=np.float64))
        action_ms = (time.perf_counter() - started) * 1000.0
        mutated = float(max_abs_diff(before, fingerprint(env)))
        h_u, h_b, h_edge = state["runtime_h"]
        started = time.perf_counter()
        rep = state["tokens"][:, :, 384 : 384 + len(REP_EDGE_FEATURES)]
        temp = state["tokens"][:, :, 384 + len(REP_EDGE_FEATURES) : 384 + len(REP_EDGE_FEATURES) + len(TEMP_EDGE_FEATURES)]
        tokens = build_tokens(h_u, h_b, h_edge, rep, temp, state["assignments"], np.stack(fresh_actions))
        head = _predict_response(predictor, _apply_affine(tokens, input_norm))
        predictor_ms = (time.perf_counter() - started) * 1000.0
        _values, value_ms = _predict_value(response_scorer, state, _to_value(head, response_norm), relational)
        return {
            "encoder_ms": encoder_ms,
            "mappo_action_ms": action_ms,
            "env_clone_ms": 0.0,
            "env_step_ms": 0.0,
            "response_extract_ms": 0.0,
            "response_predictor_ms": predictor_ms,
            "value_ms": value_ms,
            "total_ms": encoder_ms + action_ms + predictor_ms + value_ms,
            "caller_mutation": mutated,
        }

    def _static_once() -> dict[str, float]:
        started = time.perf_counter()
        _edge_features(frozen, batch)
        encoder_ms = (time.perf_counter() - started) * 1000.0
        _values, value_ms = _predict_value(static_scorer, state, zero_response, relational)
        return {
            "encoder_ms": encoder_ms,
            "mappo_action_ms": 0.0,
            "env_clone_ms": 0.0,
            "env_step_ms": 0.0,
            "response_extract_ms": 0.0,
            "response_predictor_ms": 0.0,
            "value_ms": value_ms,
            "total_ms": encoder_ms + value_ms,
        }

    for _index in range(int(RUNTIME_WARMUP)):
        _true_once()
        _predicted_once()
        _static_once()
    collected = {"TRUE-H1": [], "PREDICTED-H1": [], "STATIC": []}
    for _index in range(int(RUNTIME_REPEATS)):
        collected["TRUE-H1"].append(_true_once())
        collected["PREDICTED-H1"].append(_predicted_once())
        collected["STATIC"].append(_static_once())
    rows = []
    for name, samples in collected.items():
        row = {"path": name, "repeats": int(RUNTIME_REPEATS)}
        for key in samples[0]:
            if key == "caller_mutation":
                row[key] = max(float(item[key]) for item in samples)
            else:
                row[key] = _mean(item[key] for item in samples)
        rows.append(row)
    true_total = float(rows[0]["total_ms"])
    predicted_total = float(rows[1]["total_ms"])
    rows[1]["speedup_vs_true_h1"] = true_total / predicted_total if predicted_total > 0.0 else float("nan")
    return {"rows": rows, "speedup": float(rows[1]["speedup_vs_true_h1"]), "mutation": float(rows[1].get("caller_mutation", 0.0))}


def _write_gate(payload: dict) -> None:
    (OUTPUT / "phase5ze_response_distillation_gate.json").write_text(json.dumps(_plain(payload), indent=2), encoding="utf-8")


def _finish_early(gate: str, reason: str) -> dict:
    payload = {
        "gate": gate,
        "reason": reason,
        "planner_ready": False,
        "deployment_verified": False,
        "trained": False,
        "opened_310118": False,
        "decide_called": False,
        "frozen_for_fresh_confirmation": False,
    }
    _write_gate(payload)
    REPORT.write_text(
        "\n".join(
            [
                "# Phase-5ZE：一步执行响应蒸馏",
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


def _assigned_mean(response: np.ndarray, assignment: np.ndarray, feature: int) -> float:
    mask = np.asarray(assignment) >= 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(response[mask, feature]))


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    spec = json.loads(PHASE5ZD_SPEC.read_text(encoding="utf-8"))
    if list(spec["features"]) != list(RESPONSE_FEATURES):
        raise RuntimeError("Phase-5ZD response schema is not the registered feature list")
    (OUTPUT / "phase5ze_protocol.json").write_text(
        json.dumps(
            {
                "training": True,
                "value_models_frozen": True,
                "encoder_frozen": True,
                "mappo_frozen": True,
                "j15_in_predictor_loss": False,
                "optimizer": "Adam",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "grad_clip": GRAD_CLIP,
                "epochs": EPOCHS,
                "checkpoint": "epoch_10",
                "seeds": list(SEEDS),
                "batch": "one state per optimizer step; every assignment and every UAV has equal weight",
                "hidden": HIDDEN,
                "loss": "mean Huber on Phase-5ZD-normalized continuous channels + mean BCE on binary channels",
                "confirmation_read_before_checkpoints": False,
                "opened_310118": False,
                "planner_ready": PLANNER_READY,
                "deployment_verified": False,
                "option_hold": 15,
                "h_dyn": 1,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUTPUT / "phase5ze_response_schema.json").write_text(
        json.dumps(
            {
                "authority": "phase5zd_response_feature_spec.json",
                "features": list(RESPONSE_FEATURES),
                "continuous": list(CONTINUOUS_FEATURES),
                "binary": list(BINARY_FEATURES),
                "groups": {name: list(indexes) for name, indexes in RESPONSE_GROUPS.items()},
                "token_blocks": [{"name": name, "dim": int(width)} for name, width in TOKEN_BLOCKS],
                "queue_and_progress": "reused inside the frozen REP and TEMP per-UAV edges",
                "payload_is_input": False,
                "j15_is_input": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
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
    temporal = torch.load(_temporal_checkpoint(0), map_location="cpu", weights_only=False)
    center = np.asarray(temporal["center"], dtype=np.float32)
    scale = np.asarray(temporal["scale"], dtype=np.float32)
    response_norm = _load_response_norm()
    train_labels = _labels("train")
    train_reference = _horizon_table(TRAIN_SCENARIOS, 1)
    train_states, train_audit = _prepare(TRAIN_SCENARIOS, train_labels, train_reference, frozen, matching, actor, config, False)
    aligned = (
        float(train_audit["largest"]) == 0.0
        and float(train_audit["parity_gap"]) <= LABEL_MATCH_ABS
        and float(train_audit["edge_gap"]) <= EDGE_MATCH_ABS
        and float(train_audit["sinr_gap"]) <= SINR_MATCH_ABS
        and float(train_audit["progress_gap"]) <= PROGRESS_MATCH_ABS
        and float(train_audit["payload_gap"]) <= PAYLOAD_MATCH_ABS
        and float(np.max(np.abs(train_audit["response_mean"] - response_norm["mean"]))) <= LABEL_MATCH_ABS
    )
    input_norm = _fit_input(train_states)
    (OUTPUT / "phase5ze_normalization_audit.json").write_text(
        json.dumps(
            {
                "response_normalization": "phase5zd training statistics, reused",
                "response_population": "310100-310109",
                "recomputed_train_count": int(train_audit["n"]),
                "recomputed_mean_max_abs": float(np.max(np.abs(train_audit["response_mean"] - response_norm["mean"]))),
                "input_population": "310100-310109 tokens only",
                "input_mean": [float(value) for value in input_norm["mean"]],
                "input_std": [float(value) for value in input_norm["std"]],
                "confirm_statistics_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUTPUT / "phase5ze_data_lineage.json").write_text(
        json.dumps(
            {
                "response_definition": "phase5zd assemble_response via frozen_h1",
                "parity_gap_vs_one_step": train_audit["parity_gap"],
                "phase5x_sinr_gap": train_audit["sinr_gap"],
                "phase5x_progress_gap": train_audit["progress_gap"],
                "phase5x_payload_gap": train_audit["payload_gap"],
                "caller_fingerprint": train_audit["largest"],
                "edge_gap": train_audit["edge_gap"],
                "j15_source": "existing Phase-5O and Phase-5R caches",
                "j15_used_as_predictor_loss": False,
                "confirmation_opened": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not aligned:
        return _finish_early(
            "DISTILL-D",
            "训练集 H=1 标签与冻结的 Phase-5ZD / Phase-5X 接口没有对齐，预测器没有训练。"
            f" fingerprint {train_audit['largest']}，parity {train_audit['parity_gap']}，edge {train_audit['edge_gap']}，"
            f"sinr {train_audit['sinr_gap']}，progress {train_audit['progress_gap']}，payload {train_audit['payload_gap']}。",
        )
    probe = OneStepExecutorResponsePredictor(token_dim())
    probe_tokens = torch.as_tensor(_apply_affine(train_states[0]["tokens"], input_norm), dtype=torch.float32)
    probe_raw = train_states[0]["response_raw"]
    loss, _huber, _bce = response_loss(
        probe(probe_tokens),
        torch.as_tensor(_apply_affine(probe_raw, response_norm), dtype=torch.float32),
        torch.as_tensor(probe_raw[..., list(BINARY_INDEX)], dtype=torch.float32),
    )
    loss.backward()
    encoder_grad = 0.0
    for parameter in list(frozen.parameters()) + list(matching.parameters()):
        if parameter.grad is not None:
            encoder_grad += float(parameter.grad.abs().sum())
    actor_grad = 0.0
    for parameter in actor.parameters():
        if parameter.grad is not None:
            actor_grad += float(parameter.grad.abs().sum())
    manifest = [{"name": name, "numel": int(parameter.numel()), "requires_grad": bool(parameter.requires_grad)} for name, parameter in probe.named_parameters()]
    (OUTPUT / "phase5ze_trainable_parameter_manifest.json").write_text(
        json.dumps(
            {
                "trainable_module": "OneStepExecutorResponsePredictor",
                "parameter_count": int(sum(row["numel"] for row in manifest)),
                "hidden": HIDDEN,
                "encoder_grad_abs_sum": encoder_grad,
                "mappo_grad_abs_sum": actor_grad,
                "value_scorer_requires_grad": False,
                "parameters": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if encoder_grad != 0.0 or actor_grad != 0.0:
        return _finish_early("DISTILL-D", "HGNN 或 MAPPO 在响应预测器的反向传播里收到了梯度。")
    del probe
    curves = []
    for seed in SEEDS:
        curves.extend(_train_one(int(seed), train_states, input_norm, response_norm))
    _write_csv("phase5ze_training_curves.csv", curves)
    missing = [f"phase5ze_response_seed{int(seed)}.pt" for seed in SEEDS if not (CHECKPOINTS / f"phase5ze_response_seed{int(seed)}.pt").is_file()]
    if missing:
        return _finish_early("DISTILL-D", f"epoch-10 checkpoint 缺失：{missing}")
    confirm_labels = _labels("confirm")
    confirm_reference = _horizon_table(CONFIRM_SCENARIOS, 1)
    confirm_ref8 = _horizon_table(CONFIRM_SCENARIOS, 8)
    confirm_states, confirm_audit = _prepare(CONFIRM_SCENARIOS, confirm_labels, confirm_reference, frozen, matching, actor, config, False)
    for state in confirm_states:
        state["q_exec8"] = np.asarray([confirm_ref8[(int(state["scenario_id"]), int(state["time_step"]), vector)]["q_exec"] for vector in state["vectors"]], dtype=np.float64)
    predictors = {int(seed): _load_predictor(int(seed)) for seed in SEEDS}
    static_models = {int(seed): _load_value("static", int(seed)) for seed in SEEDS}
    response_models = {int(seed): _load_value("response", int(seed)) for seed in SEEDS}
    rank_rows = []
    downstream = []
    response_rows = []
    scorebook = {}
    rawbook = {}
    for state in confirm_states:
        relational = _relational(state, center, scale)
        zero_response = _apply_affine(np.zeros_like(state["response_raw"]), response_norm)
        true_response = _apply_affine(state["response_raw"], response_norm)
        for seed in SEEDS:
            head = _predict_response(predictors[int(seed)], _apply_affine(state["tokens"], input_norm))
            predicted_raw = _to_raw(head, response_norm)
            predicted_value = _to_value(head, response_norm)
            static_scores, _elapsed = _predict_value(static_models[int(seed)], state, zero_response, relational)
            predicted_scores, _elapsed = _predict_value(response_models[int(seed)], state, predicted_value, relational)
            true_scores, _elapsed = _predict_value(response_models[int(seed)], state, true_response, relational)
            ref8_scores = state["q_exec8"]
            bundled = {"S": static_scores, "P1": predicted_scores, "R1": true_scores, "REF8": ref8_scores, "O": state["j_mappo"]}
            choices = {}
            for model, scores in bundled.items():
                row = _rank_row(model, int(seed), state, scores)
                rank_rows.append(row)
                choices[model] = row["assignment_vector"]
                scorebook[(model, int(seed), int(state["scenario_id"]), int(state["time_step"]))] = scores
            for row in rank_rows[-len(bundled) :]:
                if row["model"] == "P1":
                    row["agreement_with_r1"] = float(choices["P1"] == choices["R1"])
            rawbook[(int(seed), int(state["scenario_id"]), int(state["time_step"]))] = predicted_raw
            for index, vector in enumerate(state["vectors"]):
                downstream.append(
                    {
                        "scenario_id": int(state["scenario_id"]),
                        "time_step": int(state["time_step"]),
                        "seed": int(seed),
                        "assignment_vector": vector,
                        "j15": float(state["j_mappo"][index]),
                        "v_s": float(static_scores[index]),
                        "v_p1": float(predicted_scores[index]),
                        "v_r1": float(true_scores[index]),
                    }
                )
                for uav in range(state["assignments"].shape[1]):
                    response_rows.append(
                        {
                            "scenario_id": int(state["scenario_id"]),
                            "time_step": int(state["time_step"]),
                            "seed": int(seed),
                            "assignment_vector": vector,
                            "uav": int(uav),
                            **{f"true_{name}": float(state["response_raw"][index, uav, feature]) for feature, name in enumerate(RESPONSE_FEATURES)},
                            **{f"pred_{name}": float(predicted_raw[index, uav, feature]) for feature, name in enumerate(RESPONSE_FEATURES)},
                        }
                    )
    feature_rows = []
    pooled_pred = []
    pooled_true = []
    for seed in SEEDS:
        pred_seed = []
        true_seed = []
        for state in confirm_states:
            pred_seed.append(rawbook[(int(seed), int(state["scenario_id"]), int(state["time_step"]))])
            true_seed.append(state["response_raw"])
        pred_cat = np.concatenate(pred_seed)
        true_cat = np.concatenate(true_seed)
        pooled_pred.append(pred_cat)
        pooled_true.append(true_cat)
        for item in _feature_block(pred_cat, true_cat):
            feature_rows.append({"seed": int(seed), **item})
    for item in _feature_block(np.concatenate(pooled_pred), np.concatenate(pooled_true)):
        feature_rows.append({"seed": "all", **item})
    _write_csv("phase5ze_feature_prediction_metrics.csv", feature_rows)
    _write_csv("phase5ze_response_predictions.csv", response_rows)
    _write_csv("phase5ze_downstream_value_predictions.csv", downstream)
    summary = []
    for model in ("S", "P1", "R1", "REF8", "O"):
        chosen = [row for row in rank_rows if row["model"] == model]
        summary.append(
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
                "optimum_rank": _mean(row["optimum_rank"] for row in chosen),
                "agreement_with_r1": _mean(row["agreement_with_r1"] for row in chosen if row["agreement_with_r1"] != ""),
            }
        )
    _write_csv("phase5ze_selected_j15.csv", summary)
    deltas = []
    scenario_rows = []
    for scenario in CONFIRM_SCENARIOS:
        gains = {}
        for left, right, name in (("P1", "S", "P1-S"), ("P1", "R1", "P1-R1"), ("R1", "S", "R1-S")):
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
            gains[name] = _mean(values)
            scenario_rows.append({"comparison": name, "scenario_id": int(scenario), "delta_j15": gains[name], "direction": _direction(gains[name])})
        scenario_rows.append(
            {
                "comparison": "retention",
                "scenario_id": int(scenario),
                "delta_j15": "",
                "direction": "",
                "gain_p1": gains["P1-S"],
                "gain_r1": gains["R1-S"],
                "retention": retention_ratio(gains["P1-S"], gains["R1-S"], TIE_BITS),
                "included": bool(abs(gains["R1-S"]) > TIE_BITS),
            }
        )
    _write_csv("phase5ze_scenario_comparison.csv", scenario_rows)
    seed_rows = []
    retention_rows = []
    for seed in SEEDS:
        gains = {}
        for model in ("S", "P1", "R1", "REF8"):
            chosen = [row for row in rank_rows if row["model"] == model and int(row["seed"]) == int(seed)]
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
                    "agreement_with_r1": _mean(row["agreement_with_r1"] for row in chosen if row["agreement_with_r1"] != ""),
                }
            )
        for name in ("P1-S", "R1-S"):
            left, right = name.split("-")
            gains[name] = _mean(
                float(row["selected_j15"])
                - float(
                    next(
                        item["selected_j15"]
                        for item in rank_rows
                        if item["model"] == right and item["seed"] == row["seed"] and item["scenario_id"] == row["scenario_id"] and item["time_step"] == row["time_step"]
                    )
                )
                for row in rank_rows
                if row["model"] == left and int(row["seed"]) == int(seed)
            )
        retention_rows.append(
            {
                "scope": "seed",
                "scenario_id": "",
                "seed": int(seed),
                "gain_p1": gains["P1-S"],
                "gain_r1": gains["R1-S"],
                "retention": retention_ratio(gains["P1-S"], gains["R1-S"], TIE_BITS),
                "included": bool(abs(gains["R1-S"]) > TIE_BITS),
            }
        )
    for row in scenario_rows:
        if row["comparison"] != "retention":
            continue
        retention_rows.append(
            {
                "scope": "scenario",
                "scenario_id": row["scenario_id"],
                "seed": "",
                "gain_p1": row["gain_p1"],
                "gain_r1": row["gain_r1"],
                "retention": row["retention"],
                "included": row["included"],
            }
        )
    comparison = {name: cluster_bootstrap([row for row in deltas if row["comparison"] == name], "delta_j15", draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED) for name in ("P1-S", "P1-R1", "R1-S")}
    retention_rows.append(
        {
            "scope": "overall",
            "scenario_id": "",
            "seed": "",
            "gain_p1": comparison["P1-S"]["mean"],
            "gain_r1": comparison["R1-S"]["mean"],
            "retention": retention_ratio(comparison["P1-S"]["mean"], comparison["R1-S"]["mean"], TIE_BITS),
            "included": bool(abs(comparison["R1-S"]["mean"]) > TIE_BITS),
        }
    )
    _write_csv("phase5ze_distillation_retention.csv", retention_rows)
    _write_csv("phase5ze_seed_stability.csv", seed_rows)
    error_rows = []
    state_error = []
    for state in confirm_states:
        for seed in SEEDS:
            predicted_raw = rawbook[(int(seed), int(state["scenario_id"]), int(state["time_step"]))]
            absolute = np.abs(predicted_raw - state["response_raw"])
            p1 = scorebook[("P1", int(seed), int(state["scenario_id"]), int(state["time_step"]))]
            r1 = scorebook[("R1", int(seed), int(state["scenario_id"]), int(state["time_step"]))]
            p1_choice = int(np.argsort(-p1, kind="stable")[0])
            r1_choice = int(np.argsort(-r1, kind="stable")[0])
            gap = float(state["j_mappo"][r1_choice] - state["j_mappo"][p1_choice])
            regret = float(metrics_at(p1, state["j_mappo"], p1_choice)["normalized_regret"])
            disagree = float(p1_choice != r1_choice)
            state_error.append(
                {
                    "level": "state",
                    "scenario_id": int(state["scenario_id"]),
                    "time_step": int(state["time_step"]),
                    "seed": int(seed),
                    "response_mae": float(np.mean(absolute)),
                    "sinr_mae": float(np.mean(absolute[..., list(RESPONSE_GROUPS["sinr"])])),
                    "progress_mae": float(np.mean(absolute[..., list(RESPONSE_GROUPS["progress"])])),
                    "interference_mae": float(np.mean(absolute[..., list(RESPONSE_GROUPS["interference"])])),
                    "motion_mae": float(np.mean(absolute[..., list(RESPONSE_GROUPS["motion"])])),
                    "value_abs_error": float(np.mean(np.abs(p1 - r1))),
                    "selected_j15_gap": gap,
                    "disagreement": disagree,
                    "p1_normalized_regret": regret,
                }
            )
            for index in range(absolute.shape[0]):
                error_rows.append(
                    {
                        "level": "assignment",
                        "scenario_id": int(state["scenario_id"]),
                        "time_step": int(state["time_step"]),
                        "seed": int(seed),
                        "assignment_index": int(index),
                        "response_mae": float(np.mean(absolute[index])),
                        "sinr_mae": float(np.mean(absolute[index][:, list(RESPONSE_GROUPS["sinr"])])),
                        "progress_mae": float(np.mean(absolute[index][:, list(RESPONSE_GROUPS["progress"])])),
                        "interference_mae": float(np.mean(absolute[index][:, list(RESPONSE_GROUPS["interference"])])),
                        "motion_mae": float(np.mean(absolute[index][:, list(RESPONSE_GROUPS["motion"])])),
                        "value_abs_error": float(abs(float(p1[index] - r1[index]))),
                        "selected_j15_gap": gap,
                        "disagreement": disagree,
                        "p1_normalized_regret": regret,
                    }
                )
    error_link = {
        "response_vs_value": spearman([row["response_mae"] for row in error_rows], [row["value_abs_error"] for row in error_rows]),
        "sinr_vs_disagreement": spearman([row["sinr_mae"] for row in state_error], [row["disagreement"] for row in state_error]),
        "progress_vs_disagreement": spearman([row["progress_mae"] for row in state_error], [row["disagreement"] for row in state_error]),
        "interference_vs_disagreement": spearman([row["interference_mae"] for row in state_error], [row["disagreement"] for row in state_error]),
        "motion_vs_disagreement": spearman([row["motion_mae"] for row in state_error], [row["disagreement"] for row in state_error]),
        "progress_vs_j15_gap": spearman([row["progress_mae"] for row in state_error], [row["selected_j15_gap"] for row in state_error]),
        "sinr_vs_j15_gap": spearman([row["sinr_mae"] for row in state_error], [row["selected_j15_gap"] for row in state_error]),
    }
    sinr_values = np.asarray([row["sinr_mae"] for row in state_error], dtype=np.float64)
    ranks = np.argsort(np.argsort(sinr_values))
    quartile_rows = []
    for bin_id in range(4):
        chosen = [row for row, rank in zip(state_error, ranks) if int(rank) * 4 // len(state_error) == bin_id]
        quartile_rows.append(
            {
                "level": "sinr_quartile",
                "quartile": int(bin_id + 1),
                "sinr_mae": _mean(row["sinr_mae"] for row in chosen),
                "p1_normalized_regret": _mean(row["p1_normalized_regret"] for row in chosen),
                "disagreement": _mean(row["disagreement"] for row in chosen),
                "n": len(chosen),
            }
        )
    error_rows.extend(state_error)
    error_rows.extend(quartile_rows)
    _write_csv("phase5ze_response_error_decision_error.csv", error_rows)
    ablation_rows = []
    for group, indexes in RESPONSE_GROUPS.items():
        disagreements = []
        drops = []
        for state in confirm_states:
            relational = _relational(state, center, scale)
            for seed in SEEDS:
                head = _predict_response(predictors[int(seed)], _apply_affine(state["tokens"], input_norm))
                value = _to_value(head, response_norm)
                value[..., list(indexes)] = 0.0
                full = scorebook[("P1", int(seed), int(state["scenario_id"]), int(state["time_step"]))]
                ablated, _elapsed = _predict_value(response_models[int(seed)], state, value, relational)
                full_choice = int(np.argsort(-full, kind="stable")[0])
                ablated_choice = int(np.argsort(-ablated, kind="stable")[0])
                disagreements.append(float(full_choice != ablated_choice))
                drops.append(float(state["j_mappo"][full_choice] - state["j_mappo"][ablated_choice]))
        ablation_rows.append({"group": group, "disagreement": _mean(disagreements), "selected_j15_drop": _mean(drops)})
    _write_csv("phase5ze_feature_ablation.csv", ablation_rows)
    case_rows = []
    case_state = next(state for state in confirm_states if int(state["scenario_id"]) == CASE_SCENARIO and int(state["time_step"]) == CASE_TIME)
    pred_index = case_state["vectors"].index(CASE_PREDICTED)
    opt_index = case_state["vectors"].index(CASE_OPTIMAL)
    for seed in SEEDS:
        predicted_raw = rawbook[(int(seed), CASE_SCENARIO, CASE_TIME)]
        for name, index in (("A_pred", pred_index), ("A_opt", opt_index)):
            truth = case_state["response_raw"][index]
            guess = predicted_raw[index]
            assignment = case_state["assignments"][index]
            case_rows.append(
                {
                    "row_type": "response",
                    "model": "",
                    "seed": int(seed),
                    "assignment": name,
                    "sinr_true": _assigned_mean(truth, assignment, 0),
                    "sinr_pred": _assigned_mean(guess, assignment, 0),
                    "progress_true": _assigned_mean(truth, assignment, 4),
                    "progress_pred": _assigned_mean(guess, assignment, 4),
                    "interference_true": _assigned_mean(truth, assignment, 2),
                    "interference_pred": _assigned_mean(guess, assignment, 2),
                    "distance_delta_true": _assigned_mean(truth, assignment, 8),
                    "distance_delta_pred": _assigned_mean(guess, assignment, 8),
                    "feature_mae": float(np.mean(np.abs(guess - truth))),
                }
            )
        for model in ("S", "P1", "R1"):
            scores = scorebook[(model, int(seed), CASE_SCENARIO, CASE_TIME)]
            case_rows.append(
                {
                    "row_type": "value",
                    "model": model,
                    "seed": int(seed),
                    "v_pred": float(scores[pred_index]),
                    "v_opt": float(scores[opt_index]),
                    "opt_rank": _named_rank(scores, case_state["vectors"], CASE_OPTIMAL),
                }
            )
    _write_csv("phase5ze_310114_t15.csv", case_rows)
    residual_rows = []
    with PHASE5W_RESIDUAL.open(encoding="utf-8", newline="") as handle:
        residual_cases = list(csv.DictReader(handle))
    for item in residual_cases:
        scenario = int(item["scenario_id"])
        moment = int(item["time_step"])
        assert_consumed(scenario)
        for seed in SEEDS:
            selected = {}
            for model in ("S", "P1", "R1", "REF8", "O"):
                selected[model] = float(
                    next(
                        row["selected_j15"]
                        for row in rank_rows
                        if row["model"] == model and int(row["seed"]) == int(seed) and int(row["scenario_id"]) == scenario and int(row["time_step"]) == moment
                    )
                )
            residual_rows.append(
                {
                    "scenario_id": scenario,
                    "time_step": moment,
                    "seed": int(seed),
                    "predicted_assignment": item["predicted_assignment"],
                    "optimal_assignment": item["optimal_assignment"],
                    "static_j15": selected["S"],
                    "predicted_j15": selected["P1"],
                    "true_j15": selected["R1"],
                    "ref8_j15": selected["REF8"],
                    "oracle_j15": selected["O"],
                }
            )
    grouped = defaultdict(list)
    for row in residual_rows:
        grouped[(row["scenario_id"], row["time_step"], row["predicted_assignment"])].append(row)
    repaired = unchanged = worsened = kept = lost = 0
    for rows in grouped.values():
        static = _mean(row["static_j15"] for row in rows)
        predicted = _mean(row["predicted_j15"] for row in rows)
        true = _mean(row["true_j15"] for row in rows)
        direction = _direction(predicted - static)
        repaired += int(direction == "positive")
        unchanged += int(direction == "tie")
        worsened += int(direction == "negative")
        if true > static + TIE_BITS:
            if predicted > static + TIE_BITS:
                kept += 1
            else:
                lost += 1
    _write_csv("phase5ze_residual_cases.csv", residual_rows)
    perm_rows = []
    largest_perm = 0.0
    audit_state = next(state for state in confirm_states if "runtime_h" in state)
    for seed in SEEDS:
        gap = _permute_audit(predictors[int(seed)], audit_state, input_norm)
        largest_perm = max(largest_perm, gap)
        perm_rows.append({"check": "uav_permutation", "seed": int(seed), "max_abs": gap, "passed": bool(gap <= PERMUTATION_ABS), "status": "tested"})
    perm_rows.append({"check": "buoy_relabel", "seed": "", "max_abs": "", "passed": "", "status": "not_applicable"})
    _write_csv("phase5ze_permutation_audit.csv", perm_rows)
    runtime = _benchmark(
        audit_state,
        actor,
        frozen,
        static_models[0],
        response_models[0],
        predictors[0],
        input_norm,
        response_norm,
        center,
        scale,
    )
    _write_csv("phase5ze_runtime.csv", runtime["rows"])
    scenario_counts = {
        name: {
            "positive": int(sum(row["direction"] == "positive" for row in scenario_rows if row["comparison"] == name)),
            "tie": int(sum(row["direction"] == "tie" for row in scenario_rows if row["comparison"] == name)),
            "negative": int(sum(row["direction"] == "negative" for row in scenario_rows if row["comparison"] == name)),
        }
        for name in ("P1-S", "P1-R1", "R1-S")
    }
    seed_direction = {}
    for name, left, right in (("P1-S", "P1", "S"), ("R1-S", "R1", "S"), ("P1-R1", "P1", "R1")):
        labels = []
        for seed in SEEDS:
            gain = _mean(row["selected_j15"] for row in seed_rows if row["model"] == left and int(row["seed"]) == int(seed))
            base = _mean(row["selected_j15"] for row in seed_rows if row["model"] == right and int(row["seed"]) == int(seed))
            labels.append(_direction(gain - base))
        seed_direction[name] = labels
    p1_summary = next(row for row in summary if row["model"] == "P1")
    case_ranks = {model: [int(row["opt_rank"]) for row in case_rows if row.get("model") == model and row["row_type"] == "value"] for model in ("S", "P1", "R1")}
    ge_count = scenario_counts["P1-S"]["positive"] + scenario_counts["P1-S"]["tie"]
    retention = retention_ratio(comparison["P1-S"]["mean"], comparison["R1-S"]["mean"], TIE_BITS)
    drop_vs_true = -float(comparison["P1-R1"]["mean"])
    confirm_aligned = (
        float(confirm_audit["largest"]) == 0.0
        and float(confirm_audit["sinr_gap"]) <= SINR_MATCH_ABS
        and float(confirm_audit["progress_gap"]) <= PROGRESS_MATCH_ABS
        and float(confirm_audit["payload_gap"]) <= PAYLOAD_MATCH_ABS
    )
    r1_reproduced = bool(
        confirm_aligned
        and float(comparison["R1-S"]["mean"]) > 0.0
        and float(comparison["R1-S"]["low"]) > 0.0
        and seed_direction["R1-S"].count("positive") >= MIN_SEED_UP
    )
    confirmed = bool(
        float(comparison["P1-S"]["mean"]) > 0.0
        and float(comparison["P1-S"]["low"]) > 0.0
        and ge_count >= MIN_SCENARIO_GE
        and scenario_counts["P1-S"]["negative"] <= MAX_SCENARIO_DOWN
        and seed_direction["P1-S"].count("positive") >= MIN_SEED_UP
        and float(p1_summary["normalized_regret"]) <= REGRET_BAR
        and float(p1_summary["top3"]) >= TOP3_BAR
        and np.isfinite(retention)
        and float(retention) >= RETENTION_BAR
        and drop_vs_true <= MAX_J15_DROP_VS_TRUE
        and float(np.median(case_ranks["P1"])) <= 3
        and np.isfinite(runtime["speedup"])
        and float(runtime["speedup"]) >= SPEEDUP_BAR
        and float(runtime["mutation"]) == 0.0
        and largest_perm <= PERMUTATION_ABS
    )
    gate = choose_gate(r1_reproduced=r1_reproduced, confirmed=confirmed, mean_improved=float(comparison["P1-S"]["mean"]) > 0.0)
    cross = []
    for predictor_seed in SEEDS:
        for value_seed in SEEDS:
            selected = []
            for state in confirm_states:
                head = _predict_response(predictors[int(predictor_seed)], _apply_affine(state["tokens"], input_norm))
                scores, _elapsed = _predict_value(response_models[int(value_seed)], state, _to_value(head, response_norm), _relational(state, center, scale))
                selected.append(float(state["j_mappo"][int(np.argsort(-scores, kind="stable")[0])]))
            cross.append({"predictor_seed": int(predictor_seed), "value_seed": int(value_seed), "selected_j15": _mean(selected), "used_for_selection": bool(int(predictor_seed) == int(value_seed))})
    payload = {
        "gate": gate,
        "planner_ready": False,
        "deployment_verified": False,
        "trained": True,
        "opened_310118": False,
        "decide_called": False,
        "frozen_for_fresh_confirmation": gate == "DISTILL-A",
        "r1_reproduced": r1_reproduced,
        "comparison": comparison,
        "scenario_counts": scenario_counts,
        "seed_directions": seed_direction,
        "retention": retention,
        "drop_vs_true": drop_vs_true,
        "p1_top3": p1_summary["top3"],
        "p1_normalized_regret": p1_summary["normalized_regret"],
        "agreement_with_r1": p1_summary["agreement_with_r1"],
        "case_ranks": case_ranks,
        "residual": {"repaired": repaired, "unchanged": unchanged, "worsened": worsened, "kept_repair": kept, "lost_repair": lost},
        "error_link": error_link,
        "speedup": runtime["speedup"],
        "permutation_max_abs": largest_perm,
        "buoy_relabel": "not_applicable",
        "caller_mutation": runtime["mutation"],
        "cross_seed_audit_not_for_selection": cross,
        "selected_j15": summary,
        "confirm_fingerprint": confirm_audit["largest"],
    }
    _write_gate(payload)
    _report(payload, summary, feature_rows, ablation_rows, case_rows, quartile_rows, runtime["rows"], scenario_counts, seed_direction)
    print(json.dumps(_plain({"gate": gate, "frozen_for_fresh_confirmation": gate == "DISTILL-A"}), indent=2), flush=True)
    return payload


def _report(payload, summary, feature_rows, ablation_rows, case_rows, quartile_rows, runtime_rows, scenario_counts, seed_direction) -> None:
    by_model = {row["model"]: row for row in summary}
    pooled = [row for row in feature_rows if row["seed"] == "all"]
    by_feature = {row["feature"]: row for row in pooled}
    gap = payload["comparison"]["P1-S"]
    versus_true = payload["comparison"]["P1-R1"]
    lines = [
        "# Phase-5ZE：一步执行响应蒸馏",
        "",
        "本阶段只训练 OneStepExecutorResponsePredictor。它根据当前 state、候选 assignment 和冻结 MAPPO 的 ACT0，预测 Phase-5ZD 冻结的 14 维 H=1 response。预测结果送入已经冻结的 ExecutorConditionedJointScorer。价值网络、HGNN、REP、TEMP、候选池和 MAPPO 都没有更新。J15 不进入预测器的损失。",
        "",
        "这个预测器不是新的低层策略，不是中间决策器，也不是另一个 assignment selector。最终 assignment 仍由高层价值函数比较后一次选出。option hold 仍是 15。H_dyn = 1 只表示执行信息第一次出现的观察尺度，高层不因此改成每步重选。",
        "",
        f"Gate = `{payload['gate']}`。`FROZEN_FOR_FRESH_CONFIRMATION = {payload['frozen_for_fresh_confirmation']}`。",
        "",
        "`planner_ready = False`",
        "",
        "`deployment_verified = False`",
        "",
        "没有打开 310118–310125，没有调用 deployment 决策接口。310110–310117 在 5 个 seed 的 epoch-10 checkpoint 写盘之后才读取。",
        "",
        "## 下游 selected J15",
        "",
        "| 路径 | mean J15 | median J15 | normalized regret | Top-1 | Top-3 | Top-5 | Spearman |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for model in ("S", "P1", "R1", "REF8", "O"):
        row = by_model[model]
        lines.append(
            f"| {model} | {row['selected_j15_mean']:.0f} | {row['selected_j15_median']:.0f} | {_fmt(row['normalized_regret'])} | {_fmt(row['top1'])} | {_fmt(row['top3'])} | {_fmt(row['top5'])} | {_fmt(row['spearman'])} |"
        )
    lines.extend(
        [
            "",
            "## Q1. H1 response 能否从 state、assignment 和 ACT0 预测出来？",
            "",
            "可以预测出一部分，而且各字段差别很大。确认集上、5 个 seed 合在一起的逐字段误差见下表。主指标仍然是后面的 selected J15，不是这些回归误差。",
            "",
            "| 字段 | MAE | RMSE | R² | bias | Pearson | Spearman |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for name in RESPONSE_FEATURES:
        row = by_feature[name]
        lines.append(
            f"| {name} | {_fmt(row['mae'])} | {_fmt(row['rmse'])} | {_fmt(row['r2'])} | {_fmt(row['bias'])} | {_fmt(row['pearson'])} | {_fmt(row['spearman'])} |"
        )
    sinr = by_feature["sinr_1"]
    progress = by_feature["service_progress_1"]
    lines.extend(
        [
            "",
            "## Q2. SINR",
            "",
            f"SINR_1 的 MAE = {_fmt(sinr['mae'])}，R² = {_fmt(sinr['r2'])}，Spearman = {_fmt(sinr['spearman'])}。",
            "",
            "## Q3. service progress",
            "",
            f"service_progress_1 的 MAE = {_fmt(progress['mae'])}，R² = {_fmt(progress['r2'])}，Spearman = {_fmt(progress['spearman'])}。",
            "",
            "## Q4–Q8. 预测响应还留住了多少价值增益？",
            "",
            f"P1 相对 S 的 mean selected J15 差是 {gap['mean']:.0f} bit，scenario-cluster 95% CI 是 [{gap['low']:.0f}, {gap['high']:.0f}]。",
            f"整体 retention = Gain_P1 / Gain_R1 = {_fmt(payload['retention'])}。Gain_R1 为 0 的 scenario 不进入这个比值。",
            f"P1 相对 R1 的 mean selected J15 差是 {versus_true['mean']:.0f} bit，CI [{versus_true['low']:.0f}, {versus_true['high']:.0f}]。真实 H1 比预测 H1 高出的部分是 {payload['drop_vs_true']:.0f} bit。",
            f"P1 与 R1 的 assignment agreement 是 {_fmt(payload['agreement_with_r1'])}。",
            f"5 个 seed 上 P1−S 的方向是 {', '.join(seed_direction['P1-S'])}。",
            f"8 个 scenario 上 P1−S：提高 {scenario_counts['P1-S']['positive']}，持平 {scenario_counts['P1-S']['tie']}，下降 {scenario_counts['P1-S']['negative']}。",
            "",
            "## Q9. 310114 t=15",
            "",
        ]
    )
    for model in ("S", "P1", "R1"):
        ranks = [str(int(row["opt_rank"])) for row in case_rows if row["row_type"] == "value" and row["model"] == model]
        lines.append(f"- {model} 的 A_opt rank：{', '.join(ranks)}。")
    lines.extend(["", "## Q10. residual cases", ""])
    residual = payload["residual"]
    lines.append(
        f"P1 相对 S：修复 {residual['repaired']}，不变 {residual['unchanged']}，恶化 {residual['worsened']}。R1 已经修好、P1 仍高于 S 的有 {residual['kept_repair']} 条，丢掉的有 {residual['lost_repair']} 条。"
    )
    lines.extend(["", "## Q11. 哪类预测误差更接近决策错误？", ""])
    link = payload["error_link"]
    for key, value in link.items():
        lines.append(f"- {key} Spearman = {_fmt(value)}。")
    lines.append("SINR 误差四分位上的 P1 normalized regret：")
    for row in quartile_rows:
        lines.append(f"- Q{row['quartile']}：SINR MAE {_fmt(row['sinr_mae'])}，regret {_fmt(row['p1_normalized_regret'])}，disagreement {_fmt(row['disagreement'])}。")
    lines.extend(["", "推理期把预测通道换成训练集均值，不重新训练：", ""])
    for row in ablation_rows:
        lines.append(f"- {row['group']}：disagreement {_fmt(row['disagreement'])}，selected J15 变化 {-float(row['selected_j15_drop']):.0f} bit。")
    true_row = next(row for row in runtime_rows if row["path"] == "TRUE-H1")
    pred_row = next(row for row in runtime_rows if row["path"] == "PREDICTED-H1")
    static_row = next(row for row in runtime_rows if row["path"] == "STATIC")
    lines.extend(
        [
            "",
            "## Q12. 120 个候选的时间",
            "",
            f"真实 H1：encoder {true_row['encoder_ms']:.1f} ms，MAPPO {true_row['mappo_action_ms']:.1f} ms，clone {true_row['env_clone_ms']:.1f} ms，step {true_row['env_step_ms']:.1f} ms，extract {true_row['response_extract_ms']:.1f} ms，value {true_row['value_ms']:.1f} ms，合计 {true_row['total_ms']:.1f} ms。",
            f"预测 H1：encoder {pred_row['encoder_ms']:.1f} ms，MAPPO {pred_row['mappo_action_ms']:.1f} ms，predictor {pred_row['response_predictor_ms']:.1f} ms，value {pred_row['value_ms']:.1f} ms，合计 {pred_row['total_ms']:.1f} ms。没有 environment clone，也没有 environment step。",
            f"静态：encoder {static_row['encoder_ms']:.1f} ms，value {static_row['value_ms']:.1f} ms，合计 {static_row['total_ms']:.1f} ms。",
            f"P1 相对真实 H1 的 speedup 是 {payload['speedup']:.2f} 倍。计时在 warm-up 之后重复 {RUNTIME_REPEATS} 次。",
            "",
            "## Q13–Q14",
            "",
            f"当前 Gate 是 {payload['gate']}。UAV permutation 最大绝对误差是 {payload['permutation_max_abs']:.3e}。浮标重编号没有现成的 encoder API，记为 not_applicable，没有伪造通过。",
            "只有 DISTILL-A 才把当前结构标成可以进入 fresh confirmation 审计。即便如此，也不在本阶段打开 310118–310125，不修改正式 planner，不进入 deployment。下一步如果进入审计，也是先确认能不能打开新的确认集，而不是继续改 HGNN 或 MAPPO。",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
