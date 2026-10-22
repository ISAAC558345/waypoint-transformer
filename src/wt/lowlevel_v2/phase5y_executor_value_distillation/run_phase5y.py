"""Train a fresh value head on frozen MAPPO 15-step payload. Confirmation is scored once."""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from wt.l3b.config import L3BConfig
from wt.lowlevel_v2.phase5l_joint_encoder.run_phase5l import _load_matching
from wt.lowlevel_v2.phase5m_joint_planner.run_phase5m import _load_proposed
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.metrics import is_collapsed, metrics_at, quantize
from wt.lowlevel_v2.phase5r_nonadditive_generalization.metrics import cluster_bootstrap
from wt.lowlevel_v2.phase5v_temporal_representation.run_phase5v import (
    _build_states,
    _checkpoint as _v1_checkpoint,
    _combine,
    _load_v1,
    _pack,
    _scale,
    _scores,
)
from wt.round30.ppo import seed_everything

from .metrics import choose_gate
from .protocol import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CASE_OPTIMUM,
    CASE_PREDICTED,
    CASE_SCENARIO,
    CASE_TIME,
    CASE_TOP_RANK,
    CHECKPOINTS,
    COLLISION_FRACTION,
    CONFIRM_SCENARIOS,
    COSINE_LIMIT,
    DEPLOYMENT_VERIFIED,
    EPOCHS,
    FINAL_REL_DIM,
    FORBIDDEN_INPUTS,
    GRAD_CLIP,
    LARGE_DROP_BITS,
    LEARNING_RATE,
    MIN_SCENARIO_UP,
    MIN_SEED_UP,
    OUTPUT,
    PAYLOAD_BLOCK_BITS,
    PHASE5W_RESIDUAL,
    PHASE5X_ROLLOUTS,
    PHASE5X_RUNTIME,
    PLANNER_READY,
    RECOVERY_EPS,
    REFERENCE_HORIZONS,
    REGRESSION_MAE,
    REGRESSION_R2,
    REGRET_CEILING,
    REPORT,
    SEEDS,
    TOP3_FLOOR,
    TRAIN_SCENARIOS,
    VALUE_GAP_BLOCKS,
    WEIGHT_DECAY,
    assert_consumed,
)
from .value import ExecutorConditionedJointValue, state_huber

ALLOWED_INPUTS = ("h_u", "h_b", "h_edge", "h_uu", "z_global", "z_rel")
BIN_NAMES = ("0", "1", "2", "3", "4+")


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value, digits: str = ".4f") -> str:
    number = float(value)
    if not np.isfinite(number):
        return "NA"
    return format(number, digits)


def _mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(np.mean(array))


def _median(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(np.median(array))


def _iqr(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(np.percentile(array, 75) - np.percentile(array, 25))


def _legal(vector: str) -> bool:
    seen = set()
    for item in str(vector).split(";"):
        buoy = int(item)
        if buoy < 0:
            continue
        if buoy in seen:
            return False
        seen.add(buoy)
    return True


def _top5(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")[:5]
    levels = quantize(labels)
    return float(np.any(levels[order] == np.max(levels)))


def _optimum_rank(scores: np.ndarray, labels: np.ndarray) -> float:
    levels = quantize(labels)
    optimal = levels == float(np.max(levels))
    return float(1 + np.sum(np.asarray(scores, dtype=np.float64) > float(np.max(np.asarray(scores)[optimal]))))


def _named_rank(scores: np.ndarray, vectors: list[str], target: str) -> int:
    index = list(vectors).index(str(target))
    values = np.asarray(scores, dtype=np.float64)
    return int(1 + np.sum(values > values[index]))


def _checkpoint(seed: int):
    return CHECKPOINTS / f"phase5y_seed{int(seed)}.pt"


def _read_rollouts() -> tuple[dict, dict]:
    labels: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    stored: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    qexec: dict[tuple[int, int, str, int], float] = {}
    with PHASE5X_ROLLOUTS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scenario = int(row["scenario_id"])
            moment = int(row["time_step"])
            assert_consumed(scenario)
            vector = str(row["assignment_vector"])
            horizon = int(row["horizon"])
            if horizon == 15:
                labels[(scenario, moment, vector)].append(float(row["payload_h"]))
                stored[(scenario, moment, vector)].append(float(row["j15"]))
            if horizon in REFERENCE_HORIZONS:
                qexec[(scenario, moment, vector, horizon)] = float(row["q_exec_h"])
    return {"payload": labels, "stored": stored}, qexec


def _h8_reference_ms() -> float:
    with PHASE5X_RUNTIME.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["horizon"]) == 8:
                return float(row["ms_per_decision_120"])
    raise RuntimeError("phase-5x H=8 runtime is missing")


def _frozen_scaler() -> tuple[np.ndarray, np.ndarray]:
    center = scale = None
    for seed in SEEDS:
        payload = torch.load(_v1_checkpoint(int(seed)), map_location="cpu", weights_only=False)
        current_center = np.asarray(payload["center"], dtype=np.float32)
        current_scale = np.asarray(payload["scale"], dtype=np.float32)
        if center is None:
            center, scale = current_center, current_scale
        elif not np.allclose(center, current_center) or not np.allclose(scale, current_scale):
            raise RuntimeError("Phase-5V temporal scaler is not shared across seeds")
    return center, scale


def _features(state: dict, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return _combine(state["z_rel"], _scale(state["z_temp_raw"], center, scale))


def _audit(states: list[dict], tables: dict) -> tuple[list[dict], dict]:
    rows = []
    conflict = False
    for key, values in tables["payload"].items():
        scenario, moment, vector = key
        payload = np.asarray(values, dtype=np.float64)
        recorded = np.asarray(tables["stored"][key], dtype=np.float64)
        span = float(np.max(payload) - np.min(payload)) if payload.size else float("nan")
        stored_span = float(np.max(recorded) - np.min(recorded)) if recorded.size else float("nan")
        match = bool(payload.size and recorded.size and abs(float(np.mean(payload)) - float(np.mean(recorded))) <= 1.0 and span <= 1.0 and stored_span <= 1.0)
        if span > 1.0 or stored_span > 1.0 or not match:
            conflict = True
        rows.append(
            {
                "row_type": "label",
                "scenario_id": int(scenario),
                "time_step": int(moment),
                "state_id": f"{int(scenario)}_{int(moment)}",
                "assignment_vector": vector,
                "j15": float(np.mean(payload)) if payload.size else "",
                "n_rows": int(payload.size),
                "j15_span": span,
                "legal": int(_legal(vector)),
                "matches_stored_label": int(match),
                "in_pool": 0,
                "check_name": "",
                "check_passed": "",
            }
        )
    label_j15 = {}
    missing = 0
    drifted = 0
    illegal_pool = 0
    by_key = {
        (int(row["scenario_id"]), int(row["time_step"]), str(row["assignment_vector"])): row
        for row in rows
        if row["row_type"] == "label"
    }
    for state in states:
        for index, vector in enumerate(state["vectors"]):
            key = (int(state["scenario_id"]), int(state["time_step"]), vector)
            if key not in tables["payload"]:
                missing += 1
                continue
            value = float(np.mean(tables["payload"][key]))
            label_j15[key] = value
            matches_state = abs(value - float(state["j_mappo"][index])) <= 1.0
            if not matches_state:
                drifted += 1
            if not _legal(vector):
                illegal_pool += 1
            by_key[key]["in_pool"] = 1
            by_key[key]["matches_stored_label"] = int(bool(by_key[key]["matches_stored_label"]) and matches_state)
    input_keys = set(ALLOWED_INPUTS)
    for name in FORBIDDEN_INPUTS:
        rows.append(
            {
                "row_type": "leakage_check",
                "scenario_id": "",
                "time_step": "",
                "state_id": "",
                "assignment_vector": "",
                "j15": "",
                "n_rows": "",
                "j15_span": "",
                "legal": "",
                "matches_stored_label": "",
                "in_pool": "",
                "check_name": name,
                "check_passed": int(name not in input_keys),
            }
        )
    for name, passed in (
        ("no_conflicting_j15", int(not conflict)),
        ("pool_labels_present", int(missing == 0)),
        ("shadow_matches_stored_j15", int(drifted == 0)),
        ("pool_assignments_legal", int(illegal_pool == 0)),
    ):
        rows.append(
            {
                "row_type": "summary",
                "scenario_id": "",
                "time_step": "",
                "state_id": "",
                "assignment_vector": "",
                "j15": "",
                "n_rows": "",
                "j15_span": "",
                "legal": "",
                "matches_stored_label": "",
                "in_pool": "",
                "check_name": name,
                "check_passed": int(passed),
            }
        )
    if conflict or missing or drifted or illegal_pool:
        _write_csv(OUTPUT / "phase5y_label_integrity.csv", rows)
        raise RuntimeError(
            f"phase-5y label audit failed conflict={conflict} missing={missing} drifted={drifted} illegal={illegal_pool}"
        )
    for state in states:
        state["j_mappo"] = np.asarray(
            [label_j15[(int(state["scenario_id"]), int(state["time_step"]), vector)] for vector in state["vectors"]],
            dtype=np.float64,
        )
    return rows, label_j15


def _reference_scores(state: dict, qexec: dict, horizon: int) -> np.ndarray:
    return np.asarray(
        [qexec[(int(state["scenario_id"]), int(state["time_step"]), vector, int(horizon))] for vector in state["vectors"]],
        dtype=np.float64,
    )


def _predict(scorer, state: dict, relational: np.ndarray) -> np.ndarray:
    features = _pack(state["embedded"], relational)
    if set(features) != set(ALLOWED_INPUTS):
        raise RuntimeError("value model received an input outside the frozen representation")
    with torch.no_grad():
        return scorer(features, torch.as_tensor(state["assignments"], dtype=torch.long)).detach().cpu().numpy()


def _latent(scorer, state: dict, relational: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return scorer.latent(_pack(state["embedded"], relational), torch.as_tensor(state["assignments"], dtype=torch.long)).detach().cpu().numpy()


def _train_one(scorer, states: list[dict], seed: int, center: np.ndarray, scale: np.ndarray) -> list[dict]:
    train_states = [state for state in states if state["split"] == "train"]
    optimizer = torch.optim.Adam(scorer.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    curves = []
    for epoch in range(1, int(EPOCHS) + 1):
        scorer.train()
        losses = []
        for state in train_states:
            if int(state["scenario_id"]) not in TRAIN_SCENARIOS:
                raise RuntimeError("confirmation state entered the value gradient")
            optimizer.zero_grad(set_to_none=True)
            relational = _features(state, center, scale)
            prediction = scorer(_pack(state["embedded"], relational), torch.as_tensor(state["assignments"], dtype=torch.long))
            target = torch.as_tensor(state["j_mappo"] / PAYLOAD_BLOCK_BITS, dtype=torch.float32)
            loss = state_huber(prediction, target)
            if not bool(torch.isfinite(loss).all()):
                raise RuntimeError(f"value loss became non-finite at seed {seed} epoch {epoch}")
            loss.backward()
            clip_grad_norm_(scorer.parameters(), GRAD_CLIP)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        curves.append({"seed": int(seed), "epoch": int(epoch), "loss": _mean(losses), "split": "train"})
        print(f"value seed {seed} epoch {epoch} loss={_mean(losses):.4f}", flush=True)
    scorer.eval()
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    return curves


def _rank_row(model: str, seed: int, state: dict, scores: np.ndarray) -> dict:
    chosen = int(np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")[0])
    measured = metrics_at(scores, state["j_mappo"], chosen)
    return {
        "split": state["split"],
        "model": model,
        "seed": int(seed),
        "scenario_id": int(state["scenario_id"]),
        "time_step": int(state["time_step"]),
        "spearman": measured["spearman"],
        "kendall": measured["kendall"],
        "top1": measured["top1"],
        "top3": measured["top3"],
        "top5": _top5(scores, state["j_mappo"]),
        "optimum_rank": _optimum_rank(scores, state["j_mappo"]),
        "normalized_regret": measured["normalized_regret"],
        "regret": measured["regret"],
        "j15": measured["j_mappo"],
        "assignment_vector": state["vectors"][chosen],
        "score_std": measured["score_std"],
        "collapsed": measured["collapsed"],
    }


def _regression_block(prediction: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels, dtype=np.float64).reshape(-1) / PAYLOAD_BLOCK_BITS
    residual = pred - truth
    square = float(np.sum((truth - float(np.mean(truth))) ** 2))
    huber = np.where(np.abs(residual) <= 1.0, 0.5 * residual**2, np.abs(residual) - 0.5)
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "huber": float(np.mean(huber)),
        "r2": float("nan") if square <= 1.0e-12 else float(1.0 - np.sum((truth - pred) ** 2) / square),
        "bias": float(np.mean(residual)),
        "prediction_std": float(np.std(pred)),
    }


def _value_bin(value: float) -> str:
    if value < 0.5:
        return "0"
    if value < 1.5:
        return "1"
    if value < 2.5:
        return "2"
    if value < 3.5:
        return "3"
    return "4+"


def _collision(states: list[dict], predictions: dict, scorers: dict, center: np.ndarray, scale: np.ndarray) -> list[dict]:
    rows = []
    for seed, scorer in scorers.items():
        decisive = 0
        value_close = 0
        embedding_close = 0
        both = 0
        cosines = []
        for state in states:
            if state["split"] != "confirmation":
                continue
            labels = np.asarray(state["j_mappo"], dtype=np.float64)
            values = predictions[(int(seed), int(state["scenario_id"]), int(state["time_step"]))]
            latent = _latent(scorer, state, _features(state, center, scale))
            norm = np.linalg.norm(latent, axis=1, keepdims=True)
            latent = latent / np.maximum(norm, 1.0e-8)
            left, right = np.triu_indices(int(labels.shape[0]), k=1)
            gap = np.abs(np.round(labels[left] / PAYLOAD_BLOCK_BITS) - np.round(labels[right] / PAYLOAD_BLOCK_BITS))
            keep = gap >= 1.0
            if not np.any(keep):
                continue
            value_gap = np.abs(values[left] - values[right])[keep]
            cosine = np.sum(latent[left] * latent[right], axis=1)[keep]
            decisive += int(value_gap.size)
            value_close += int(np.sum(value_gap < VALUE_GAP_BLOCKS))
            embedding_close += int(np.sum(cosine >= COSINE_LIMIT))
            both += int(np.sum((value_gap < VALUE_GAP_BLOCKS) & (cosine >= COSINE_LIMIT)))
            cosines.extend(cosine[value_gap < VALUE_GAP_BLOCKS].tolist())
        rows.append(
            {
                "seed": int(seed),
                "decisive_pairs": decisive,
                "value_collision_fraction": float(value_close / decisive) if decisive else float("nan"),
                "embedding_collision_fraction": float(embedding_close / decisive) if decisive else float("nan"),
                "representation_limit_fraction": float(both / decisive) if decisive else float("nan"),
                "mean_cosine_when_value_collides": _mean(cosines),
            }
        )
    return rows


def _write_report(summary: dict) -> None:
    lines = [
        "# HTW Phase-5Y Executor-Conditioned Joint Value Distillation",
        "",
        "**planner_ready = False**",
        "",
        "**deployment_verified = False**",
        "",
        "本阶段只重新初始化 assignment-level value head。Phase-5L encoder、Phase-5U 关系特征和 Phase-5V 时间特征保持冻结。监督标签是 frozen MAPPO 执行 15 步后的 payload block。J15、短程 rollout 和未来 SINR 都不是网络输入。310110–310117 只评估一次。",
        "",
        "## 1. 标签",
        "",
        f"标签审计通过：{summary['integrity_summary']}。",
        "",
        "## 2. 排序",
        "",
        "| model | Spearman | Top-1 | Top-3 | Top-5 | normalized regret | optimum rank | selected J15 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        *summary["rank_lines"],
        "",
        "## 3. 相对 Phase-5V",
        "",
        f"mean ΔJ15 = {summary['delta_mean']}，95% CI [{summary['delta_low']}, {summary['delta_high']}]。",
        f"scenario 上升/持平/下降 = {summary['scenario_counts']}。seed 上升/持平/下降 = {summary['seed_counts']}。",
        "",
        "## 4. 回归与校准",
        "",
        f"confirmation：MAE {summary['mae']} block，RMSE {summary['rmse']}，Huber {summary['huber']}，R² {summary['r2']}，bias {summary['bias']}。",
        f"collapsed fraction = {summary['collapsed']}。",
        "",
        "| bin | n | mean predicted | mean true |",
        "| --- | --- | --- | --- |",
        *summary["calibration_lines"],
        "",
        "## 5. 310114、residual、recovery",
        "",
        summary["case_text"],
        "",
        summary["residual_text"],
        "",
        summary["recovery_text"],
        "",
        f"representation limit fraction = {summary['collision_fraction']}。",
        "",
        "## 6. 时间",
        "",
        summary["runtime_text"],
        "",
        "## 7. 结论",
        "",
        f"Gate 是 **{summary['gate']}**。",
        "",
        summary["advice"],
        "",
        "Q1. " + summary["answers"]["q1"],
        "",
        "Q2. " + summary["answers"]["q2"],
        "",
        "Q3. " + summary["answers"]["q3"],
        "",
        "Q4. " + summary["answers"]["q4"],
        "",
        "Q5. " + summary["answers"]["q5"],
        "",
        "Q6. " + summary["answers"]["q6"],
        "",
        "Q7. " + summary["answers"]["q7"],
        "",
        "Q8. " + summary["answers"]["q8"],
        "",
        "Q9. " + summary["answers"]["q9"],
        "",
        "Q10. " + summary["answers"]["q10"],
        "",
        "Q11. " + summary["answers"]["q11"],
        "",
        "`planner_ready = False`",
        "",
        "`deployment_verified = False`",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "phase5y_protocol.json").write_text(
        json.dumps(
            {
                "training": True,
                "encoder": "frozen Phase-5L",
                "representation": "frozen Phase-5U relational plus Phase-5V temporal",
                "value_head": "randomly initialized ExecutorConditionedJointValue",
                "loaded_phase5v_head": False,
                "loss": "state-mean Huber, delta=1 payload block",
                "loss_note": "Phase-5S steps once per state with the same Adam settings. Pairwise ranking is replaced by Huber on J15/1e6. Every state has the same candidate count, and the code still averages inside the state before averaging states.",
                "optimizer": "Adam",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "grad_clip": GRAD_CLIP,
                "epochs": EPOCHS,
                "checkpoint": "epoch_10",
                "early_stopping": False,
                "seeds": list(SEEDS),
                "train_scenarios": list(TRAIN_SCENARIOS),
                "evaluation_scenarios": list(CONFIRM_SCENARIOS),
                "opened_310118": False,
                "forbidden_inputs": list(FORBIDDEN_INPUTS),
                "top3_floor": TOP3_FLOOR,
                "regret_ceiling": REGRET_CEILING,
                "collision_fraction": COLLISION_FRACTION,
                "regression_r2": REGRESSION_R2,
                "regression_mae": REGRESSION_MAE,
                "planner_ready": PLANNER_READY,
                "deployment_verified": DEPLOYMENT_VERIFIED,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    tables, qexec = _read_rollouts()
    config = L3BConfig()
    frozen = _load_proposed()
    matching = _load_matching()
    for parameter in list(frozen.parameters()) + list(matching.parameters()):
        parameter.requires_grad_(False)
    states = _build_states(frozen, matching, config)
    integrity_rows, _label_j15 = _audit(states, tables)
    center, scale = _frozen_scaler()
    curves = []
    for seed in SEEDS:
        seed_everything(int(seed))
        scorer = ExecutorConditionedJointValue()
        seed_curves = _train_one(scorer, states, int(seed), center, scale)
        curves.extend(seed_curves)
        torch.save(
            {
                "seed": int(seed),
                "epoch": int(EPOCHS),
                "state_dict": scorer.state_dict(),
                "center": center,
                "scale": scale,
                "rel_dim": FINAL_REL_DIM,
                "frozen_for_fresh_confirmation": False,
            },
            _checkpoint(int(seed)),
        )
        print(f"saved {_checkpoint(int(seed)).name}", flush=True)

    value_models = {}
    for seed in SEEDS:
        payload = torch.load(_checkpoint(int(seed)), map_location="cpu", weights_only=False)
        if int(payload["epoch"]) != int(EPOCHS):
            raise RuntimeError("checkpoint is not epoch 10")
        scorer = ExecutorConditionedJointValue()
        scorer.load_state_dict(payload["state_dict"])
        scorer.eval()
        value_models[int(seed)] = scorer
    phase5v, _unused, v_center, v_scale = _load_v1()
    if not np.allclose(center, v_center) or not np.allclose(scale, v_scale):
        raise RuntimeError("value training used a different temporal scaler")

    rank_rows = []
    prediction_rows = []
    predictions = {}
    regression_rows = []
    calibration = []
    runtime_ms = []
    warmed = False
    for seed, scorer in value_models.items():
        pooled_pred = []
        pooled_label = []
        per_state_std = []
        for state in states:
            relational = _features(state, center, scale)
            if not warmed and state["split"] == "confirmation":
                _predict(scorer, state, relational)
                warmed = True
            started = time.perf_counter()
            values = _predict(scorer, state, relational)
            if state["split"] == "confirmation" and int(seed) == int(SEEDS[0]):
                runtime_ms.append((time.perf_counter() - started) * 1000.0)
            predictions[(int(seed), int(state["scenario_id"]), int(state["time_step"]))] = values
            rank_rows.append(_rank_row("phase5y", int(seed), state, values))
            if state["split"] == "confirmation":
                pooled_pred.append(values)
                pooled_label.append(state["j_mappo"])
                per_state_std.append(float(np.std(values)))
                for vector, value, label in zip(state["vectors"], values, state["j_mappo"]):
                    prediction_rows.append(
                        {
                            "split": "confirmation",
                            "scenario_id": int(state["scenario_id"]),
                            "time_step": int(state["time_step"]),
                            "seed": int(seed),
                            "assignment_vector": vector,
                            "v_exec": float(value),
                            "y_exec": float(label) / PAYLOAD_BLOCK_BITS,
                            "j15": float(label),
                        }
                    )
        measured = _regression_block(np.concatenate(pooled_pred), np.concatenate(pooled_label))
        measured.update(
            {
                "split": "confirmation",
                "seed": int(seed),
                "per_state_score_std": _median(per_state_std),
                "collapsed_fraction": _mean(is_collapsed(row) for row in pooled_pred),
            }
        )
        regression_rows.append(measured)
        bins = {name: [] for name in BIN_NAMES}
        flat_pred = np.concatenate(pooled_pred)
        flat_true = np.concatenate(pooled_label) / PAYLOAD_BLOCK_BITS
        for value, truth in zip(flat_pred, flat_true):
            bins[_value_bin(float(value))].append((float(value), float(truth)))
        for name in BIN_NAMES:
            pairs = bins[name]
            calibration.append(
                {
                    "seed": int(seed),
                    "bin": name,
                    "n": len(pairs),
                    "mean_predicted": _mean(item[0] for item in pairs),
                    "mean_true": _mean(item[1] for item in pairs),
                }
            )
    for state in states:
        for seed, scorer in phase5v.items():
            rank_rows.append(_rank_row("phase5v", int(seed), state, _scores(scorer, state, _features(state, v_center, v_scale))))
        for horizon in REFERENCE_HORIZONS:
            rank_rows.append(_rank_row(f"score_b_h{horizon}", -1, state, _reference_scores(state, qexec, int(horizon))))
        rank_rows.append(_rank_row("oracle", -1, state, np.asarray(state["j_mappo"], dtype=np.float64)))
        stacked = np.stack([predictions[(int(seed), int(state["scenario_id"]), int(state["time_step"]))] for seed in SEEDS], axis=0)
        rank_rows.append(_rank_row("phase5y_ensemble", -1, state, np.mean(stacked, axis=0)))

    confirmation = [state for state in states if state["split"] == "confirmation"]
    paired = []
    for state in confirmation:
        for seed in SEEDS:
            proposed = next(row for row in rank_rows if row["model"] == "phase5y" and int(row["seed"]) == int(seed) and int(row["scenario_id"]) == int(state["scenario_id"]) and int(row["time_step"]) == int(state["time_step"]))
            baseline = next(row for row in rank_rows if row["model"] == "phase5v" and int(row["seed"]) == int(seed) and int(row["scenario_id"]) == int(state["scenario_id"]) and int(row["time_step"]) == int(state["time_step"]))
            paired.append(
                {
                    "scenario_id": int(state["scenario_id"]),
                    "time_step": int(state["time_step"]),
                    "seed": int(seed),
                    "delta_j15": float(proposed["j15"] - baseline["j15"]),
                }
            )
    delta = cluster_bootstrap(paired, "delta_j15", draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    by_scenario = defaultdict(list)
    by_seed = defaultdict(list)
    for row in paired:
        by_scenario[int(row["scenario_id"])].append(float(row["delta_j15"]))
        by_seed[int(row["seed"])].append(float(row["delta_j15"]))
    scenario_up = sum(float(np.mean(value)) > 0.0 for value in by_scenario.values())
    scenario_down = sum(float(np.mean(value)) < 0.0 for value in by_scenario.values())
    scenario_tie = len(by_scenario) - scenario_up - scenario_down
    seed_up = sum(float(np.mean(value)) > 0.0 for value in by_seed.values())
    seed_down = sum(float(np.mean(value)) < 0.0 for value in by_seed.values())
    seed_tie = len(by_seed) - seed_up - seed_down
    large_drop = any(float(np.mean(value)) <= -float(LARGE_DROP_BITS) for value in by_scenario.values())
    scenario_rows = [
        {
            "scenario_id": scenario,
            "mean_delta_j15": float(np.mean(values)),
            "median_delta_j15": float(np.median(values)),
            "direction": "up" if float(np.mean(values)) > 0.0 else "down" if float(np.mean(values)) < 0.0 else "tie",
        }
        for scenario, values in sorted(by_scenario.items())
    ]

    def confirm_rows(model: str, seed: int | None = None) -> list[dict]:
        selected = [row for row in rank_rows if row["model"] == model and row["split"] == "confirmation"]
        if seed is not None:
            selected = [row for row in selected if int(row["seed"]) == int(seed)]
        return selected

    value_rows = confirm_rows("phase5y")
    mean_top3 = _mean(row["top3"] for row in value_rows)
    mean_regret = _mean(row["normalized_regret"] for row in value_rows)
    mean_top1 = _mean(row["top1"] for row in value_rows)
    mean_spearman = _mean(row["spearman"] for row in value_rows)
    collapsed = _mean(row["collapsed_fraction"] for row in regression_rows)
    mae = _mean(row["mae"] for row in regression_rows)
    rmse = _mean(row["rmse"] for row in regression_rows)
    huber = _mean(row["huber"] for row in regression_rows)
    r2 = _mean(row["r2"] for row in regression_rows)
    bias = _mean(row["bias"] for row in regression_rows)
    case_state = next(state for state in confirmation if int(state["scenario_id"]) == CASE_SCENARIO and int(state["time_step"]) == CASE_TIME)
    case_rows = []
    for seed in SEEDS:
        values = predictions[(int(seed), CASE_SCENARIO, CASE_TIME)]
        rank = _named_rank(values, case_state["vectors"], CASE_OPTIMUM)
        chosen = int(np.argsort(-values, kind="stable")[0])
        opt_index = case_state["vectors"].index(CASE_OPTIMUM)
        case_rows.append(
            {
                "seed": int(seed),
                "optimum_value": float(values[opt_index]),
                "optimum_rank": rank,
                "optimum_j15": float(case_state["j_mappo"][opt_index]),
                "selected_assignment": case_state["vectors"][chosen],
                "selected_value": float(values[chosen]),
                "selected_j15": float(case_state["j_mappo"][chosen]),
                "top1": int(rank <= 1),
                "top3": int(rank <= 3),
                "top5": int(rank <= 5),
            }
        )
    case_rank = _median(row["optimum_rank"] for row in case_rows)
    no_collapse = bool(collapsed < 0.5 and _mean(row["prediction_std"] for row in regression_rows) > 1.0e-4)
    confirmed = bool(
        float(delta["mean"]) > 0.0
        and float(delta["low"]) > 0.0
        and scenario_up >= MIN_SCENARIO_UP
        and not large_drop
        and seed_up >= MIN_SEED_UP
        and mean_top3 >= TOP3_FLOOR
        and mean_regret <= REGRET_CEILING
        and case_rank <= CASE_TOP_RANK
        and no_collapse
    )
    collision_rows = _collision(confirmation, predictions, value_models, center, scale)
    collision_fraction = _mean(row["representation_limit_fraction"] for row in collision_rows)
    representation_limit = bool(collision_fraction >= COLLISION_FRACTION and float(delta["mean"]) <= 0.0)
    regression_without_decision = bool(r2 >= REGRESSION_R2 and mae <= REGRESSION_MAE and float(delta["mean"]) <= 0.0 and not representation_limit)
    promising = bool(float(delta["mean"]) > 0.0 and not confirmed)
    gate = choose_gate(
        confirmed=confirmed,
        promising=promising,
        regression_without_decision=regression_without_decision,
        representation_limit=representation_limit,
    )
    advice = {
        "Y-A": "冻结 encoder、representation、value architecture、loss、训练预算和 candidate pool。下一阶段只在 310118–310125 做 fresh confirmation，不再调模型。本阶段不打开这些 scenario。",
        "Y-B": "平均 selected J15 有提高，但还没有达到稳定确认。下一阶段只做 value optimization stability audit，不增加网络深度，不打开 310118–310125。",
        "Y-C": "回归指标和选中 J15 不一致。下一阶段只审计 value resolution、ties 和 rank sensitivity，不立即加入 ranking loss。",
        "Y-D": "J15 差至少一个 block 时，frozen joint embedding 仍然频繁撞在一起，value head 分不开。下一阶段只做 executor-relevant representation redesign，不简单增加 HGNN 层。",
        "Y-E": "价值模型既没有可靠回归 J15，也没有改善排序，representation collision 也不明显。先复查标签、状态恢复、MAPPO 确定性和候选对齐。不打开 fresh confirmation。",
    }[gate]
    if gate == "Y-A":
        for seed in SEEDS:
            payload = torch.load(_checkpoint(int(seed)), map_location="cpu", weights_only=False)
            payload["frozen_for_fresh_confirmation"] = True
            torch.save(payload, _checkpoint(int(seed)))

    residual_cases = list(csv.DictReader(PHASE5W_RESIDUAL.open(encoding="utf-8")))
    residual_rows = []
    repaired = 0
    for item in residual_cases:
        scenario = int(item["scenario_id"])
        moment = int(item["time_step"])
        state = next(row for row in confirmation if int(row["scenario_id"]) == scenario and int(row["time_step"]) == moment)
        oracle = float(np.max(state["j_mappo"]))
        h5 = next(row for row in rank_rows if row["model"] == "score_b_h5" and int(row["scenario_id"]) == scenario and int(row["time_step"]) == moment)
        h8 = next(row for row in rank_rows if row["model"] == "score_b_h8" and int(row["scenario_id"]) == scenario and int(row["time_step"]) == moment)
        value_j = _median(row["j15"] for row in confirm_rows("phase5y") if int(row["scenario_id"]) == scenario and int(row["time_step"]) == moment)
        phase_j = _median(row["j15"] for row in confirm_rows("phase5v") if int(row["scenario_id"]) == scenario and int(row["time_step"]) == moment)
        value_regret = oracle - float(value_j)
        phase_regret = oracle - float(phase_j)
        fixed = int(value_regret < PAYLOAD_BLOCK_BITS and phase_regret >= PAYLOAD_BLOCK_BITS)
        repaired += fixed
        residual_rows.append(
            {
                "scenario_id": scenario,
                "time_step": moment,
                "phase5v_j15": phase_j,
                "h5_j15": float(h5["j15"]),
                "h8_j15": float(h8["j15"]),
                "phase5y_j15": value_j,
                "oracle_j15": oracle,
                "phase5v_regret": phase_regret,
                "h5_regret": oracle - float(h5["j15"]),
                "h8_regret": oracle - float(h8["j15"]),
                "phase5y_regret": value_regret,
                "repaired": fixed,
            }
        )
    recovery_values = []
    recovery_file = []
    for state in confirmation:
        h8 = next(row for row in rank_rows if row["model"] == "score_b_h8" and int(row["scenario_id"]) == int(state["scenario_id"]) and int(row["time_step"]) == int(state["time_step"]))
        for seed in SEEDS:
            baseline = next(row for row in rank_rows if row["model"] == "phase5v" and int(row["seed"]) == int(seed) and int(row["scenario_id"]) == int(state["scenario_id"]) and int(row["time_step"]) == int(state["time_step"]))
            proposed = next(row for row in rank_rows if row["model"] == "phase5y" and int(row["seed"]) == int(seed) and int(row["scenario_id"]) == int(state["scenario_id"]) and int(row["time_step"]) == int(state["time_step"]))
            delta_h8 = float(h8["j15"] - baseline["j15"])
            delta_value = float(proposed["j15"] - baseline["j15"])
            ratio = ""
            if delta_h8 > 0.0:
                ratio = float(delta_value / (delta_h8 + RECOVERY_EPS))
                recovery_values.append(ratio)
            recovery_file.append(
                {
                    "scenario_id": int(state["scenario_id"]),
                    "time_step": int(state["time_step"]),
                    "seed": int(seed),
                    "delta_h8": delta_h8,
                    "delta_value": delta_value,
                    "recovery": ratio,
                }
            )
    seed_rows = []
    for seed in SEEDS:
        rows = confirm_rows("phase5y", int(seed))
        base = confirm_rows("phase5v", int(seed))
        reg = next(row for row in regression_rows if int(row["seed"]) == int(seed))
        seed_rows.append(
            {
                "seed": int(seed),
                "selected_j15": _mean(row["j15"] for row in rows),
                "normalized_regret": _mean(row["normalized_regret"] for row in rows),
                "top1": _mean(row["top1"] for row in rows),
                "top3": _mean(row["top3"] for row in rows),
                "mae": reg["mae"],
                "delta_j15": _mean(float(row["j15"]) - float(base_row["j15"]) for row, base_row in zip(rows, base)),
            }
        )
    ensemble_rows = confirm_rows("phase5y_ensemble")
    seed_rows.append(
        {
            "seed": -1,
            "selected_j15": _mean(row["j15"] for row in ensemble_rows),
            "normalized_regret": _mean(row["normalized_regret"] for row in ensemble_rows),
            "top1": _mean(row["top1"] for row in ensemble_rows),
            "top3": _mean(row["top3"] for row in ensemble_rows),
            "mae": "",
            "delta_j15": "",
        }
    )
    inference_ms = _mean(runtime_ms)
    h8_ms = _h8_reference_ms()
    runtime_rows = [
        {
            "model": "phase5y",
            "ms_per_decision_120": inference_ms,
            "h8_ms_per_decision_120": h8_ms,
            "speedup_vs_h8": float(h8_ms / inference_ms) if inference_ms > 0.0 else float("nan"),
        }
    ]

    def line_for(model: str, seed: int | None = None) -> str:
        rows = confirm_rows(model, seed)
        return "| " + " | ".join(
            (
                model if seed is None else f"{model} seed {seed}",
                _fmt(_mean(row["spearman"] for row in rows)),
                _fmt(_mean(row["top1"] for row in rows)),
                _fmt(_mean(row["top3"] for row in rows)),
                _fmt(_mean(row["top5"] for row in rows)),
                _fmt(_mean(row["normalized_regret"] for row in rows)),
                _fmt(_median(row["optimum_rank"] for row in rows), ".1f"),
                _fmt(_mean(row["j15"] for row in rows), ".4e"),
            )
        ) + " |"

    rank_lines = [line_for("phase5v"), line_for("score_b_h5"), line_for("score_b_h8"), line_for("phase5y"), line_for("phase5y_ensemble"), line_for("oracle")]
    for seed in SEEDS:
        rank_lines.append(line_for("phase5y", int(seed)))
    calibration_lines = []
    for name in BIN_NAMES:
        selected = [row for row in calibration if row["bin"] == name]
        calibration_lines.append(
            f"| {name} | {int(sum(int(row['n']) for row in selected))} | {_fmt(_mean(row['mean_predicted'] for row in selected if int(row['n']) > 0))} | {_fmt(_mean(row['mean_true'] for row in selected if int(row['n']) > 0))} |"
        )
    answers = {
        "q1": f"可以评估。confirmation R² = {_fmt(r2)}，MAE = {_fmt(mae)} block。Gate 是 {gate}。",
        "q2": f"mean Δ selected J15 = {_fmt(delta['mean'], '.4e')} bit，95% CI [{_fmt(delta['low'], '.4e')}, {_fmt(delta['high'], '.4e')}]。",
        "q3": f"在 ΔH8 > 0 的 state 上，recovery mean = {_fmt(_mean(recovery_values))}，median = {_fmt(_median(recovery_values))}，IQR = {_fmt(_iqr(recovery_values))}。",
        "q4": f"Top-1 = {_fmt(mean_top1)}，Top-3 = {_fmt(mean_top3)}，normalized regret = {_fmt(mean_regret)}，Spearman = {_fmt(mean_spearman)}。",
        "q5": f"命名 optimum 的 seed 中位名次是 {_fmt(case_rank, '.1f')}。各种子名次是 {', '.join(str(row['optimum_rank']) for row in case_rows)}。",
        "q6": f"10 个 residual case 中，Phase-5Y 的 seed 中位选择把 regret 降到 1 个 block 以下的有 {repaired} 个。",
        "q7": f"seed 上升/持平/下降 = {seed_up}/{seed_tie}/{seed_down}。",
        "q8": f"decisive pair 中，embedding cosine >= {COSINE_LIMIT} 且 |ΔV| < {VALUE_GAP_BLOCKS} block 的比例是 {_fmt(collision_fraction)}。",
        "q9": advice,
        "q10": f"120 个 assignment 的 value inference 平均 { _fmt(inference_ms, '.2f') } ms。相对 H=8 的 { _fmt(h8_ms, '.1f') } ms，加速 { _fmt(h8_ms / inference_ms if inference_ms else float('nan'), '.1f') } 倍。",
        "q11": "达到 Y-A。架构可以冻结，但本阶段仍不打开 310118–310125。" if gate == "Y-A" else f"没有达到 Y-A。当前 Gate 是 {gate}。不打开 310118–310125。",
    }
    summary = {
        "integrity_summary": "conflicting J15 = 0，missing pool label = 0，illegal pool assignment = 0",
        "rank_lines": rank_lines,
        "delta_mean": _fmt(delta["mean"], ".4e"),
        "delta_low": _fmt(delta["low"], ".4e"),
        "delta_high": _fmt(delta["high"], ".4e"),
        "scenario_counts": f"{scenario_up}/{scenario_tie}/{scenario_down}",
        "seed_counts": f"{seed_up}/{seed_tie}/{seed_down}",
        "mae": _fmt(mae),
        "rmse": _fmt(rmse),
        "huber": _fmt(huber),
        "r2": _fmt(r2),
        "bias": _fmt(bias),
        "collapsed": _fmt(collapsed),
        "calibration_lines": calibration_lines,
        "case_text": f"310114 t=15 命名 optimum 中位名次 {_fmt(case_rank, '.1f')}。选中 J15 的 seed 中位数是 {_fmt(_median(row['selected_j15'] for row in case_rows), '.4e')}。",
        "residual_text": f"residual 修复 {repaired}/10。",
        "recovery_text": f"executor information recovery mean {_fmt(_mean(recovery_values))}，median {_fmt(_median(recovery_values))}，IQR {_fmt(_iqr(recovery_values))}，可用 state×seed 样本 {len(recovery_values)}。",
        "collision_fraction": _fmt(collision_fraction),
        "runtime_text": f"Phase-5Y 120 candidates { _fmt(inference_ms, '.2f') } ms；Phase-5X H=8 { _fmt(h8_ms, '.1f') } ms；speedup { _fmt(h8_ms / inference_ms if inference_ms else float('nan'), '.1f') }。",
        "gate": gate,
        "advice": advice,
        "answers": answers,
    }
    _write_csv(OUTPUT / "phase5y_label_integrity.csv", integrity_rows)
    _write_csv(OUTPUT / "phase5y_training_curves.csv", curves)
    _write_csv(OUTPUT / "phase5y_value_predictions.csv", prediction_rows)
    _write_csv(OUTPUT / "phase5y_ranking_metrics.csv", [row for row in rank_rows if row["split"] == "confirmation"])
    _write_csv(OUTPUT / "phase5y_regression_metrics.csv", regression_rows)
    _write_csv(OUTPUT / "phase5y_calibration.csv", calibration)
    _write_csv(OUTPUT / "phase5y_seed_stability.csv", seed_rows)
    _write_csv(OUTPUT / "phase5y_scenario_comparison.csv", scenario_rows)
    _write_csv(OUTPUT / "phase5y_residual_cases.csv", residual_rows)
    _write_csv(OUTPUT / "phase5y_310114_t15.csv", case_rows)
    _write_csv(OUTPUT / "phase5y_executor_information_recovery.csv", recovery_file)
    _write_csv(OUTPUT / "phase5y_representation_collision.csv", collision_rows)
    _write_csv(OUTPUT / "phase5y_runtime.csv", runtime_rows)
    payload = {
        "gate": gate,
        "planner_ready": PLANNER_READY,
        "deployment_verified": DEPLOYMENT_VERIFIED,
        "frozen_for_fresh_confirmation": gate == "Y-A",
        "delta_j15_mean": delta["mean"],
        "delta_j15_low": delta["low"],
        "delta_j15_high": delta["high"],
        "scenario_up": scenario_up,
        "scenario_tie": scenario_tie,
        "scenario_down": scenario_down,
        "seed_up": seed_up,
        "seed_tie": seed_tie,
        "seed_down": seed_down,
        "top1": mean_top1,
        "top3": mean_top3,
        "normalized_regret": mean_regret,
        "mae": mae,
        "r2": r2,
        "case_rank_median": case_rank,
        "residual_repaired": repaired,
        "recovery_mean": _mean(recovery_values),
        "collision_fraction": collision_fraction,
        "inference_ms": inference_ms,
        "speedup_vs_h8": float(h8_ms / inference_ms) if inference_ms else float("nan"),
        "trained_encoder": False,
        "opened_310118": False,
    }
    (OUTPUT / "phase5y_executor_value_gate.json").write_text(json.dumps(_plain(payload), indent=2), encoding="utf-8")
    _write_report(summary)
    print(json.dumps(_plain({"gate": gate, "delta": delta["mean"], "top3": mean_top3, "case_rank": case_rank})), flush=True)
    return summary


if __name__ == "__main__":
    run()
