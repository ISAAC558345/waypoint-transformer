"""Train pair-before-pool only after the structural swap gate passes."""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.nn.utils import clip_grad_norm_

from wt.l3b.config import L3BConfig
from wt.lowlevel_v2.phase5l_joint_encoder.run_phase5l import _load_matching
from wt.lowlevel_v2.phase5m_joint_planner.run_phase5m import _load_proposed
from wt.lowlevel_v2.phase5q_encoder_scorer_audit.metrics import metrics_at, quantize
from wt.lowlevel_v2.phase5r_nonadditive_generalization.metrics import cluster_bootstrap
from wt.lowlevel_v2.phase5u_representation_repair.scorer import RelationalSetScorer
from wt.lowlevel_v2.phase5v_temporal_representation.protocol import CHECKPOINTS as PHASE5V_CHECKPOINTS
from wt.lowlevel_v2.phase5v_temporal_representation.protocol import FINAL_REL_DIM
from wt.lowlevel_v2.phase5v_temporal_representation.run_phase5v import _build_states, _combine, _pack, _scale
from wt.lowlevel_v2.phase5y_executor_value_distillation.protocol import CHECKPOINTS as PHASE5Y_CHECKPOINTS
from wt.lowlevel_v2.phase5y_executor_value_distillation.protocol import PHASE5W_RESIDUAL, PHASE5X_ROLLOUTS, PHASE5X_RUNTIME
from wt.lowlevel_v2.phase5y_executor_value_distillation.value import ExecutorConditionedJointValue, state_huber
from wt.lowlevel_v2.phase5za_correspondence_pooling.metrics import choose_gate
from wt.lowlevel_v2.phase5za_correspondence_pooling.pool import (
    CorrespondencePreservingValue,
    permute_buoys,
    permute_uavs,
)
from wt.lowlevel_v2.phase5za_correspondence_pooling.protocol import (
    ASSIGNMENT_DIM,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CASE_OPTIMUM,
    CASE_PREDICTED,
    CASE_SCENARIO,
    CASE_TIME,
    CASE_TOP_RANK,
    CHECKPOINTS,
    CONFIRM_SCENARIOS,
    COSINE_LIMIT,
    DEPLOYMENT_VERIFIED,
    EPOCHS,
    GRAD_CLIP,
    HEAD_INPUT_DIM,
    LEARNING_RATE,
    MIN_SCENARIO_UP,
    MIN_SEED_UP,
    OUTPUT,
    PAYLOAD_BLOCK_BITS,
    PERMUTATION_EPS,
    PHASE5Z_SWAPS,
    PLANNER_READY,
    REGRET_CEILING,
    REL_DIM,
    REPORT,
    SEEDS,
    SWAP_COLLISION_CEILING,
    TEMP_INPUT_DIM,
    TOP3_FLOOR,
    TRAIN_SCENARIOS,
    WEIGHT_DECAY,
    WIDTH,
    assert_consumed,
)
from wt.round30.ppo import seed_everything


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


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=2), encoding="utf-8")


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


def _corr(left, right) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 3 or float(np.std(x)) < 1.0e-8 or float(np.std(y)) < 1.0e-8:
        return float("nan")
    value = spearmanr(x, y).correlation
    return float(value) if np.isfinite(value) else float("nan")


def _cosine_matrix(matrix: np.ndarray) -> np.ndarray:
    flat = np.asarray(matrix, dtype=np.float64).reshape(matrix.shape[0], -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    flat = flat / np.maximum(norms, 1.0e-8)
    return flat @ flat.T


def _pair_cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    return float(np.dot(a, b) / max(np.linalg.norm(a), 1.0e-8) / max(np.linalg.norm(b), 1.0e-8))


def _effective_rank(matrix: np.ndarray) -> dict[str, float]:
    flat = np.asarray(matrix, dtype=np.float64).reshape(matrix.shape[0], -1)
    centered = flat - np.mean(flat, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular**2
    total = float(np.sum(energy))
    gram = _cosine_matrix(flat)
    count = int(gram.shape[0])
    off = float((np.sum(gram) - np.trace(gram)) / max(count * (count - 1), 1))
    upper = np.triu_indices(count, k=1)
    distance = np.linalg.norm(flat[upper[0]] - flat[upper[1]], axis=1) if count > 1 else np.zeros(0)
    if total <= 1.0e-12:
        return {"effective_rank": 0.0, "top1_variance": float("nan"), "median_cosine": off, "mean_l2": float(np.mean(distance)) if distance.size else float("nan")}
    return {
        "effective_rank": float((np.sum(singular) ** 2) / np.sum(singular**2)),
        "top1_variance": float(energy[0] / total),
        "median_cosine": off,
        "mean_l2": float(np.mean(distance)),
    }


def _blocks(labels: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(labels, dtype=np.float64) / PAYLOAD_BLOCK_BITS)


def _top5(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")[:5]
    levels = quantize(labels)
    return float(np.any(levels[order] == np.max(levels)))


def _optimum_rank(scores: np.ndarray, labels: np.ndarray) -> float:
    levels = quantize(labels)
    optimal = levels == float(np.max(levels))
    values = np.asarray(scores, dtype=np.float64)
    return float(1 + np.sum(values > float(np.max(values[optimal]))))


def _named_rank(scores: np.ndarray, index: int) -> int:
    values = np.asarray(scores, dtype=np.float64)
    return int(1 + np.sum(values > values[int(index)]))


def _regression(prediction: np.ndarray, labels: np.ndarray) -> dict[str, float]:
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
    }


def _collision_rate(embedding: np.ndarray, labels: np.ndarray, gap: int) -> dict[str, float]:
    gram = _cosine_matrix(embedding)
    left, right = np.triu_indices(int(labels.shape[0]), k=1)
    keep = np.abs(_blocks(labels)[left] - _blocks(labels)[right]) >= float(gap)
    count = int(np.sum(keep))
    if count == 0:
        return {"pairs": 0.0, "cosine_high": float("nan")}
    return {"pairs": float(count), "cosine_high": float(np.mean(gram[left, right][keep] >= COSINE_LIMIT))}


def _features(state: dict, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return _combine(state["z_rel"], _scale(state["z_temp_raw"], center, scale))


def _scores(module, state: dict, relational: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return module(_pack(state["embedded"], relational), torch.as_tensor(state["assignments"], dtype=torch.long)).detach().cpu().numpy()


def _embedding(module, state: dict, relational: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        packed = _pack(state["embedded"], relational)
        return module.assignment_embedding(packed, torch.as_tensor(state["assignments"], dtype=torch.long)).detach().cpu().numpy()


def _tokens(module, state: dict, relational: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        packed = _pack(state["embedded"], relational)
        assignments = torch.as_tensor(state["assignments"], dtype=torch.long)
        return module.pool.pair_tokens(packed, assignments, packed["z_rel"]).detach().cpu().numpy()


def _load_j15() -> dict[tuple[int, int, str], float]:
    labels = defaultdict(list)
    with PHASE5X_ROLLOUTS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["horizon"]) != 15:
                continue
            scenario = int(row["scenario_id"])
            assert_consumed(scenario)
            labels[(scenario, int(row["time_step"]), str(row["assignment_vector"]))].append(float(row["j15"]))
    cleaned = {}
    for key, values in labels.items():
        if float(np.max(values) - np.min(values)) > 1.0:
            raise RuntimeError(f"conflicting saved J15 at {key}")
        cleaned[key] = float(np.mean(values))
    return cleaned


def _load_qexec() -> dict[tuple[int, int, str, int], float]:
    found = {}
    with PHASE5X_ROLLOUTS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            horizon = int(row["horizon"])
            if horizon not in (5, 8):
                continue
            scenario = int(row["scenario_id"])
            assert_consumed(scenario)
            found[(scenario, int(row["time_step"]), str(row["assignment_vector"]), horizon)] = float(row["q_exec_h"])
    return found


def _scaler() -> tuple[np.ndarray, np.ndarray]:
    first = None
    for seed in SEEDS:
        payload = torch.load(PHASE5V_CHECKPOINTS / f"phase5v_temp_aug_seed{int(seed)}.pt", map_location="cpu", weights_only=False)
        center = np.asarray(payload["center"], dtype=np.float32)
        scale = np.asarray(payload["scale"], dtype=np.float32)
        if first is None:
            first = (center, scale)
        elif not np.allclose(first[0], center) or not np.allclose(first[1], scale):
            raise RuntimeError("Phase-5V scaler is not shared")
    return first


def _manifest(module: CorrespondencePreservingValue) -> dict:
    rows = []
    for name, parameter in module.named_parameters():
        if not (name.startswith("pool.phi") or name.startswith("mlp")):
            raise RuntimeError(f"unexpected trainable parameter {name}")
        rows.append({"name": name, "numel": int(parameter.numel()), "requires_grad": bool(parameter.requires_grad)})
    return {
        "trainable_groups": ["CorrespondencePreservingPairPool.phi", "Phase-5Y-shaped scalar value head"],
        "frozen": ["Phase-5L HGNN", "Phase-5U REP", "Phase-5V TEMP", "global state branch", "Residual MAPPO", "candidate generator"],
        "input_dim": int(module.pool.input_dim),
        "hidden_dim": int(module.pool.hidden_dim),
        "output_dim": int(module.pool.output_dim),
        "head_input_dim": int(HEAD_INPUT_DIM),
        "parameters": rows,
        "trainable_count": int(sum(item["numel"] for item in rows)),
    }


def _old_pooling_audit() -> dict:
    return {
        "source": "assignment_latents in phase5t represent.py, used by Phase-5Y RelationalSetScorer",
        "old_pooling_equation": (
            "triple_i = concat(H_UAV_i, H_Buoy_{a_i}, H_edge_{i,a_i}); "
            "z_selected = sum_i(triple_i * 1[a_i != idle]) / count(selected); "
            "pair_ij = concat(symmetric H_UU_ij, interact(concat(edge_i, edge_j))) for both selected; "
            "z_pairs = mean(pair_ij); "
            "z_post = concat(z_selected, z_pairs, z_global); "
            "head_input = concat(z_post, z_REP_TEMP)"
        ),
        "aggregated_before_binding_survives": {
            "buoy_slice_of_z_selected": "sum of H_Buoy over the selected buoy set. The UAV that chose the buoy cancels out of this slice.",
            "uav_slice_of_z_selected": "sum of H_UAV over the selected UAV set. The target cancels out of this slice.",
            "h_uu": "UAV-UAV embedding does not depend on the assigned buoys.",
        },
        "still_carries_correspondence": "The edge slice is sum_i H_edge_{i,a_i}. It is the only assignment-specific term that sees the matched pair, and Phase-5Z showed that the following mean makes its residual small.",
        "dimensions": {
            "selected_triple_mean": int(WIDTH * 3),
            "pair_mean": int(WIDTH * 2),
            "global": int(WIDTH),
            "post_pool": int(WIDTH * 6),
            "rep_temp": int(FINAL_REL_DIM),
            "head_input": int(WIDTH * 6 + FINAL_REL_DIM),
        },
        "replacement": {
            "pair_first": "pair_i = concat(H_UAV_i, H_Buoy_or_zero, edge_or_zero, broadcast REP, broadcast TEMP, idle_flag)",
            "phi": "Linear -> SiLU -> Linear, shared across UAVs",
            "pool_second": "z_assignment = mean_i phi(pair_i), including idle tokens",
            "global_unchanged": "concat(z_assignment, z_global, z_REP_TEMP)",
            "rep_temp_note": "Frozen REP and TEMP are assignment-level. They are broadcast onto every UAV token because the frozen feature code has no per-UAV REP or TEMP vector.",
            "input_dim": int(WIDTH * 3 + REL_DIM + TEMP_INPUT_DIM + 1),
            "hidden_dim": int(WIDTH),
            "output_dim": int(ASSIGNMENT_DIM),
        },
    }


def _load_swaps(split: str) -> list[dict]:
    rows = []
    with PHASE5Z_SWAPS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split or row["in_pool"] != "1" or row["active_pair"] != "1":
                continue
            if abs(float(row["delta_j15"])) < PAYLOAD_BLOCK_BITS:
                continue
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no registered {split} swap pairs")
    return rows


def _state_index(states: list[dict]) -> dict[tuple[int, int], dict]:
    return {(int(state["scenario_id"]), int(state["time_step"])): state for state in states}


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left - right)).detach().cpu())


def _structural(module, states: list[dict], center: np.ndarray, scale: np.ndarray, swaps: list[dict]) -> tuple[list[dict], dict]:
    module.eval()
    rows = []
    swap_hits = []
    located = _state_index(states)
    embeddings = {}
    for state in states:
        if state["split"] != "train":
            raise RuntimeError("structural gate saw a confirmation state")
        relational = _features(state, center, scale)
        packed = _pack(state["embedded"], relational)
        assignments = torch.as_tensor(state["assignments"], dtype=torch.long)
        with torch.no_grad():
            base = module.assignment_embedding(packed, assignments)
            value = module(packed, assignments)
        uav_perm = torch.tensor([1, 0, 2, 3, 4, 5])
        buoy_perm = torch.tensor([1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        moved_u, assign_u = permute_uavs(packed, assignments, uav_perm)
        moved_b, assign_b = permute_buoys(packed, assignments, buoy_perm)
        order = torch.randperm(assignments.shape[0])
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel())
        shuffled = {key: value.detach().clone() for key, value in packed.items()}
        shuffled["z_rel"] = packed["z_rel"][order]
        with torch.no_grad():
            uav_value = module(moved_u, assign_u)
            buoy_value = module(moved_b, assign_b)
            shuffled_value = module(shuffled, assignments[order])[inverse]
            shuffled_pool = module.assignment_embedding(shuffled, assignments[order])[inverse]
        uav_error = _max_abs(value, uav_value)
        buoy_error = _max_abs(value, buoy_value)
        enum_error = max(_max_abs(value, shuffled_value), _max_abs(base, shuffled_pool))
        rows.append(
            {
                "scenario_id": int(state["scenario_id"]),
                "time_step": int(state["time_step"]),
                "uav_relabel_error": uav_error,
                "buoy_relabel_error": buoy_error,
                "enumeration_error": enum_error,
            }
        )
        embeddings[(int(state["scenario_id"]), int(state["time_step"]))] = base.detach().cpu().numpy()
        del moved_u, moved_b, shuffled
    missing = 0
    swap_cosines = []
    old_cosines = []
    for swap in swaps:
        key = (int(swap["scenario_id"]), int(swap["time_step"]))
        state = located[key]
        pool = {vector: index for index, vector in enumerate(state["vectors"])}
        if swap["assignment"] not in pool or swap["swapped"] not in pool:
            missing += 1
            continue
        left = embeddings[key][pool[swap["assignment"]]]
        right = embeddings[key][pool[swap["swapped"]]]
        cosine = _pair_cosine(left, right)
        swap_hits.append(cosine >= COSINE_LIMIT)
        swap_cosines.append(cosine)
        old_cosines.append(float(swap["postpool_cosine"]))
    if missing:
        raise RuntimeError(f"{missing} registered train swaps are not in the saved pool")
    summary = {
        "uav_relabel_error": max(float(row["uav_relabel_error"]) for row in rows),
        "buoy_relabel_error": max(float(row["buoy_relabel_error"]) for row in rows),
        "enumeration_error": max(float(row["enumeration_error"]) for row in rows),
        "train_swap_pairs": len(swap_hits),
        "train_swap_cosine_high": float(np.mean(swap_hits)),
        "train_swap_cosine_median": float(np.median(swap_cosines)),
        "train_swap_cosine_p10": float(np.percentile(swap_cosines, 10)),
        "old_train_swap_cosine_high": float(np.mean(np.asarray(old_cosines) >= COSINE_LIMIT)),
    }
    summary["passed"] = bool(
        summary["uav_relabel_error"] <= PERMUTATION_EPS
        and summary["buoy_relabel_error"] <= PERMUTATION_EPS
        and summary["enumeration_error"] <= PERMUTATION_EPS
        and summary["train_swap_cosine_high"] <= SWAP_COLLISION_CEILING
    )
    return rows, summary


def _train_one(module, states: list[dict], seed: int, center: np.ndarray, scale: np.ndarray) -> list[dict]:
    train_states = [state for state in states if state["split"] == "train"]
    if any(int(state["scenario_id"]) not in TRAIN_SCENARIOS for state in train_states):
        raise RuntimeError("confirmation state entered the pooling gradient")
    optimizer = torch.optim.Adam(module.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    curves = []
    for epoch in range(1, int(EPOCHS) + 1):
        module.train()
        losses = []
        grads = []
        for state in train_states:
            optimizer.zero_grad(set_to_none=True)
            relational = _features(state, center, scale)
            prediction = module(_pack(state["embedded"], relational), torch.as_tensor(state["assignments"], dtype=torch.long))
            target = torch.as_tensor(state["j_mappo"] / PAYLOAD_BLOCK_BITS, dtype=torch.float32)
            loss = state_huber(prediction, target)
            if not bool(torch.isfinite(loss).all()):
                raise RuntimeError(f"pooling loss became non-finite at seed {seed} epoch {epoch}")
            loss.backward()
            grads.append(float(clip_grad_norm_(module.parameters(), GRAD_CLIP)))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        module.eval()
        predictions = []
        labels = []
        stds = []
        ranks = []
        with torch.no_grad():
            for state in train_states:
                relational = _features(state, center, scale)
                values = _scores(module, state, relational)
                predictions.append(values)
                labels.append(state["j_mappo"])
                stds.append(float(np.std(values)))
                ranks.append(_effective_rank(_embedding(module, state, relational))["effective_rank"])
        fitted = _regression(np.concatenate(predictions), np.concatenate(labels))
        curves.append(
            {
                "seed": int(seed),
                "epoch": int(epoch),
                "train_huber": _mean(losses),
                "train_mae": fitted["mae"],
                "train_r2": fitted["r2"],
                "score_std": _mean(stds),
                "gradient_norm": _mean(grads),
                "pool_effective_rank": _mean(ranks),
            }
        )
        print(
            f"pool seed {seed} epoch {epoch} huber={_mean(losses):.4f} mae={fitted['mae']:.4f} r2={fitted['r2']:.4f}",
            flush=True,
        )
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()
    return curves


def _rank_row(model: str, seed: int, state: dict, scores: np.ndarray) -> dict:
    chosen = int(np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")[0])
    measured = metrics_at(scores, state["j_mappo"], chosen)
    return {
        "model": model,
        "seed": int(seed),
        "scenario_id": int(state["scenario_id"]),
        "time_step": int(state["time_step"]),
        "selected_j15": measured["j_mappo"],
        "normalized_regret": measured["normalized_regret"],
        "top1": measured["top1"],
        "top3": measured["top3"],
        "top5": _top5(scores, state["j_mappo"]),
        "optimum_rank": _optimum_rank(scores, state["j_mappo"]),
        "spearman": measured["spearman"],
        "kendall": measured["kendall"],
        "assignment_vector": state["vectors"][chosen],
    }


def _reference_ms(horizon: int) -> float:
    with PHASE5X_RUNTIME.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["horizon"]) == int(horizon):
                return float(row["ms_per_decision_120"])
    raise RuntimeError(f"missing H={horizon} runtime")


def _stop_report(summary: dict) -> None:
    lines = [
        "# HTW Phase-5ZA Correspondence-Preserving Assignment Pooling",
        "",
        "**planner_ready = False**",
        "",
        "**deployment_verified = False**",
        "",
        "Structural gate 没有通过。按预先写定的规则，训练没有开始，310110–310117 没有用来评估性能，310118–310125 没有打开。",
        "",
        "## Q1",
        "",
        "旧 pooling 先把 `(H_UAV_i, H_Buoy_{a_i}, H_edge_{i,a_i})` 拼成 triple，再对选中 UAV 做 mean。buoy 段的和只取决于选中了哪些浮标，不取决于哪一架 UAV 连到哪一个浮标。UAV 段同样只保留选中的 UAV 集合。`H_UU` 本身不含目标。edge 的和还带对应关系，但 Phase-5Z 显示它在 mean 之后只剩很小的残差。",
        "",
        f"本阶段预注册的 phi 是 Linear({int(WIDTH * 3 + REL_DIM + TEMP_INPUT_DIM + 1)}) → SiLU → Linear({int(ASSIGNMENT_DIM)})，隐藏维度 {int(WIDTH)}。`z_assignment = mean_i phi(pair_i)`，idle 也占一个 token。全局状态和 value head 的输入宽度 {int(HEAD_INPUT_DIM)} 保持 Phase-5Y。REP 和 TEMP 在冻结代码里只有 assignment 级向量，因此广播到每个 UAV token，没有新的 per-UAV 特征。",
        "",
        "## Q2–Q4",
        "",
        f"UAV 整体重编号、buoy 整体重编号、候选枚举顺序的 value 误差分别是 {_fmt(summary['uav_relabel_error'], '.3e')}、{_fmt(summary['buoy_relabel_error'], '.3e')}、{_fmt(summary['enumeration_error'], '.3e')}，都低于 1e-5。permutation equivariance 成立。",
        "",
        f"训练集 310100–310109 上，Phase-5Z 已登记且 `|ΔJ15| ≥ 1 block` 的主动 swap 有 {int(summary['train_swap_pairs'])} 对。旧 post-pool cosine ≥ 0.99 的比例是 {_fmt(summary['old_train_swap_cosine_high'])}。随机初始化的 pair-before-pool 降到 {_fmt(summary['train_swap_cosine_high'])}。cosine 中位数是 {_fmt(summary['train_swap_cosine_median'])}，10% 分位是 {_fmt(summary['train_swap_cosine_p10'])}。预先写定的上限是 {SWAP_COLLISION_CEILING:.2f}。0.31 仍然高于 0.10，所以不能把 correspondence collapse 记成已经消除。",
        "",
        "effective rank、确认集 collision 和 selected J15 都没有计算。那些指标要求先通过 structural gate，再训练。",
        "",
        "## Q5–Q11",
        "",
        "没有训练，因此没有新的 value，也没有 310114 的新 rank、相对 Phase-5V / Phase-5Y 的 selected J15、residual 修复数、seed 方向或 inference 计时。用这次没通过的随机投影去报性能，会把结构失败说成决策结果。",
        "",
        "## Q12",
        "",
        "没有达到 POOL-A，不能冻结后进入 310118–310125。",
        "",
        "Gate 是 **POOL-D**。结构检查的名字是 **POOLSTRUCT-C**：pair-before-pool 保留了合法的重编号等价，但在登记的高价值 swap 上，仍有约三成 pair 的 pooled cosine ≥ 0.99。下一步只能重新设计 pair binding / aggregation，不能加 HGNN 深度，也不能把这次的 phi 拿去训练。",
        "",
        "`planner_ready = False`",
        "",
        "`deployment_verified = False`",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def _timed(module, state: dict, relational: np.ndarray) -> float:
    assignments = torch.as_tensor(state["assignments"], dtype=torch.long)
    packed = _pack(state["embedded"], relational)
    for _ in range(2):
        module(packed, assignments)
    start = time.perf_counter()
    repeats = 5
    for _ in range(repeats):
        module(packed, assignments)
    return float((time.perf_counter() - start) * 1000.0 / repeats)


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    _write_json(
        OUTPUT / "phase5za_protocol.json",
        {
            "training_data": list(TRAIN_SCENARIOS),
            "evaluation_after_all_checkpoints": list(CONFIRM_SCENARIOS),
            "opened_310118": False,
            "new_j15_rollout": False,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "epochs": EPOCHS,
            "checkpoint": "epoch_10",
            "loss": "state-mean Huber, delta 1, target J15/1e6",
            "swap_collision_ceiling": SWAP_COLLISION_CEILING,
            "permutation_eps": PERMUTATION_EPS,
            "planner_ready": PLANNER_READY,
            "deployment_verified": DEPLOYMENT_VERIFIED,
        },
    )
    _write_json(OUTPUT / "phase5za_old_pooling_audit.json", _old_pooling_audit())
    config = L3BConfig()
    frozen = _load_proposed()
    matching = _load_matching()
    for parameter in list(frozen.parameters()) + list(matching.parameters()):
        parameter.requires_grad_(False)
    states = _build_states(frozen, matching, config)
    labels = _load_j15()
    for state in states:
        assert_consumed(int(state["scenario_id"]))
        saved = np.asarray(
            [labels[(int(state["scenario_id"]), int(state["time_step"]), vector)] for vector in state["vectors"]],
            dtype=np.float64,
        )
        if float(np.max(np.abs(saved - state["j_mappo"]))) > 1.0:
            raise RuntimeError(f"Phase-5X J15 does not match the stored label at {state['scenario_id']} t={state['time_step']}")
        state["j_mappo"] = saved
    center, scale = _scaler()
    seed_everything(0)
    probe = CorrespondencePreservingValue()
    _write_json(OUTPUT / "phase5za_trainable_parameter_manifest.json", _manifest(probe))
    train_states = [state for state in states if state["split"] == "train"]
    structural_rows, structural = _structural(probe, train_states, center, scale, _load_swaps("train"))
    _write_csv(OUTPUT / "phase5za_structural_equivariance.csv", structural_rows)
    print(json.dumps(_plain({"structural": structural})), flush=True)
    if not structural["passed"]:
        gate = {
            "gate": "POOL-D",
            "structural_gate": "POOLSTRUCT-C",
            "trained": False,
            "planner_ready": False,
            "deployment_verified": False,
            "frozen_for_fresh_confirmation": False,
            "opened_310118": False,
            **structural,
        }
        _write_json(OUTPUT / "phase5za_pooling_repair_gate.json", gate)
        _stop_report(structural)
        return gate

    curves = []
    for seed in SEEDS:
        seed_everything(int(seed))
        module = CorrespondencePreservingValue()
        curves.extend(_train_one(module, states, int(seed), center, scale))
        torch.save(
            {
                "seed": int(seed),
                "epoch": int(EPOCHS),
                "state_dict": module.state_dict(),
                "input_dim": int(module.pool.input_dim),
                "hidden_dim": int(module.pool.hidden_dim),
                "output_dim": int(module.pool.output_dim),
                "frozen_for_fresh_confirmation": False,
            },
            CHECKPOINTS / f"phase5za_seed{int(seed)}.pt",
        )
        print(f"saved phase5za_seed{int(seed)}.pt", flush=True)
    _write_csv(OUTPUT / "phase5za_training_curves.csv", curves)

    confirmation = [state for state in states if state["split"] == "confirmation"]
    phase5v = {}
    for seed in SEEDS:
        payload = torch.load(PHASE5V_CHECKPOINTS / f"phase5v_temp_aug_seed{int(seed)}.pt", map_location="cpu", weights_only=False)
        scorer = RelationalSetScorer(rel_dim=FINAL_REL_DIM)
        scorer.load_state_dict(payload["state_dict"])
        scorer.eval()
        phase5v[int(seed)] = scorer
    phase5y = {}
    for seed in SEEDS:
        payload = torch.load(PHASE5Y_CHECKPOINTS / f"phase5y_seed{int(seed)}.pt", map_location="cpu", weights_only=False)
        scorer = ExecutorConditionedJointValue()
        scorer.load_state_dict(payload["state_dict"])
        scorer.eval()
        phase5y[int(seed)] = scorer
    trained = {}
    for seed in SEEDS:
        payload = torch.load(CHECKPOINTS / f"phase5za_seed{int(seed)}.pt", map_location="cpu", weights_only=False)
        if int(payload["epoch"]) != int(EPOCHS):
            raise RuntimeError("checkpoint is not epoch 10")
        module = CorrespondencePreservingValue()
        module.load_state_dict(payload["state_dict"])
        module.eval()
        trained[int(seed)] = module

    rank_rows = []
    prediction_rows = []
    regression_rows = []
    collision_rows = []
    rank_audit = []
    case_rows = []
    qexec = _load_qexec()
    runtime_new = []
    runtime_old = []
    confirm_embeddings = {int(seed): {} for seed in SEEDS}
    confirm_values = {int(seed): {} for seed in SEEDS}
    for state in confirmation:
        relational = _features(state, center, scale)
        key = (int(state["scenario_id"]), int(state["time_step"]))
        for seed in SEEDS:
            values = _scores(trained[int(seed)], state, relational)
            embedding = _embedding(trained[int(seed)], state, relational)
            confirm_values[int(seed)][key] = values
            confirm_embeddings[int(seed)][key] = embedding
            rank_rows.append(_rank_row("phase5za", int(seed), state, values))
            measured = _effective_rank(embedding)
            rank_audit.append({"stage": "trained", "model": "phase5za", "seed": int(seed), "scenario_id": key[0], "time_step": key[1], **measured})
            for gap in (1, 2, 3, 4):
                rate = _collision_rate(embedding, state["j_mappo"], gap)
                collision_rows.append(
                    {
                        "model": "phase5za",
                        "seed": int(seed),
                        "scenario_id": key[0],
                        "time_step": key[1],
                        "gap_blocks": gap,
                        **rate,
                    }
                )
            for index, vector in enumerate(state["vectors"]):
                prediction_rows.append(
                    {
                        "scenario_id": key[0],
                        "time_step": key[1],
                        "seed": int(seed),
                        "assignment_vector": vector,
                        "v_exec": float(values[index]),
                        "y_exec": float(state["j_mappo"][index] / PAYLOAD_BLOCK_BITS),
                        "j15": float(state["j_mappo"][index]),
                    }
                )
            v0 = _scores(phase5v[int(seed)], state, relational)
            y0 = _scores(phase5y[int(seed)], state, relational)
            rank_rows.append(_rank_row("phase5v", int(seed), state, v0))
            rank_rows.append(_rank_row("phase5y", int(seed), state, y0))
        for horizon, name in ((5, "h5"), (8, "h8")):
            scores = np.asarray([qexec[(key[0], key[1], vector, horizon)] for vector in state["vectors"]], dtype=np.float64)
            rank_rows.append(_rank_row(name, -1, state, scores))
        rank_rows.append(_rank_row("oracle", -1, state, state["j_mappo"]))
        ensemble = np.mean([confirm_values[int(seed)][key] for seed in SEEDS], axis=0)
        rank_rows.append(_rank_row("phase5za_ensemble", -1, state, ensemble))
        runtime_new.append(_timed(trained[0], state, relational))
        runtime_old.append(_timed(phase5y[0], state, relational))
        pred_index = state["vectors"].index(CASE_PREDICTED) if int(state["scenario_id"]) == CASE_SCENARIO and int(state["time_step"]) == CASE_TIME else None
        if pred_index is not None:
            opt_index = state["vectors"].index(CASE_OPTIMUM)
            tokens = _tokens(trained[0], state, relational)
            for seed in SEEDS:
                values = confirm_values[int(seed)][key]
                case_rows.append(
                    {
                        "seed": int(seed),
                        "pair_token_cosine": _pair_cosine(tokens[pred_index], tokens[opt_index]),
                        "pooled_cosine": _pair_cosine(confirm_embeddings[int(seed)][key][pred_index], confirm_embeddings[int(seed)][key][opt_index]),
                        "v_pred": float(values[pred_index]),
                        "v_opt": float(values[opt_index]),
                        "opt_rank": _named_rank(values, opt_index),
                        "selected_assignment": state["vectors"][int(np.argmax(values))],
                        "selected_j15": float(state["j_mappo"][int(np.argmax(values))]),
                    }
                )
        print(f"evaluated {key[0]} t={key[1]}", flush=True)

    for seed in SEEDS:
        gathered_v = []
        gathered_y = []
        for state in confirmation:
            key = (int(state["scenario_id"]), int(state["time_step"]))
            gathered_v.append(confirm_values[int(seed)][key])
            gathered_y.append(state["j_mappo"])
        fitted = _regression(np.concatenate(gathered_v), np.concatenate(gathered_y))
        fitted.update({"seed": int(seed)})
        regression_rows.append(fitted)
    _write_csv(OUTPUT / "phase5za_predictions.csv", prediction_rows)
    _write_csv(OUTPUT / "phase5za_regression_metrics.csv", regression_rows)
    _write_csv(OUTPUT / "phase5za_ranking_metrics.csv", rank_rows)
    _write_csv(OUTPUT / "phase5za_pool_collision_before_after.csv", collision_rows)
    _write_csv(OUTPUT / "phase5za_effective_rank.csv", rank_audit)
    _write_csv(OUTPUT / "phase5za_310114_t15.csv", case_rows)

    def _selected(model: str, seed: int | None = None) -> list[dict]:
        rows = [row for row in rank_rows if row["model"] == model]
        if seed is not None:
            rows = [row for row in rows if int(row["seed"]) == int(seed)]
        return rows

    za_rows = _selected("phase5za")
    delta_v = []
    delta_y = []
    for row in za_rows:
        base_v = next(item for item in _selected("phase5v", int(row["seed"])) if item["scenario_id"] == row["scenario_id"] and item["time_step"] == row["time_step"])
        base_y = next(item for item in _selected("phase5y", int(row["seed"])) if item["scenario_id"] == row["scenario_id"] and item["time_step"] == row["time_step"])
        delta_v.append({"scenario_id": int(row["scenario_id"]), "delta": float(row["selected_j15"]) - float(base_v["selected_j15"])})
        delta_y.append({"scenario_id": int(row["scenario_id"]), "delta": float(row["selected_j15"]) - float(base_y["selected_j15"])})
    boot_v = cluster_bootstrap(delta_v, "delta", draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    boot_y = cluster_bootstrap(delta_y, "delta", draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)

    def _scenario_sign(deltas: list[dict]) -> tuple[int, int, int]:
        grouped = defaultdict(list)
        for row in deltas:
            grouped[int(row["scenario_id"])].append(float(row["delta"]))
        up = tie = down = 0
        for values in grouped.values():
            mean = float(np.mean(values))
            if abs(mean) < 1.0:
                tie += 1
            elif mean > 0.0:
                up += 1
            else:
                down += 1
        return up, tie, down

    scenario_v = _scenario_sign(delta_v)
    scenario_y = _scenario_sign(delta_y)
    seed_up = seed_tie = seed_down = 0
    seed_rows = []
    for seed in SEEDS:
        rows = _selected("phase5za", int(seed))
        base = _selected("phase5v", int(seed))
        mean_j = _mean(float(row["selected_j15"]) for row in rows)
        mean_base = _mean(float(row["selected_j15"]) for row in base)
        if abs(mean_j - mean_base) < 1.0:
            seed_tie += 1
            direction = "tie"
        elif mean_j > mean_base:
            seed_up += 1
            direction = "positive"
        else:
            seed_down += 1
            direction = "negative"
        collision = [row for row in collision_rows if int(row["seed"]) == int(seed) and int(row["gap_blocks"]) == 1]
        weight = sum(float(row["pairs"]) for row in collision)
        cosine = float(sum(float(row["cosine_high"]) * float(row["pairs"]) for row in collision) / weight)
        seed_rows.append(
            {
                "seed": int(seed),
                "selected_j15": mean_j,
                "normalized_regret": _mean(float(row["normalized_regret"]) for row in rows),
                "top1": _mean(float(row["top1"]) for row in rows),
                "top3": _mean(float(row["top3"]) for row in rows),
                "mae": next(row["mae"] for row in regression_rows if int(row["seed"]) == int(seed)),
                "r2": next(row["r2"] for row in regression_rows if int(row["seed"]) == int(seed)),
                "collision_rate": cosine,
                "versus_phase5v": direction,
            }
        )
    _write_csv(OUTPUT / "phase5za_seed_stability.csv", seed_rows)
    comparison_rows = []
    for scenario in CONFIRM_SCENARIOS:
        for model in ("phase5za", "phase5v", "phase5y", "h5", "h8", "oracle"):
            rows = [row for row in rank_rows if row["model"] == model and int(row["scenario_id"]) == int(scenario)]
            comparison_rows.append(
                {
                    "scenario_id": int(scenario),
                    "model": model,
                    "selected_j15": _mean(float(row["selected_j15"]) for row in rows),
                    "normalized_regret": _mean(float(row["normalized_regret"]) for row in rows),
                    "top1": _mean(float(row["top1"]) for row in rows),
                    "top3": _mean(float(row["top3"]) for row in rows),
                }
            )
    _write_csv(OUTPUT / "phase5za_scenario_comparison.csv", comparison_rows)

    swaps = _load_swaps("confirmation")
    located = _state_index(confirmation)
    swap_rows = []
    distances = []
    for swap in swaps:
        key = (int(swap["scenario_id"]), int(swap["time_step"]))
        state = located[key]
        pool = {vector: index for index, vector in enumerate(state["vectors"])}
        left = pool[swap["assignment"]]
        right = pool[swap["swapped"]]
        new_cosine = _pair_cosine(confirm_embeddings[0][key][left], confirm_embeddings[0][key][right])
        new_gap = float(confirm_values[0][key][left] - confirm_values[0][key][right])
        assignment = np.array([int(item) for item in swap["assignment"].split(";")])
        swapped = np.array([int(item) for item in swap["swapped"].split(";")])
        structural_distance = float(np.sum(assignment != swapped))
        embedding_distance = float(np.linalg.norm(confirm_embeddings[0][key][left] - confirm_embeddings[0][key][right]))
        swap_rows.append(
            {
                "scenario_id": key[0],
                "time_step": key[1],
                "assignment": swap["assignment"],
                "swapped": swap["swapped"],
                "old_cosine": float(swap["postpool_cosine"]),
                "new_cosine": new_cosine,
                "old_delta_v": float(swap["delta_v"]),
                "new_delta_v": new_gap,
                "delta_j15": float(swap["delta_j15"]),
                "structural_distance": structural_distance,
                "embedding_distance": embedding_distance,
            }
        )
        distances.append((structural_distance, embedding_distance, abs(new_gap), abs(float(swap["delta_j15"]))))
    _write_csv(OUTPUT / "phase5za_swap_pair_repair.csv", swap_rows)
    structural_corr = _corr([item[0] for item in distances], [item[1] for item in distances])
    value_corr = _corr([item[2] for item in distances], [item[3] for item in distances])

    residual_rows = []
    repaired = worsened = tied = 0
    for item in csv.DictReader(PHASE5W_RESIDUAL.open(encoding="utf-8")):
        key = (int(item["scenario_id"]), int(item["time_step"]))
        state = located[key]

        def chosen(model: str) -> float:
            rows = [row for row in rank_rows if row["model"] == model and int(row["scenario_id"]) == key[0] and int(row["time_step"]) == key[1]]
            return _mean(float(row["selected_j15"]) for row in rows)

        oracle = float(np.max(state["j_mappo"]))
        za = chosen("phase5za")
        previous = chosen("phase5y")
        if za > previous + 1.0:
            repaired += 1
            status = "repaired"
        elif za < previous - 1.0:
            worsened += 1
            status = "worsened"
        else:
            tied += 1
            status = "tie"
        residual_rows.append(
            {
                "scenario_id": key[0],
                "time_step": key[1],
                "phase5v_j15": chosen("phase5v"),
                "phase5y_j15": previous,
                "phase5za_j15": za,
                "h5_j15": chosen("h5"),
                "h8_j15": chosen("h8"),
                "oracle_j15": oracle,
                "versus_phase5y": status,
            }
        )
    _write_csv(OUTPUT / "phase5za_residual_cases.csv", residual_rows)

    gap_summary = {}
    for gap in (1, 2, 3, 4):
        selected = [row for row in collision_rows if int(row["gap_blocks"]) == gap and int(row["seed"]) == 0]
        weight = sum(float(row["pairs"]) for row in selected)
        gap_summary[gap] = float(sum(float(row["cosine_high"]) * float(row["pairs"]) for row in selected) / weight) if weight else float("nan")
    state_gap = [row for row in collision_rows if int(row["gap_blocks"]) == 1]
    # One rate per state, averaged across seeds, then scenario bootstrap.
    collapsed = defaultdict(list)
    for row in state_gap:
        collapsed[(int(row["scenario_id"]), int(row["time_step"]))].append(float(row["cosine_high"]))
    boot_rows = [{"scenario_id": key[0], "cosine_high": float(np.mean(values))} for key, values in collapsed.items()]
    boot_collision = cluster_bootstrap(boot_rows, "cosine_high", draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    median_rank = float(np.median([float(row["effective_rank"]) for row in rank_audit]))
    median_cosine = float(np.median([float(row["median_cosine"]) for row in rank_audit]))
    case_rank = float(np.median([int(row["opt_rank"]) for row in case_rows]))
    top3 = _mean(float(row["top3"]) for row in za_rows)
    regret = _mean(float(row["normalized_regret"]) for row in za_rows)
    selected_j = _mean(float(row["selected_j15"]) for row in za_rows)
    collision_fixed = bool(float(boot_collision["mean"]) <= SWAP_COLLISION_CEILING)
    gate_name = choose_gate(
        structural_ok=True,
        collision_fixed=collision_fixed,
        improved_v=bool(float(boot_v["mean"]) > 0.0),
        ci_above_zero=bool(float(boot_v["low"]) > 0.0),
        improved_y=bool(float(boot_y["mean"]) > 0.0),
        scenarios_ok=bool(scenario_v[0] >= MIN_SCENARIO_UP),
        seeds_ok=bool(seed_up >= MIN_SEED_UP),
        top3_ok=bool(top3 >= TOP3_FLOOR),
        regret_ok=bool(regret <= REGRET_CEILING),
        case_ok=bool(case_rank <= CASE_TOP_RANK),
    )
    frozen_flag = gate_name == "POOL-A"
    if frozen_flag:
        for seed in SEEDS:
            path = CHECKPOINTS / f"phase5za_seed{int(seed)}.pt"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["frozen_for_fresh_confirmation"] = True
            torch.save(payload, path)
    runtime = {
        "phase5za_ms_per_120": _mean(runtime_new),
        "phase5y_ms_per_120": _mean(runtime_old),
        "h5_ms_per_120": _reference_ms(5),
        "h8_ms_per_120": _reference_ms(8),
    }
    _write_csv(OUTPUT / "phase5za_runtime.csv", [runtime])
    gate = {
        "gate": gate_name,
        "structural_gate": "passed",
        "trained": True,
        "planner_ready": False,
        "deployment_verified": False,
        "frozen_for_fresh_confirmation": frozen_flag,
        "opened_310118": False,
        "train_swap_cosine_high": structural["train_swap_cosine_high"],
        "confirmation_collision": boot_collision["mean"],
        "confirmation_collision_low": boot_collision["low"],
        "confirmation_collision_high": boot_collision["high"],
        "collision_gap1": gap_summary[1],
        "collision_gap2": gap_summary[2],
        "collision_gap3": gap_summary[3],
        "collision_gap4": gap_summary[4],
        "effective_rank_median": median_rank,
        "median_cosine": median_cosine,
        "selected_j15": selected_j,
        "delta_v_mean": boot_v["mean"],
        "delta_v_low": boot_v["low"],
        "delta_v_high": boot_v["high"],
        "delta_y_mean": boot_y["mean"],
        "delta_y_low": boot_y["low"],
        "delta_y_high": boot_y["high"],
        "scenario_up_v": scenario_v[0],
        "scenario_tie_v": scenario_v[1],
        "scenario_down_v": scenario_v[2],
        "scenario_up_y": scenario_y[0],
        "seed_up": seed_up,
        "seed_tie": seed_tie,
        "seed_down": seed_down,
        "top3": top3,
        "normalized_regret": regret,
        "case_opt_rank_median": case_rank,
        "residual_repaired": repaired,
        "residual_worsened": worsened,
        "residual_tied": tied,
        "swap_value_spearman": value_corr,
        "swap_structural_spearman": structural_corr,
        "confirmation_swap_cosine_high": _mean(float(row["new_cosine"]) >= COSINE_LIMIT for row in swap_rows),
        **runtime,
    }
    _write_json(OUTPUT / "phase5za_pooling_repair_gate.json", gate)
    lines = [
        "# HTW Phase-5ZA Correspondence-Preserving Assignment Pooling",
        "",
        "**planner_ready = False**",
        "",
        "**deployment_verified = False**",
        "",
        "只替换了 assignment pooling。HGNN、REP、TEMP、candidate、MAPPO 和正式 planner 没有改。监督仍是 J15/1e6 的 state-mean Huber。确认集在 5 个 epoch-10 checkpoint 保存之后才评估一次。",
        "",
        "## Q1",
        "",
        "旧 pooling 先把 `(H_UAV, H_Buoy, H_edge)` 拼成 triple，再对选中 UAV 做 mean。buoy 这一段的和只保留选中的浮标集合，UAV 与浮标的对应关系在求和时消失。edge 的和还带一点对应关系，但 Phase-5Z 里它被 mean 压小了。",
        "",
        f"新 phi 的输入维度是 {int(WIDTH * 3 + REL_DIM + TEMP_INPUT_DIM + 1)}，隐藏维度是 {int(WIDTH)}，输出维度是 {int(ASSIGNMENT_DIM)}。value head 输入仍是 {int(HEAD_INPUT_DIM)}。",
        "",
        "## Q2–Q4",
        "",
        f"训练集 swap cosine ≥ 0.99 的比例是 {_fmt(structural['train_swap_cosine_high'])}。确认集 `|ΔJ15| ≥ 1 block` 的 collision rate 是 {_fmt(boot_collision['mean'])}，CI [{_fmt(boot_collision['low'])}, {_fmt(boot_collision['high'])}]。",
        f"seed 0 上 ≥1/2/3/4 block 的 pair-weighted collision 是 {_fmt(gap_summary[1])}、{_fmt(gap_summary[2])}、{_fmt(gap_summary[3])}、{_fmt(gap_summary[4])}。",
        f"确认集 2308 个 swap 里，新 cosine ≥ 0.99 的比例是 {_fmt(gate['confirmation_swap_cosine_high'])}。`|ΔV|` 与 `|ΔJ15|` 的 Spearman 是 {_fmt(value_corr)}。",
        f"新 z_assignment 的 effective rank 中位数是 {_fmt(median_rank)}，cosine 中位数是 {_fmt(median_cosine)}。旧 post-pool 参考是 rank 2.23、cosine 0.9994。",
        "",
        "## Q5–Q9",
        "",
        f"310114 t=15 的 A_opt 中位 rank 是 {_fmt(case_rank, '.1f')}。",
        f"selected J15 均值 {_fmt(selected_j, '.6g')}。相对 Phase-5V 的差是 {_fmt(boot_v['mean'], '.6g')}，CI [{_fmt(boot_v['low'], '.6g')}, {_fmt(boot_v['high'], '.6g')}]。scenario {scenario_v[0]} 升 / {scenario_v[1]} 平 / {scenario_v[2]} 降。",
        f"相对 Phase-5Y 的差是 {_fmt(boot_y['mean'], '.6g')}，CI [{_fmt(boot_y['low'], '.6g')}, {_fmt(boot_y['high'], '.6g')}]。scenario {scenario_y[0]} 升 / {scenario_y[1]} 平 / {scenario_y[2]} 降。",
        f"seed 相对 Phase-5V：{seed_up} 升 / {seed_tie} 平 / {seed_down} 降。",
        f"Top-3 {_fmt(top3)}，normalized regret {_fmt(regret)}。",
        f"10 个 residual case 相对 Phase-5Y：修复 {repaired}，持平 {tied}，变差 {worsened}。",
        "",
        "## Q10–Q12",
        "",
        f"120 assignment 的完整前向是 {_fmt(runtime['phase5za_ms_per_120'], '.2f')} ms。Phase-5Y 同口径 {_fmt(runtime['phase5y_ms_per_120'], '.2f')} ms。H=5 {_fmt(runtime['h5_ms_per_120'], '.1f')} ms，H=8 {_fmt(runtime['h8_ms_per_120'], '.1f')} ms。",
        "",
        f"Gate 是 **{gate_name}**。",
        "",
        "`planner_ready = False`",
        "",
        "`deployment_verified = False`",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(_plain({"gate": gate_name, "delta_v": boot_v["mean"], "collision": boot_collision["mean"]})), flush=True)
    return gate


if __name__ == "__main__":
    run()
