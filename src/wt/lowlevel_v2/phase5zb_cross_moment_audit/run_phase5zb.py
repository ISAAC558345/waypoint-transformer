"""Offline cross-moment audit. No parameter is updated."""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr

from wt.l3b.config import L3BConfig
from wt.lowlevel_v2.phase5l_joint_encoder.run_phase5l import _load_matching
from wt.lowlevel_v2.phase5m_joint_planner.run_phase5m import _load_proposed
from wt.lowlevel_v2.phase5r_nonadditive_generalization.metrics import cluster_bootstrap
from wt.lowlevel_v2.phase5s_multiscenario_nonadditive.protocol import TRAIN_SCENARIOS
from wt.lowlevel_v2.phase5v_temporal_representation.protocol import CHECKPOINTS as PHASE5V_CHECKPOINTS
from wt.lowlevel_v2.phase5v_temporal_representation.run_phase5v import _build_states, _combine, _pack, _scale
from wt.lowlevel_v2.phase5y_executor_value_distillation.protocol import PHASE5X_ROLLOUTS
from wt.lowlevel_v2.phase5za_correspondence_pooling.pool import CorrespondencePreservingValue
from wt.lowlevel_v2.phase5zb_cross_moment_audit.metrics import choose_gate
from wt.lowlevel_v2.phase5zb_cross_moment_audit.moments import (
    cross_moment,
    edge_mean,
    optional_terms,
    pair_mean,
    permute_buoys,
    permute_uavs,
)
from wt.lowlevel_v2.phase5zb_cross_moment_audit.protocol import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CASE_OPTIMUM,
    CASE_PREDICTED,
    CASE_SCENARIO,
    CASE_TIME,
    COLLISION_1,
    COLLISION_2,
    COSINE_LIMIT,
    DEPLOYMENT_VERIFIED,
    HELP_DROP,
    NN_MARGIN,
    NO_REDUCTION_EPS,
    OUTPUT,
    PAYLOAD_BLOCK_BITS,
    PERMUTATION_EPS,
    PHASE5Z_SWAPS,
    PLANNER_READY,
    REPORT,
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


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=2), encoding="utf-8")


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
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return format(number, digits)


def _mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _corr(left, right, kind: str = "spearman") -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 3 or float(np.std(x)) < 1.0e-8 or float(np.std(y)) < 1.0e-8:
        return float("nan")
    method = spearmanr if kind == "spearman" else kendalltau
    value = method(x, y).correlation
    return float(value) if np.isfinite(value) else float("nan")


def _blocks(labels: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(labels, dtype=np.float64) / PAYLOAD_BLOCK_BITS)


def _geometry(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = np.asarray(matrix, dtype=np.float64).reshape(matrix.shape[0], -1)
    norms = np.linalg.norm(flat, axis=1)
    unit = flat / np.maximum(norms[:, None], 1.0e-8)
    gram = unit @ unit.T
    return gram, norms, flat


def _normalized_l2(norms: np.ndarray, gram: np.ndarray, left: int, right: int) -> float:
    dot = float(gram[left, right]) * float(norms[left]) * float(norms[right])
    distance = np.sqrt(max(float(norms[left] ** 2 + norms[right] ** 2 - 2.0 * dot), 0.0))
    scale = 0.5 * (float(norms[left]) + float(norms[right])) + 1.0e-8
    return float(distance / scale)


def _effective_rank(matrix: np.ndarray) -> dict[str, float]:
    flat = np.asarray(matrix, dtype=np.float64).reshape(matrix.shape[0], -1)
    centered = flat - np.mean(flat, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = singular**2
    total = float(np.sum(energy))
    gram, _, _ = _geometry(flat)
    count = int(gram.shape[0])
    median_cosine = float((np.sum(gram) - np.trace(gram)) / max(count * (count - 1), 1))
    if total <= 1.0e-12:
        return {"effective_rank": 0.0, "top1_variance": float("nan"), "top3_variance": float("nan"), "median_cosine": median_cosine}
    share = energy / total
    return {
        "effective_rank": float((np.sum(singular) ** 2) / np.sum(singular**2)),
        "top1_variance": float(share[0]),
        "top3_variance": float(np.sum(share[: min(3, share.size)])),
        "median_cosine": median_cosine,
    }


def _load_j15(allowed: set[int]) -> dict[tuple[int, int, str], float]:
    labels = defaultdict(list)
    with PHASE5X_ROLLOUTS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["horizon"]) != 15:
                continue
            scenario = int(row["scenario_id"])
            if scenario not in allowed:
                continue
            assert_consumed(scenario)
            labels[(scenario, int(row["time_step"]), str(row["assignment_vector"]))].append(float(row["j15"]))
    cleaned = {}
    for key, values in labels.items():
        if float(np.max(values) - np.min(values)) > 1.0:
            raise RuntimeError(f"conflicting saved J15 at {key}")
        cleaned[key] = float(np.mean(values))
    return cleaned


def _load_swaps() -> list[dict]:
    rows = []
    with PHASE5Z_SWAPS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scenario = int(row["scenario_id"])
            if row["split"] != "train" or scenario not in TRAIN_SCENARIOS:
                continue
            if row["in_pool"] != "1" or row["active_pair"] != "1":
                continue
            if abs(float(row["delta_j15"])) < PAYLOAD_BLOCK_BITS:
                continue
            assert_consumed(scenario)
            rows.append(row)
    if len(rows) != 2834:
        raise RuntimeError(f"expected the 2834 Phase-5ZA train swaps, found {len(rows)}")
    return rows


def _scaler() -> tuple[np.ndarray, np.ndarray]:
    payload = torch.load(PHASE5V_CHECKPOINTS / "phase5v_temp_aug_seed0.pt", map_location="cpu", weights_only=False)
    return np.asarray(payload["center"], dtype=np.float32), np.asarray(payload["scale"], dtype=np.float32)


def _features_of(h_u, h_b, h_edge, assignments, relational) -> dict[str, np.ndarray]:
    moment = cross_moment(h_u, h_b, assignments)
    return {
        "cm": moment["representation"],
        "edge": edge_mean(h_edge, assignments),
        "pairmean": pair_mean(h_u, h_b, h_edge, assignments, relational),
        "moment": moment["moment"],
        "m_idle": moment["m_idle"],
        "idle_fraction": moment["idle_fraction"],
    }


def _permutation_row(h_u, h_b, h_edge, assignments) -> dict[str, float]:
    base = cross_moment(h_u, h_b, assignments)["representation"]
    uav_order = np.array([1, 0, 2, 3, 4, 5], dtype=np.int64)
    buoy_order = np.arange(h_b.shape[0] - 1, -1, -1, dtype=np.int64)
    h_u2, h_edge2, assign_u = permute_uavs(h_u, h_edge, assignments, uav_order)
    uav_error = float(np.max(np.abs(base - cross_moment(h_u2, h_b, assign_u)["representation"])))
    h_b2, h_edge3, assign_b = permute_buoys(h_b, h_edge, assignments, buoy_order)
    buoy_error = float(np.max(np.abs(base - cross_moment(h_u, h_b2, assign_b)["representation"])))
    order = np.arange(assignments.shape[0])[::-1]
    inverse = np.empty(order.shape[0], dtype=np.int64)
    inverse[order] = np.arange(order.shape[0])
    moved = cross_moment(h_u, h_b, assignments[order])["representation"][inverse]
    enum_error = float(np.max(np.abs(base - moved)))
    return {"uav_relabel_error": uav_error, "buoy_relabel_error": buoy_error, "enumeration_error": enum_error}


def _upper_distances(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gram, norms, _flat = _geometry(matrix)
    left, right = np.triu_indices(matrix.shape[0], k=1)
    cosine = gram[left, right]
    distance = np.empty(left.shape[0], dtype=np.float64)
    for index, (a, b) in enumerate(zip(left, right)):
        distance[index] = _normalized_l2(norms, gram, int(a), int(b))
    return left, right, cosine, distance


def _incidence_distance(assignments: np.ndarray, buoys: int) -> np.ndarray:
    count, uavs = assignments.shape
    flat = np.zeros((count, uavs * buoys), dtype=np.float64)
    for uav in range(uavs):
        active = assignments[:, uav] >= 0
        flat[np.flatnonzero(active), uav * buoys + assignments[active, uav]] = 1.0
    norms2 = np.sum(flat * flat, axis=1)
    gram = flat @ flat.T
    left, right = np.triu_indices(count, k=1)
    raw = np.sqrt(np.maximum(norms2[left] + norms2[right] - 2.0 * gram[left, right], 0.0))
    return raw / np.sqrt(float(uavs * buoys))


def _nn_rates(matrix: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    gram, _norms, _flat = _geometry(matrix)
    np.fill_diagonal(gram, -np.inf)
    neighbor = np.argmax(gram, axis=1)
    rows = np.arange(labels.shape[0])
    cosine = gram[rows, neighbor]
    gap = np.abs(_blocks(labels)[rows] - _blocks(labels)[neighbor])
    close = cosine >= COSINE_LIMIT
    return {
        "contradiction_1": float(np.mean(close & (gap >= 1.0))),
        "contradiction_2": float(np.mean(close & (gap >= 2.0))),
        "contradiction_3": float(np.mean(close & (gap >= 3.0))),
        "median_neighbor_cosine": float(np.median(cosine)),
    }


def _rate(cosines: list[float], gaps: list[float], minimum: float) -> dict[str, float]:
    cosine = np.asarray(cosines, dtype=np.float64)
    gap = np.asarray(gaps, dtype=np.float64)
    keep = gap >= minimum
    count = int(np.sum(keep))
    if count == 0:
        return {"pairs": 0.0, "cosine_high": float("nan"), "median_cosine": float("nan")}
    return {
        "pairs": float(count),
        "cosine_high": float(np.mean(cosine[keep] >= COSINE_LIMIT)),
        "median_cosine": float(np.median(cosine[keep])),
    }


def _correction() -> str:
    return "\n".join(
        [
            "# Phase-5Z / Phase-5ZA 机制修正",
            "",
            "旧代码 `assignment_latents` 已经先构造",
            "",
            "`triple_i = concat(H_UAV_i, H_Buoy_{a_i}, H_edge_{i,a_i})`",
            "",
            "然后才对选中 UAV 做 mean。Phase-5ZA 的 `phi(pair_i)` 再 mean，仍是这个一阶集合统计。",
            "",
            "因此不能再说“旧结构没有先绑定 UAV-target”。",
            "",
            "准确名称是 `FIRST_ORDER_SET_AGGREGATION_LOSS`。",
            "",
            "一阶求和保留的是 UAV 集合和 buoy 集合。`sum_i u_i b_i^T` 才会随具体配对改变。",
            "",
            "Phase-5ZA 在 2834 个 `|ΔJ15| ≥ 1 block` 的训练集 swap 上，把 cosine ≥ 0.99 的比例从 1.0 降到 0.3119，仍高于 0.10。pair binding 不是完整修复。",
            "",
        ]
    )


def run() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_json(
        OUTPUT / "phase5zb_protocol.json",
        {
            "training": False,
            "population": "310100-310109 registered active swaps with |delta J15| >= 1 block",
            "named_case": f"{CASE_SCENARIO} t={CASE_TIME}",
            "opened_310118": False,
            "confirmation_used_to_choose_aggregation": False,
            "collision_1": COLLISION_1,
            "collision_2": COLLISION_2,
            "nn_margin": NN_MARGIN,
            "help_drop": HELP_DROP,
            "planner_ready": PLANNER_READY,
            "deployment_verified": DEPLOYMENT_VERIFIED,
        },
    )
    (OUTPUT / "phase5zb_old_mechanism_correction.md").write_text(_correction(), encoding="utf-8")
    allowed = set(int(item) for item in TRAIN_SCENARIOS) | {int(CASE_SCENARIO)}
    config = L3BConfig()
    frozen = _load_proposed()
    matching = _load_matching()
    for parameter in list(frozen.parameters()) + list(matching.parameters()):
        parameter.requires_grad_(False)
    built = _build_states(frozen, matching, config)
    labels = _load_j15(allowed)
    kept = []
    for state in built:
        scenario = int(state["scenario_id"])
        moment = int(state["time_step"])
        if scenario in TRAIN_SCENARIOS or (scenario == CASE_SCENARIO and moment == CASE_TIME):
            assert_consumed(scenario)
            saved = np.asarray(
                [labels[(scenario, moment, vector)] for vector in state["vectors"]],
                dtype=np.float64,
            )
            if float(np.max(np.abs(saved - state["j_mappo"]))) > 1.0:
                raise RuntimeError(f"saved J15 drifted at {scenario} t={moment}")
            state["j_mappo"] = saved
            kept.append(state)
    center, scale = _scaler()
    seed_everything(0)
    probe = CorrespondencePreservingValue()
    probe.eval()
    for parameter in probe.parameters():
        parameter.requires_grad_(False)
    cache = {}
    permutation_rows = []
    rank_rows = []
    nn_rows = []
    distance_rows = []
    gap_rows = []
    max_error = {"uav_relabel_error": 0.0, "buoy_relabel_error": 0.0, "enumeration_error": 0.0}
    for state in kept:
        scenario = int(state["scenario_id"])
        moment = int(state["time_step"])
        h_u = state["embedded"]["h_u"][0].detach().cpu().numpy()
        h_b = state["embedded"]["h_b"][0].detach().cpu().numpy()
        h_edge = state["embedded"]["h_edge"][0].detach().cpu().numpy()
        assignments = np.asarray(state["assignments"], dtype=np.int64)
        relational = _combine(state["z_rel"], _scale(state["z_temp_raw"], center, scale)).astype(np.float64)
        features = _features_of(h_u, h_b, h_edge, assignments, relational)
        with torch.no_grad():
            packed = _pack(state["embedded"], relational.astype(np.float32))
            features["phi"] = probe.assignment_embedding(packed, torch.as_tensor(assignments, dtype=torch.long)).detach().cpu().numpy()
        key = (scenario, moment)
        grams = {}
        for name in ("cm", "pairmean", "edge", "phi"):
            grams[name] = _geometry(features[name])[:2]
        cache[key] = {
            "grams": grams,
            "features": features,
            "labels": np.asarray(state["j_mappo"], dtype=np.float64),
            "assignments": assignments,
            "vectors": list(state["vectors"]),
            "h_u": h_u,
            "h_b": h_b,
            "h_edge": h_edge,
            "scenario_id": scenario,
            "time_step": moment,
            "train": scenario in TRAIN_SCENARIOS,
        }
        if scenario in TRAIN_SCENARIOS:
            errors = _permutation_row(h_u, h_b, h_edge, assignments)
            errors.update({"scenario_id": scenario, "time_step": moment})
            permutation_rows.append(errors)
            for name in ("uav_relabel_error", "buoy_relabel_error", "enumeration_error"):
                max_error[name] = max(max_error[name], float(errors[name]))
            incidence = _incidence_distance(assignments, h_b.shape[0])
            left, right, _cosine, _distance = _upper_distances(features["cm"])
            label_gap = np.abs(_blocks(cache[key]["labels"])[left] - _blocks(cache[key]["labels"])[right])
            distance_row = {"scenario_id": scenario, "time_step": moment}
            gap_row = {"scenario_id": scenario, "time_step": moment}
            for name in ("cm", "pairmean", "edge", "phi"):
                _left, _right, cosine, distance = _upper_distances(features[name])
                distance_row[f"spearman_{name}"] = _corr(incidence, distance)
                gap_row[f"spearman_{name}"] = _corr(label_gap, distance)
                gap_row[f"kendall_{name}"] = _corr(label_gap, distance, "kendall")
                measured = _effective_rank(features[name])
                rank_rows.append({"scenario_id": scenario, "time_step": moment, "method": name, **measured})
                neighbor = _nn_rates(features[name], cache[key]["labels"])
                nn_rows.append({"scenario_id": scenario, "time_step": moment, "method": name, **neighbor})
            distance_rows.append(distance_row)
            gap_rows.append(gap_row)
        print(f"represented {scenario} t={moment}", flush=True)
    _write_csv(OUTPUT / "phase5zb_permutation_audit.csv", permutation_rows)
    permutation_ok = all(float(max_error[name]) <= PERMUTATION_EPS for name in max_error)
    swaps = _load_swaps()
    swap_rows = []
    series = {name: {"cosine": [], "gap": [], "nl2": []} for name in ("old", "phi", "pairmean", "edge", "cm")}
    for swap in swaps:
        key = (int(swap["scenario_id"]), int(swap["time_step"]))
        state = cache[key]
        pool = {vector: index for index, vector in enumerate(state["vectors"])}
        left = pool[swap["assignment"]]
        right = pool[swap["swapped"]]
        gap = abs(float(swap["delta_j15"])) / PAYLOAD_BLOCK_BITS
        row = {
            "scenario_id": key[0],
            "time_step": key[1],
            "assignment": swap["assignment"],
            "swapped": swap["swapped"],
            "delta_j15": float(swap["delta_j15"]),
            "cosine_old": float(swap["postpool_cosine"]),
        }
        series["old"]["cosine"].append(float(swap["postpool_cosine"]))
        series["old"]["gap"].append(gap)
        series["old"]["nl2"].append(float("nan"))
        for name in ("phi", "pairmean", "edge", "cm"):
            gram, norms = state["grams"][name]
            cosine = float(gram[left, right])
            distance = _normalized_l2(norms, gram, left, right)
            row[f"cosine_{name}"] = cosine
            row[f"normalized_l2_{name}"] = distance
            series[name]["cosine"].append(cosine)
            series[name]["gap"].append(gap)
            series[name]["nl2"].append(distance)
        swap_rows.append(row)
    _write_csv(OUTPUT / "phase5zb_swap_sensitivity.csv", swap_rows)
    comparison_rows = []
    large_rows = []
    for name in ("old", "phi", "pairmean", "edge", "cm"):
        for gap in (1, 2, 3, 4):
            measured = _rate(series[name]["cosine"], series[name]["gap"], float(gap))
            finite = np.asarray(series[name]["nl2"], dtype=np.float64)
            keep = np.asarray(series[name]["gap"], dtype=np.float64) >= float(gap)
            measured["median_normalized_l2"] = float(np.nanmedian(finite[keep])) if np.any(keep) else float("nan")
            measured.update({"method": name, "gap_blocks": gap})
            comparison_rows.append(measured)
            if gap >= 2:
                large_rows.append(measured)
    _write_csv(OUTPUT / "phase5zb_collision_comparison.csv", comparison_rows)
    _write_csv(OUTPUT / "phase5zb_large_gap_collision.csv", large_rows)
    _write_csv(OUTPUT / "phase5zb_assignment_distance.csv", distance_rows)
    _write_csv(OUTPUT / "phase5zb_j15_gap_distance.csv", gap_rows)
    _write_csv(OUTPUT / "phase5zb_nearest_neighbor.csv", nn_rows)
    _write_csv(OUTPUT / "phase5zb_effective_rank.csv", rank_rows)

    def _lookup(method: str, gap: int, field: str = "cosine_high") -> float:
        found = next(row for row in comparison_rows if row["method"] == method and int(row["gap_blocks"]) == gap)
        return float(found[field])

    cm_rate_1 = _lookup("cm", 1)
    cm_rate_2 = _lookup("cm", 2)
    phi_rate_1 = _lookup("phi", 1)
    pair_rate_1 = _lookup("pairmean", 1)
    edge_rate_1 = _lookup("edge", 1)
    nn_cm = _mean(float(row["contradiction_1"]) for row in nn_rows if row["method"] == "cm")
    nn_pair = _mean(float(row["contradiction_1"]) for row in nn_rows if row["method"] == "pairmean")
    nn_edge = _mean(float(row["contradiction_1"]) for row in nn_rows if row["method"] == "edge")
    spearman_cm = _mean(float(row["spearman_cm"]) for row in distance_rows)
    spearman_pair = _mean(float(row["spearman_pairmean"]) for row in distance_rows)
    spearman_edge = _mean(float(row["spearman_edge"]) for row in distance_rows)
    j15_cm = _mean(float(row["spearman_cm"]) for row in gap_rows)
    j15_pair = _mean(float(row["spearman_pairmean"]) for row in gap_rows)
    j15_edge = _mean(float(row["spearman_edge"]) for row in gap_rows)
    rank_cm = _mean(float(row["effective_rank"]) for row in rank_rows if row["method"] == "cm")
    rank_pair = _mean(float(row["effective_rank"]) for row in rank_rows if row["method"] == "pairmean")
    case = cache[(CASE_SCENARIO, CASE_TIME)]
    pred = case["vectors"].index(CASE_PREDICTED)
    opt = case["vectors"].index(CASE_OPTIMUM)
    case_gram, case_norms, _flat = _geometry(case["features"]["cm"])
    case_cosine = float(case_gram[pred, opt])
    case_l2 = _normalized_l2(case_norms, case_gram, pred, opt)
    moment_gap = float(np.linalg.norm(case["features"]["moment"][pred] - case["features"]["moment"][opt]))
    idle_gap = float(np.linalg.norm(case["features"]["m_idle"][pred] - case["features"]["m_idle"][opt]))
    neighbor = [int(index) for index in np.argsort(-case_gram[pred]) if int(index) != pred]
    opt_neighbor_rank = int(neighbor.index(opt) + 1)
    case_rows = [
        {
            "cosine_cm": case_cosine,
            "normalized_l2_cm": case_l2,
            "moment_frobenius": moment_gap,
            "idle_l2": idle_gap,
            "idle_fraction_pred": float(case["features"]["idle_fraction"][pred]),
            "idle_fraction_opt": float(case["features"]["idle_fraction"][opt]),
            "opt_neighbor_rank": opt_neighbor_rank,
            "collision": int(case_cosine >= COSINE_LIMIT),
            "cosine_pairmean": float(_geometry(case["features"]["pairmean"])[0][pred, opt]),
            "cosine_edge": float(_geometry(case["features"]["edge"])[0][pred, opt]),
            "cosine_phi": float(_geometry(case["features"]["phi"])[0][pred, opt]),
        }
    ]
    separated_case = bool(case_cosine < COSINE_LIMIT)
    cm_a = bool(
        permutation_ok
        and cm_rate_1 <= COLLISION_1
        and cm_rate_2 <= COLLISION_2
        and nn_cm + NN_MARGIN <= nn_pair
        and spearman_cm > spearman_pair
        and separated_case
    )
    optional_rows = []
    cm_all_passes = False
    if not cm_a and permutation_ok:
        optional_series = {name: {"cosine": [], "gap": []} for name in ("cm_ub", "cm_ub_ue", "cm_ub_be", "cm_all")}
        optional_case = {}
        all_nn = []
        all_spearman = []
        swaps_by_state = defaultdict(list)
        for swap in swaps:
            swaps_by_state[(int(swap["scenario_id"]), int(swap["time_step"]))].append(swap)
        for key, state in cache.items():
            if not state["train"] and key != (CASE_SCENARIO, CASE_TIME):
                continue
            terms = optional_terms(state["h_u"], state["h_b"], state["h_edge"], state["assignments"])
            grams = {name: _geometry(matrix)[:2] for name, matrix in terms.items()}
            if key == (CASE_SCENARIO, CASE_TIME):
                for name, (gram, norms) in grams.items():
                    optional_case[name] = {
                        "cosine": float(gram[pred, opt]),
                        "normalized_l2": _normalized_l2(norms, gram, pred, opt),
                    }
            if not state["train"]:
                continue
            all_matrix = terms["cm_all"]
            all_nn.append(_nn_rates(all_matrix, state["labels"])["contradiction_1"])
            _left, _right, _cosine, distance = _upper_distances(all_matrix)
            incidence = _incidence_distance(state["assignments"], state["h_b"].shape[0])
            all_spearman.append(_corr(incidence, distance))
            pool = {vector: index for index, vector in enumerate(state["vectors"])}
            for swap in swaps_by_state[key]:
                left = pool[swap["assignment"]]
                right = pool[swap["swapped"]]
                gap = abs(float(swap["delta_j15"])) / PAYLOAD_BLOCK_BITS
                for name, (gram, _norms) in grams.items():
                    optional_series[name]["cosine"].append(float(gram[left, right]))
                    optional_series[name]["gap"].append(gap)
        for name, values in optional_series.items():
            for gap in (1, 2, 3, 4):
                measured = _rate(values["cosine"], values["gap"], float(gap))
                measured.update({"method": name, "gap_blocks": gap, "case_cosine": optional_case.get(name, {}).get("cosine", "")})
                optional_rows.append(measured)
        all_rate_1 = float(next(row["cosine_high"] for row in optional_rows if row["method"] == "cm_all" and int(row["gap_blocks"]) == 1))
        all_rate_2 = float(next(row["cosine_high"] for row in optional_rows if row["method"] == "cm_all" and int(row["gap_blocks"]) == 2))
        all_case = float(optional_case["cm_all"]["cosine"])
        cm_all_passes = bool(
            permutation_ok
            and all_rate_1 <= COLLISION_1
            and all_rate_2 <= COLLISION_2
            and _mean(all_nn) + NN_MARGIN <= nn_pair
            and _mean(all_spearman) > spearman_pair
            and all_case < COSINE_LIMIT
        )
        _write_csv(OUTPUT / "phase5zb_optional_cross_terms.csv", optional_rows)
    edge_summary = [
        {
            "method": "edge",
            "collision_1": edge_rate_1,
            "collision_2": _lookup("edge", 2),
            "nn_contradiction_1": nn_edge,
            "assignment_spearman": spearman_edge,
            "j15_spearman": j15_edge,
        },
        {
            "method": "cm",
            "collision_1": cm_rate_1,
            "collision_2": cm_rate_2,
            "nn_contradiction_1": nn_cm,
            "assignment_spearman": spearman_cm,
            "j15_spearman": j15_cm,
        },
        {
            "method": "pairmean",
            "collision_1": pair_rate_1,
            "collision_2": _lookup("pairmean", 2),
            "nn_contradiction_1": nn_pair,
            "assignment_spearman": spearman_pair,
            "j15_spearman": j15_pair,
        },
    ]
    _write_csv(OUTPUT / "phase5zb_selected_edge_vs_cross_moment.csv", edge_summary)
    _write_csv(OUTPUT / "phase5zb_310114_t15.csv", case_rows)
    helped = bool(cm_rate_1 <= 1.0 - HELP_DROP)
    no_reduction = bool(cm_rate_1 >= min(phi_rate_1, pair_rate_1) - NO_REDUCTION_EPS)
    gate_name = choose_gate(
        permutation_ok=permutation_ok,
        cm_a=cm_a,
        cm_all_passes=cm_all_passes,
        helped=helped,
        no_reduction=no_reduction,
    )
    state_rates = []
    grouped = defaultdict(list)
    for row in swap_rows:
        grouped[(int(row["scenario_id"]), int(row["time_step"]))].append(float(row["cosine_cm"]) >= COSINE_LIMIT)
    for (scenario, moment), values in grouped.items():
        state_rates.append({"scenario_id": scenario, "time_step": moment, "cosine_high": float(np.mean(values))})
    boot = cluster_bootstrap(state_rates, "cosine_high", draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    next_train = gate_name == "CM-A" or (gate_name == "CM-B" and cm_all_passes)
    gate = {
        "gate": gate_name,
        "trained": False,
        "planner_ready": False,
        "deployment_verified": False,
        "opened_310118": False,
        "confirmation_used_to_choose_aggregation": False,
        "permutation_ok": permutation_ok,
        **max_error,
        "phi_collision_1": phi_rate_1,
        "pairmean_collision_1": pair_rate_1,
        "edge_collision_1": edge_rate_1,
        "cm_collision_1": cm_rate_1,
        "cm_collision_2": cm_rate_2,
        "cm_collision_3": _lookup("cm", 3),
        "cm_collision_4": _lookup("cm", 4),
        "cm_collision_bootstrap_mean": boot["mean"],
        "cm_collision_bootstrap_low": boot["low"],
        "cm_collision_bootstrap_high": boot["high"],
        "nn_cm": nn_cm,
        "nn_pairmean": nn_pair,
        "nn_edge": nn_edge,
        "spearman_cm": spearman_cm,
        "spearman_pairmean": spearman_pair,
        "spearman_edge": spearman_edge,
        "j15_spearman_cm": j15_cm,
        "j15_spearman_pairmean": j15_pair,
        "j15_spearman_edge": j15_edge,
        "effective_rank_cm": rank_cm,
        "effective_rank_pairmean": rank_pair,
        "case_cosine": case_cosine,
        "case_normalized_l2": case_l2,
        "case_neighbor_rank": opt_neighbor_rank,
        "case_separated": separated_case,
        "cm_all_passes": cm_all_passes,
        "next_phase_may_train_low_rank_pool": next_train,
    }
    _write_json(OUTPUT / "phase5zb_cross_moment_gate.json", gate)
    lines = [
        "# HTW Phase-5ZB Second-Order UAV-Target Correspondence Audit",
        "",
        "**planner_ready = False**",
        "",
        "**deployment_verified = False**",
        "",
        "本阶段没有训练。结构统计只用 310100–310109 的 2834 个已登记 swap。310114 t=15 是预先指定的个案，没有用 310110–310117 来挑选聚合。310118–310125 没有打开。",
        "",
        "## Q1–Q2",
        "",
        "Phase-5ZA 没有通过，是因为 `phi(pair_i)` 之后的 mean 仍然是一阶集合统计。旧代码并不是没绑定：`assignment_latents` 先形成 `(H_UAV_i, H_Buoy_{a_i}, H_edge_{i,a_i})`，再对 UAV 做 mean。buoy 段的和只保留浮标集合。准确名称是 `FIRST_ORDER_SET_AGGREGATION_LOSS`。",
        "",
        f"随机 phi+mean 在这 2834 对上的 cosine ≥ 0.99 比例是 {_fmt(phi_rate_1)}，复现 Phase-5ZA 的 0.3119。一阶控制组 PAIRMEAN 是 {_fmt(pair_rate_1)}，mean selected-edge 是 {_fmt(edge_rate_1)}。",
        "",
        "## Q3–Q5",
        "",
        f"C_UB 的 `|ΔJ15| ≥ 1 block` collision rate 是 {_fmt(cm_rate_1)}。scenario-cluster 95% CI 是 [{_fmt(boot['low'])}, {_fmt(boot['high'])}]。",
        f"≥ 2 / 3 / 4 block 分别是 {_fmt(cm_rate_2)}、{_fmt(_lookup('cm', 3))}、{_fmt(_lookup('cm', 4))}。",
        f"310114 t=15 的 C_UB cosine 是 {_fmt(case_cosine)}，normalized L2 是 {_fmt(case_l2)}，A_opt 在 A_pred 的近邻序里排第 {opt_neighbor_rank}。collision = {int(case_cosine >= COSINE_LIMIT)}。",
        "",
        "## Q6–Q9",
        "",
        f"mean selected-edge 的 collision 是 {_fmt(edge_rate_1)}，assignment-distance Spearman {_fmt(spearman_edge)}，`|ΔJ15|` Spearman {_fmt(j15_edge)}。C_UB 对应是 collision {_fmt(cm_rate_1)}、assignment Spearman {_fmt(spearman_cm)}、J15 Spearman {_fmt(j15_cm)}。PAIRMEAN 的 assignment Spearman 是 {_fmt(spearman_pair)}，J15 Spearman 是 {_fmt(j15_pair)}。",
        f"最近邻 contradiction：C_UB {_fmt(nn_cm)}，PAIRMEAN {_fmt(nn_pair)}，edge {_fmt(nn_edge)}。",
        f"训练集 effective rank 均值：C_UB {_fmt(rank_cm)}，PAIRMEAN {_fmt(rank_pair)}。旧确认集 post-pool 参考是 2.23，pre-pool 参考是 5.08。",
        f"UAV / buoy / 枚举重编号最大误差是 {_fmt(max_error['uav_relabel_error'], '.3e')} / {_fmt(max_error['buoy_relabel_error'], '.3e')} / {_fmt(max_error['enumeration_error'], '.3e')}。",
        "",
        f"CM-ALL 是否通过同一数值门槛：{cm_all_passes}。",
        "",
        "## Q10",
        "",
        f"Gate 是 **{gate_name}**。下一阶段是否可以训练 low-rank second-order pooling：{next_train}。",
        "",
        "`planner_ready = False`",
        "",
        "`deployment_verified = False`",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(_plain({"gate": gate_name, "cm": cm_rate_1, "phi": phi_rate_1})), flush=True)
    return gate


if __name__ == "__main__":
    run()
