"""Offline check of whether assignment scores can express swap interactions.

No training, no environment rollout, and no writes into existing checkpoints.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import PRIMARY_TIMES, TRAIN_POOL, VALIDATION_POOL
from wt.benchmark_v2.phase6im_local_swap.audit import single_swaps
from wt.benchmark_v2.phase6is_assignment_head.audit import (
    _checkpoint_hashes,
    _encode,
    _load_bundle,
    _load_decoder,
    _load_hgnn,
    _score_matching,
)
from wt.benchmark_v2.phase6is_assignment_head.features import assignment_raw_features
from wt.benchmark_v2.phase6is_assignment_head.head import AssignmentConditionedHead
from wt.benchmark_v2.phase6is_assignment_head.protocol import EPOCH6_CKPT, OUTPUT as HEAD_OUTPUT, SCHEME_C_CKPT
from wt.benchmark_v2.phase6it_interaction.protocol import (
    AGREEMENT_BAR,
    DRAWS,
    MATERIAL_BITS,
    OUTPUT,
    SCORE_DUST,
    SEED,
    assert_output_path,
    assign_gate,
)
def _same_buoy_set(assignments: np.ndarray, labels: np.ndarray, scores: dict[str, np.ndarray], base_index: int) -> dict:
    """Spread of labels and scores among matchings that serve the same buoys."""
    rows = np.asarray(assignments, dtype=np.int64)
    base = rows[int(base_index)]
    target = tuple(sorted(int(item) for item in base if int(item) >= 0))
    chosen = [
        index
        for index, row in enumerate(rows)
        if tuple(sorted(int(item) for item in row if int(item) >= 0)) == target
    ]
    payload = np.asarray(labels, dtype=np.float64)[chosen]
    return {
        "count": int(len(chosen)),
        "j15_range": float(np.ptp(payload)) if len(chosen) else None,
        "score_range": {name: float(np.ptp(np.asarray(values, dtype=np.float64)[chosen])) for name, values in scores.items()},
    }

ROOT = Path(__file__).resolve().parents[4]
SAVED_SCHEME = ROOT / "results/phase6im_local_swap" / "states.json"
MODEL_NAMES = ("epoch6", "scheme_c", "seed0", "seed1")


def pair_scores(matrix: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    """Sum selected pair logits. Idle contributes nothing."""
    rows = np.asarray(assignments, dtype=np.int64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    valid = rows >= 0
    safe = np.where(valid, rows, 0)
    uav = np.arange(rows.shape[1], dtype=np.int64)[None, :]
    return np.where(valid, matrix[uav, safe], 0.0).sum(axis=1)


def decompositions(base: np.ndarray) -> list[dict]:
    """Unique two-swap assignments and the single swaps used to build them."""
    origin = np.asarray(base, dtype=np.int64).reshape(-1)
    singles = single_swaps(origin)
    seen = {tuple(int(item) for item in origin)}
    for _left, _right, row in singles:
        seen.add(tuple(int(item) for item in row))
    found = []
    for left, right, row in singles:
        for other_left, other_right, child in single_swaps(row):
            key = tuple(int(item) for item in child)
            if key in seen:
                continue
            seen.add(key)
            indexes = {int(left), int(right), int(other_left), int(other_right)}
            kind = "disjoint" if len(indexes) == 4 else "cycle" if len(indexes) == 3 else "other"
            second = origin.copy()
            second[int(other_left)], second[int(other_right)] = second[int(other_right)], second[int(other_left)]
            found.append({"kind": kind, "joint": np.asarray(child, dtype=np.int64), "first": np.asarray(row, dtype=np.int64), "second": second})
    return found


def context_groups(assignments: np.ndarray) -> list[list[tuple[int, int]]]:
    """Assignment pairs that exchange one buoy pair and keep the other UAVs fixed."""
    rows = np.asarray(assignments, dtype=np.int64)
    groups = []
    width = int(rows.shape[1])
    for left in range(width):
        for right in range(left + 1, width):
            others = [index for index in range(width) if index not in {left, right}]
            buckets: dict[tuple[int, int], dict[tuple[int, ...], int]] = {}
            for index, row in enumerate(rows):
                if int(row[left]) == int(row[right]):
                    continue
                key = (int(row[left]), int(row[right]))
                context = tuple(int(row[item]) for item in others)
                buckets.setdefault(key, {})[context] = index
            for first, second in buckets:
                if first >= second:
                    continue
                opposite = buckets.get((second, first))
                if not opposite:
                    continue
                paired = []
                for context, index in buckets[(first, second)].items():
                    other = opposite.get(context)
                    if other is not None:
                        paired.append((index, other))
                if len(paired) >= 2:
                    groups.append(paired)
    return groups


def context_ranges(assignments: np.ndarray, values: np.ndarray) -> list[float]:
    """Range of a same-buoy-pair exchange across different backgrounds."""
    payload = np.asarray(values, dtype=np.float64)
    ranges = []
    for group in context_groups(assignments):
        deltas = np.asarray([payload[other] - payload[index] for index, other in group], dtype=np.float64)
        ranges.append(float(np.ptp(deltas)))
    return ranges


def _index(assignments: np.ndarray) -> dict[tuple[int, ...], int]:
    return {tuple(int(item) for item in row): index for index, row in enumerate(np.asarray(assignments, dtype=np.int64))}


def _delta(table: dict[tuple[int, ...], int], values: np.ndarray, row: np.ndarray, base_index: int) -> float | None:
    index = table.get(tuple(int(item) for item in np.asarray(row, dtype=np.int64)))
    if index is None:
        return None
    return float(values[index] - values[base_index])


def _range_of(values: np.ndarray, group: list[tuple[int, int]]) -> float:
    deltas = np.asarray([float(values[other] - values[index]) for index, other in group], dtype=np.float64)
    return float(np.ptp(deltas))


def _median_abs(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.abs(np.asarray(values, dtype=np.float64))))


def _max_abs(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.max(np.abs(np.asarray(values, dtype=np.float64))))


def _interval(samples: list[float]) -> dict:
    array = np.asarray([value for value in samples if value is not None], dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": None, "low": None, "high": None}
    generator = np.random.default_rng(SEED)
    draws = generator.integers(0, array.size, size=(DRAWS, array.size))
    means = array[draws].mean(axis=1)
    return {"n": int(array.size), "mean": float(np.mean(array)), "low": float(np.percentile(means, 2.5)), "high": float(np.percentile(means, 97.5))}


def _load_head(seed: int, device: torch.device) -> AssignmentConditionedHead:
    head = AssignmentConditionedHead().to(device)
    payload = torch.load(HEAD_OUTPUT / f"seed_{seed}" / "head.pt", map_location="cpu", weights_only=False)
    head.load_state_dict(payload["head"])
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head


def _head_scores(model, head, sample: dict, arrays: dict, device: torch.device, config) -> np.ndarray:
    h_u, h_b, edge_h = _encode(model, sample, device)
    raw = torch.as_tensor(assignment_raw_features(arrays, sample["assignments"], config), dtype=torch.float32)
    record = {
        "h_u": h_u,
        "h_b": h_b,
        "edge_h": edge_h,
        "raw": raw,
        "assignments": torch.as_tensor(sample["assignments"], dtype=torch.long),
    }
    with torch.no_grad():
        return head(
            record["h_u"].to(device),
            record["h_b"].to(device),
            record["edge_h"].to(device),
            record["raw"].to(device),
            record["assignments"].to(device),
        ).detach().cpu().numpy()


def analyze_state(sample: dict, scores: dict[str, np.ndarray]) -> dict:
    labels = np.asarray(sample["j15_bits"], dtype=np.float64)
    assignments = np.asarray(sample["assignments"], dtype=np.int64)
    table = _index(assignments)
    base_index = int(np.argsort(-scores["scheme_c"], kind="stable")[0])
    base = assignments[base_index]
    missing = 0
    j15_values = {"disjoint": [], "cycle": []}
    score_values = {name: {"disjoint": [], "cycle": []} for name in scores}
    examples = {"disjoint": None, "cycle": None}
    for item in decompositions(base):
        joint = _delta(table, labels, item["joint"], base_index)
        first = _delta(table, labels, item["first"], base_index)
        second = _delta(table, labels, item["second"], base_index)
        if joint is None or first is None or second is None:
            missing += 1
            continue
        interaction = float(joint - first - second)
        j15_values[item["kind"]].append(interaction)
        score_pieces = {}
        for name, values in scores.items():
            pieces = [_delta(table, values, item[key], base_index) for key in ("joint", "first", "second")]
            residual = float(pieces[0] - pieces[1] - pieces[2])
            score_values[name][item["kind"]].append(residual)
            score_pieces[name] = {
                "joint": float(pieces[0]),
                "first": float(pieces[1]),
                "second": float(pieces[2]),
                "interaction": residual,
            }
        current = examples[item["kind"]]
        if current is None or abs(interaction) > abs(float(current["j15_interaction"])):
            examples[item["kind"]] = {
                "j15_joint": float(joint),
                "j15_first": float(first),
                "j15_second": float(second),
                "j15_interaction": interaction,
                "scores": score_pieces,
            }
    groups = context_groups(assignments)
    j15_context = [_range_of(labels, group) for group in groups]
    score_context = {name: [_range_of(values, group) for group in groups] for name, values in scores.items()}
    signs = {}
    for name in scores:
        paired = [
            (j15_value, score_value)
            for j15_value, score_value in zip(j15_values["disjoint"], score_values[name]["disjoint"])
            if abs(j15_value) > MATERIAL_BITS and abs(score_value) > SCORE_DUST
        ]
        signs[name] = None if not paired else float(np.mean([np.sign(left) == np.sign(right) for left, right in paired]))
    gaps = {}
    for name, values in scores.items():
        order = np.argsort(-values, kind="stable")
        gaps[name] = float(values[order[0]] - values[order[1]])
    return {
        "scenario_id": int(sample["scenario_id"]),
        "time_step": int(sample["time_step"]),
        "base": [int(item) for item in base],
        "missing": int(missing),
        "disjoint_count": int(len(j15_values["disjoint"])),
        "cycle_count": int(len(j15_values["cycle"])),
        "j15_disjoint_abs_median": _median_abs(j15_values["disjoint"]),
        "j15_cycle_abs_median": _median_abs(j15_values["cycle"]),
        "material_disjoint": int(sum(abs(value) > MATERIAL_BITS for value in j15_values["disjoint"])),
        "tie_disjoint": int(sum(abs(value) <= 1.0 for value in j15_values["disjoint"])),
        "score_disjoint_abs_max": {name: _max_abs(score_values[name]["disjoint"]) for name in scores},
        "score_cycle_abs_median": {name: _median_abs(score_values[name]["cycle"]) for name in scores},
        "context_groups": int(len(j15_context)),
        "context_j15_over_material": int(sum(value > MATERIAL_BITS for value in j15_context)),
        "context_score_range_max": {name: _max_abs(score_context[name]) for name in scores},
        "sign_agreement": signs,
        "top_gap": gaps,
        "same_buoy_set": _same_buoy_set(assignments, labels, scores, base_index),
        "largest_abs_interaction": examples,
    }


def _saved_scheme_matches(rows: list[dict]) -> int:
    saved = {int(row["scenario_id"]): [int(item) for item in row["assignments"]["scheme_c"]] for row in json.loads(SAVED_SCHEME.read_text(encoding="utf-8"))}
    return int(sum(row["base"] == saved[int(row["scenario_id"])] for row in rows if int(row["time_step"]) == 15 and int(row["scenario_id"]) in saved))


def _population(rows: list[dict], split: str, time_step: int) -> dict:
    chosen = [row for row in rows if row["split"] == split and int(row["time_step"]) == int(time_step)]
    return {
        "states": int(len(chosen)),
        "missing": int(sum(row["missing"] for row in chosen)),
        "disjoint": int(sum(row["disjoint_count"] for row in chosen)),
        "cycles": int(sum(row["cycle_count"] for row in chosen)),
        "material_disjoint": int(sum(row["material_disjoint"] for row in chosen)),
        "tie_disjoint": int(sum(row["tie_disjoint"] for row in chosen)),
        "j15_disjoint_abs_median": _median_abs([row["j15_disjoint_abs_median"] for row in chosen if row["j15_disjoint_abs_median"] is not None]),
        "score_disjoint_abs_max": {name: _max_abs([row["score_disjoint_abs_max"][name] for row in chosen if row["score_disjoint_abs_max"][name] is not None]) for name in MODEL_NAMES},
        "context_states": int(
            sum(
                row["context_j15_over_material"] > 0
                and float(row["context_score_range_max"]["epoch6"] or 0.0) < SCORE_DUST
                and float(row["context_score_range_max"]["scheme_c"] or 0.0) < SCORE_DUST
                for row in chosen
            )
        ),
        "seed0_sign": _interval([row["sign_agreement"]["seed0"] for row in chosen]),
        "seed1_sign": _interval([row["sign_agreement"]["seed1"] for row in chosen]),
        "top_gap_median": {name: _median_abs([row["top_gap"][name] for row in chosen]) for name in MODEL_NAMES},
    }


def run_audit() -> dict:
    device = torch.device("cpu")
    config = L3BBitQueueEnvV2().config
    before = _checkpoint_hashes()
    decoder = _load_decoder(device)
    epoch6 = _load_hgnn(EPOCH6_CKPT, device)
    scheme = _load_hgnn(SCHEME_C_CKPT, device)
    heads = {0: _load_head(0, device), 1: _load_head(1, device)}
    rows = []
    for split, pool in (("train", TRAIN_POOL), ("validation", VALIDATION_POOL)):
        for scenario in pool:
            for time_step in PRIMARY_TIMES:
                sample, arrays = _load_bundle(int(scenario), int(time_step))
                scores = {
                    "epoch6": _score_matching(epoch6, decoder, sample, device),
                    "scheme_c": _score_matching(scheme, decoder, sample, device),
                    "seed0": _head_scores(epoch6, heads[0], sample, arrays, device, config),
                    "seed1": _head_scores(epoch6, heads[1], sample, arrays, device, config),
                }
                row = analyze_state(sample, scores)
                row["split"] = split
                rows.append(row)
                print(f"{split} {scenario} t={time_step} missing={row['missing']} material={row['material_disjoint']}", flush=True)
    after = _checkpoint_hashes()
    validation_t15 = [row for row in rows if row["split"] == "validation" and int(row["time_step"]) == 15]
    populations = {
        f"{split}_t{time_step:03d}": _population(rows, split, time_step)
        for split in ("train", "validation")
        for time_step in PRIMARY_TIMES
    }
    focus = populations["validation_t015"]
    additive_flat = all(float(focus["score_disjoint_abs_max"][name]) < SCORE_DUST for name in ("epoch6", "scheme_c"))
    sign_tracks = bool(focus["seed0_sign"]["low"] is not None and float(focus["seed0_sign"]["low"]) > AGREEMENT_BAR)
    blocked = bool(before != after or int(focus["missing"]) > 0 or _saved_scheme_matches(validation_t15) != 16)
    gate = assign_gate(
        blocked=blocked,
        material_swaps=int(focus["material_disjoint"]),
        additive_flat=additive_flat,
        context_states=int(focus["context_states"]),
        sign_tracks=sign_tracks,
    )
    result = {
        "gate": gate,
        "blocked": int(blocked),
        "hashes_unchanged": before == after,
        "saved_scheme_matches": _saved_scheme_matches(validation_t15),
        "populations": populations,
        "rows": rows,
        "reserved_scenarios_read": 0,
    }
    path = assert_output_path(OUTPUT / "summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate": gate, "focus": focus}, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    run_audit()
