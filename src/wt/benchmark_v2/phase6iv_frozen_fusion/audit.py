"""Equal-weight rank fusion of frozen high-level scorers.

No training, no checkpoint writes, and no reserved-test reads.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import PRIMARY_TIMES
from wt.benchmark_v2.phase6is_assignment_head.audit import (
    _load_bundle,
    _load_decoder,
    _load_hgnn,
    _rss_bytes,
    _score_head,
    _score_matching,
    _state_record,
)
from wt.benchmark_v2.phase6is_assignment_head.head import AssignmentConditionedHead
from wt.benchmark_v2.phase6is_assignment_head.protocol import EPOCH6_CKPT
from wt.benchmark_v2.phase6iu_pairwise_head.audit import verify_split
from wt.benchmark_v2.phase6iu_pairwise_head.head import PairwiseInteractionHead
from wt.benchmark_v2.phase6iv_frozen_fusion.protocol import (
    BIT_TIE,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CONCENTRATION,
    EXPECTED_EPOCH,
    HEAD_OUTPUT,
    MIN_POSITIVE_SCENARIOS,
    OUTPUT,
    PAIR_OUTPUT,
    PUBLISHED_TOP1_COUNTS,
    RULES,
    SCHEME_C_CKPT,
    VALIDATION_POOL,
    assert_output_path,
    assign_gate,
    rule_supported,
)
from wt.lowlevel_v2.phase5l_joint_encoder.model import append_queue_edge, batch_tensors


def average_rank(scores: np.ndarray) -> np.ndarray:
    """Higher score gets rank 1. Tied scores share the average rank."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("rank fusion received a non-finite score")
    return rankdata(-values, method="average")


def fuse_ranks(parts: list[np.ndarray], scheme_rank: np.ndarray) -> np.ndarray:
    """Equal-weight mean of rank vectors. Lower is better."""
    stacked = np.vstack([np.asarray(part, dtype=np.float64) for part in parts])
    return 0.5 * np.asarray(scheme_rank, dtype=np.float64) + 0.5 * np.mean(stacked, axis=0)


def fusion_order(fused: np.ndarray, scheme_rank: np.ndarray) -> np.ndarray:
    """Break remaining ties by scheme C rank, then by candidate index."""
    return np.lexsort((np.arange(fused.shape[0]), np.asarray(scheme_rank, dtype=np.float64), np.asarray(fused, dtype=np.float64)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_paths() -> dict[str, Path]:
    paths = {"epoch6": EPOCH6_CKPT, "scheme_c": SCHEME_C_CKPT}
    for seed in (0, 1):
        paths[f"conditioned_{seed}"] = HEAD_OUTPUT / f"seed_{seed}" / "head.pt"
    for seed in range(5):
        paths[f"pair_{seed}"] = PAIR_OUTPUT / f"seed_{seed}" / "head.pt"
    return paths


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _metrics(order: np.ndarray, labels: np.ndarray, ranks: np.ndarray, fused: np.ndarray) -> dict:
    truth = np.asarray(labels, dtype=np.float64)
    groups = np.asarray(ranks, dtype=np.int64)
    optimal = groups == 1
    chosen = int(order[0])
    best = float(np.max(truth))
    payload = float(truth[chosen])
    regret = best - payload
    place = np.empty(order.shape[0], dtype=np.int64)
    place[order] = np.arange(1, order.shape[0] + 1)
    optimal_scores = fused[optimal]
    other_scores = fused[~optimal]
    return {
        "top1": float(bool(optimal[chosen])),
        "top3": float(np.any(optimal[order[: min(3, order.size)]])),
        "top5": float(np.any(optimal[order[: min(5, order.size)]])),
        "payload": payload,
        "oracle_payload": best,
        "regret": float(regret),
        "normalized_regret": float(regret / (best + 1.0)),
        "chosen": float(chosen),
        "exact_rank": float(place[int(np.argmax(truth))]),
        "group_rank": float(np.min(place[optimal])),
        "fused_rank_gap": float(fused[int(order[1])] - fused[int(order[0])]),
        "high_gain_margin": float(np.min(other_scores) - np.min(optimal_scores)),
    }


def _summarize(rows: list[dict]) -> dict:
    payloads = np.asarray([float(row["payload"]) for row in rows], dtype=np.float64)
    return {
        "n_states": int(len(rows)),
        "top1": float(np.mean([row["top1"] for row in rows])),
        "top3": float(np.mean([row["top3"] for row in rows])),
        "top5": float(np.mean([row["top5"] for row in rows])),
        "normalized_regret": float(np.mean([row["normalized_regret"] for row in rows])),
        "payload_mean": float(np.mean(payloads)),
        "payload_median": float(np.median(payloads)),
        "exact_rank_median": float(np.median([row["exact_rank"] for row in rows])),
        "group_rank_median": float(np.median([row["group_rank"] for row in rows])),
        "fused_rank_gap_median": float(np.median([row["fused_rank_gap"] for row in rows])),
        "high_gain_margin_median": float(np.median([row["high_gain_margin"] for row in rows])),
        "top1_hits": int(sum(int(row["top1"]) for row in rows)),
    }


def _slice_summary(rows: list[dict], time_step: int) -> dict:
    packed = _summarize([row for row in rows if int(row["time_step"]) == int(time_step)])
    packed["time_step"] = int(time_step)
    return packed


def _paired_matrix(rows: list[dict], baseline_rows: list[dict]) -> np.ndarray:
    baseline = {(int(row["scenario_id"]), int(row["time_step"])): row for row in baseline_rows}
    matrix = np.zeros((len(VALIDATION_POOL), len(PRIMARY_TIMES)), dtype=np.float64)
    for row_index, scenario in enumerate(VALIDATION_POOL):
        for column, time_step in enumerate(PRIMARY_TIMES):
            current = next(row for row in rows if int(row["scenario_id"]) == int(scenario) and int(row["time_step"]) == int(time_step))
            matrix[row_index, column] = float(current["payload"]) - float(baseline[(int(scenario), int(time_step))]["payload"])
    return matrix


def _interval(matrix: np.ndarray) -> dict:
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws = generator.integers(0, matrix.shape[0], size=(BOOTSTRAP_DRAWS, matrix.shape[0]))
    means = matrix[draws].reshape(BOOTSTRAP_DRAWS, -1).mean(axis=1)
    totals = matrix.sum(axis=1)
    return {
        "mean": float(np.mean(matrix)),
        "mean_low": float(np.percentile(means, 2.5)),
        "mean_high": float(np.percentile(means, 97.5)),
        "median": float(np.median(matrix)),
        "positive_scenarios": int(np.sum(totals > BIT_TIE)),
        "negative_scenarios": int(np.sum(totals < -BIT_TIE)),
    }


def _decision(rows: list[dict], scheme_rows: list[dict]) -> dict:
    summary = _summarize(rows)
    scheme = _summarize(scheme_rows)
    versus = _interval(_paired_matrix(rows, scheme_rows))
    deltas = _paired_matrix(rows, scheme_rows).reshape(-1)
    positive = deltas[deltas > BIT_TIE]
    concentrated = positive.size == 0 or float(np.max(positive)) >= CONCENTRATION * float(np.sum(positive))
    scheme_miss = [row for row, base in zip(rows, scheme_rows) if float(base["top1"]) == 0.0]
    recovered = int(sum(float(row["top1"]) == 1.0 for row in scheme_miss))
    lost = int(sum(float(row["top1"]) == 0.0 and float(base["top1"]) == 1.0 for row, base in zip(rows, scheme_rows)))
    report = {
        "summary": summary,
        "t0": _slice_summary(rows, 0),
        "t15": _slice_summary(rows, 15),
        "t30": _slice_summary(rows, 30),
        "versus_scheme_c": versus,
        "top1_above_scheme_c": bool(summary["top1"] > scheme["top1"]),
        "regret_not_worse": bool(summary["normalized_regret"] <= scheme["normalized_regret"]),
        "mean_ci_above_scheme_c": bool(versus["mean_low"] > 0.0),
        "scenarios_consistent": bool(
            versus["positive_scenarios"] >= MIN_POSITIVE_SCENARIOS and versus["positive_scenarios"] > versus["negative_scenarios"]
        ),
        "gain_not_concentrated": bool(not concentrated),
        "scheme_misses": int(len(scheme_miss)),
        "recovered_misses": recovered,
        "lost_hits": lost,
        "chosen_disagreements": int(sum(int(row["chosen"]) != int(base["chosen"]) for row, base in zip(rows, scheme_rows))),
    }
    report["supported"] = rule_supported(report)
    return report


def _single_order(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")


def _load_conditioned(seed: int, device: torch.device) -> AssignmentConditionedHead:
    head = AssignmentConditionedHead().to(device)
    payload = torch.load(HEAD_OUTPUT / f"seed_{seed}" / "head.pt", map_location="cpu", weights_only=False)
    if int(payload["epoch"]) != int(EXPECTED_EPOCH[f"conditioned_{seed}"]):
        raise RuntimeError(f"conditioned seed {seed} epoch is {payload['epoch']}")
    head.load_state_dict(payload["head"])
    head.eval()
    return head


def _load_pair(seed: int, device: torch.device) -> PairwiseInteractionHead:
    head = PairwiseInteractionHead().to(device)
    payload = torch.load(PAIR_OUTPUT / f"seed_{seed}" / "head.pt", map_location="cpu", weights_only=False)
    if int(payload["epoch"]) != 1 or float(payload["lambda_pair"]) != 1.0:
        raise RuntimeError(f"pair seed {seed} checkpoint is not the fixed epoch-1 head")
    head.load_state_dict(payload["head"])
    head.eval()
    return head


def _score_pair(head: PairwiseInteractionHead, record: dict, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        return head(record["edge_h"].to(device), record["assignments"].to(device), record["unary"].to(device)).detach().cpu().numpy()


def _pair_record(model, decoder, sample: dict, device: torch.device) -> dict:
    graph = append_queue_edge(sample["graph"])
    tensors = {key: value.to(device) for key, value in batch_tensors(graph).items()}
    assignments = torch.as_tensor(sample["assignments"], dtype=torch.long, device=device)
    with torch.no_grad():
        h_u, h_b = model.condition(tensors)
        unary = decoder.score_assignments(h_u, h_b, assignments).detach().cpu()
        edge_h = model.encoder.edge_enc(tensors["edge"])[0].detach().cpu()
    return {"edge_h": edge_h, "assignments": assignments.cpu(), "unary": unary}


def run_audit() -> dict:
    device = torch.device("cpu")
    config = L3BBitQueueEnvV2().config
    paths = _checkpoint_paths()
    before = {name: _sha256(path) for name, path in paths.items()}
    split = verify_split()
    failures = []
    epoch6_model = _load_hgnn(EPOCH6_CKPT, device)
    scheme_model = _load_hgnn(SCHEME_C_CKPT, device)
    decoder = _load_decoder(device)
    conditioned = {seed: _load_conditioned(seed, device) for seed in (0, 1)}
    pairs = {seed: _load_pair(seed, device) for seed in range(5)}
    bundle = []
    model_seconds = {name: [] for name in ("epoch6", "scheme_c", "conditioned_0", "conditioned_1", "pair")}
    for scenario in VALIDATION_POOL:
        for time_step in PRIMARY_TIMES:
            sample, _arrays = _load_bundle(int(scenario), int(time_step))
            labels = np.asarray(sample["j15_bits"], dtype=np.float64)
            assignments = np.asarray(sample["assignments"], dtype=np.int64)
            if int(assignments.shape[0]) != 13327 or labels.shape != (13327,):
                failures.append(f"{scenario} t={time_step} candidate alignment")
                continue
            started = time.perf_counter()
            epoch6_scores = _score_matching(epoch6_model, decoder, sample, device)
            model_seconds["epoch6"].append(time.perf_counter() - started)
            started = time.perf_counter()
            scheme_scores = _score_matching(scheme_model, decoder, sample, device)
            model_seconds["scheme_c"].append(time.perf_counter() - started)
            record = _state_record(int(scenario), int(time_step), epoch6_model, device, config)
            if not np.array_equal(np.asarray(record["sample"]["assignments"]), assignments):
                failures.append(f"{scenario} t={time_step} label order")
            conditioned_scores = {}
            for seed, head in conditioned.items():
                started = time.perf_counter()
                conditioned_scores[seed] = _score_head(head, record, device).detach().cpu().numpy()
                model_seconds[f"conditioned_{seed}"].append(time.perf_counter() - started)
            pair_state = _pair_record(epoch6_model, decoder, sample, device)
            if not np.allclose(pair_state["unary"].numpy(), epoch6_scores, atol=1.0e-5, rtol=0.0):
                failures.append(f"{scenario} t={time_step} unary alignment")
            pair_scores = {}
            for seed, head in pairs.items():
                started = time.perf_counter()
                pair_scores[seed] = _score_pair(head, pair_state, device)
                model_seconds["pair"].append(time.perf_counter() - started)
            bundle.append(
                {
                    "scenario_id": int(scenario),
                    "time_step": int(time_step),
                    "labels": labels,
                    "ranks": np.asarray(sample["utility_rank"], dtype=np.int64),
                    "scores": {"epoch6": epoch6_scores, "scheme_c": scheme_scores, "conditioned": conditioned_scores, "pair": pair_scores},
                }
            )
            print(f"scored {scenario} t={time_step}", flush=True)
    single_rows = {name: [] for name in ("epoch6", "scheme_c", "conditioned_0", "conditioned_1", "pair_0", "pair_1", "pair_2", "pair_3", "pair_4")}
    fusion_rows = {rule: [] for rule in ("A", *RULES)}
    fusion_seconds = []
    spearman = {name: [] for name in ("epoch6", "conditioned_0", "conditioned_1", "pair_0", "pair_1", "pair_2", "pair_3", "pair_4")}
    for item in bundle:
        labels = item["labels"]
        groups = item["ranks"]
        scheme_rank = average_rank(item["scores"]["scheme_c"])
        epoch_rank = average_rank(item["scores"]["epoch6"])
        conditioned_ranks = {seed: average_rank(item["scores"]["conditioned"][seed]) for seed in (0, 1)}
        pair_ranks = {seed: average_rank(item["scores"]["pair"][seed]) for seed in range(5)}
        started = time.perf_counter()
        fused = {
            "A": scheme_rank,
            "B": fuse_ranks([epoch_rank], scheme_rank),
            "C": fuse_ranks([conditioned_ranks[0]], scheme_rank),
            "D": fuse_ranks([conditioned_ranks[1]], scheme_rank),
            "E": fuse_ranks([pair_ranks[seed] for seed in range(5)], scheme_rank),
        }
        fusion_seconds.append(time.perf_counter() - started)
        orders = {
            "epoch6": _single_order(item["scores"]["epoch6"]),
            "scheme_c": _single_order(item["scores"]["scheme_c"]),
            "conditioned_0": _single_order(item["scores"]["conditioned"][0]),
            "conditioned_1": _single_order(item["scores"]["conditioned"][1]),
        }
        for seed in range(5):
            orders[f"pair_{seed}"] = _single_order(item["scores"]["pair"][seed])
        for name, order in orders.items():
            source = item["scores"]["scheme_c"]
            if name == "epoch6":
                source = item["scores"]["epoch6"]
            elif name.startswith("conditioned"):
                source = item["scores"]["conditioned"][int(name[-1])]
            elif name.startswith("pair_"):
                source = item["scores"]["pair"][int(name[-1])]
            row = _metrics(order, labels, groups, -np.asarray(source, dtype=np.float64))
            row["scenario_id"] = item["scenario_id"]
            row["time_step"] = item["time_step"]
            single_rows[name].append(row)
            if name != "scheme_c":
                left = average_rank(item["scores"]["scheme_c"])
                right = average_rank(source)
                spearman[name].append(float(np.corrcoef(left, right)[0, 1]))
        for rule, values in fused.items():
            order = fusion_order(values, scheme_rank) if rule != "A" else orders["scheme_c"]
            row = _metrics(order, labels, groups, values)
            row["scenario_id"] = item["scenario_id"]
            row["time_step"] = item["time_step"]
            fusion_rows[rule].append(row)
    alignment = {name: int(sum(int(row["top1"]) for row in rows)) for name, rows in single_rows.items()}
    for name, expected in PUBLISHED_TOP1_COUNTS.items():
        if alignment.get(name) != int(expected):
            failures.append(f"{name} top1 {alignment.get(name)} != {expected}")
    after = {name: _sha256(path) for name, path in paths.items()}
    blocked = bool(failures) or before != after or split != {"train": 48, "validation": 16}
    scheme_rows = fusion_rows["A"]
    rules = []
    if not blocked:
        for rule in RULES:
            report = _decision(fusion_rows[rule], scheme_rows)
            report["rule"] = rule
            rules.append(report)
            print(f"rule {rule} top1={report['summary']['top1_hits']} supported={report['supported']}", flush=True)
    gate = assign_gate(blocked, not failures and before == after, rules)
    result = {
        "gate": gate,
        "blocked": int(blocked),
        "failures": failures,
        "split": split,
        "hashes_unchanged": before == after,
        "checkpoint_hashes": before,
        "alignment_top1_hits": alignment,
        "reserved_scenarios_read": 0,
        "tie_rule": "average rank for equal scores; fused ties break by scheme C rank, then candidate index",
        "rules": {
            "A": "scheme C rank",
            "B": "equal weight of epoch6 rank and scheme C rank",
            "C": "equal weight of scheme C rank and conditioned seed 0 rank",
            "D": "equal weight of scheme C rank and conditioned seed 1 rank",
            "E": "equal weight of scheme C rank and the mean rank of pair seeds 0-4",
        },
        "timing_seconds_median": {name: float(np.median(values)) for name, values in model_seconds.items() if values},
        "fusion_seconds_median": float(np.median(fusion_seconds)) if fusion_seconds else None,
        "rss_bytes": _rss_bytes(),
        "rank_correlation_with_scheme_c_median": {name: float(np.median(values)) for name, values in spearman.items() if values},
        "single_model": {
            name: {"summary": _summarize(rows), "t0": _slice_summary(rows, 0), "t15": _slice_summary(rows, 15), "t30": _slice_summary(rows, 30)}
            for name, rows in single_rows.items()
        },
        "fusion": {
            rule: {"summary": _summarize(rows), "t0": _slice_summary(rows, 0), "t15": _slice_summary(rows, 15), "t30": _slice_summary(rows, 30), "rows": rows}
            for rule, rows in fusion_rows.items()
        },
        "rule_reports": rules,
    }
    path = assert_output_path(OUTPUT / "summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
    print(json.dumps({"gate": gate, "failures": failures, "alignment": alignment}, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    run_audit()
