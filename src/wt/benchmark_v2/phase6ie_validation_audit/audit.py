"""Validation-set attribution for the frozen encoder and epoch 6.

This reads checkpoints and the validation revision. It does not train or write labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import EPS_BITS
from wt.benchmark_v2.phase6ic_hgnn_smoke.dataset import DecisionDataset
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import GRAPH_KEYS, RESERVED_TEST
from wt.benchmark_v2.phase6id_joint_training.metrics import state_metrics
from wt.benchmark_v2.phase6id_joint_training.protocol import ENCODER_CKPT, OUTPUT, selection_key
from wt.benchmark_v2.phase6id_joint_training.train import _build, _finite_tensor, _forward, _sha256


ROOT = Path(__file__).resolve().parents[4]
AUDIT_DIR = ROOT / "results/phase6ie_validation_audit"
EPOCH6 = OUTPUT / "epoch_006.pt"


def _assignment(rows: np.ndarray, index: int) -> str:
    return ",".join(str(int(item)) for item in np.asarray(rows[int(index)], dtype=np.int64).reshape(-1))


def _score_view(scores: np.ndarray, labels: np.ndarray, ranks: np.ndarray) -> dict:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels, dtype=np.float64).reshape(-1)
    groups = np.asarray(ranks, dtype=np.int64).reshape(-1)
    order = np.argsort(-values, kind="stable")
    chosen = int(order[0])
    oracle = int(np.argmax(truth))
    rank_of = np.empty(values.size, dtype=np.int64)
    rank_of[order] = np.arange(1, values.size + 1)
    true_rank = int(1 + np.sum(truth > truth[chosen]))
    best = truth[groups == 1]
    strict_inside_best = 0
    ordered_best = np.sort(best)
    for value in ordered_best:
        strict_inside_best += int(np.searchsorted(ordered_best, value, side="left"))
    ordered = np.sort(truth)
    strict_pairs = 0
    for value in ordered:
        strict_pairs += int(np.searchsorted(ordered, value, side="left"))
    equal_pairs = int(truth.size * (truth.size - 1) / 2 - strict_pairs)
    metric = state_metrics(values, truth, groups)
    return {
        "metric": metric,
        "chosen": chosen,
        "oracle_index": oracle,
        "oracle_assignment_ready": oracle,
        "predicted_rank_of_oracle": int(rank_of[oracle]),
        "predicted_rank_of_best_group": int(np.min(rank_of[groups == 1])),
        "true_rank_of_chosen": true_rank,
        "rank_gap": int(true_rank - 1),
        "score_margin": float(values[order[0]] - values[order[1]]),
        "score_tie_count": int(np.sum(values == values[order[0]])),
        "score_min": float(np.min(values)),
        "score_max": float(np.max(values)),
        "score_std": float(np.std(values)),
        "best_group_size": int(np.sum(groups == 1)),
        "exact_max_count": int(np.sum(truth == truth[oracle])),
        "strict_pairs": int(strict_pairs),
        "equal_pairs": int(equal_pairs),
        "strict_pairs_inside_best_group": int(strict_inside_best),
        "top1_matches_regret_rule": int(bool(metric["top1"]) == bool(float(metric["regret"]) <= float(EPS_BITS))),
    }


def _stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _group(rows: list[dict], mask: np.ndarray) -> dict:
    chosen = [rows[index] for index in np.flatnonzero(mask)]
    return _stats([float(row["payload_delta"]) for row in chosen])


def aggregate(rows: list[dict]) -> dict:
    delta = np.asarray([float(row["payload_delta"]) for row in rows], dtype=np.float64)
    frozen_top = np.asarray([float(row["frozen_top1"]) for row in rows], dtype=np.float64)
    trained_top = np.asarray([float(row["epoch6_top1"]) for row in rows], dtype=np.float64)
    positive = delta[delta > 0.0]
    positive_share = float("nan")
    if positive.size and float(np.sum(np.abs(delta))) > 0.0:
        order = np.argsort(-positive)
        positive_share = float(np.sum(positive[order[: min(5, positive.size)]]) / np.sum(positive))
    return {
        "n_states": int(len(rows)),
        "improved": _group(rows, delta > 0.0),
        "tied": _group(rows, delta == 0.0),
        "declined": _group(rows, delta < 0.0),
        "frozen_hit_epoch6_miss": _group(rows, (frozen_top == 1.0) & (trained_top == 0.0)),
        "frozen_miss_epoch6_hit": _group(rows, (frozen_top == 0.0) & (trained_top == 1.0)),
        "same_top1_payload_up": _group(rows, (frozen_top == trained_top) & (delta > 0.0)),
        "same_top1_payload_down": _group(rows, (frozen_top == trained_top) & (delta < 0.0)),
        "same_top1_payload_tied": _group(rows, (frozen_top == trained_top) & (delta == 0.0)),
        "delta": _stats(delta.tolist()),
        "largest5_positive_share": positive_share,
        "median_delta": float(np.median(delta)),
    }


def _reproduce(saved: list[dict], fresh: list[dict], name: str) -> None:
    if len(saved) != len(fresh):
        raise RuntimeError(f"{name} row count drifted")
    for left, right in zip(saved, fresh):
        if int(left["scenario_id"]) != int(right["scenario_id"]) or int(left["time_step"]) != int(right["time_step"]):
            raise RuntimeError(f"{name} state order drifted")
        if int(left["chosen"]) != int(right["chosen"]):
            raise RuntimeError(f"{name} chosen candidate drifted at {int(right['scenario_id'])} t={int(right['time_step'])}")
        if abs(float(left["payload"]) - float(right["payload"])) > 1.0e-4:
            raise RuntimeError(f"{name} payload drifted")
        if int(left["top1"]) != int(right["top1"]):
            raise RuntimeError(f"{name} top1 drifted")


def _selection_audit() -> dict:
    history = json.loads((OUTPUT / "history.json").read_text(encoding="utf-8"))["history"]
    keys = [selection_key(row) for row in history]
    best = int(np.argmax([key[0] for key in keys]))
    # Lexicographic key, not Spearman alone.
    winner = max(range(len(history)), key=lambda index: keys[index])
    return {
        "epochs": [int(row["epoch"]) for row in history],
        "selection_winner_epoch": int(history[winner]["epoch"]),
        "spearman_argmax_epoch": int(history[best]["epoch"]),
        "keys": [list(key) for key in keys],
    }


def run_audit() -> dict:
    if any(int(scenario) in RESERVED_TEST for scenario, _time in DecisionDataset("validation").index):
        raise RuntimeError("validation index contains a reserved scenario")
    device = torch.device("cpu")
    frozen, decoder, _optimizer = _build(device)
    epoch_payload = torch.load(EPOCH6, map_location="cpu", weights_only=False)
    if int(epoch_payload["epoch"]) != 6:
        raise RuntimeError("epoch 6 checkpoint header is not 6")
    if any("norm" in key.lower() for key in epoch_payload["model"]):
        raise RuntimeError("epoch 6 checkpoint contains a normalization parameter")
    trained, _decoder_again, _optimizer_again = _build(device)
    trained.load_state_dict(epoch_payload["model"])
    trained.to(device)
    trained.eval()
    frozen.eval()
    rows = []
    with torch.no_grad():
        for sample in DecisionDataset("validation"):
            if tuple(sample["graph"]) != GRAPH_KEYS:
                raise RuntimeError("graph keys drifted")
            frozen_scores, _frozen_loss = _forward(frozen, decoder, sample, device)
            trained_scores, _trained_loss = _forward(trained, decoder, sample, device)
            if not _finite_tensor(frozen_scores) or not _finite_tensor(trained_scores):
                raise RuntimeError("non-finite audit score")
            labels = np.asarray(sample["j15_bits"], dtype=np.float64)
            ranks = np.asarray(sample["utility_rank"], dtype=np.int64)
            assignments = np.asarray(sample["assignments"], dtype=np.int64)
            left = _score_view(frozen_scores.detach().cpu().numpy(), labels, ranks)
            right = _score_view(trained_scores.detach().cpu().numpy(), labels, ranks)
            oracle = int(left["oracle_index"])
            if int(right["oracle_index"]) != oracle:
                raise RuntimeError("oracle index depends on the model")
            row = {
                "scenario_id": int(sample["scenario_id"]),
                "time_step": int(sample["time_step"]),
                "candidate_count": int(labels.shape[0]),
                "oracle_assignment": _assignment(assignments, oracle),
                "oracle_payload": float(labels[oracle]),
                "frozen_assignment": _assignment(assignments, left["chosen"]),
                "epoch6_assignment": _assignment(assignments, right["chosen"]),
                "frozen_payload": float(left["metric"]["payload"]),
                "epoch6_payload": float(right["metric"]["payload"]),
                "payload_delta": float(right["metric"]["payload"] - left["metric"]["payload"]),
                "frozen_regret": float(left["metric"]["normalized_regret"]),
                "epoch6_regret": float(right["metric"]["normalized_regret"]),
                "frozen_top1": float(left["metric"]["top1"]),
                "epoch6_top1": float(right["metric"]["top1"]),
                "frozen_top3": float(left["metric"]["top3"]),
                "epoch6_top3": float(right["metric"]["top3"]),
                "frozen_top5": float(left["metric"]["top5"]),
                "epoch6_top5": float(right["metric"]["top5"]),
                "frozen_rank_gap": int(left["rank_gap"]),
                "epoch6_rank_gap": int(right["rank_gap"]),
                "frozen_oracle_predicted_rank": int(left["predicted_rank_of_oracle"]),
                "epoch6_oracle_predicted_rank": int(right["predicted_rank_of_oracle"]),
                "frozen_best_group_predicted_rank": int(left["predicted_rank_of_best_group"]),
                "epoch6_best_group_predicted_rank": int(right["predicted_rank_of_best_group"]),
                "frozen_margin": float(left["score_margin"]),
                "epoch6_margin": float(right["score_margin"]),
                "frozen_score_tie_count": int(left["score_tie_count"]),
                "epoch6_score_tie_count": int(right["score_tie_count"]),
                "frozen_score_std": float(left["score_std"]),
                "epoch6_score_std": float(right["score_std"]),
                "frozen_score_min": float(left["score_min"]),
                "frozen_score_max": float(left["score_max"]),
                "epoch6_score_min": float(right["score_min"]),
                "epoch6_score_max": float(right["score_max"]),
                "best_group_size": int(left["best_group_size"]),
                "exact_max_count": int(left["exact_max_count"]),
                "equal_pairs": int(left["equal_pairs"]),
                "strict_pairs": int(left["strict_pairs"]),
                "strict_pairs_inside_best_group": int(left["strict_pairs_inside_best_group"]),
                "frozen_rule_match": int(left["top1_matches_regret_rule"]),
                "epoch6_rule_match": int(right["top1_matches_regret_rule"]),
                "chosen_frozen": int(left["chosen"]),
                "chosen_epoch6": int(right["chosen"]),
            }
            rows.append(row)
            del frozen_scores, trained_scores
            print(f"audit {row['scenario_id']} t={row['time_step']} delta={row['payload_delta']:.3f}", flush=True)
    saved_frozen = json.loads((OUTPUT / "baseline_states.json").read_text(encoding="utf-8"))
    saved_best = json.loads((OUTPUT / "best_states.json").read_text(encoding="utf-8"))
    _reproduce(
        saved_frozen,
        [{"scenario_id": row["scenario_id"], "time_step": row["time_step"], "chosen": row["chosen_frozen"], "payload": row["frozen_payload"], "top1": row["frozen_top1"]} for row in rows],
        "frozen",
    )
    _reproduce(
        saved_best,
        [{"scenario_id": row["scenario_id"], "time_step": row["time_step"], "chosen": row["chosen_epoch6"], "payload": row["epoch6_payload"], "top1": row["epoch6_top1"]} for row in rows],
        "epoch6",
    )
    summary = aggregate(rows)
    summary["selection"] = _selection_audit()
    summary["encoder_sha256"] = _sha256(ENCODER_CKPT)
    summary["epoch6_sha_matches_loaded_header"] = True
    summary["normalization"] = "no checkpoint normalization; both models use hgnn_tensors"
    summary["score_tie_states_frozen"] = int(sum(row["frozen_score_tie_count"] > 1 for row in rows))
    summary["score_tie_states_epoch6"] = int(sum(row["epoch6_score_tie_count"] > 1 for row in rows))
    summary["rule_mismatches"] = int(sum((row["frozen_rule_match"] == 0) or (row["epoch6_rule_match"] == 0) for row in rows))
    summary["states_with_exact_j15_ties_at_max"] = int(sum(row["exact_max_count"] > 1 for row in rows))
    summary["states_with_best_group_larger_than_exact_max"] = int(sum(row["best_group_size"] > row["exact_max_count"] for row in rows))
    summary["states_with_strict_pairs_inside_best_group"] = int(sum(row["strict_pairs_inside_best_group"] > 0 for row in rows))
    for name, key in (
        ("frozen_margin", "frozen_margin"),
        ("epoch6_margin", "epoch6_margin"),
        ("frozen_score_std", "frozen_score_std"),
        ("epoch6_score_std", "epoch6_score_std"),
        ("frozen_oracle_rank", "frozen_oracle_predicted_rank"),
        ("epoch6_oracle_rank", "epoch6_oracle_predicted_rank"),
        ("frozen_rank_gap", "frozen_rank_gap"),
        ("epoch6_rank_gap", "epoch6_rank_gap"),
        ("best_group_size", "best_group_size"),
    ):
        summary[name] = _stats([float(row[key]) for row in rows])
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / "states.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (AUDIT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps({"wrote": str(AUDIT_DIR)}), flush=True)
    run_audit()
