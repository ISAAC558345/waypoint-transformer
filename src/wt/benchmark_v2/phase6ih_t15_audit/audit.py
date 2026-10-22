"""Offline t=15 observability audit. No training, rollout, or label edits."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from wt.benchmark_v2.phase6e.generators import is_legal
from wt.benchmark_v2.phase6ib_highlevel_dataset.hgnn_revision import FROZEN_DATASET, hgnn_tensors
from wt.benchmark_v2.phase6ib_highlevel_dataset.protocol import EPS_BITS
from wt.benchmark_v2.phase6ic_hgnn_smoke.dataset import load_decision
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import (
    RESERVED_TEST,
    REVISION_DATASET,
    TRAIN_POOL,
    VALIDATION_POOL,
)
from wt.benchmark_v2.phase6id_joint_training.protocol import ENCODER_CKPT
from wt.benchmark_v2.phase6id_joint_training.train import _build, _finite_tensor, _sha256
from wt.benchmark_v2.phase6if_ranking_loss.losses import LISTNET_BETA, top_regret_loss
from wt.benchmark_v2.phase6if_ranking_loss.train import EPOCH6, _scores
from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.lowlevel_v2.phase5l_joint_encoder.model import append_queue_edge, batch_tensors


ROOT = Path(__file__).resolve().parents[4]
AUDIT_DIR = ROOT / "results/phase6ih_t15_audit"
SCHEME_C = ROOT / "results/phase6if_ranking_loss" / "runs" / "c" / "epoch_001.pt"
EDGE_NAMES = ("distance_over_diag", "clipped_rate", "eta_over_horizon", "queue_ratio")
GAP_EDGES = (0.0, 1.0, 1.0e4, 1.0e6, 1.0e7, 1.0e12)


def lexicographic_rows(assignment: np.ndarray) -> bool:
    rows = np.asarray(assignment, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[0] < 2:
        return False
    previous = tuple(int(item) for item in rows[0])
    for row in rows[1:]:
        current = tuple(int(item) for item in row)
        if current <= previous:
            return False
        previous = current
    return True


def utility_rank_matches(labels: np.ndarray, ranks: np.ndarray, eps: float = EPS_BITS) -> bool:
    truth = np.asarray(labels, dtype=np.float64)
    stored = np.asarray(ranks, dtype=np.int64)
    expected = np.ones(truth.shape[0], dtype=np.int64)
    order = np.argsort(-truth, kind="mergesort")
    group = 1
    best = float(truth[order[0]])
    for index in order:
        if float(truth[index]) < best - float(eps):
            group += 1
            best = float(truth[index])
        expected[int(index)] = group
    return bool(np.array_equal(stored, expected))


def alignment_report(sample: dict, manifest: dict, frozen_candidates: dict[str, np.ndarray]) -> dict:
    labels = np.asarray(sample["j15_bits"], dtype=np.float64)
    ranks = np.asarray(sample["utility_rank"], dtype=np.int64)
    assignment = np.asarray(sample["assignments"], dtype=np.int64)
    candidate_id = np.asarray(frozen_candidates["candidate_id"], dtype=np.int64)
    failures = []
    if int(candidate_id.shape[0]) != int(labels.shape[0]) or not np.array_equal(candidate_id, np.arange(labels.shape[0])):
        failures.append("candidate_id")
    if assignment.shape != (labels.shape[0], 6):
        failures.append("assignment_shape")
    if not np.array_equal(assignment, np.asarray(frozen_candidates["assignment"], dtype=np.int64)):
        failures.append("assignment_copy")
    if not np.array_equal(labels, np.asarray(frozen_candidates["j15_bits"], dtype=np.float64)):
        failures.append("j15_copy")
    if not lexicographic_rows(assignment):
        failures.append("lexicographic_order")
    if any(not is_legal(row, 6, 12) for row in assignment):
        failures.append("legality")
    if not utility_rank_matches(labels, ranks):
        failures.append("utility_rank")
    manifest_best = tuple(int(item) for item in str(manifest["argmax_assignment"]).split(","))
    matches = [int(index) for index, row in enumerate(assignment) if tuple(int(item) for item in row) == manifest_best]
    if len(matches) != 1:
        failures.append("argmax_assignment_missing")
        manifest_index = -1
        manifest_gap = float("nan")
    else:
        manifest_index = matches[0]
        manifest_gap = float(np.max(labels) - labels[manifest_index])
        if manifest_gap > float(EPS_BITS) or int(ranks[manifest_index]) != 1:
            failures.append("argmax_assignment_not_in_best_group")
    return {
        "candidate_count": int(labels.shape[0]),
        "failures": failures,
        "exact_max_count": int(np.sum(labels == float(np.max(labels)))),
        "best_group_size": int(np.sum(ranks == 1)),
        "float32_changed": int(np.sum(labels.astype(np.float32).astype(np.float64) != labels)),
        "first_exact_index": int(np.flatnonzero(labels == float(np.max(labels)))[0]),
        "manifest_argmax_index": int(manifest_index),
        "manifest_argmax_gap_bits": manifest_gap,
        "idle_count_mean": float(np.mean(np.sum(assignment < 0, axis=1))),
    }


def _selected_edges(edge: np.ndarray, assignment: np.ndarray) -> np.ndarray:
    chosen = np.asarray(assignment, dtype=np.int64).reshape(-1)
    total = np.zeros(edge.shape[-1], dtype=np.float64)
    for uav, buoy in enumerate(chosen):
        if int(buoy) >= 0:
            total += edge[int(uav), int(buoy)]
    return total


def _pair_logit_sum(matrix: np.ndarray, assignment: np.ndarray) -> float:
    total = 0.0
    for uav, buoy in enumerate(np.asarray(assignment, dtype=np.int64).reshape(-1)):
        if int(buoy) >= 0:
            total += float(matrix[int(uav), int(buoy)])
    return total


def listnet_parts(scores: np.ndarray, labels: np.ndarray) -> dict:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels, dtype=np.float64).reshape(-1)
    span = max(float(truth.max() - truth.min()), 1.0e-6)
    quality = (truth - float(truth.min())) / span
    logits = LISTNET_BETA * (quality - float(quality.max()))
    logits -= float(logits.max())
    target = np.exp(logits)
    target /= float(target.sum())
    shifted = values - float(values.max())
    probability = np.exp(shifted)
    probability /= float(probability.sum())
    order = np.argsort(-values, kind="stable")
    exact = truth == float(truth.max())
    bins = []
    gap = float(truth.max()) - truth
    for low, high in zip(GAP_EDGES[:-1], GAP_EDGES[1:]):
        mask = (gap > low) & (gap <= high) if low > 0.0 else gap <= high
        bins.append(
            {
                "gap_low": float(low),
                "gap_high": float(high),
                "count": int(np.sum(mask)),
                "target_mass": float(target[mask].sum()) if np.any(mask) else 0.0,
                "probability_mass": float(probability[mask].sum()) if np.any(mask) else 0.0,
                "loss_contribution": float(-(target[mask] * np.log(np.clip(probability[mask], 1.0e-30, None))).sum()) if np.any(mask) else 0.0,
                "absolute_gradient": float(np.abs(probability[mask] - target[mask]).sum()) if np.any(mask) else 0.0,
            }
        )
    return {
        "target_on_exact_max": float(target[exact].sum()),
        "target_on_best_group": float(target[truth >= float(truth.max()) - 1.0].sum()),
        "target_within_1e6": float(target[truth >= float(truth.max()) - 1.0e6].sum()),
        "probability_on_exact_max": float(probability[exact].sum()),
        "chosen_target": float(target[int(order[0])]),
        "exact_max_rank": int(1 + np.sum(values > float(np.max(values[exact])))),
        "score_gap_exact_to_top": float(values[int(order[0])] - float(np.max(values[exact]))),
        "bins": bins,
    }


def _score_matrix(model, decoder, sample: dict, device: torch.device) -> np.ndarray:
    conditioned = append_queue_edge(sample["graph"])
    tensors = {key: value.to(device) for key, value in batch_tensors(conditioned).items()}
    embeddings = model.condition(tensors)
    return decoder.score_matrix(embeddings[0], embeddings[1]).detach().cpu().numpy()


def _candidate_paths(scenario: int, time_step: int) -> tuple[Path, Path, Path]:
    split = "train" if int(scenario) in TRAIN_POOL else "validation"
    folder = REVISION_DATASET / split / f"scenario_{int(scenario)}"
    stem = f"t{int(time_step):03d}"
    frozen = FROZEN_DATASET / split / f"scenario_{int(scenario)}"
    return folder / f"{stem}_candidates.npz", folder / f"{stem}_manifest.json", frozen / f"{stem}_candidates.npz"


def _load_frozen_candidates(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def run_audit() -> dict:
    if any(int(scenario) in RESERVED_TEST for scenario in list(VALIDATION_POOL) + list(TRAIN_POOL)):
        raise RuntimeError("pool overlaps the reserved test")
    before = {path.name: _sha256(path) for path in (ENCODER_CKPT, EPOCH6, SCHEME_C)}
    device = torch.device("cpu")
    frozen, decoder, _optimizer = _build(device)
    epoch6, _decoder_b, _optimizer_b = _build(device)
    scheme, _decoder_c, _optimizer_c = _build(device)
    epoch6.load_state_dict(torch.load(EPOCH6, map_location="cpu", weights_only=False)["model"])
    scheme_payload = torch.load(SCHEME_C, map_location="cpu", weights_only=False)
    if int(scheme_payload["epoch"]) != 1 or scheme_payload.get("scheme") != "c":
        raise RuntimeError("scheme C header is not epoch 1")
    scheme.load_state_dict(scheme_payload["model"])
    for model in (frozen, epoch6, scheme, decoder):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    config = L3BBitQueueEnvV2().config
    rows = []
    alignment_failures = []
    with torch.no_grad():
        for scenario in VALIDATION_POOL:
            sample = load_decision(int(scenario), 15)
            revision_path, manifest_path, frozen_path = _candidate_paths(scenario, 15)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            frozen_candidates = _load_frozen_candidates(frozen_path)
            report = alignment_report(sample, manifest, frozen_candidates)
            if report["failures"]:
                alignment_failures.append({"scenario_id": int(scenario), "failures": report["failures"]})
            labels = np.asarray(sample["j15_bits"], dtype=np.float64)
            assignment = np.asarray(sample["assignments"], dtype=np.int64)
            state_path = revision_path.parent / "t015_state.npz"
            with np.load(state_path) as state_file:
                arrays = {key: np.asarray(state_file[key]) for key in state_file.files}
            tensors = hgnn_tensors(arrays, config)
            edge = np.asarray(tensors["edge"], dtype=np.float64)
            exact = int(np.flatnonzero(labels == float(np.max(labels)))[0])
            low = int(np.argmin(labels))
            scored = {}
            matrices = {}
            for name, model in (("frozen", frozen), ("epoch6", epoch6), ("scheme_c", scheme)):
                tensor = _scores(model, decoder, sample, device)
                if not _finite_tensor(tensor) or int(tensor.shape[0]) != int(labels.shape[0]):
                    raise RuntimeError(f"{name} scores are not row-aligned")
                values = tensor.detach().cpu().numpy().astype(np.float64)
                matrix = _score_matrix(model, decoder, sample, device)
                order = np.argsort(-values, kind="stable")
                rank_of = np.empty(values.shape[0], dtype=np.int64)
                rank_of[order] = np.arange(1, values.shape[0] + 1)
                if abs(_pair_logit_sum(matrix, assignment[int(order[0])]) - float(values[int(order[0])])) > 1.0e-4:
                    raise RuntimeError(f"{name} assignment score does not match the selected pair logits")
                exact_rows = labels == float(np.max(labels))
                group_rows = np.asarray(sample["utility_rank"], dtype=np.int64) == 1
                scored[name] = {
                    "top1": int(order[0]),
                    "top5": [int(index) for index in order[:5]],
                    "exact_rank": int(np.min(rank_of[exact_rows])),
                    "group_rank": int(np.min(rank_of[group_rows])),
                    "manifest_rank": int(rank_of[int(report["manifest_argmax_index"])]),
                    "score_top1": float(values[int(order[0])]),
                    "score_top2": float(values[int(order[1])]),
                    "score_top5": float(values[int(order[4])]),
                    "score_exact": float(np.max(values[exact_rows])),
                    "margin_12": float(values[int(order[0])] - values[int(order[1])]),
                    "margin_15": float(values[int(order[0])] - values[int(order[4])]),
                    "tie_count": int(np.sum(values == float(values[int(order[0])]))),
                }
                matrices[name] = matrix
                del tensor
            oracle_edges = _selected_edges(edge, assignment[exact])
            top_edges = _selected_edges(edge, assignment[scored["scheme_c"]["top1"]])
            low_edges = _selected_edges(edge, assignment[low])
            logit_gap = []
            rate_gap = []
            matrix = matrices["scheme_c"]
            for uav, (left, right) in enumerate(zip(assignment[exact], assignment[scored["scheme_c"]["top1"]])):
                if int(left) == int(right):
                    continue
                left_logit = float(matrix[uav, int(left)]) if int(left) >= 0 else 0.0
                right_logit = float(matrix[uav, int(right)]) if int(right) >= 0 else 0.0
                logit_gap.append(abs(left_logit - right_logit))
                if int(left) >= 0 and int(right) >= 0:
                    rate_gap.append(abs(float(edge[uav, int(left), 1] - edge[uav, int(right), 1])))
            rows.append(
                {
                    "scenario_id": int(scenario),
                    "alignment": report,
                    "oracle_index": exact,
                    "oracle_j15": float(labels[exact]),
                    "oracle_assignment": [int(item) for item in assignment[exact]],
                    "low_j15": float(labels[low]),
                    "models": scored,
                    "edge_sum_oracle": [float(item) for item in oracle_edges],
                    "edge_sum_scheme_top1": [float(item) for item in top_edges],
                    "edge_sum_low": [float(item) for item in low_edges],
                    "assignment_hamming_oracle_top1": int(np.sum(assignment[exact] != assignment[scored["scheme_c"]["top1"]])),
                "per_uav_logit_gap_median": float(np.median(logit_gap)) if logit_gap else 0.0,
                "per_uav_rate_gap_median": float(np.median(rate_gap)) if rate_gap else 0.0,
                    "scheme_top1_j15": float(labels[scored["scheme_c"]["top1"]]),
                    "scheme_top1_assignment": [int(item) for item in assignment[scored["scheme_c"]["top1"]]],
                    "uav_energy_channel_max": float(np.max(np.abs(tensors["uav"][:, 7]))),
                    "node_features_identical_across_candidates": 1,
                }
            )
            print(f"audit {scenario} rank={scored['scheme_c']['exact_rank']} failures={report['failures']}", flush=True)
        train_parts = []
        for scenario in TRAIN_POOL:
            if int(scenario) in RESERVED_TEST:
                raise RuntimeError("training audit opened a reserved scenario")
            sample = load_decision(int(scenario), 15)
            tensor = _scores(scheme, decoder, sample, device)
            loss = top_regret_loss(tensor, torch.as_tensor(sample["j15_bits"], dtype=tensor.dtype))
            if not _finite_tensor(tensor) or not _finite_tensor(loss):
                raise RuntimeError("non-finite scheme C train score")
            part = listnet_parts(tensor.detach().cpu().numpy(), np.asarray(sample["j15_bits"], dtype=np.float64))
            part["scenario_id"] = int(scenario)
            part["loss"] = float(loss.detach().cpu())
            train_parts.append(part)
            del tensor, loss
            if int(scenario) % 8 == 0:
                print(f"train {scenario} rank={part['exact_max_rank']}", flush=True)
    if {path.name: _sha256(path) for path in (ENCODER_CKPT, EPOCH6, SCHEME_C)} != before:
        raise RuntimeError("a checkpoint changed during the audit")
    summary = _summarize(rows, train_parts, alignment_failures)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / "states.json").write_text(json.dumps(rows), encoding="utf-8")
    (AUDIT_DIR / "train_t15.json").write_text(json.dumps(train_parts), encoding="utf-8")
    (AUDIT_DIR / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return summary


def _mean_key(rows: list[dict], model: str, key: str) -> float:
    return float(np.mean([float(row["models"][model][key]) for row in rows]))


def _summarize(rows: list[dict], train_parts: list[dict], alignment_failures: list[dict]) -> dict:
    edge_gap = np.asarray([np.abs(np.asarray(row["edge_sum_oracle"]) - np.asarray(row["edge_sum_scheme_top1"])) for row in rows], dtype=np.float64)
    low_gap = np.asarray([np.abs(np.asarray(row["edge_sum_oracle"]) - np.asarray(row["edge_sum_low"])) for row in rows], dtype=np.float64)
    score_gap = np.asarray([row["models"]["scheme_c"]["score_top1"] - row["models"]["scheme_c"]["score_exact"] for row in rows], dtype=np.float64)
    j15_gap = np.asarray([row["oracle_j15"] - row["scheme_top1_j15"] for row in rows])
    distinguishable = np.any(edge_gap > 1.0e-3, axis=1)
    flat_score = score_gap < 1.0e-3
    low_score = score_gap >= 0.1
    bin_mean = []
    for index in range(len(train_parts[0]["bins"])):
        bin_mean.append(
            {
                "gap_low": train_parts[0]["bins"][index]["gap_low"],
                "gap_high": train_parts[0]["bins"][index]["gap_high"],
                "count": float(np.mean([part["bins"][index]["count"] for part in train_parts])),
                "target_mass": float(np.mean([part["bins"][index]["target_mass"] for part in train_parts])),
                "probability_mass": float(np.mean([part["bins"][index]["probability_mass"] for part in train_parts])),
                "loss_contribution": float(np.mean([part["bins"][index]["loss_contribution"] for part in train_parts])),
                "absolute_gradient": float(np.mean([part["bins"][index]["absolute_gradient"] for part in train_parts])),
            }
        )
    total_loss = float(np.mean([part["loss"] for part in train_parts]))
    queue_gap = edge_gap[:, 3]
    rate_gap = np.asarray([float(row["per_uav_rate_gap_median"]) for row in rows], dtype=np.float64)
    gate = "T15_AUDIT_INCONCLUSIVE"
    if alignment_failures:
        gate = "T15_LABEL_ALIGNMENT_BLOCKED"
    elif float(np.median(j15_gap)) > 1.0e6 and float(np.median(queue_gap)) <= 1.0e-8 and float(np.median(rate_gap)) < 1.0e-3:
        # The true optimum and the selected candidate differ by millions of bits,
        # while the candidate-conditioned rate and queue features do not.
        gate = "T15_FEATURE_OBSERVABILITY_BLOCKED"
    elif int(np.sum(distinguishable)) >= 12 and int(np.sum(flat_score)) >= 12:
        gate = "T15_RANKING_OBJECTIVE_MISMATCH"
    return {
        "gate": gate,
        "alignment_failure_count": len(alignment_failures),
        "alignment_failures": alignment_failures,
        "states": len(rows),
        "scheme_exact_rank_median": float(np.median([row["models"]["scheme_c"]["exact_rank"] for row in rows])),
        "scheme_group_rank_median": float(np.median([row["models"]["scheme_c"]["group_rank"] for row in rows])),
        "epoch6_exact_rank_median": float(np.median([row["models"]["epoch6"]["exact_rank"] for row in rows])),
        "epoch6_group_rank_median": float(np.median([row["models"]["epoch6"]["group_rank"] for row in rows])),
        "frozen_exact_rank_median": float(np.median([row["models"]["frozen"]["exact_rank"] for row in rows])),
        "frozen_group_rank_median": float(np.median([row["models"]["frozen"]["group_rank"] for row in rows])),
        "manifest_gap_max": float(np.max([row["alignment"]["manifest_argmax_gap_bits"] for row in rows])),
        "scheme_score_gap_median": float(np.median(score_gap)),
        "scheme_score_gap_max": float(np.max(score_gap)),
        "flat_score_states": int(np.sum(flat_score)),
        "low_oracle_score_states": int(np.sum(low_score)),
        "edge_distinguishable_states": int(np.sum(distinguishable)),
        "edge_gap_median": [float(np.median(edge_gap[:, index])) for index in range(4)],
        "low_edge_gap_median": [float(np.median(low_gap[:, index])) for index in range(4)],
        "hamming_median": float(np.median([row["assignment_hamming_oracle_top1"] for row in rows])),
        "per_uav_logit_gap_median": float(np.median([row["per_uav_logit_gap_median"] for row in rows])),
        "per_uav_rate_gap_median": float(np.median([row["per_uav_rate_gap_median"] for row in rows])),
        "j15_regret_median": float(np.median(j15_gap)),
        "energy_channel_max": float(np.max([row["uav_energy_channel_max"] for row in rows])),
        "train_states": len(train_parts),
        "train_loss_mean": total_loss,
        "train_target_exact": float(np.mean([part["target_on_exact_max"] for part in train_parts])),
        "train_target_group": float(np.mean([part["target_on_best_group"] for part in train_parts])),
        "train_target_1e6": float(np.mean([part["target_within_1e6"] for part in train_parts])),
        "train_probability_exact": float(np.mean([part["probability_on_exact_max"] for part in train_parts])),
        "train_chosen_target": float(np.mean([part["chosen_target"] for part in train_parts])),
        "train_exact_rank_median": float(np.median([part["exact_max_rank"] for part in train_parts])),
        "train_bins": bin_mean,
        "margins": {
            model: {
                "margin_12_median": float(np.median([row["models"][model]["margin_12"] for row in rows])),
                "margin_15_median": float(np.median([row["models"][model]["margin_15"] for row in rows])),
            }
            for model in ("frozen", "epoch6", "scheme_c")
        },
    }


if __name__ == "__main__":
    print(json.dumps({"gate": run_audit()["gate"]}), flush=True)
