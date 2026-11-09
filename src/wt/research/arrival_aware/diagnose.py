"""Read-only diagnosis of the L3 t15 selections. No training and no extra scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import log_softmax, softmax

from wt.algorithms.ranking_loss.losses import LISTNET_BETA, top_regret_loss
from wt.models.edge_bound.const import BUOY_DIM
from wt.research.arrival_aware.loader import load_package
from wt.research.arrival_aware.runner import _choose_local
from wt.models.joint_decoder.decoders import JointAssignmentDecoder, gather_assignment_score
from wt.models.joint_encoder.model import JointUtilityHGNN, append_queue_edge, batch_tensors
from wt.common.paths import repo_root


EXPECTED = {
    "E6": "7859f9c94940d9259ab1bece1f8dddf9655a4dcb1cd45d537ad127347526bcde",
    "A10": "87f7c7673555518d80303a8226ae699057bcdaeacb4307512331b7dbad4e60d8",
    "B10": "6024f4c42cdf2de2e8452b49da6ed1824ff8d6e56e84204681a6a37161b81141",
    "decoder": "de1809fab8290b5be3e89cd0d31e22cd04638fd9a43207957aa20b1de255eca5",
}
MODEL_ORDER = ("E6", "A10", "B10")
L3_FILES = {"E6": "init_reference.json", "A10": "explore_arm_a.json", "B10": "explore_arm_b.json"}


def _repo_root() -> Path:
    return repo_root()
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configure() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    np.random.seed(0)


def _load_encoder(path: Path, kind: str, device: torch.device) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if kind == "E6" and int(payload["epoch"]) != 6:
        raise RuntimeError(f"E6 epoch is {payload.get('epoch')}")
    model = JointUtilityHGNN(buoy_dim=int(BUOY_DIM) + 1)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    model.eval()
    return model


def _load_decoder(path: Path, device: torch.device) -> JointAssignmentDecoder:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    decoder = JointAssignmentDecoder("matching")
    decoder.load_state_dict(payload["state_dict"], strict=True)
    decoder.to(device)
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    return decoder


def _score_once(model, decoder, state: dict, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    graph = state["graph"]
    if "zero_payload" in graph or "real_payload" in graph or "scenario_id" in graph:
        raise RuntimeError("graph contains a label or identity field")
    conditioned = append_queue_edge(graph)
    tensors = {key: value.to(device) for key, value in batch_tensors(conditioned).items()}
    assignments = torch.as_tensor(state["assignments"], dtype=torch.long, device=device)
    with torch.no_grad():
        hidden_u, hidden_b = model.condition(tensors)
        matrix = decoder.score_matrix(hidden_u, hidden_b)
        scores = gather_assignment_score(matrix, assignments, logarithmic=False)
    if matrix.ndim != 2 or not torch.isfinite(matrix).all() or not torch.isfinite(scores).all():
        raise RuntimeError("non-finite matching matrix or scores")
    return matrix.detach().cpu().numpy().astype(np.float32, copy=True), scores.detach().cpu().numpy().astype(np.float32, copy=True)


def _target(labels: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    values = torch.as_tensor(labels, dtype=dtype)
    span = (values.max() - values.min()).clamp_min(1.0e-6)
    quality = (values - values.min()) / span
    return softmax(float(LISTNET_BETA) * (quality - quality.max()), dim=0)


def _entropy(probability: torch.Tensor) -> float:
    logged = probability.clamp_min(torch.finfo(probability.dtype).tiny).log()
    return float((-(probability * logged).sum()).detach().cpu())


def _label_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    score_tensor = torch.as_tensor(scores)
    target = _target(labels, score_tensor.dtype)
    cross_entropy = float(top_regret_loss(score_tensor, torch.as_tensor(labels, dtype=score_tensor.dtype)).detach().cpu())
    target_entropy = _entropy(target)
    model_probability = softmax(score_tensor, dim=0)
    return {
        "target_entropy": target_entropy,
        "cross_entropy": cross_entropy,
        "kl": cross_entropy - target_entropy,
        "model_entropy": _entropy(model_probability),
        "target": target.detach().cpu().numpy(),
        "model_probability": model_probability.detach().cpu().numpy(),
    }


def _best_index(values: np.ndarray, candidate_ids: np.ndarray, pool: np.ndarray) -> int:
    best = np.max(values[pool])
    tied = pool[np.flatnonzero(values[pool] == best)]
    return int(tied[int(np.argmin(candidate_ids[tied]))])


def _distribution(values: np.ndarray) -> dict:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise RuntimeError("edge distribution is empty")
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "negative_fraction": float(np.mean(array < 0.0)),
    }


def _active_mask(assignment: np.ndarray) -> str:
    return "".join("1" if int(value) >= 0 else "0" for value in assignment)


def _edges(matrix: np.ndarray, assignment: np.ndarray) -> np.ndarray:
    chosen = [float(matrix[uav, int(buoy)]) for uav, buoy in enumerate(assignment) if int(buoy) >= 0]
    return np.asarray(chosen, dtype=np.float64)


def _unique_edges(matrix: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    seen: set[tuple[int, int]] = set()
    values = []
    for assignment in assignments:
        for uav, buoy in enumerate(assignment):
            pair = (int(uav), int(buoy))
            if pair[1] >= 0 and pair not in seen:
                seen.add(pair)
                values.append(float(matrix[pair]))
    return np.asarray(values, dtype=np.float64)


def _record(state: dict, matrix: np.ndarray, scores: np.ndarray) -> dict:
    local = _choose_local(scores, state["candidate_id"])
    assignments = state["assignments"]
    active = np.sum(assignments >= 0, axis=1).astype(np.int64)
    zero_metrics = _label_metrics(scores, state["zero_payload"])
    real_metrics = _label_metrics(scores, state["real_payload"])
    real_oracle = _best_index(state["real_payload"], state["candidate_id"], np.arange(128))
    chosen_mask = _active_mask(assignments[local])
    same_mask = np.asarray([index for index, row in enumerate(assignments) if _active_mask(row) == chosen_mask], dtype=np.int64)
    same_k = np.flatnonzero(active == int(active[local]))
    mask_best = _best_index(state["real_payload"], state["candidate_id"], same_mask)
    k_best = _best_index(state["real_payload"], state["candidate_id"], same_k)
    bins = []
    for count in sorted(set(int(value) for value in active)):
        pool = np.flatnonzero(active == count)
        score_best = _best_index(scores.astype(np.float64), state["candidate_id"], pool)
        label_best = _best_index(state["real_payload"], state["candidate_id"], pool)
        bins.append(
            {
                "active_uavs": count,
                "candidates": int(pool.size),
                "max_score": float(np.max(scores[pool])),
                "mean_score": float(np.mean(scores[pool])),
                "model_probability_mass": float(np.sum(real_metrics["model_probability"][pool])),
                "zero_target_mass": float(np.sum(zero_metrics["target"][pool])),
                "real_target_mass": float(np.sum(real_metrics["target"][pool])),
                "highest_score_candidate_id": int(state["candidate_id"][score_best]),
                "highest_score_real_payload_bits": float(state["real_payload"][score_best]),
                "label_best_real_payload_bits": float(state["real_payload"][label_best]),
            }
        )
    chosen_edges = _edges(matrix, assignments[local])
    return {
        "scenario_id": int(state["scenario_id"]),
        "time_step": int(state["time_step"]),
        "group": state["group"],
        "split": f"{'train' if state['group'] == 'train' else 'explore'}_t{int(state['time_step'])}",
        "local_position": local,
        "candidate_id": int(state["candidate_id"][local]),
        "assignment": state["assignment_text"][local],
        "active_uavs": int(active[local]),
        "active_mask": chosen_mask,
        "score": float(scores[local]),
        "tie_count": int(np.sum(scores == np.max(scores))),
        "real_payload_bits": float(state["real_payload"][local]),
        "real_regret_bits": float(np.max(state["real_payload"]) - state["real_payload"][local]),
        "zero_payload_bits": float(state["zero_payload"][local]),
        "zero_regret_bits": float(np.max(state["zero_payload"]) - state["zero_payload"][local]),
        "zero_target_entropy": zero_metrics["target_entropy"],
        "zero_cross_entropy": zero_metrics["cross_entropy"],
        "zero_kl": zero_metrics["kl"],
        "real_target_entropy": real_metrics["target_entropy"],
        "real_cross_entropy": real_metrics["cross_entropy"],
        "real_kl": real_metrics["kl"],
        "model_entropy": real_metrics["model_entropy"],
        "real_oracle_candidate_id": int(state["candidate_id"][real_oracle]),
        "real_oracle_active_uavs": int(active[real_oracle]),
        "real_oracle_score": float(scores[real_oracle]),
        "score_minus_real_oracle": float(scores[local] - scores[real_oracle]),
        "payload_minus_real_oracle_bits": float(state["real_payload"][local] - state["real_payload"][real_oracle]),
        "within_active_count_regret_bits": float(state["real_payload"][k_best] - state["real_payload"][local]),
        "within_active_count_candidates": int(same_k.size),
        "within_mask_regret_bits": float(state["real_payload"][mask_best] - state["real_payload"][local]),
        "within_mask_candidates": int(same_mask.size),
        "model_expected_active_uavs": float(np.sum(real_metrics["model_probability"] * active)),
        "zero_target_expected_active_uavs": float(np.sum(zero_metrics["target"] * active)),
        "real_target_expected_active_uavs": float(np.sum(real_metrics["target"] * active)),
        "matrix_mean": float(np.mean(matrix)),
        "matrix_negative_fraction": float(np.mean(matrix < 0.0)),
        "chosen_edges": _distribution(chosen_edges),
        "candidate_edges": _distribution(_unique_edges(matrix, assignments)),
        "activity_bins": bins,
    }


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _nested_mean(rows: list[dict], key: str, stat: str) -> float:
    return float(np.mean([float(row[key][stat]) for row in rows]))


def _group_size_report(rows: list[dict], size_key: str, regret_key: str) -> dict:
    comparable = [row for row in rows if int(row[size_key]) > 1]
    return {
        "group_sizes": [int(row[size_key]) for row in rows],
        "singleton_states": int(sum(int(row[size_key]) == 1 for row in rows)),
        "comparable_states": len(comparable),
        "positive_regret_states": int(sum(float(row[regret_key]) > 0.0 for row in rows)),
        "positive_regret_among_comparable_states": int(sum(float(row[regret_key]) > 0.0 for row in comparable)),
        "singleton_zero_regret_is_not_ranking_evidence": True,
    }


def _group(records: dict[str, list[dict]]) -> dict:
    summary = {}
    for name, rows in records.items():
        buckets: dict[str, list[dict]] = {}
        for row in rows:
            buckets.setdefault(row["split"], []).append(row)
        summary[name] = {}
        for split, group_rows in buckets.items():
            summary[name][split] = {
                "states": len(group_rows),
                "mean_real_payload_bits": _mean(group_rows, "real_payload_bits"),
                "mean_real_regret_bits": _mean(group_rows, "real_regret_bits"),
                "mean_zero_payload_bits": _mean(group_rows, "zero_payload_bits"),
                "mean_active_uavs": _mean(group_rows, "active_uavs"),
                "mean_model_expected_active_uavs": _mean(group_rows, "model_expected_active_uavs"),
                "mean_real_target_expected_active_uavs": _mean(group_rows, "real_target_expected_active_uavs"),
                "mean_zero_kl": _mean(group_rows, "zero_kl"),
                "mean_real_kl": _mean(group_rows, "real_kl"),
                "primary_used_candidate_edges": {
                    "definition": "unique valid (uav, buoy) edges appearing in the 128 candidates",
                    "mean_count": _nested_mean(group_rows, "candidate_edges", "count"),
                    "mean_of_state_means": _nested_mean(group_rows, "candidate_edges", "mean"),
                    "mean_negative_fraction": _nested_mean(group_rows, "candidate_edges", "negative_fraction"),
                },
                "auxiliary_full_matrix": {
                    "definition": "every matrix entry, including edges the 128 candidates never use",
                    "mean_of_state_means": _mean(group_rows, "matrix_mean"),
                    "mean_negative_fraction": _mean(group_rows, "matrix_negative_fraction"),
                },
                "states_with_negative_chosen_score": int(sum(float(row["score"]) < 0.0 for row in group_rows)),
                "states_active_at_most_two": int(sum(int(row["active_uavs"]) <= 2 for row in group_rows)),
                "within_active_count": _group_size_report(
                    group_rows, "within_active_count_candidates", "within_active_count_regret_bits"
                ),
                "within_mask": _group_size_report(group_rows, "within_mask_candidates", "within_mask_regret_bits"),
                "states_active_count_differs_from_real_oracle": int(
                    sum(int(row["active_uavs"]) != int(row["real_oracle_active_uavs"]) for row in group_rows)
                ),
            }
    return summary


def _online_epoch10(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    epoch10 = [row for row in rows if int(row["epoch"]) == 10]
    result = {}
    for arm in ("A", "B"):
        losses = [float(row["loss"]) for row in epoch10 if row["arm"] == arm]
        result[arm] = {"states": len(losses), "mean_online_loss_before_update": float(np.mean(losses))}
    return result


def _reproduce(records: dict[str, list[dict]], l3_dir: Path) -> dict:
    mismatches = []
    for model, filename in L3_FILES.items():
        saved = json.loads((l3_dir / filename).read_text(encoding="utf-8"))
        fresh = {
            (int(row["scenario_id"]), int(row["time_step"])): row
            for row in records[model]
            if row["group"] == "explore_eval"
        }
        if len(saved) != 4 or len(fresh) != 4:
            raise RuntimeError(f"{model} explore rows are not 4")
        for old in saved:
            key = (int(old["scenario_id"]), int(old["time_step"]))
            new = fresh[key]
            payload_gap = max(
                abs(float(new["real_payload_bits"]) - float(old["real_payload_bits"])),
                abs(float(new["zero_payload_bits"]) - float(old["zero_payload_bits"])),
            )
            if int(new["candidate_id"]) != int(old["candidate_id"]) or new["assignment"] != old["assignment"] or payload_gap != 0.0:
                mismatches.append(
                    {
                        "model": model,
                        "scenario_id": key[0],
                        "time_step": key[1],
                        "l3_candidate_id": int(old["candidate_id"]),
                        "l4_candidate_id": int(new["candidate_id"]),
                        "l3_assignment": old["assignment"],
                        "l4_assignment": new["assignment"],
                        "payload_abs_gap_bits": payload_gap,
                    }
                )
    return {"status": "passed" if not mismatches else "failed", "compared_rows": 12, "mismatches": mismatches}


def _paths(repo: Path) -> dict[str, Path]:
    return {
        "E6": repo / "checkpoints/initialization/joint_utility_epoch6.pt",
        "A10": repo / "results/research/arrival_aware/listnet_comparison/arm_a_epoch_010.pt",
        "B10": repo / "results/research/arrival_aware/listnet_comparison/arm_b_epoch_010.pt",
        "decoder": repo / "checkpoints/formal/matching_decoder.pt",
    }


def diagnose(repo: Path, data: Path, l3_dir: Path, output: Path) -> dict:
    if (output / "manifest.json").exists() or (output / "failure.json").exists():
        raise RuntimeError("refusing to repeat an existing L4 output")
    output.mkdir(parents=True, exist_ok=True)
    consumption = {"environment_steps": 0, "low_level_inferences": 0, "scoring_forwards": 0, "training_updates": 0, "backward_passes": 0}
    try:
        paths = _paths(repo)
        hashes = {name: _sha256(path) for name, path in paths.items()}
        for name, digest in EXPECTED.items():
            if hashes[name] != digest:
                raise RuntimeError(f"{name} hash changed")
        _configure()
        device = torch.device("cpu")
        loaded = load_package(data, repo)
        states = loaded["train"] + loaded["explore_eval"]
        if len(states) != 16:
            raise RuntimeError("diagnosis did not receive 16 states")
        decoder = _load_decoder(paths["decoder"], device)
        matrices: dict[str, np.ndarray] = {}
        score_rows: dict[str, np.ndarray] = {}
        records: dict[str, list[dict]] = {}
        for model_name in MODEL_ORDER:
            model = _load_encoder(paths[model_name], model_name, device)
            records[model_name] = []
            for state in states:
                matrix, scores = _score_once(model, decoder, state, device)
                consumption["scoring_forwards"] += 1
                key = f"{model_name}_{int(state['scenario_id'])}_t{int(state['time_step']):03d}"
                matrices[key] = matrix
                score_rows[key] = scores
                records[model_name].append(_record(state, matrix, scores))
            del model
        if consumption["scoring_forwards"] != 48:
            raise RuntimeError(f"scoring forwards are {consumption['scoring_forwards']}")
        np.savez(output / "raw_matrices.npz", **matrices)
        np.savez(output / "raw_scores.npz", **score_rows)
        reproduction = _reproduce(records, l3_dir)
        (output / "state_records.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        (output / "reproduction.json").write_text(json.dumps(reproduction, indent=2) + "\n", encoding="utf-8")
        if reproduction["status"] != "passed":
            failure = {"error": "L3 selection reproduction failed", "reproduction": reproduction, **consumption}
            (output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(failure, indent=2), flush=True)
            return failure
        summary = _group(records)
        online = _online_epoch10(l3_dir / "train_log.jsonl")
        (output / "group_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (output / "online_epoch10.json").write_text(json.dumps(online, indent=2) + "\n", encoding="utf-8")
        manifest = {
            "package": "legacy_arrival_aware_ab_l4",
            "parent_instruction": "b37ff6f691d86809812618cb2725278f8b1f2ebd",
            "command": "py -3 -m wt.research.arrival_aware.diagnose --run --data results/research/arrival_aware/state_preparation --l3 results/research/arrival_aware/listnet_comparison --output results/research/arrival_aware/selector_diagnosis",
            "code_sha256": _sha256(Path(__file__).resolve()),
            "input_sha256": hashes,
            "reproduction": reproduction,
            "online_epoch10_is_not_final_checkpoint_eval": online,
            "group_summary": summary,
            **consumption,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"scoring_forwards": 48, "reproduction": "passed", "training_updates": 0}, sort_keys=True), flush=True)
        return manifest
    except Exception as exc:
        if not (output / "failure.json").exists():
            (output / "failure.json").write_text(
                json.dumps({"error": f"{type(exc).__name__}: {exc}", **consumption}, indent=2) + "\n",
                encoding="utf-8",
            )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("results/research/arrival_aware/state_preparation"))
    parser.add_argument("--l3", type=Path, default=Path("results/research/arrival_aware/listnet_comparison"))
    parser.add_argument("--output", type=Path, default=Path("results/research/arrival_aware/selector_diagnosis"))
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        print(json.dumps({"mode": "config_only", "scoring_forwards_budget": 48, "training_updates": 0}, indent=2), flush=True)
        return
    repo = _repo_root()
    diagnose(
        repo,
        args.data if args.data.is_absolute() else repo / args.data,
        args.l3 if args.l3.is_absolute() else repo / args.l3,
        args.output if args.output.is_absolute() else repo / args.output,
    )


if __name__ == "__main__":
    main()