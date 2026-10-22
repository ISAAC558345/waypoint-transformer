"""Offline interface audit for scheme C. No planner call and no rollout."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from wt.benchmark_v2.phase6ib_highlevel_dataset.hgnn_revision import FROZEN_DATASET
from wt.benchmark_v2.phase6ic_hgnn_smoke.dataset import load_decision
from wt.benchmark_v2.phase6ic_hgnn_smoke.protocol import PRIMARY_TIMES, assert_open
from wt.benchmark_v2.phase6is_assignment_head.audit import _load_hgnn, _rss_bytes, _score_matching
from wt.benchmark_v2.phase6is_assignment_head.protocol import EPOCH6_CKPT
from wt.benchmark_v2.phase6iu_pairwise_head.audit import verify_split
from wt.benchmark_v2.phase6iw_inference_adapter.interface import (
    SchemeCAdapter,
    assignment_is_legal,
    g1_rate_scores,
    stable_choice,
)
from wt.benchmark_v2.phase6iw_inference_adapter.protocol import (
    CANDIDATE_COUNT,
    MAPPO_CKPT,
    OUTPUT,
    PROBE,
    PUBLISHED_TOP1_COUNTS,
    REPEAT_ATOL,
    SCHEME_C_CKPT,
    VALIDATION_POOL,
    assert_output_path,
    assign_gate,
)
from wt.lowlevel_v2.phase5l_joint_encoder.model import EDGE_DIM


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _frozen_folder(scenario: int) -> Path:
    split = "train" if int(scenario) < 322048 else "validation"
    return FROZEN_DATASET / split / f"scenario_{int(scenario)}"


def _frozen_manifest(scenario: int, time_step: int) -> dict:
    path = _frozen_folder(scenario) / f"t{int(time_step):03d}_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _frozen_rate(scenario: int, time_step: int) -> np.ndarray:
    path = _frozen_folder(scenario) / f"t{int(time_step):03d}_state.npz"
    with np.load(path) as state_file:
        return np.asarray(state_file["uav_buoy_hypothetical_rate"], dtype=np.float64)


def _parse_assignment(text: str) -> list[int]:
    return [int(item) for item in str(text).split(",")]


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


def _pool_illegal(assignments: np.ndarray) -> int:
    rows = np.asarray(assignments, dtype=np.int64)
    out_of_range = np.any((rows < -1) | (rows >= 12), axis=1)
    occupied = rows >= 0
    duplicate = np.zeros(rows.shape[0], dtype=bool)
    for column in range(rows.shape[1]):
        duplicate |= occupied[:, column] & np.any((rows[:, column + 1 :] == rows[:, column : column + 1]) & occupied[:, column + 1 :], axis=1)
    return int(np.sum(out_of_range | duplicate))


def _checkpoint_epoch(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload["epoch"])


def run_audit() -> dict:
    device = torch.device("cpu")
    failures = []
    if _checkpoint_epoch(SCHEME_C_CKPT) != 1 or _checkpoint_epoch(EPOCH6_CKPT) != 6:
        failures.append("checkpoint epoch")
    if not MAPPO_CKPT.is_file() or not SCHEME_C_CKPT.is_file() or not EPOCH6_CKPT.is_file():
        failures.append("missing checkpoint")
    before = {
        "scheme_c": _sha256(SCHEME_C_CKPT),
        "epoch6": _sha256(EPOCH6_CKPT),
        "mappo": _sha256(MAPPO_CKPT) if MAPPO_CKPT.is_file() else "",
    }
    split = verify_split()
    adapter = SchemeCAdapter(device)
    adapter.checkpoint_sha256 = before["scheme_c"]
    epoch6 = _load_hgnn(EPOCH6_CKPT, device)
    decoder = adapter.decoder
    probe_rows = []
    rows = []
    disagreements = []
    top1 = {"scheme_c": 0, "epoch6": 0}
    g1_manifest_mismatches = 0
    illegal_rows = 0
    for scenario, time_step in PROBE:
        if int(scenario) in VALIDATION_POOL:
            continue
        assert_open(int(scenario), "train")
        sample = load_decision(int(scenario), int(time_step))
        graph = sample["graph"]
        assignments = np.asarray(sample["assignments"], dtype=np.int64)
        result = adapter.infer(graph, assignments)
        gap = adapter.repeat_gap(graph, assignments)
        probe_rows.append(
            {
                "scenario_id": int(scenario),
                "time_step": int(time_step),
                "split": "train",
                "uav_shape": list(graph["uav"].shape),
                "buoy_shape": list(graph["buoy"].shape),
                "edge_shape": list(graph["edge"].shape),
                "edge_uu_shape": list(graph["edge_uu"].shape),
                "global_shape": list(graph["global"].shape),
                "candidate_count": int(assignments.shape[0]),
                "repeat_gap": gap,
                "chosen_legal": int(assignment_is_legal(result.chosen_assignment)),
                "finite": int(np.all(np.isfinite(result.scores))),
            }
        )
        if gap > REPEAT_ATOL or int(assignments.shape[0]) != CANDIDATE_COUNT:
            failures.append(f"train probe {scenario} t={time_step}")
    for scenario in VALIDATION_POOL:
        for time_step in PRIMARY_TIMES:
            assert_open(int(scenario), "validation")
            sample = load_decision(int(scenario), int(time_step))
            graph = sample["graph"]
            assignments = np.asarray(sample["assignments"], dtype=np.int64)
            if int(assignments.shape[0]) != CANDIDATE_COUNT or graph["uav"].shape[-1] != 8 or graph["edge"].shape[-1] != int(EDGE_DIM) - 1:
                failures.append(f"shape {scenario} t={time_step}")
            if any(name in graph for name in ("j15_bits", "utility_rank")):
                failures.append(f"label leakage {scenario} t={time_step}")
            illegal_rows += _pool_illegal(assignments)
            result = adapter.infer(graph, assignments)
            started = time.perf_counter()
            epoch6_scores = _score_matching(epoch6, decoder, {"graph": graph, "assignments": assignments}, device)
            epoch6_seconds = time.perf_counter() - started
            started = time.perf_counter()
            g1_scores = g1_rate_scores(_frozen_rate(int(scenario), int(time_step)), assignments)
            g1_seconds = time.perf_counter() - started
            _g1_index, g1_assignment = stable_choice(g1_scores, assignments)
            epoch_index, epoch_assignment = stable_choice(epoch6_scores, assignments)
            manifest = _frozen_manifest(int(scenario), int(time_step))
            recorded = _parse_assignment(manifest["additive_assignment"])
            if g1_assignment != recorded:
                g1_manifest_mismatches += 1
            recorded_present = bool(np.any(np.all(assignments == np.asarray(recorded, dtype=np.int64)[None, :], axis=1)))
            if not recorded_present:
                failures.append(f"recorded g1 missing {scenario} t={time_step}")
            legal_flags = [
                assignment_is_legal(result.chosen_assignment),
                assignment_is_legal(epoch_assignment),
                assignment_is_legal(g1_assignment),
                assignment_is_legal(recorded),
            ]
            if not all(legal_flags):
                failures.append(f"illegal choice {scenario} t={time_step}")
            ranks = np.asarray(sample["utility_rank"], dtype=np.int64)
            top1["scheme_c"] += int(ranks[result.chosen_index] == 1)
            top1["epoch6"] += int(ranks[epoch_index] == 1)
            same = result.chosen_assignment == epoch_assignment == recorded
            row = {
                "scenario_id": int(scenario),
                "time_step": int(time_step),
                "scheme_c": result.chosen_assignment,
                "epoch6": epoch_assignment,
                "g1_rate": recorded,
                "same": int(same),
                "scheme_seconds": result.seconds,
                "epoch6_seconds": float(epoch6_seconds),
                "g1_seconds": float(g1_seconds),
                "scheme_top_gap": float(result.scores[result.order[0]] - result.scores[result.order[1]]),
            }
            rows.append(row)
            if not same:
                disagreements.append(
                    {
                        "scenario_id": int(scenario),
                        "time_step": int(time_step),
                        "scheme_c": result.chosen_assignment,
                        "epoch6": epoch_assignment,
                        "g1_rate": recorded,
                        "scheme_scores": result.scores,
                        "epoch6_scores": epoch6_scores,
                        "g1_scores": g1_scores,
                    }
                )
            if (int(scenario), int(time_step)) in PROBE:
                gap = adapter.repeat_gap(graph, assignments)
                probe_rows.append(
                    {
                        "scenario_id": int(scenario),
                        "time_step": int(time_step),
                        "split": "validation",
                        "uav_shape": list(graph["uav"].shape),
                        "buoy_shape": list(graph["buoy"].shape),
                        "edge_shape": list(graph["edge"].shape),
                        "edge_uu_shape": list(graph["edge_uu"].shape),
                        "global_shape": list(graph["global"].shape),
                        "candidate_count": int(assignments.shape[0]),
                        "repeat_gap": gap,
                        "chosen_legal": int(legal_flags[0]),
                        "finite": int(np.all(np.isfinite(result.scores))),
                    }
                )
                if gap > REPEAT_ATOL:
                    failures.append(f"repeat gap {scenario} t={time_step}")
            print(f"scored {scenario} t={time_step} same={int(same)}", flush=True)
    for name, expected in PUBLISHED_TOP1_COUNTS.items():
        if int(top1[name]) != int(expected):
            failures.append(f"{name} top1 {top1[name]} != {expected}")
    if g1_manifest_mismatches:
        failures.append(f"g1 reconstruction mismatches {g1_manifest_mismatches}")
    if illegal_rows:
        failures.append(f"illegal candidate rows {illegal_rows}")
    after = {
        "scheme_c": _sha256(SCHEME_C_CKPT) if SCHEME_C_CKPT.is_file() else "",
        "epoch6": _sha256(EPOCH6_CKPT) if EPOCH6_CKPT.is_file() else "",
        "mappo": _sha256(MAPPO_CKPT) if MAPPO_CKPT.is_file() else "",
    }
    if MAPPO_CKPT.is_file() and before != after:
        failures.append("checkpoint hash changed")
    blocked = bool(failures) or split != {"train": 48, "validation": 16}
    gate = assign_gate(blocked)
    result = {
        "gate": gate,
        "blocked": int(blocked),
        "failures": failures,
        "split": split,
        "hashes_unchanged": before == after,
        "checkpoint_hashes": before,
        "mappo_loaded_for_prediction": 0,
        "candidate_pool": "frozen Phase-6I-B G1_RATE k=6, 13327 legal assignments; Phase-6F bring-up constant remains k=5 and is not regenerated",
        "label_name": "J15_MAPPO_CURRENT",
        "hold": 15,
        "environment": "L3BBitQueueEnvV2",
        "model_version": adapter.model_version,
        "probe": probe_rows,
        "top1_hits_alignment_only": top1,
        "states": int(len(rows)),
        "same_choice_states": int(sum(int(row["same"]) for row in rows)),
        "scheme_c_equals_g1": int(sum(row["scheme_c"] == row["g1_rate"] for row in rows)),
        "scheme_c_equals_epoch6": int(sum(row["scheme_c"] == row["epoch6"] for row in rows)),
        "epoch6_equals_g1": int(sum(row["epoch6"] == row["g1_rate"] for row in rows)),
        "g1_manifest_mismatches": int(g1_manifest_mismatches),
        "illegal_candidate_rows": int(illegal_rows),
        "scheme_seconds_median": float(np.median([row["scheme_seconds"] for row in rows])),
        "scheme_seconds_max": float(np.max([row["scheme_seconds"] for row in rows])),
        "epoch6_seconds_median": float(np.median([row["epoch6_seconds"] for row in rows])),
        "g1_seconds_median": float(np.median([row["g1_seconds"] for row in rows])),
        "rss_bytes": _rss_bytes(),
        "rows": rows,
        "disagreement_count": int(len(disagreements)),
        "reserved_scenarios_read": 0,
        "planner_ready": False,
        "deployment_verified": False,
    }
    folder = assert_output_path(OUTPUT)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.json").write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
    if disagreements:
        payload = {
            "scenario_id": np.asarray([item["scenario_id"] for item in disagreements], dtype=np.int32),
            "time_step": np.asarray([item["time_step"] for item in disagreements], dtype=np.int32),
            "scheme_c": np.asarray([item["scheme_c"] for item in disagreements], dtype=np.int16),
            "epoch6": np.asarray([item["epoch6"] for item in disagreements], dtype=np.int16),
            "g1_rate": np.asarray([item["g1_rate"] for item in disagreements], dtype=np.int16),
            "scheme_scores": np.asarray([item["scheme_scores"] for item in disagreements], dtype=np.float32),
            "epoch6_scores": np.asarray([item["epoch6_scores"] for item in disagreements], dtype=np.float32),
            "g1_scores": np.asarray([item["g1_scores"] for item in disagreements], dtype=np.float32),
        }
        np.savez_compressed(folder / "disagreements.npz", **payload)
    print(json.dumps({"gate": gate, "failures": failures, "disagreements": len(disagreements)}, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    run_audit()
