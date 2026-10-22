"""Deployable-only frozen Task-Waypoint caches for Round-22 train/validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .data import H_TASK, R22_OUTPUT, FrozenDeepSetsPredictor, _filtered_rows


def _state_hash(graph: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(graph, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest().upper()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def materialize_predicted_task_waypoint_cache(predictor: FrozenDeepSetsPredictor, output: Path = R22_OUTPUT) -> dict[str, Any]:
    """Cache only predicted deployable Phi and offline movement labels.

    The cache contains no true future Task-Waypoint label, no fresh-dev data,
    no development-test row and no untouched-evaluation row.
    """
    manifest_rows: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        samples, actions = _filtered_rows(split)
        graph = {key: np.asarray([row["current_graph_state"][key] for row in samples], dtype=np.bool_ if key.endswith("mask") else np.float32) for key in ("uav_nodes", "buoy_nodes", "uav_mask", "buoy_mask", "global_features")}
        predicted = predictor.phi(graph)
        rows: list[dict[str, Any]] = []
        for index, sample in enumerate(samples):
            action = actions[(str(sample["episode_id"]), int(sample["t"]))]["joint_action_applied_t"]
            rows.append({"episode_id": sample["episode_id"], "scenario_id": sample["scenario_id"], "timestep": int(sample["t"]), "split": split, "state_sha256": _state_hash(sample["current_graph_state"]), "uav_mask": graph["uav_mask"][index].astype(bool).tolist(), "buoy_mask": graph["buoy_mask"][index].astype(bool).tolist(), "predictor_sha256": predictor.sha256, "predicted_service_responsibility": predicted[index, :, :5].tolist(), "predicted_gross_service_progress": predicted[index, :, 5:10].tolist(), "predicted_team_h_step_return": predicted[index, 0, 10].item(), "action_label": action})
        path = output / f"{split}_predicted_task_waypoint_cache.jsonl"
        _write_jsonl(path, rows)
        manifest_rows.append({"split": split, "path": path.as_posix(), "record_count": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper()})
    manifest = {"schema_version": "round22_predicted_task_waypoint_cache_v1", "h_task": H_TASK, "predictor_sha256": predictor.sha256, "splits": manifest_rows, "true_future_task_waypoint_stored": False, "fresh_dev_stored": False, "development_test_stored": False, "untouched_eval_stored": False}
    path = output / "predicted_task_waypoint_cache_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


__all__ = ["materialize_predicted_task_waypoint_cache"]
