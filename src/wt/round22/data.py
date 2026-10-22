"""Frozen Round-20 H5 policy sequences and frozen Round-21 predictor adapter."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from wt.round21.data import bridge_flags, conflict_flags, load_task_samples
from wt.round21.training import load_run


R20_OUTPUT = Path("round_20/outputs/round20_multi_uav_dataset_20260828_2100")
R22_OUTPUT = Path("round_22/outputs/round22_deepsets_task_waypoint_policy_20260829_1153")
TRAIN = "train"
VALIDATION = "validation"
H_TASK = 5
CONTEXT_LENGTH = 10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


class FrozenDeepSetsPredictor:
    """Read-only Round-21 DeepSets H5 predictor with no autograd path."""

    def __init__(self, selection_path: Path = R22_OUTPUT / "selected_deepsets_task_waypoint_predictor.json") -> None:
        self.selection = json.loads(selection_path.read_text(encoding="utf-8"))
        self.path = Path(self.selection["checkpoint"])
        expected = self.selection["checkpoint_sha256_before_policy_training"]
        if self.selection["family"] != "deepsets" or self.selection["h_task"] != H_TASK or sha256(self.path) != expected:
            raise RuntimeError("Round-22 predictor provenance is invalid")
        self.model, self.normalizer, _ = load_run(self.path)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("frozen predictor unexpectedly has trainable parameters")

    @property
    def sha256(self) -> str:
        return sha256(self.path)

    def phi(self, graph: Mapping[str, np.ndarray], batch_size: int = 256) -> np.ndarray:
        n = int(graph["uav_nodes"].shape[0])
        pieces: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, n, batch_size):
                stop = min(n, start + batch_size)
                batch = {}
                for key in ("uav_nodes", "buoy_nodes", "uav_mask", "buoy_mask", "global_features"):
                    values = np.asarray(graph[key][start:stop])
                    if key in self.normalizer.means:
                        values = (values - self.normalizer.means[key]) / self.normalizer.stds[key]
                    batch[key] = torch.as_tensor(values, dtype=torch.bool if key.endswith("mask") else torch.float32)
                output = self.model(batch)
                service = output["service"].cpu().numpy()
                progress = output["progress"].cpu().numpy()
                returned = output["team_return"].cpu().numpy()[:, None, None]
                pieces.append(np.concatenate((service, progress, np.repeat(returned, service.shape[1], axis=1)), axis=-1).astype(np.float32))
        return np.concatenate(pieces, axis=0)


@dataclass(frozen=True)
class PolicyNormalizer:
    means: Mapping[str, np.ndarray]
    stds: Mapping[str, np.ndarray]

    @classmethod
    def fit(cls, sequences: "PolicySequences") -> "PolicyNormalizer":
        values: dict[str, list[np.ndarray]] = {"uav_nodes": [], "buoy_nodes": [], "global_features": []}
        for row in range(len(sequences)):
            valid = sequences.attention_mask[row]
            values["uav_nodes"].append(sequences.uav_nodes[row, valid][sequences.uav_mask[row, valid]])
            values["buoy_nodes"].append(sequences.buoy_nodes[row, valid][sequences.buoy_mask[row, valid]])
            values["global_features"].append(sequences.global_features[row, valid])
        means, stds = {}, {}
        for key, chunks in values.items():
            joined = np.concatenate(chunks, axis=0)
            means[key] = joined.mean(axis=0, keepdims=False).astype(np.float32)
            stds[key] = np.maximum(joined.std(axis=0, keepdims=False), 1.0e-6).astype(np.float32)
        return cls(MappingProxyType(means), MappingProxyType(stds))

    def apply(self, sequences: "PolicySequences") -> "PolicySequences":
        return sequences.replace(uav_nodes=(sequences.uav_nodes - self.means["uav_nodes"]) / self.stds["uav_nodes"], buoy_nodes=(sequences.buoy_nodes - self.means["buoy_nodes"]) / self.stds["buoy_nodes"], global_features=(sequences.global_features - self.means["global_features"]) / self.stds["global_features"])

    def serializable(self) -> dict[str, Any]:
        return {"fit_split": TRAIN, "means": {key: value.tolist() for key, value in self.means.items()}, "stds": {key: value.tolist() for key, value in self.stds.items()}}

    @classmethod
    def from_serializable(cls, payload: Mapping[str, Any]) -> "PolicyNormalizer":
        if payload.get("fit_split") != TRAIN:
            raise ValueError("policy normalization must be train-only")
        return cls(MappingProxyType({key: np.asarray(value, dtype=np.float32) for key, value in payload["means"].items()}), MappingProxyType({key: np.asarray(value, dtype=np.float32) for key, value in payload["stds"].items()}))


@dataclass(frozen=True)
class PolicySequences:
    split: str
    uav_nodes: np.ndarray
    buoy_nodes: np.ndarray
    uav_mask: np.ndarray
    buoy_mask: np.ndarray
    global_features: np.ndarray
    predicted_phi: np.ndarray
    oracle_phi: np.ndarray
    actions: np.ndarray
    attention_mask: np.ndarray
    metadata: Mapping[str, np.ndarray]

    def __len__(self) -> int: return int(self.actions.shape[0])

    def replace(self, **updates: Any) -> "PolicySequences":
        fields = {name: getattr(self, name) for name in self.__dataclass_fields__}; fields.update(updates)
        return PolicySequences(**fields)

    def batch(self, indices: Iterable[int], device: str = "cpu") -> dict[str, torch.Tensor]:
        idx = np.asarray(list(indices), dtype=np.int64)
        return {key: torch.as_tensor(value[idx], dtype=torch.bool if key.endswith("mask") else torch.float32, device=device) for key, value in {"uav_nodes": self.uav_nodes, "buoy_nodes": self.buoy_nodes, "uav_mask": self.uav_mask, "buoy_mask": self.buoy_mask, "global_features": self.global_features, "predicted_phi": self.predicted_phi, "oracle_phi": self.oracle_phi, "actions": self.actions, "attention_mask": self.attention_mask}.items()}


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _bridge_conflict_by_sample(split: str) -> dict[str, tuple[bool, bool]]:
    task = load_task_samples(H_TASK, {split})
    bridge, conflict = bridge_flags(task), conflict_flags(task)
    return {str(row["sample_id"]): (bool(bridge[index]), bool(conflict[index])) for index, row in enumerate(task.metadata)}


def _filtered_rows(split: str) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    def permitted(path: Path) -> Iterable[dict[str, Any]]:
        # Development-test rows are routed using only their split token before
        # JSON decode, matching the Round-21 protected-split adapter.
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip() or '"split":"development_test"' in line or '"split": "development_test"' in line:
                    continue
                row = json.loads(line)
                if row["split"] == split:
                    yield row
    samples = [row for row in permitted(R20_OUTPUT / "multi_uav_task_waypoint_samples.jsonl") if int(row["H_task"]) == H_TASK]
    actions = {(str(row["episode_id"]), int(row["timestep"])): row for row in permitted(R20_OUTPUT / "multi_uav_transitions.jsonl")}
    if not samples or not actions:
        raise RuntimeError(f"Round-20 {split} H5/action data is unavailable")
    return samples, actions


def load_policy_sequences(split: str, predictor: FrozenDeepSetsPredictor) -> PolicySequences:
    if split not in {TRAIN, VALIDATION}:
        raise PermissionError("Round-22 permits only Round-20 train/validation policy sequences")
    samples, actions = _filtered_rows(split)
    flags = _bridge_conflict_by_sample(split)
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in samples: by_episode.setdefault(str(row["episode_id"]), []).append(row)
    ordered: list[list[dict[str, Any]]] = []
    for rows in by_episode.values():
        rows.sort(key=lambda row: int(row["t"]))
        expected = list(range(len(rows)))
        if [int(row["t"]) for row in rows] != expected:
            raise RuntimeError("H5 policy anchors must be contiguous episode prefixes")
        for index in range(len(rows)): ordered.append(rows[max(0, index - CONTEXT_LENGTH + 1):index + 1])
    count = len(ordered); h, m, k = CONTEXT_LENGTH, 2, 5
    uav = np.zeros((count, h, m, 8), dtype=np.float32); buoy = np.zeros((count, h, k, 7), dtype=np.float32); uav_mask = np.zeros((count, h, m), dtype=bool); buoy_mask = np.zeros((count, h, k), dtype=bool); global_ = np.zeros((count, h, 6), dtype=np.float32); oracle = np.zeros((count, h, m, 11), dtype=np.float32); attention = np.zeros((count, h), dtype=bool); action = np.zeros((count, m, 2), dtype=np.float32)
    metadata: dict[str, list[Any]] = {name: [] for name in ("sample_id", "episode_id", "scenario_id", "timestep", "num_buoys", "interference_tier", "quality_tier", "behavior_generator", "bridge", "conflict")}
    for row_index, history in enumerate(ordered):
        offset = h - len(history)
        for local, row in enumerate(history):
            dest = offset + local; graph = row["current_graph_state"]
            uav[row_index, dest] = np.asarray(graph["uav_nodes"], dtype=np.float32); buoy[row_index, dest] = np.asarray(graph["buoy_nodes"], dtype=np.float32); uav_mask[row_index, dest] = np.asarray(graph["uav_mask"], dtype=bool); buoy_mask[row_index, dest] = np.asarray(graph["buoy_mask"], dtype=bool); global_[row_index, dest] = np.asarray(graph["global_features"], dtype=np.float32)
            service = np.asarray(row["per_uav_service_responsibility_target"], dtype=np.float32); progress = np.asarray(row["per_uav_gross_service_progress_normalized"], dtype=np.float32); returned = np.full((m, 1), float(row["team_h_step_return_normalized"]), dtype=np.float32)
            oracle[row_index, dest] = np.concatenate((service, progress, returned), axis=-1); attention[row_index, dest] = True
        current = history[-1]; transition = actions.get((str(current["episode_id"]), int(current["t"])))
        if transition is None: raise RuntimeError("H5 policy target action is missing")
        action[row_index] = np.asarray(transition["joint_action_applied_t"], dtype=np.float32)
        bridge, conflict = flags[str(current["sample_id"])]
        source = {"sample_id": current["sample_id"], "episode_id": current["episode_id"], "scenario_id": current["scenario_id"], "timestep": int(current["t"]), "num_buoys": int(current["current_graph_state"]["num_buoys"]), "interference_tier": current["interference_tier"], "quality_tier": current["quality_tier"], "behavior_generator": current["behavior_generator"], "bridge": bridge, "conflict": conflict}
        for key, value in source.items(): metadata[key].append(value)
    flat = {"uav_nodes": uav.reshape(count * h, m, 8), "buoy_nodes": buoy.reshape(count * h, k, 7), "uav_mask": uav_mask.reshape(count * h, m), "buoy_mask": buoy_mask.reshape(count * h, k), "global_features": global_.reshape(count * h, 6)}
    predicted = predictor.phi(flat).reshape(count, h, m, 11)
    # Padding has no semantic state/condition and is excluded by attention.
    predicted[~attention] = 0.0
    return PolicySequences(split, uav, buoy, uav_mask, buoy_mask, global_, predicted, oracle, action, attention, MappingProxyType({key: np.asarray(value) for key, value in metadata.items()}))


__all__ = ["CONTEXT_LENGTH", "FrozenDeepSetsPredictor", "H_TASK", "PolicyNormalizer", "PolicySequences", "R20_OUTPUT", "R22_OUTPUT", "TRAIN", "VALIDATION", "load_policy_sequences", "sha256"]
