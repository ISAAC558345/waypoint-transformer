"""Read-only Round-20 Task-Waypoint adapters with protected-split guards.

This module is deliberately the sole route from the locked JSONL samples to a
Round-21 model.  It selects only physical graph tensors and current masks;
the serialized raw assignment index arrays are never included in its outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


R20_OUTPUT = Path("round_20/outputs/round20_multi_uav_dataset_20260828_2100")
TASK_SAMPLES = R20_OUTPUT / "multi_uav_task_waypoint_samples.jsonl"
GRAPH_SCHEMA = R20_OUTPUT / "multi_uav_graph_schema_v1.json"
RETURN_NORMALIZATION = R20_OUTPUT / "multi_uav_task_waypoint_normalization.json"
DEV_SPLIT = "development_test"

# These lists are an allowlist, not an exclusion list.  In particular, the
# JSON's uav_to_buoy_assignment and matched_uav_by_buoy arrays never cross this
# adapter boundary.
SAFE_GRAPH_KEYS = (
    "uav_nodes", "buoy_nodes", "ub_edges", "uu_edges", "uav_mask", "buoy_mask", "global_features",
)
EXCLUDED_GRAPH_KEYS = ("uav_to_buoy_assignment", "matched_uav_by_buoy", "num_uavs", "num_buoys", "schema_version")
EXCLUDED_MODEL_METADATA = (
    "episode_id", "scenario_id", "sample_id", "behavior_generator", "quality_tier", "interference_tier",
    "split", "future_assignment_signature", "future_actual_buoy_service_bits", "H_task", "t",
)
INTERFERENCE_EDGE_FIELDS = {
    "ub_edges": ("cross_interference_potential_w",),
    "uu_edges": ("receiver_cross_interference_power_w", "receiver_interference_to_signal_ratio"),
}


def _as_float(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _dev_line(line: str) -> bool:
    """Route the protected split without decoding its state or target fields."""
    return '"split":"development_test"' in line or '"split": "development_test"' in line


def read_schema() -> dict[str, Any]:
    return json.loads(GRAPH_SCHEMA.read_text(encoding="utf-8"))


def return_normalization() -> dict[str, float]:
    data = json.loads(RETURN_NORMALIZATION.read_text(encoding="utf-8"))
    values = data["team_h_step_return"]
    return {"mean": float(values["mean"]), "std": float(values["std"]), "scale_used": float(values["scale_used"])}


@dataclass
class TaskDataset:
    """Numpy-backed immutable arrays; metadata is intentionally non-model data."""

    uav_nodes: np.ndarray
    buoy_nodes: np.ndarray
    ub_edges: np.ndarray
    uu_edges: np.ndarray
    uav_mask: np.ndarray
    buoy_mask: np.ndarray
    global_features: np.ndarray
    service_target: np.ndarray
    service_valid: np.ndarray
    progress_normalized: np.ndarray
    progress_raw: np.ndarray
    return_normalized: np.ndarray
    return_raw: np.ndarray
    metadata: list[dict[str, Any]]
    future_assignment_signatures: np.ndarray
    future_actual_buoy_service_bits: np.ndarray

    def __len__(self) -> int:
        return int(self.uav_nodes.shape[0])

    @property
    def input_feature_names(self) -> tuple[str, ...]:
        return SAFE_GRAPH_KEYS

    def subset(self, indices: np.ndarray) -> "TaskDataset":
        indices = np.asarray(indices, dtype=np.int64)
        return TaskDataset(
            *(value[indices] for value in self.arrays()),
            [self.metadata[int(index)] for index in indices],
            self.future_assignment_signatures[indices],
            self.future_actual_buoy_service_bits[indices],
        )

    def arrays(self) -> tuple[np.ndarray, ...]:
        return (
            self.uav_nodes, self.buoy_nodes, self.ub_edges, self.uu_edges, self.uav_mask, self.buoy_mask,
            self.global_features, self.service_target, self.service_valid, self.progress_normalized,
            self.progress_raw, self.return_normalized, self.return_raw,
        )

    def batch(self, indices: Sequence[int], device: str = "cpu") -> dict[str, Any]:
        import torch

        item = np.asarray(indices, dtype=np.int64)
        names = (
            "uav_nodes", "buoy_nodes", "ub_edges", "uu_edges", "uav_mask", "buoy_mask", "global_features",
            "service_target", "service_valid", "progress_normalized", "progress_raw", "return_normalized", "return_raw",
        )
        values = self.arrays()
        result: dict[str, Any] = {name: torch.from_numpy(value[item]).to(device) for name, value in zip(names, values)}
        result["metadata"] = [self.metadata[int(index)] for index in item]
        result["indices"] = item
        return result

    def with_interference_features_removed(self) -> "TaskDataset":
        schema = read_schema()["features"]
        ub = self.ub_edges.copy()
        uu = self.uu_edges.copy()
        for name in INTERFERENCE_EDGE_FIELDS["ub_edges"]:
            ub[..., schema["ub_edges"].index(name)] = 0.0
        for name in INTERFERENCE_EDGE_FIELDS["uu_edges"]:
            uu[..., schema["uu_edges"].index(name)] = 0.0
        return TaskDataset(
            self.uav_nodes.copy(), self.buoy_nodes.copy(), ub, uu, self.uav_mask.copy(), self.buoy_mask.copy(),
            self.global_features.copy(), self.service_target.copy(), self.service_valid.copy(),
            self.progress_normalized.copy(), self.progress_raw.copy(), self.return_normalized.copy(), self.return_raw.copy(),
            list(self.metadata), self.future_assignment_signatures.copy(), self.future_actual_buoy_service_bits.copy(),
        )


def load_task_samples(
    horizon: int,
    splits: Iterable[str],
    *,
    allow_development_test: bool = False,
    sample_path: Path = TASK_SAMPLES,
) -> TaskDataset:
    """Decode only requested split rows.

    While training/validation is active, protected development rows are routed
    from their split token before JSON decoding, so no protected state, labels,
    graph arrays, or metadata are materialized.
    """
    selected = set(splits)
    if DEV_SPLIT in selected and not allow_development_test:
        raise PermissionError("development-test decoding requires the frozen-selection guard")
    if DEV_SPLIT not in selected and allow_development_test:
        raise ValueError("development-test permission may be granted only for its explicit evaluation")
    records: list[dict[str, Any]] = []
    with sample_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            is_dev = _dev_line(line)
            if is_dev and DEV_SPLIT not in selected:
                continue
            if not is_dev and selected == {DEV_SPLIT}:
                continue
            row = json.loads(line)
            if row["split"] not in selected or int(row["H_task"]) != int(horizon):
                continue
            records.append(row)
    if not records:
        raise ValueError(f"no rows for H={horizon}, splits={sorted(selected)}")
    graph = [row["current_graph_state"] for row in records]
    for value in graph:
        unknown = set(value).intersection(EXCLUDED_GRAPH_KEYS)
        if not unknown.issuperset({"uav_to_buoy_assignment", "matched_uav_by_buoy"}):
            raise ValueError("expected raw identity arrays were not present for adapter audit")
    metadata = [
        {
            "sample_id": row["sample_id"], "episode_id": row["episode_id"], "scenario_id": row["scenario_id"],
            "timestep": int(row["t"]), "split": row["split"], "quality_tier": row["quality_tier"],
            "behavior_generator": row["behavior_generator"], "interference_tier": row["interference_tier"],
            "num_buoys": int(np.sum(np.asarray(row["buoy_mask"], dtype=bool))), "horizon": int(row["H_task"]),
            "queue_capacity_bits": float(row["current_graph_state"]["global_features"][3]),
        }
        for row in records
    ]
    return TaskDataset(
        *(_as_float([value[key] for value in graph]) for key in SAFE_GRAPH_KEYS[:4]),
        np.asarray([value["uav_mask"] for value in graph], dtype=bool),
        np.asarray([value["buoy_mask"] for value in graph], dtype=bool),
        _as_float([value["global_features"] for value in graph]),
        _as_float([row["per_uav_service_responsibility_target"] for row in records]),
        _as_float([row["service_valid_uav_mask"] for row in records]),
        _as_float([row["per_uav_gross_service_progress_normalized"] for row in records]),
        _as_float([row["per_uav_gross_service_progress_bits"] for row in records]),
        _as_float([row["team_h_step_return_normalized"] for row in records]),
        _as_float([row["team_h_step_return_raw"] for row in records]),
        metadata,
        np.asarray([row["future_assignment_signature"] for row in records], dtype=np.int64),
        _as_float([row["future_actual_buoy_service_bits"] for row in records]),
    )


@dataclass(frozen=True)
class GraphNormalizer:
    """Train-split-only physical feature standardization."""

    means: Mapping[str, np.ndarray]
    stds: Mapping[str, np.ndarray]

    @staticmethod
    def _masked_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return values[mask]

    @classmethod
    def fit(cls, dataset: TaskDataset) -> "GraphNormalizer":
        um = dataset.uav_mask
        bm = dataset.buoy_mask
        edge = um[:, :, None] & bm[:, None, :]
        uu = um[:, :, None] & um[:, None, :]
        sources = {
            "uav_nodes": cls._masked_values(dataset.uav_nodes, um),
            "buoy_nodes": cls._masked_values(dataset.buoy_nodes, bm),
            "ub_edges": cls._masked_values(dataset.ub_edges, edge),
            "uu_edges": cls._masked_values(dataset.uu_edges, uu),
            "global_features": dataset.global_features,
        }
        means = {name: values.mean(axis=0, dtype=np.float64).astype(np.float32) for name, values in sources.items()}
        stds = {
            name: np.where(values.std(axis=0, dtype=np.float64) > 1.0e-6, values.std(axis=0, dtype=np.float64), 1.0).astype(np.float32)
            for name, values in sources.items()
        }
        return cls(means=means, stds=stds)

    def apply(self, dataset: TaskDataset) -> TaskDataset:
        return TaskDataset(
            (dataset.uav_nodes - self.means["uav_nodes"]) / self.stds["uav_nodes"],
            (dataset.buoy_nodes - self.means["buoy_nodes"]) / self.stds["buoy_nodes"],
            (dataset.ub_edges - self.means["ub_edges"]) / self.stds["ub_edges"],
            (dataset.uu_edges - self.means["uu_edges"]) / self.stds["uu_edges"],
            dataset.uav_mask.copy(), dataset.buoy_mask.copy(),
            (dataset.global_features - self.means["global_features"]) / self.stds["global_features"],
            dataset.service_target.copy(), dataset.service_valid.copy(), dataset.progress_normalized.copy(),
            dataset.progress_raw.copy(), dataset.return_normalized.copy(), dataset.return_raw.copy(),
            list(dataset.metadata), dataset.future_assignment_signatures.copy(), dataset.future_actual_buoy_service_bits.copy(),
        )

    def serializable(self) -> dict[str, Any]:
        return {"means": {name: value.tolist() for name, value in self.means.items()}, "stds": {name: value.tolist() for name, value in self.stds.items()}, "fit_split": "train_only"}


def bridge_flags(dataset: TaskDataset) -> np.ndarray:
    """Recompute a local coordination-bridge diagnostic within this allowed split.

    It is an output-only audit: future signatures/targets are used here but are
    absent from every model input tensor.
    """
    count = len(dataset)
    if count < 3:
        return np.zeros(count, dtype=bool)
    # Safe current-state vector, with masks making K=3 padding explicit.
    vector = np.concatenate((
        dataset.uav_nodes.reshape(count, -1), dataset.buoy_nodes.reshape(count, -1),
        dataset.ub_edges.reshape(count, -1), dataset.uu_edges.reshape(count, -1),
        dataset.global_features,
    ), axis=1).astype(np.float64)
    vector = (vector - vector.mean(axis=0)) / np.where(vector.std(axis=0) > 1.0e-9, vector.std(axis=0), 1.0)
    try:
        from scipy.spatial import cKDTree
        distances, nearest = cKDTree(vector).query(vector, k=2)
        nearest_distance = np.asarray(distances[:, 1], dtype=np.float64)
        nearest = np.asarray(nearest[:, 1], dtype=np.int64)
    except Exception:  # pragma: no cover - conservative dependency-free fallback
        nearest = np.empty(count, dtype=np.int64)
        nearest_distance = np.empty(count, dtype=np.float64)
        for start in range(0, count, 16):
            stop = min(start + 16, count)
            distances = np.sqrt(np.maximum(0.0, ((vector[start:stop, None, :] - vector[None, :, :]) ** 2).sum(axis=2)))
            distances[np.arange(stop - start), np.arange(start, stop)] = np.inf
            nearest[start:stop] = distances.argmin(axis=1)
            nearest_distance[start:stop] = distances[np.arange(stop - start), nearest[start:stop]]
    threshold = float(np.quantile(nearest_distance, 0.25))
    labels = dataset.service_target
    returns = dataset.return_raw
    result = np.zeros(count, dtype=bool)
    for index, other in enumerate(nearest):
        assignment_changed = not np.array_equal(dataset.future_assignment_signatures[index], dataset.future_assignment_signatures[other])
        service_changed = bool(np.abs(labels[index] - labels[other]).sum() > 1.0e-6)
        return_changed = bool(abs(float(returns[index] - returns[other])) > 1.0e-6)
        result[index] = bool(nearest_distance[index] <= threshold and assignment_changed and service_changed and return_changed)
    return result


def conflict_flags(dataset: TaskDataset, threshold: float = 0.25) -> np.ndarray:
    """Current physical ISR criterion from the frozen Round-20 definition."""
    isr = dataset.uu_edges[..., 8]
    mask = dataset.uav_mask[:, :, None] & dataset.uav_mask[:, None, :]
    return np.max(np.where(mask, isr, -np.inf), axis=(1, 2)) >= float(threshold)


__all__ = [
    "DEV_SPLIT", "EXCLUDED_GRAPH_KEYS", "EXCLUDED_MODEL_METADATA", "GraphNormalizer", "INTERFERENCE_EDGE_FIELDS",
    "R20_OUTPUT", "SAFE_GRAPH_KEYS", "TASK_SAMPLES", "TaskDataset", "bridge_flags", "conflict_flags", "load_task_samples",
    "read_schema", "return_normalization",
]
