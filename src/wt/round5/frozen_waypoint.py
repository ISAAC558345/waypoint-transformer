"""Strict frozen Round-4 waypoint inference and auditable Round-5 caches."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

from ..maritime.dataset_v1 import SPLIT_NAMES_V1
from ..round4.round4_training import (
    CHECKPOINT_FORMAT_VERSION,
    load_spatial_waypoint_checkpoint,
    resolve_device,
)
from .round5_data import (
    Round5TransitionDataset,
    Round5TransitionSplit,
)


FROZEN_WAYPOINT_CACHE_SCHEMA_VERSION = "round05_frozen_waypoint_cache_v1"
FROZEN_WAYPOINT_NORMALIZATION_VERSION = "round04_horizon_delta_v1"
SELECTED_WAYPOINT_CHECKPOINT_SHA256 = (
    "F92133AACA1FF6F0839EE7C817F40B696F7FD8FB15973A6118D222F83F01368D"
)


@dataclass(frozen=True)
class FrozenWaypointPredictor:
    """A verified checkpoint whose parameters can never be optimized here."""

    model: torch.nn.Module
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_format_version: str
    horizon: int
    input_dim: int
    output_dim: int
    normalization_distance_m: float
    maximum_step_distance_m: float
    source_run_id: str
    selected_epoch: int
    selected_seed: int
    device: torch.device
    payload_provenance: Mapping[str, Any]
    parameter_sha256: str

    def __post_init__(self) -> None:
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        if self.checkpoint_format_version != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("unsupported frozen waypoint checkpoint format")
        if self.horizon < 1 or self.input_dim < 1 or self.output_dim < 1:
            raise ValueError("frozen predictor dimensions/horizon must be positive")
        if self.normalization_distance_m <= 0.0:
            raise ValueError("waypoint normalization distance must be positive")
        if self.model.training:
            raise ValueError("frozen waypoint model must be in eval mode")
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise ValueError("frozen waypoint parameters must disable gradients")
        if any(parameter.grad is not None for parameter in self.model.parameters()):
            raise ValueError("frozen waypoint parameters must not hold gradients")
        if _module_sha256(self.model) != self.parameter_sha256:
            raise ValueError("frozen waypoint parameter fingerprint changed")

    def predictor_inputs(self, split: Round5TransitionSplit) -> np.ndarray:
        """Build the exact Round-4 goal-conditioned input order."""

        if not isinstance(split, Round5TransitionSplit):
            raise TypeError("split must be Round5TransitionSplit")
        matrix = np.concatenate(
            (
                np.asarray(split.states, dtype=np.float32),
                np.asarray(split.goals, dtype=np.float32),
                np.asarray(split.node_masks, dtype=np.float32),
            ),
            axis=1,
        ).astype(np.float32, copy=False)
        if matrix.shape != (len(split), self.input_dim):
            raise ValueError("frozen waypoint input dimension disagrees with schema")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("frozen waypoint input contains NaN or Inf")
        return np.ascontiguousarray(matrix)

    def predict(
        self,
        split: Round5TransitionSplit,
        *,
        batch_size: int = 1024,
    ) -> np.ndarray:
        """Predict H-step normalized deltas under forced eval/no-grad mode."""

        if isinstance(batch_size, bool) or int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        before = _module_sha256(self.model)
        if before != self.parameter_sha256:
            raise RuntimeError("frozen waypoint parameters changed before inference")
        self.model.eval()
        chunks: List[np.ndarray] = []
        matrix = self.predictor_inputs(split)
        with torch.no_grad():
            for start in range(0, len(split), int(batch_size)):
                batch = torch.from_numpy(
                    matrix[start : start + int(batch_size)]
                ).to(self.device)
                prediction = self.model(batch)
                if prediction.ndim != 2 or prediction.shape[1] != self.output_dim:
                    raise RuntimeError("frozen waypoint model returned invalid shape")
                chunks.append(
                    prediction.detach().to("cpu", dtype=torch.float32).numpy()
                )
        result = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        if result.shape != (len(split), self.output_dim):
            raise RuntimeError("frozen waypoint prediction count changed")
        if not np.all(np.isfinite(result)):
            raise RuntimeError("frozen waypoint prediction contains NaN or Inf")
        after = _module_sha256(self.model)
        if before != after or after != self.parameter_sha256:
            raise RuntimeError("frozen waypoint inference mutated parameters")
        if self.model.training or any(
            parameter.requires_grad or parameter.grad is not None
            for parameter in self.model.parameters()
        ):
            raise RuntimeError("frozen waypoint inference violated freeze contract")
        return result


@dataclass(frozen=True)
class FrozenWaypointCacheSplit:
    """Prediction rows and per-row provenance for one immutable split."""

    split_name: str
    predicted_waypoints: np.ndarray
    predicted_waypoints_m: np.ndarray
    sample_id: np.ndarray
    episode_id: np.ndarray
    t: np.ndarray
    state_sha256: np.ndarray
    goal_sha256: np.ndarray
    node_mask_sha256: np.ndarray
    predictor_input_sha256: np.ndarray
    row_sha256: np.ndarray
    checkpoint_sha256: str
    prediction_sha256: str

    def __post_init__(self) -> None:
        if self.split_name not in SPLIT_NAMES_V1:
            raise ValueError("invalid frozen waypoint cache split")
        count = int(self.predicted_waypoints.shape[0])
        if self.predicted_waypoints.ndim != 2 or count < 1:
            raise ValueError("predicted waypoints must be non-empty [N,D]")
        if self.predicted_waypoints_m.shape != self.predicted_waypoints.shape:
            raise ValueError("raw and normalized waypoint shapes disagree")
        if not np.all(np.isfinite(self.predicted_waypoints)) or not np.all(
            np.isfinite(self.predicted_waypoints_m)
        ):
            raise ValueError("frozen waypoint cache contains NaN or Inf")
        vector_names = (
            "sample_id",
            "episode_id",
            "t",
            "state_sha256",
            "goal_sha256",
            "node_mask_sha256",
            "predictor_input_sha256",
            "row_sha256",
        )
        for name in vector_names:
            value = getattr(self, name)
            if value.shape != (count,) or value.dtype.kind == "O":
                raise ValueError(f"{name} must be a non-object [N] vector")
        for name in (
            "state_sha256",
            "goal_sha256",
            "node_mask_sha256",
            "predictor_input_sha256",
            "row_sha256",
        ):
            if any(len(str(value)) != 64 for value in getattr(self, name)):
                raise ValueError(f"{name} contains an invalid SHA-256 digest")
        if len(set(self.row_sha256.tolist())) != count:
            raise ValueError("frozen waypoint cache row hashes are not unique")
        observed = _array_sha256(self.predicted_waypoints)
        if observed.lower() != self.prediction_sha256.lower():
            raise ValueError("frozen waypoint prediction tensor hash changed")

    def __len__(self) -> int:
        return int(self.predicted_waypoints.shape[0])

    @property
    def predicted_waypoints_normalized(self) -> np.ndarray:
        return self.predicted_waypoints

    @property
    def normalized_delta_xy(self) -> np.ndarray:
        return self.predicted_waypoints

    @property
    def raw_delta_xy_m(self) -> np.ndarray:
        return self.predicted_waypoints_m

    def arrays(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType(
            {
                "predicted_waypoints": self.predicted_waypoints,
                "predicted_waypoints_m": self.predicted_waypoints_m,
                "sample_id": self.sample_id,
                "episode_id": self.episode_id,
                "t": self.t,
                "state_sha256": self.state_sha256,
                "goal_sha256": self.goal_sha256,
                "node_mask_sha256": self.node_mask_sha256,
                "predictor_input_sha256": self.predictor_input_sha256,
                "row_sha256": self.row_sha256,
            }
        )


@dataclass(frozen=True)
class FrozenWaypointCache:
    """Complete three-split cache bound to one dataset and checkpoint."""

    schema_version: str
    source_dir: Path
    source_run_id: str
    checkpoint_path: Path
    checkpoint_sha256: str
    horizon: int
    input_dim: int
    output_dim: int
    normalization_distance_m: float
    normalization_version: str
    splits: Mapping[str, FrozenWaypointCacheSplit]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != FROZEN_WAYPOINT_CACHE_SCHEMA_VERSION:
            raise ValueError("unexpected frozen waypoint cache schema")
        if set(self.splits) != set(SPLIT_NAMES_V1):
            raise ValueError("frozen waypoint cache requires all three splits")
        if self.normalization_version != FROZEN_WAYPOINT_NORMALIZATION_VERSION:
            raise ValueError("unexpected waypoint normalization version")
        if self.horizon < 1 or self.input_dim < 1 or self.output_dim < 1:
            raise ValueError("invalid frozen waypoint cache dimensions")
        sample_ids: List[str] = []
        for split_name in SPLIT_NAMES_V1:
            split = self.splits[split_name]
            if split.split_name != split_name:
                raise ValueError("cache split mapping and object disagree")
            if split.predicted_waypoints.shape[1] != self.output_dim:
                raise ValueError("cache output dimension changed across splits")
            if split.checkpoint_sha256.lower() != self.checkpoint_sha256.lower():
                raise ValueError("cache split checkpoint hash disagrees")
            sample_ids.extend(map(str, split.sample_id.tolist()))
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("cache rows leak across splits")

    @property
    def train(self) -> FrozenWaypointCacheSplit:
        return self.splits["train"]

    @property
    def validation(self) -> FrozenWaypointCacheSplit:
        return self.splits["validation"]

    @property
    def test(self) -> FrozenWaypointCacheSplit:
        return self.splits["test"]

    def split(self, split_name: str) -> FrozenWaypointCacheSplit:
        if split_name not in SPLIT_NAMES_V1:
            raise ValueError(f"split must be one of {SPLIT_NAMES_V1}")
        return self.splits[split_name]

    def summary(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_run_id": self.source_run_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "horizon": self.horizon,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "normalization_distance_m": self.normalization_distance_m,
            "normalization_version": self.normalization_version,
            "sample_counts": {
                name: len(self.splits[name]) for name in SPLIT_NAMES_V1
            },
        }


def load_frozen_waypoint_predictor(
    checkpoint_path: Any,
    dataset: Round5TransitionDataset,
    *,
    expected_sha256: str = SELECTED_WAYPOINT_CHECKPOINT_SHA256,
    device: str | torch.device = "cpu",
) -> FrozenWaypointPredictor:
    """Strictly verify, load, freeze, and fingerprint the selected H=30 model."""

    if not isinstance(dataset, Round5TransitionDataset):
        raise TypeError("dataset must be Round5TransitionDataset")
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_hash = str(expected_sha256).strip().lower()
    if len(expected_hash) != 64:
        raise ValueError("expected checkpoint SHA-256 must contain 64 hex digits")
    actual_hash = _file_sha256(path)
    if actual_hash.lower() != expected_hash:
        raise ValueError(
            "selected Round-4 waypoint checkpoint SHA-256 mismatch: "
            f"expected {expected_hash.upper()}, observed {actual_hash.upper()}"
        )
    resolved = device if isinstance(device, torch.device) else resolve_device(device)
    expected_input = dataset.state_dim + dataset.goal_dim + dataset.node_mask_dim
    model, payload = load_spatial_waypoint_checkpoint(
        path,
        device=resolved,
        expected_horizon=dataset.waypoint_horizon,
        expected_model_kind="goal_conditioned",
        expected_input_dim=expected_input,
    )
    if payload.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("checkpoint format changed after strict load")
    if payload.get("output_representation") != "relative_delta":
        raise ValueError("frozen checkpoint output must be a relative delta")
    architecture = payload.get("architecture")
    provenance = payload.get("provenance")
    if not isinstance(architecture, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("checkpoint lacks architecture/provenance metadata")
    output_dim = int(architecture.get("output_dim", -1))
    if output_dim != dataset.action_dim:
        raise ValueError("checkpoint output dimension disagrees with action schema")
    scale = float(payload.get("displacement_scale_m", math.nan))
    maximum_step = float(payload.get("maximum_step_distance_m", math.nan))
    if not math.isclose(
        scale, dataset.waypoint_normalization_distance_m, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("checkpoint waypoint normalization scale changed")
    if not math.isclose(
        maximum_step, dataset.maximum_step_distance_m, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("checkpoint maximum step distance changed")
    if str(provenance.get("source_run_id")) != dataset.source_run_id:
        raise ValueError("checkpoint source run differs from Round-5 dataset")
    if provenance.get("source_read_only") is not True or provenance.get(
        "round3_lock_verified"
    ) is not True:
        raise ValueError("checkpoint does not prove locked read-only source use")
    source_hashes = provenance.get("source_hashes")
    if not isinstance(source_hashes, Mapping) or dict(source_hashes) != dict(
        dataset.source_hashes
    ):
        raise ValueError("checkpoint source hashes differ from current locked data")

    model.to(resolved)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    fingerprint = _module_sha256(model)
    return FrozenWaypointPredictor(
        model=model,
        checkpoint_path=path,
        checkpoint_sha256=actual_hash.upper(),
        checkpoint_format_version=str(payload["checkpoint_format_version"]),
        horizon=int(payload["horizon"]),
        input_dim=int(architecture["input_dim"]),
        output_dim=output_dim,
        normalization_distance_m=scale,
        maximum_step_distance_m=maximum_step,
        source_run_id=str(provenance["source_run_id"]),
        selected_epoch=int(payload["epoch"]),
        selected_seed=int(payload["seed"]),
        device=resolved,
        payload_provenance=MappingProxyType(dict(provenance)),
        parameter_sha256=fingerprint,
    )


def build_frozen_waypoint_cache(
    dataset: Round5TransitionDataset,
    predictor: FrozenWaypointPredictor,
    *,
    batch_size: int = 1024,
) -> FrozenWaypointCache:
    """Run frozen inference for all three splits and bind every output row."""

    _validate_dataset_predictor_pair(dataset, predictor)
    splits: Dict[str, FrozenWaypointCacheSplit] = {}
    split_provenance: Dict[str, Any] = {}
    for split_name in SPLIT_NAMES_V1:
        transition = dataset.split(split_name)
        inputs = predictor.predictor_inputs(transition)
        predictions = predictor.predict(transition, batch_size=batch_size)
        raw = predictions.astype(np.float64) * predictor.normalization_distance_m
        state_hashes = _row_hashes(transition.states, transition.state_masks)
        goal_hashes = _row_hashes(transition.goals, transition.goal_masks)
        node_hashes = _row_hashes(transition.node_masks)
        input_hashes = _row_hashes(inputs)
        row_hashes = np.asarray(
            [
                _cache_row_sha256(
                    split_name=split_name,
                    sample_id=str(transition.sample_id[index]),
                    episode_id=str(transition.episode_id[index]),
                    t=int(transition.t[index]),
                    state_sha256=str(state_hashes[index]),
                    goal_sha256=str(goal_hashes[index]),
                    node_mask_sha256=str(node_hashes[index]),
                    input_sha256=str(input_hashes[index]),
                    checkpoint_sha256=predictor.checkpoint_sha256,
                )
                for index in range(len(transition))
            ],
            dtype="<U64",
        )
        prediction_hash = _array_sha256(predictions)
        cache_split = FrozenWaypointCacheSplit(
            split_name=split_name,
            predicted_waypoints=predictions,
            predicted_waypoints_m=raw,
            sample_id=transition.sample_id.copy(),
            episode_id=transition.episode_id.copy(),
            t=transition.t.copy(),
            state_sha256=state_hashes,
            goal_sha256=goal_hashes,
            node_mask_sha256=node_hashes,
            predictor_input_sha256=input_hashes,
            row_sha256=row_hashes,
            checkpoint_sha256=predictor.checkpoint_sha256,
            prediction_sha256=prediction_hash,
        )
        splits[split_name] = cache_split
        split_provenance[split_name] = {
            "sample_count": len(cache_split),
            "episode_count": len(set(cache_split.episode_id.tolist())),
            "predictor_input_shape": list(inputs.shape),
            "predictor_input_sha256": _array_sha256(inputs),
            "prediction_shape": list(predictions.shape),
            "prediction_sha256": prediction_hash,
            "raw_prediction_sha256": _array_sha256(raw),
            "sample_id_sha256": _array_sha256(cache_split.sample_id),
            "episode_id_sha256": _array_sha256(cache_split.episode_id),
            "t_sha256": _array_sha256(cache_split.t),
            "row_sha256_vector_sha256": _array_sha256(row_hashes),
        }
    provenance = MappingProxyType(
        {
            "source_read_only": True,
            "source_run_id": dataset.source_run_id,
            "source_dir": str(dataset.source_dir),
            "source_hashes": dict(dataset.source_hashes),
            "round3_lock_verified": bool(
                dataset.provenance.get("round3_lock_verified")
            ),
            "checkpoint_path": str(predictor.checkpoint_path),
            "checkpoint_sha256": predictor.checkpoint_sha256,
            "checkpoint_parameter_sha256": predictor.parameter_sha256,
            "checkpoint_format_version": predictor.checkpoint_format_version,
            "checkpoint_model_kind": "goal_conditioned",
            "checkpoint_selected_epoch": predictor.selected_epoch,
            "checkpoint_selected_seed": predictor.selected_seed,
            "waypoint_horizon": predictor.horizon,
            "input_recipe": ["masked_state_t", "masked_goal", "valid_node_mask_t"],
            "action_labels_used_as_input": False,
            "inference_mode": "eval_and_torch_no_grad",
            "parameters_require_grad": False,
            "parameters_mutated": False,
            "normalization": {
                "version": FROZEN_WAYPOINT_NORMALIZATION_VERSION,
                "formula": "predicted_delta_m / (H * maximum_step_distance_m)",
                "distance_m": predictor.normalization_distance_m,
                "maximum_step_distance_m": predictor.maximum_step_distance_m,
            },
            "splits": split_provenance,
            "training_performed": False,
        }
    )
    return FrozenWaypointCache(
        schema_version=FROZEN_WAYPOINT_CACHE_SCHEMA_VERSION,
        source_dir=dataset.source_dir,
        source_run_id=dataset.source_run_id,
        checkpoint_path=predictor.checkpoint_path,
        checkpoint_sha256=predictor.checkpoint_sha256,
        horizon=predictor.horizon,
        input_dim=predictor.input_dim,
        output_dim=predictor.output_dim,
        normalization_distance_m=predictor.normalization_distance_m,
        normalization_version=FROZEN_WAYPOINT_NORMALIZATION_VERSION,
        splits=MappingProxyType(splits),
        provenance=provenance,
    )


def materialize_frozen_waypoint_cache(
    cache: FrozenWaypointCache,
    output_dir: Any,
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Persist pickle-free NPZ rows plus complete per-split JSON provenance."""

    if not isinstance(cache, FrozenWaypointCache):
        raise TypeError("cache must be FrozenWaypointCache")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "cache_manifest.json"
    expected_paths = [manifest_path]
    for split_name in SPLIT_NAMES_V1:
        expected_paths.extend(
            [
                root / f"predicted_waypoints_{split_name}.npz",
                root / f"predicted_waypoints_{split_name}.json",
            ]
        )
    existing = [path for path in expected_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "frozen waypoint cache already exists: "
            + ", ".join(str(path) for path in existing)
        )

    split_manifests: Dict[str, Any] = {}
    for split_name in SPLIT_NAMES_V1:
        split = cache.split(split_name)
        npz_path = root / f"predicted_waypoints_{split_name}.npz"
        json_path = root / f"predicted_waypoints_{split_name}.json"
        arrays = dict(split.arrays())
        _atomic_save_npz(npz_path, arrays)
        array_hashes = {
            name: _array_sha256(value) for name, value in arrays.items()
        }
        split_manifest = {
            "schema_version": FROZEN_WAYPOINT_CACHE_SCHEMA_VERSION,
            "split": split_name,
            "sample_count": len(split),
            "npz_file": npz_path.name,
            "npz_sha256": _file_sha256(npz_path),
            "array_fields": {
                name: {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "sha256": array_hashes[name],
                }
                for name, value in arrays.items()
            },
            "checkpoint_path": str(cache.checkpoint_path),
            "checkpoint_sha256": cache.checkpoint_sha256,
            "source_run_id": cache.source_run_id,
            "source_dir": str(cache.source_dir),
            "waypoint_horizon": cache.horizon,
            "normalization_version": cache.normalization_version,
            "normalization_distance_m": cache.normalization_distance_m,
            "input_dim": cache.input_dim,
            "output_dim": cache.output_dim,
            "input_recipe": ["masked_state_t", "masked_goal", "valid_node_mask_t"],
            "action_labels_used_as_input": False,
            "training_performed": False,
            "split_provenance": cache.provenance["splits"][split_name],
        }
        _atomic_write_json(json_path, split_manifest)
        split_manifests[split_name] = {
            "npz_file": npz_path.name,
            "metadata_file": json_path.name,
            "npz_sha256": split_manifest["npz_sha256"],
            "metadata_sha256": _file_sha256(json_path),
            "sample_count": len(split),
            "prediction_sha256": split.prediction_sha256,
        }

    manifest = {
        "schema_version": FROZEN_WAYPOINT_CACHE_SCHEMA_VERSION,
        "summary": cache.summary(),
        "splits": split_manifests,
        "provenance": dict(cache.provenance),
        "pickle_allowed": False,
        "training_performed": False,
    }
    _atomic_write_json(manifest_path, manifest)
    result = dict(manifest)
    result["manifest_file"] = str(manifest_path)
    result["manifest_sha256"] = _file_sha256(manifest_path)
    return result


def load_frozen_waypoint_cache(
    output_dir: Any,
    dataset: Round5TransitionDataset,
    *,
    expected_checkpoint_sha256: str = SELECTED_WAYPOINT_CHECKPOINT_SHA256,
) -> FrozenWaypointCache:
    """Reload and fully verify a materialized cache against current rows."""

    if not isinstance(dataset, Round5TransitionDataset):
        raise TypeError("dataset must be Round5TransitionDataset")
    root = Path(output_dir).expanduser().resolve()
    manifest = _read_json(root / "cache_manifest.json")
    if manifest.get("schema_version") != FROZEN_WAYPOINT_CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported frozen waypoint cache manifest")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("frozen waypoint cache lacks provenance")
    checkpoint_hash = str(provenance.get("checkpoint_sha256", ""))
    if checkpoint_hash.lower() != str(expected_checkpoint_sha256).lower():
        raise ValueError("frozen waypoint cache checkpoint hash is not selected H=30")
    if str(provenance.get("source_run_id")) != dataset.source_run_id:
        raise ValueError("frozen waypoint cache source run changed")
    if dict(provenance.get("source_hashes", {})) != dict(dataset.source_hashes):
        raise ValueError("frozen waypoint cache source hashes changed")
    normalization = provenance.get("normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError("frozen waypoint cache lacks normalization provenance")
    scale = float(normalization.get("distance_m", math.nan))
    if not math.isclose(
        scale,
        dataset.waypoint_normalization_distance_m,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("frozen waypoint cache normalization distance changed")

    split_entries = manifest.get("splits")
    if not isinstance(split_entries, Mapping) or set(split_entries) != set(
        SPLIT_NAMES_V1
    ):
        raise ValueError("frozen waypoint cache split coverage changed")
    splits: Dict[str, FrozenWaypointCacheSplit] = {}
    for split_name in SPLIT_NAMES_V1:
        entry = split_entries[split_name]
        if not isinstance(entry, Mapping):
            raise ValueError("cache split manifest entry must be an object")
        npz_path = root / str(entry["npz_file"])
        metadata_path = root / str(entry["metadata_file"])
        if _file_sha256(npz_path) != str(entry["npz_sha256"]):
            raise ValueError(f"frozen waypoint NPZ checksum changed: {split_name}")
        if _file_sha256(metadata_path) != str(entry["metadata_sha256"]):
            raise ValueError(f"frozen waypoint JSON checksum changed: {split_name}")
        metadata = _read_json(metadata_path)
        if metadata.get("split") != split_name:
            raise ValueError("cache metadata split changed")
        if metadata.get("checkpoint_sha256", "").lower() != checkpoint_hash.lower():
            raise ValueError("cache split checkpoint hash changed")
        fields = metadata.get("array_fields")
        if not isinstance(fields, Mapping):
            raise ValueError("cache split lacks array field descriptors")
        with np.load(str(npz_path), allow_pickle=False) as archive:
            if set(archive.files) != set(fields):
                raise ValueError("cache NPZ keys differ from metadata")
            arrays = {name: archive[name].copy() for name in archive.files}
        for name, descriptor in fields.items():
            if not isinstance(descriptor, Mapping):
                raise ValueError("cache array descriptor must be an object")
            value = arrays[name]
            if str(value.dtype) != descriptor.get("dtype") or list(
                value.shape
            ) != descriptor.get("shape"):
                raise ValueError(f"cache array shape/dtype changed: {name}")
            if _array_sha256(value) != descriptor.get("sha256"):
                raise ValueError(f"cache array content hash changed: {name}")
        transition = dataset.split(split_name)
        if not np.array_equal(arrays["sample_id"].astype(str), transition.sample_id):
            raise ValueError("cache sample order differs from locked transitions")
        if not np.array_equal(arrays["episode_id"].astype(str), transition.episode_id):
            raise ValueError("cache episode order differs from locked transitions")
        if not np.array_equal(arrays["t"].astype(np.int64), transition.t):
            raise ValueError("cache timestep order differs from locked transitions")
        expected_state_hash = _row_hashes(
            transition.states, transition.state_masks
        )
        expected_goal_hash = _row_hashes(
            transition.goals, transition.goal_masks
        )
        expected_node_hash = _row_hashes(transition.node_masks)
        expected_inputs = np.concatenate(
            (
                transition.states,
                transition.goals,
                transition.node_masks.astype(np.float32),
            ),
            axis=1,
        ).astype(np.float32, copy=False)
        expected_input_hash = _row_hashes(expected_inputs)
        for name, expected in (
            ("state_sha256", expected_state_hash),
            ("goal_sha256", expected_goal_hash),
            ("node_mask_sha256", expected_node_hash),
            ("predictor_input_sha256", expected_input_hash),
        ):
            if not np.array_equal(arrays[name].astype(str), expected):
                raise ValueError(f"cache row provenance changed: {name}")
        prediction = np.asarray(arrays["predicted_waypoints"], dtype=np.float32)
        raw = np.asarray(arrays["predicted_waypoints_m"], dtype=np.float64)
        if not np.allclose(
            prediction.astype(np.float64) * scale,
            raw,
            rtol=2e-6,
            atol=2e-6,
        ):
            raise ValueError("cache raw/normalized predictions disagree")
        splits[split_name] = FrozenWaypointCacheSplit(
            split_name=split_name,
            predicted_waypoints=prediction,
            predicted_waypoints_m=raw,
            sample_id=arrays["sample_id"].astype(str),
            episode_id=arrays["episode_id"].astype(str),
            t=np.asarray(arrays["t"], dtype=np.int64),
            state_sha256=arrays["state_sha256"].astype(str),
            goal_sha256=arrays["goal_sha256"].astype(str),
            node_mask_sha256=arrays["node_mask_sha256"].astype(str),
            predictor_input_sha256=arrays["predictor_input_sha256"].astype(str),
            row_sha256=arrays["row_sha256"].astype(str),
            checkpoint_sha256=checkpoint_hash,
            prediction_sha256=_array_sha256(prediction),
        )

    checkpoint_path = Path(str(provenance["checkpoint_path"]))
    return FrozenWaypointCache(
        schema_version=FROZEN_WAYPOINT_CACHE_SCHEMA_VERSION,
        source_dir=dataset.source_dir,
        source_run_id=dataset.source_run_id,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_hash,
        horizon=int(provenance["waypoint_horizon"]),
        input_dim=dataset.state_dim + dataset.goal_dim + dataset.node_mask_dim,
        output_dim=dataset.action_dim,
        normalization_distance_m=scale,
        normalization_version=str(normalization["version"]),
        splits=MappingProxyType(splits),
        provenance=MappingProxyType(dict(provenance)),
    )


def _validate_dataset_predictor_pair(
    dataset: Round5TransitionDataset, predictor: FrozenWaypointPredictor
) -> None:
    if not isinstance(dataset, Round5TransitionDataset):
        raise TypeError("dataset must be Round5TransitionDataset")
    if not isinstance(predictor, FrozenWaypointPredictor):
        raise TypeError("predictor must be FrozenWaypointPredictor")
    if predictor.source_run_id != dataset.source_run_id:
        raise ValueError("predictor and dataset source runs differ")
    if predictor.horizon != dataset.waypoint_horizon:
        raise ValueError("predictor and dataset waypoint horizons differ")
    if predictor.input_dim != (
        dataset.state_dim + dataset.goal_dim + dataset.node_mask_dim
    ):
        raise ValueError("predictor input dimension disagrees with data schema")
    if predictor.output_dim != dataset.action_dim:
        raise ValueError("predictor output dimension disagrees with data schema")
    if not math.isclose(
        predictor.normalization_distance_m,
        dataset.waypoint_normalization_distance_m,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("predictor and dataset waypoint scales differ")


def _cache_row_sha256(
    *,
    split_name: str,
    sample_id: str,
    episode_id: str,
    t: int,
    state_sha256: str,
    goal_sha256: str,
    node_mask_sha256: str,
    input_sha256: str,
    checkpoint_sha256: str,
) -> str:
    values = (
        FROZEN_WAYPOINT_CACHE_SCHEMA_VERSION,
        split_name,
        sample_id,
        episode_id,
        str(t),
        state_sha256,
        goal_sha256,
        node_mask_sha256,
        input_sha256,
        checkpoint_sha256.upper(),
        FROZEN_WAYPOINT_NORMALIZATION_VERSION,
    )
    return hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()


def _row_hashes(*arrays: np.ndarray) -> np.ndarray:
    if not arrays:
        raise ValueError("at least one array is required for row hashes")
    normalized = [np.asarray(value) for value in arrays]
    count = int(normalized[0].shape[0])
    if any(value.ndim < 1 or int(value.shape[0]) != count for value in normalized):
        raise ValueError("row hash arrays must share their leading dimension")
    return np.asarray(
        [_arrays_sha256(*(value[index] for value in normalized)) for index in range(count)],
        dtype="<U64",
    )


def _arrays_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.asarray(value)
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return _arrays_sha256(np.asarray(array))


def _module_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().to("cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            _json_compatible(dict(value)),
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
    os.replace(temporary, path)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist())
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON metadata contains NaN or Inf")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"JSON metadata cannot encode {type(value).__name__}")


# Runner-friendly aliases.
build_waypoint_cache = build_frozen_waypoint_cache
load_waypoint_cache = load_frozen_waypoint_cache
load_waypoint_predictor = load_frozen_waypoint_predictor
save_waypoint_cache = materialize_frozen_waypoint_cache


__all__ = [
    "FROZEN_WAYPOINT_CACHE_SCHEMA_VERSION",
    "FROZEN_WAYPOINT_NORMALIZATION_VERSION",
    "FrozenWaypointCache",
    "FrozenWaypointCacheSplit",
    "FrozenWaypointPredictor",
    "SELECTED_WAYPOINT_CHECKPOINT_SHA256",
    "build_frozen_waypoint_cache",
    "build_waypoint_cache",
    "load_frozen_waypoint_cache",
    "load_frozen_waypoint_predictor",
    "load_waypoint_cache",
    "load_waypoint_predictor",
    "materialize_frozen_waypoint_cache",
    "save_waypoint_cache",
]
