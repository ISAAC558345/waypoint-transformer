"""Train/validation-only Round-12 transition loading and frozen-Phi inference.

Round-10's task-waypoint package supplies the authorized H10 task labels and
split registration.  The matching Round-7 episode files are opened only for
the episode IDs pre-registered by those Round-10 *train/validation* rows, to
recover their recorded two-dimensional action labels.  This loader never
opens old-development-test, untouched-holdout, or fresh-development content.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .contracts import (
    ACTION_DIM,
    ALLOWED_SPLITS,
    FORBIDDEN_SPLITS,
    H_TASK,
    PREDICTOR_SHA256,
    assert_predictor_frozen,
    assert_selected_predictor,
    load_frozen_predictor,
    predicted_phi,
    sha256,
)


ROUND12_SOURCE_DIR = Path("round_10/outputs/round10_task_waypoint_20260810_0110")
ROUND12_ACTION_SOURCE_DIR = Path("round_07/outputs/round07_formal_20260809_1740/development_dataset_v2")
ROUND12_DATA_SCHEMA = "round12_task_waypoint_policy_data_v1"
# Round-7's scenario generator fixed this configured horizon before every
# rollout.  It is known at reset; the realised terminal episode length is not
# a deployable policy feature and must never be used here.
ROUND12_EPISODE_HORIZON = 128


@dataclass(frozen=True)
class Round12TransitionSplit:
    """Numerical policy inputs/labels for one permitted split.

    Metadata is retained exclusively for sampling/audit.  It is deliberately
    not included in :meth:`model_arrays`, preventing it from becoming a model
    feature by accident.
    """

    split_name: str
    states: np.ndarray
    node_masks: np.ndarray
    actions: np.ndarray
    predicted_phi: np.ndarray
    oracle_phi: np.ndarray
    remaining_time: np.ndarray
    episode_id: np.ndarray
    scenario_id: np.ndarray
    behavior_generator: np.ndarray
    quality: np.ndarray
    difficulty: np.ndarray
    num_buoys: np.ndarray
    timestep: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray

    def __post_init__(self) -> None:
        if self.split_name not in ALLOWED_SPLITS:
            raise PermissionError(f"Round-12 may not load split {self.split_name!r}")
        count = int(self.states.shape[0])
        if count < 1 or self.states.ndim != 2:
            raise ValueError("states must be a nonempty [N,state_dim] matrix")
        if self.node_masks.ndim != 2 or self.node_masks.shape[0] != count:
            raise ValueError("node_masks must have one row per transition")
        if self.actions.shape != (count, ACTION_DIM):
            raise ValueError("Round-12 actions must have shape [N,2]")
        phi_dim = self.node_masks.shape[1] * 2 + 1
        for name in ("predicted_phi", "oracle_phi"):
            value = getattr(self, name)
            if value.shape != (count, phi_dim):
                raise ValueError(f"{name} has an invalid Task-Waypoint shape")
        if self.remaining_time.shape != (count, 1):
            raise ValueError("remaining_time must have shape [N,1]")
        for name in ("states", "node_masks", "actions", "predicted_phi", "oracle_phi", "remaining_time"):
            value = getattr(self, name)
            if value.dtype.kind == "O" or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite and non-object")
        if np.any(self.node_masks < 0.0) or np.any(self.node_masks > 1.0):
            raise ValueError("node masks must be binary-valued")
        invalid = ~self.node_masks.astype(bool)
        buoy_count = self.node_masks.shape[1]
        if np.any(self.predicted_phi[:, :buoy_count][invalid] != 0.0):
            raise ValueError("predicted service padding must be zero")
        if np.any(self.predicted_phi[:, buoy_count : 2 * buoy_count][invalid] != 0.0):
            raise ValueError("predicted progress padding must be zero")
        if np.any(self.oracle_phi[:, :buoy_count][invalid] != 0.0):
            raise ValueError("oracle service padding must be zero")
        if np.any(self.oracle_phi[:, buoy_count : 2 * buoy_count][invalid] != 0.0):
            raise ValueError("oracle progress padding must be zero")
        for name in (
            "episode_id", "scenario_id", "behavior_generator", "quality", "difficulty",
            "num_buoys", "timestep", "terminated", "truncated",
        ):
            if np.asarray(getattr(self, name)).shape != (count,):
                raise ValueError(f"metadata field {name!r} has an invalid shape")

    def __len__(self) -> int:
        return int(self.states.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[1])

    @property
    def max_buoys(self) -> int:
        return int(self.node_masks.shape[1])

    @property
    def phi_dim(self) -> int:
        return int(self.predicted_phi.shape[1])

    def model_arrays(self) -> Mapping[str, np.ndarray]:
        """Return only numeric arrays that can participate in policy training."""
        return MappingProxyType(
            {
                "states": self.states,
                "node_masks": self.node_masks,
                "actions": self.actions,
                "predicted_phi": self.predicted_phi,
                "oracle_phi": self.oracle_phi,
                "remaining_time": self.remaining_time,
            }
        )

    def sampling_metadata(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType(
            {
                "episode_id": self.episode_id,
                "scenario_id": self.scenario_id,
                "behavior_generator": self.behavior_generator,
                "quality": self.quality,
                "difficulty": self.difficulty,
                "num_buoys": self.num_buoys,
                "timestep": self.timestep,
                "terminated": self.terminated,
                "truncated": self.truncated,
            }
        )

    def take(self, indices: Sequence[int]) -> "Round12TransitionSplit":
        selected = np.asarray(indices, dtype=np.int64)
        if selected.ndim != 1 or selected.size < 1:
            raise ValueError("take requires a nonempty one-dimensional index array")
        return Round12TransitionSplit(
            split_name=self.split_name,
            **{name: np.asarray(getattr(self, name))[selected].copy() for name in (
                "states", "node_masks", "actions", "predicted_phi", "oracle_phi", "remaining_time",
                "episode_id", "scenario_id", "behavior_generator", "quality", "difficulty",
                "num_buoys", "timestep", "terminated", "truncated",
            )},
        )


@dataclass(frozen=True)
class Round12PolicyDataset:
    """The two approved data splits and their frozen-predictor provenance."""

    splits: Mapping[str, Round12TransitionSplit]
    predictor_sha256: str
    source_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        if set(self.splits) != set(ALLOWED_SPLITS):
            raise ValueError("Round-12 dataset must contain only train and validation")
        train, validation = self.train, self.validation
        dimensions = (train.state_dim, train.max_buoys, train.phi_dim)
        if dimensions != (validation.state_dim, validation.max_buoys, validation.phi_dim):
            raise ValueError("Round-12 dimensions disagree between train and validation")
        if self.predictor_sha256 != PREDICTOR_SHA256:
            raise RuntimeError("Round-12 dataset was not made by the selected predictor")
        if set(train.episode_id.tolist()) & set(validation.episode_id.tolist()):
            raise ValueError("episode leakage between train and validation")

    @property
    def train(self) -> Round12TransitionSplit:
        return self.splits["train"]

    @property
    def validation(self) -> Round12TransitionSplit:
        return self.splits["validation"]

    @property
    def state_dim(self) -> int:
        return self.train.state_dim

    @property
    def max_buoys(self) -> int:
        return self.train.max_buoys

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": ROUND12_DATA_SCHEMA,
            "allowed_splits": sorted(ALLOWED_SPLITS),
            "sample_counts": {name: len(value) for name, value in self.splits.items()},
            "state_dim": self.state_dim,
            "max_buoys": self.max_buoys,
            "action_dim": ACTION_DIM,
            "predictor_sha256": self.predictor_sha256,
            "metadata_model_input": False,
            "old_development_test_accessed": False,
            "untouched_holdout_accessed": False,
            "fresh_development_accessed": False,
        }


def _read_source_rows(source_dir: Path) -> tuple[list[dict[str, Any]], Mapping[str, np.ndarray]]:
    metadata_path = source_dir / "task_waypoint_metadata.jsonl"
    arrays_path = source_dir / "task_waypoint_samples_v1.npz"
    if not metadata_path.is_file() or not arrays_path.is_file():
        raise FileNotFoundError("Round-10 task-waypoint source package is incomplete")
    with metadata_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if len(rows) != len(arrays.get("H_task", ())):
        raise RuntimeError("Round-10 JSONL/NPZ row alignment failed")
    return rows, arrays


def _masked_phi(phi: np.ndarray, masks: np.ndarray) -> np.ndarray:
    result = np.asarray(phi, dtype=np.float32).copy()
    width = masks.shape[1]
    result[:, :width] *= masks
    result[:, width : 2 * width] *= masks
    return result


def _authorized_action_lookup(
    root: Path,
    selected_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Open action arrays only for Round-10 registered train/validation IDs."""
    source = root / ROUND12_ACTION_SOURCE_DIR
    manifest_path = source / "dataset_manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = {str(entry["episode_id"]): entry for entry in manifest.get("episodes", [])}
    wanted = {str(row["episode_id"]) for row in selected_rows}
    result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for episode_id in sorted(wanted):
        entry = entries.get(episode_id)
        if entry is None:
            raise RuntimeError("Round-10 row is not registered in the locked Round-7 data package")
        with np.load(source / str(entry["npz_file"]), allow_pickle=False) as archive:
            actions = archive["actions"].astype(np.float32, copy=True)
            terminated = archive["terminated"].astype(np.bool_, copy=True)
            truncated = archive["truncated"].astype(np.bool_, copy=True)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ValueError("authorized action source has an invalid action dimension")
        result[episode_id] = (actions, terminated, truncated)
    return MappingProxyType(result)


def load_round12_policy_dataset(
    root: Path,
    *,
    source_dir: Path | None = None,
    limits: Mapping[str, int] | None = None,
) -> Round12PolicyDataset:
    """Load only H=10 train/validation samples and predict Phi without gradients."""
    root = Path(root)
    source = root / (ROUND12_SOURCE_DIR if source_dir is None else Path(source_dir))
    rows, arrays = _read_source_rows(source)
    available: dict[str, list[int]] = {name: [] for name in ALLOWED_SPLITS}
    for index, row in enumerate(rows):
        split = str(row.get("split", ""))
        if split in FORBIDDEN_SPLITS or split not in ALLOWED_SPLITS:
            raise PermissionError("Round-12 source contains a prohibited split row")
        if int(row.get("H_task", arrays["H_task"][index])) != int(arrays["H_task"][index]):
            raise RuntimeError("Round-10 JSONL/NPZ horizon alignment failed")
        if int(arrays["H_task"][index]) != H_TASK:
            continue
        if row.get("task_goal") not in (None, []):
            raise RuntimeError("constant or task-goal content is prohibited in Round-12")
        available[split].append(index)
    selected = {name: list(values) for name, values in available.items()}
    for split in ALLOWED_SPLITS:
        if limits is not None and split in limits:
            limit = int(limits[split])
            if limit < 1:
                raise ValueError("Round-12 limits must be positive")
            selected[split] = selected[split][:limit]
        if not selected[split]:
            raise RuntimeError(f"Round-12 has no permitted H10 {split} samples")

    # Infer the predictor's fixed padding width from the complete permitted
    # package, not from a deliberately small smoke subset.  A smoke composed
    # solely of three-buoy rows still has to use the frozen 44-wide model.
    active = np.asarray(available["train"] + available["validation"], dtype=np.int64)
    state_dim = max(len(rows[int(index)]["state"]) for index in active)
    max_buoys = int(np.asarray(arrays["node_mask"])[active].shape[1])
    predictor_before = assert_selected_predictor(root)
    model = load_frozen_predictor(root, state_dim, max_buoys)
    assert_predictor_frozen(model)
    payload_config = model.config

    all_selected_rows = [rows[int(index)] for split in sorted(ALLOWED_SPLITS) for index in selected[split]]
    action_lookup = _authorized_action_lookup(root, all_selected_rows)
    splits: dict[str, Round12TransitionSplit] = {}
    for split in sorted(ALLOWED_SPLITS):
        indices = np.asarray(selected[split], dtype=np.int64)
        states = np.zeros((len(indices), state_dim), dtype=np.float32)
        for destination, source_index in enumerate(indices):
            value = np.asarray(rows[int(source_index)]["state"], dtype=np.float32)
            states[destination, : value.size] = value
        masks = np.asarray(arrays["node_mask"])[indices].astype(np.float32, copy=True)
        split_rows = [rows[int(index)] for index in indices]
        actions = np.asarray([
            action_lookup[str(row["episode_id"])][0][int(row["t"])]
            for row in split_rows
        ], dtype=np.float32)
        if actions.shape != (len(indices), ACTION_DIM):
            raise ValueError("Round-10 action labels are not two-dimensional")
        oracle = np.concatenate(
            (
                np.asarray(arrays["service_target"])[indices].astype(np.float32),
                np.asarray(arrays["queue_progress_service"])[indices].astype(np.float32)
                / float(payload_config.progress_scale),
                ((np.asarray(arrays["return_H"])[indices].astype(np.float32) - float(payload_config.return_mean))
                 / float(payload_config.return_std))[:, None],
            ),
            axis=1,
        )
        oracle = _masked_phi(oracle, masks)
        predicted = np.zeros_like(oracle)
        with torch.no_grad():
            for start in range(0, len(indices), 4096):
                stop = min(start + 4096, len(indices))
                predicted[start:stop] = predicted_phi(
                    model,
                    torch.from_numpy(states[start:stop]),
                    torch.from_numpy(masks[start:stop]),
                ).cpu().numpy()
        assert_predictor_frozen(model)
        splits[split] = Round12TransitionSplit(
            split_name=split,
            states=states,
            node_masks=masks,
            actions=actions,
            predicted_phi=predicted,
            oracle_phi=oracle,
            remaining_time=np.asarray(
                [[(ROUND12_EPISODE_HORIZON - int(row["t"])) / ROUND12_EPISODE_HORIZON]
                 for row in split_rows],
                dtype=np.float32,
            ),
            episode_id=np.asarray([str(row["episode_id"]) for row in split_rows]),
            scenario_id=np.asarray([str(row["scenario_id"]) for row in split_rows]),
            behavior_generator=np.asarray([str(row["behavior_generator"]) for row in split_rows]),
            quality=np.asarray([str(row.get("quality", "unavailable")) for row in split_rows]),
            difficulty=np.asarray([str(row.get("difficulty", "unavailable")) for row in split_rows]),
            num_buoys=np.asarray([int(row["num_buoys"]) for row in split_rows], dtype=np.int64),
            timestep=np.asarray([int(row["t"]) for row in split_rows], dtype=np.int64),
            terminated=np.asarray([
                bool(action_lookup[str(row["episode_id"])][1][int(row["t"])]) for row in split_rows
            ], dtype=np.bool_),
            truncated=np.asarray([
                bool(action_lookup[str(row["episode_id"])][2][int(row["t"])]) for row in split_rows
            ], dtype=np.bool_),
        )
    if sha256(predictor_before) != PREDICTOR_SHA256:
        raise RuntimeError("selected predictor hash changed while constructing Round-12 data")
    return Round12PolicyDataset(
        splits=MappingProxyType(splits),
        predictor_sha256=PREDICTOR_SHA256,
        source_hashes=MappingProxyType({
            "task_waypoint_metadata.jsonl": sha256(source / "task_waypoint_metadata.jsonl"),
            "task_waypoint_samples_v1.npz": sha256(source / "task_waypoint_samples_v1.npz"),
            "authorized_action_dataset_manifest.json": sha256(
                root / ROUND12_ACTION_SOURCE_DIR / "dataset_manifest.json"
            ),
        }),
    )


def build_cache(root: Path, split: str, output: Path) -> dict[str, Any]:
    """Persist an optional, provenance-bound Phi cache for one allowed split."""
    if split not in ALLOWED_SPLITS:
        raise PermissionError("Round-12 cache construction permits only train/validation")
    dataset = load_round12_policy_dataset(root)
    values = dataset.splits[split]
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        states=values.states,
        node_masks=values.node_masks,
        actions=values.actions,
        predicted_phi=values.predicted_phi,
        oracle_phi=values.oracle_phi,
        remaining_time=values.remaining_time,
        episode_id=values.episode_id,
        timestep=values.timestep,
    )
    return {
        "schema_version": ROUND12_DATA_SCHEMA,
        "split": split,
        "records": len(values),
        "predictor_sha256": PREDICTOR_SHA256,
        "true_future_labels_cached": True,
        "oracle_use_only": True,
        "old_development_test_accessed": False,
        "untouched_holdout_accessed": False,
    }


__all__ = [
    "ROUND12_DATA_SCHEMA",
    "ROUND12_ACTION_SOURCE_DIR",
    "ROUND12_EPISODE_HORIZON",
    "ROUND12_SOURCE_DIR",
    "Round12PolicyDataset",
    "Round12TransitionSplit",
    "build_cache",
    "load_round12_policy_dataset",
]
