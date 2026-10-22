"""Read-only Round-4 data adapter for the locked Round-3 maritime dataset.

The adapter deliberately keeps predictor tensors, supervised targets, and
evaluation-only metadata separate.  It never writes to the source directory
and it never re-splits transitions.  Spatial targets are reconstructed from
the raw physical episode arrays rather than from the normalized waypoint
labels stored by Round-3.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..maritime.dataset_v1 import load_dataset_v1
from ..maritime.waypoint_samples import (
    GOAL_SOURCES,
    SPLIT_NAMES,
    build_waypoint_sample_dataset,
    load_waypoint_samples,
    validate_waypoint_samples,
)


ROUND4_DATA_SCHEMA_VERSION = "round04_spatial_waypoint_data_v1"
CANONICAL_ROUND3_RUN_ID = "round03_pilot_canonical_20260730_230000"
CANONICAL_GOAL_SOURCE = "achieved_terminal_goal"
OUTPUT_DIM = 2

# These fields may participate in predictor construction.  Masks are applied
# before reduction/concatenation and are not themselves appended, except for
# the explicit current-node mask required by the experiment definition.
STATE_ONLY_SOURCE_FIELDS = (
    "state_t",
    "state_mask_t",
    "valid_node_mask_t",
)
GOAL_CONDITIONED_SOURCE_FIELDS = (
    "state_t",
    "state_mask_t",
    "goal",
    "goal_mask",
    "valid_node_mask_t",
)

# Evaluation metadata may remain available on Round4SpatialSplit, but none of
# the names below may be requested as an additional predictor feature.
PREDICTOR_LEAKAGE_DENYLIST = frozenset(
    {
        "waypoint_target_t_plus_k",
        "waypoint_mask_t_plus_k",
        "valid_node_mask_t_plus_k",
        "terminated_between_t_and_t_plus_k",
        "truncated_between_t_and_t_plus_k",
        "endpoint_terminated",
        "endpoint_truncated",
        "episode_length",
        "episode_id",
        "sample_id",
        "arrival_trace_id",
        "policy_id",
        "split",
        "split_name",
        "scenario_id",
        "schema_version",
        "trajectory_schema_version",
        "goal_spec_version",
        "seed",
        "t",
        "target_t",
        "target_transition_index",
        "actions",
        "future_actions",
        "rewards",
        "future_rewards",
        "returns",
        "future_arrivals",
        "future_velocity",
        "q_current_m",
        "q_future_m",
        "current_velocity_mps",
        "raw_delta_xy_m",
        "normalized_delta_xy",
        "quality_class",
        "num_buoys",
        "validation_statistics",
        "test_statistics",
    }
)

_REQUIRED_TOP_LEVEL_FILES = (
    "baseline_episode_metrics.csv",
    "dataset_manifest.json",
    "dataset_schema.json",
    "run_config.json",
    "waypoint_k_statistics.csv",
    "waypoint_sample_manifest.json",
    "waypoint_sample_schema.json",
    "waypoint_samples_train.npz",
    "waypoint_samples_validation.npz",
    "waypoint_samples_test.npz",
)


@dataclass(frozen=True)
class GoalSourceAudit:
    """Observed, non-synthetic variation for one materialized goal source."""

    goal_source: str
    materialization: str
    sample_count: int
    unique_goal_rows: int
    unique_goal_mask_rows: int
    unique_goal_and_mask_pairs: int
    unique_episode_goal_and_mask_pairs: int
    position_mask_true_rows: int
    changed_from_canonical_rows: int
    is_hindsight_goal: bool
    primary_experiment_eligible: bool
    limitation: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "goal_source": self.goal_source,
            "materialization": self.materialization,
            "sample_count": self.sample_count,
            "unique_goal_rows": self.unique_goal_rows,
            "unique_goal_mask_rows": self.unique_goal_mask_rows,
            "unique_goal_and_mask_pairs": (
                self.unique_goal_and_mask_pairs
            ),
            "unique_episode_goal_and_mask_pairs": (
                self.unique_episode_goal_and_mask_pairs
            ),
            "position_mask_true_rows": self.position_mask_true_rows,
            "changed_from_canonical_rows": (
                self.changed_from_canonical_rows
            ),
            "is_hindsight_goal": self.is_hindsight_goal,
            "primary_experiment_eligible": (
                self.primary_experiment_eligible
            ),
            "limitation": self.limitation,
        }


@dataclass(frozen=True)
class Round4SpatialSplit:
    """One immutable-split view of a fixed-horizon spatial prediction task."""

    split_name: str
    horizon: int
    goal_source: str
    goal_conditioned_inputs: np.ndarray
    state_only_inputs: np.ndarray
    normalized_delta_xy: np.ndarray
    raw_delta_xy_m: np.ndarray
    q_current_m: np.ndarray
    q_future_m: np.ndarray
    current_velocity_mps: np.ndarray
    masked_state_t: np.ndarray
    state_mask_t: np.ndarray
    masked_goal: np.ndarray
    goal_mask: np.ndarray
    valid_node_mask_t: np.ndarray
    sample_id: np.ndarray
    episode_id: np.ndarray
    scenario_id: np.ndarray
    seed: np.ndarray
    policy_id: np.ndarray
    arrival_trace_id: np.ndarray
    num_buoys: np.ndarray
    quality_class: np.ndarray
    t: np.ndarray
    target_t: np.ndarray
    target_transition_index: np.ndarray
    episode_length: np.ndarray
    endpoint_terminated: np.ndarray
    endpoint_truncated: np.ndarray

    def __post_init__(self) -> None:
        if self.split_name not in SPLIT_NAMES:
            raise ValueError("invalid Round-4 split name")
        if self.goal_source not in GOAL_SOURCES:
            raise ValueError("unsupported goal source")
        if isinstance(self.horizon, bool) or int(self.horizon) < 1:
            raise ValueError("horizon must be a positive integer")

        count = int(self.sample_id.shape[0])
        if count < 1:
            raise ValueError("each Round-4 split must contain samples")
        matrix_widths = {
            "goal_conditioned_inputs": self.goal_conditioned_inputs.shape,
            "state_only_inputs": self.state_only_inputs.shape,
            "normalized_delta_xy": self.normalized_delta_xy.shape,
            "raw_delta_xy_m": self.raw_delta_xy_m.shape,
            "q_current_m": self.q_current_m.shape,
            "q_future_m": self.q_future_m.shape,
            "current_velocity_mps": self.current_velocity_mps.shape,
            "masked_state_t": self.masked_state_t.shape,
            "state_mask_t": self.state_mask_t.shape,
            "masked_goal": self.masked_goal.shape,
            "goal_mask": self.goal_mask.shape,
            "valid_node_mask_t": self.valid_node_mask_t.shape,
        }
        for name, shape in matrix_widths.items():
            if len(shape) != 2 or int(shape[0]) != count:
                raise ValueError(
                    "{} must be a rank-two sample matrix".format(name)
                )
        for name in (
            "normalized_delta_xy",
            "raw_delta_xy_m",
            "q_current_m",
            "q_future_m",
            "current_velocity_mps",
        ):
            if getattr(self, name).shape[1] != OUTPUT_DIM:
                raise ValueError("{} must have width two".format(name))

        vector_names = (
            "sample_id",
            "episode_id",
            "scenario_id",
            "seed",
            "policy_id",
            "arrival_trace_id",
            "num_buoys",
            "quality_class",
            "t",
            "target_t",
            "target_transition_index",
            "episode_length",
            "endpoint_terminated",
            "endpoint_truncated",
        )
        for name in vector_names:
            if getattr(self, name).shape != (count,):
                raise ValueError("{} must have shape [B]".format(name))

        numeric_names = (
            "goal_conditioned_inputs",
            "state_only_inputs",
            "normalized_delta_xy",
            "raw_delta_xy_m",
            "q_current_m",
            "q_future_m",
            "current_velocity_mps",
            "masked_state_t",
            "masked_goal",
        )
        for name in numeric_names:
            if not np.all(np.isfinite(getattr(self, name))):
                raise ValueError("{} contains NaN or Inf".format(name))
        for name in vector_names + tuple(matrix_widths):
            if getattr(self, name).dtype.kind == "O":
                raise ValueError("object dtype is forbidden: {}".format(name))

        node_width = int(self.valid_node_mask_t.shape[1])
        state_width = int(self.masked_state_t.shape[1])
        goal_width = int(self.masked_goal.shape[1])
        if self.state_mask_t.shape != (count, state_width):
            raise ValueError("state mask width disagrees with state width")
        if self.goal_mask.shape != (count, goal_width):
            raise ValueError("goal mask width disagrees with goal width")
        if self.state_only_inputs.shape[1] != state_width + node_width:
            raise ValueError("state-only input width is inconsistent")
        if self.goal_conditioned_inputs.shape[1] != (
            state_width + goal_width + node_width
        ):
            raise ValueError("goal-conditioned input width is inconsistent")
        if np.any(self.masked_state_t[~self.state_mask_t] != 0.0):
            raise ValueError("masked state contains non-zero padding")
        if np.any(self.masked_goal[~self.goal_mask] != 0.0):
            raise ValueError("masked goal contains non-zero padding")
        if np.any(self.target_t != self.t + int(self.horizon)):
            raise ValueError("target_t does not equal t + horizon")
        if np.any(
            self.target_transition_index
            != self.t + int(self.horizon) - 1
        ):
            raise ValueError("target transition index is misaligned")
        if np.any(self.target_t > self.episode_length):
            raise ValueError("target crosses an episode boundary")
        if len(set(self.sample_id.tolist())) != count:
            raise ValueError("sample IDs are not unique inside the split")

    def __len__(self) -> int:
        return int(self.sample_id.shape[0])

    @property
    def targets(self) -> np.ndarray:
        """Alias used by trainers; output is normalized ``[dx, dy]``."""

        return self.normalized_delta_xy

    def inputs(self, goal_conditioned: bool = True) -> np.ndarray:
        """Return the already-audited predictor matrix."""

        return (
            self.goal_conditioned_inputs
            if goal_conditioned
            else self.state_only_inputs
        )

    def evaluation_metadata(self) -> Mapping[str, np.ndarray]:
        """Return grouping/provenance arrays, never predictor features."""

        values = {
            "split": np.full(
                (len(self),), self.split_name, dtype="<U10"
            ),
            "goal_source": np.full(
                (len(self),), self.goal_source, dtype="<U22"
            ),
            "sample_id": self.sample_id,
            "episode_id": self.episode_id,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "policy_id": self.policy_id,
            "arrival_trace_id": self.arrival_trace_id,
            "num_buoys": self.num_buoys,
            "quality_class": self.quality_class,
            "t": self.t,
            "target_t": self.target_t,
            "target_transition_index": self.target_transition_index,
            "episode_length": self.episode_length,
            "endpoint_terminated": self.endpoint_terminated,
            "endpoint_truncated": self.endpoint_truncated,
            "q_current_m": self.q_current_m,
            "q_future_m": self.q_future_m,
            "current_velocity_mps": self.current_velocity_mps,
            "raw_delta_xy_m": self.raw_delta_xy_m,
        }
        return MappingProxyType(values)


@dataclass(frozen=True)
class Round4SpatialDataset:
    """Complete fixed-horizon, fixed-split dataset returned to a trainer."""

    schema_version: str
    source_dir: Path
    source_run_id: str
    horizon: int
    goal_source: str
    max_speed_mps: float
    delta_t_seconds: float
    max_step_distance_m: float
    horizon_normalization_distance_m: float
    state_dim: int
    goal_dim: int
    max_num_buoys: int
    state_only_input_dim: int
    goal_conditioned_input_dim: int
    output_dim: int
    splits: Mapping[str, Round4SpatialSplit]
    schema: Mapping[str, Any]
    provenance: Mapping[str, Any]
    source_hashes: Mapping[str, str]
    goal_source_audits: Mapping[str, GoalSourceAudit]

    def __post_init__(self) -> None:
        if self.schema_version != ROUND4_DATA_SCHEMA_VERSION:
            raise ValueError("unexpected Round-4 adapter schema")
        if set(self.splits) != set(SPLIT_NAMES):
            raise ValueError("dataset must expose the three locked splits")
        seen_episode_ids = []
        seen_sample_ids = []
        for split_name in SPLIT_NAMES:
            split = self.splits[split_name]
            if split.split_name != split_name:
                raise ValueError("split mapping and split object disagree")
            if split.horizon != self.horizon:
                raise ValueError("split horizon disagrees with dataset")
            if split.goal_source != self.goal_source:
                raise ValueError("split goal source disagrees with dataset")
            if split.state_only_inputs.shape[1] != self.state_only_input_dim:
                raise ValueError("state-only dimension disagrees with split")
            if (
                split.goal_conditioned_inputs.shape[1]
                != self.goal_conditioned_input_dim
            ):
                raise ValueError(
                    "goal-conditioned dimension disagrees with split"
                )
            seen_episode_ids.append(set(split.episode_id.tolist()))
            seen_sample_ids.extend(split.sample_id.tolist())
        for index, left in enumerate(seen_episode_ids):
            for right in seen_episode_ids[index + 1 :]:
                if left & right:
                    raise ValueError("episode leakage across Round-4 splits")
        if len(seen_sample_ids) != len(set(seen_sample_ids)):
            raise ValueError("sample IDs leak across Round-4 splits")

    @property
    def train(self) -> Round4SpatialSplit:
        return self.splits["train"]

    @property
    def validation(self) -> Round4SpatialSplit:
        return self.splits["validation"]

    @property
    def test(self) -> Round4SpatialSplit:
        return self.splits["test"]

    def split(self, split_name: str) -> Round4SpatialSplit:
        if split_name not in SPLIT_NAMES:
            raise ValueError("split must be train, validation, or test")
        return self.splits[split_name]

    def input_dim(self, goal_conditioned: bool = True) -> int:
        return (
            self.goal_conditioned_input_dim
            if goal_conditioned
            else self.state_only_input_dim
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_run_id": self.source_run_id,
            "horizon": self.horizon,
            "goal_source": self.goal_source,
            "sample_counts": {
                name: len(self.splits[name]) for name in SPLIT_NAMES
            },
            "state_dim": self.state_dim,
            "goal_dim": self.goal_dim,
            "max_num_buoys": self.max_num_buoys,
            "state_only_input_dim": self.state_only_input_dim,
            "goal_conditioned_input_dim": (
                self.goal_conditioned_input_dim
            ),
            "output_dim": self.output_dim,
            "max_step_distance_m": self.max_step_distance_m,
            "horizon_normalization_distance_m": (
                self.horizon_normalization_distance_m
            ),
        }

    @property
    def manifest(self) -> Dict[str, Any]:
        """Return a JSON-compatible manifest fragment for Round-4 runners."""

        value = self.summary()
        value.update(
            {
                "source_dir": str(self.source_dir),
                "max_speed_mps": self.max_speed_mps,
                "delta_t_seconds": self.delta_t_seconds,
                "source_hashes": dict(self.source_hashes),
                "goal_source_audits": {
                    name: audit.as_dict()
                    for name, audit in self.goal_source_audits.items()
                },
                "provenance": dict(self.provenance),
                "training_performed": False,
            }
        )
        return value


def assert_predictor_fields_safe(
    feature_names: Iterable[str], *, goal_conditioned: bool
) -> Tuple[str, ...]:
    """Reject target/metadata leakage and unknown predictor recipes.

    The accepted value is exactly one of the two documented source recipes;
    accepting arbitrary extra fields would defeat the denylist.
    """

    names = tuple(str(value) for value in feature_names)
    denied = sorted(set(names) & PREDICTOR_LEAKAGE_DENYLIST)
    if denied:
        raise ValueError(
            "predictor feature leakage is forbidden: " + ", ".join(denied)
        )
    expected = (
        GOAL_CONDITIONED_SOURCE_FIELDS
        if goal_conditioned
        else STATE_ONLY_SOURCE_FIELDS
    )
    if names != expected:
        raise ValueError(
            "predictor source fields must be exactly {}".format(expected)
        )
    return names


def load_round4_spatial_dataset(
    source_dir: Any,
    horizon: int,
    goal_source: str = CANONICAL_GOAL_SOURCE,
    *,
    reachability_tolerance_m: float = 1.0e-6,
    verify_round3_lock: bool = True,
) -> Round4SpatialDataset:
    """Load and audit one fixed-horizon Round-4 spatial dataset.

    ``desired_goal`` is supported only through deterministic in-memory
    materialization from each authoritative episode's stored ``desired_goal``.
    No noise or synthetic position variation is added.  The primary locked
    experiment should continue to request ``achieved_terminal_goal``.
    """

    root = Path(source_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    horizon_value = _positive_integer("horizon", horizon)
    source = str(goal_source)
    if source not in GOAL_SOURCES:
        raise ValueError(
            "goal_source must be one of {}".format(GOAL_SOURCES)
        )
    tolerance = float(reachability_tolerance_m)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            "reachability_tolerance_m must be finite and non-negative"
        )

    for name in _REQUIRED_TOP_LEVEL_FILES:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    dataset_manifest = _read_json(root / "dataset_manifest.json")
    dataset_schema = _read_json(root / "dataset_schema.json")
    run_config = _read_json(root / "run_config.json")
    waypoint_manifest = _read_json(
        root / "waypoint_sample_manifest.json"
    )
    waypoint_schema = _read_json(root / "waypoint_sample_schema.json")
    _validate_manifest_paths(root, dataset_manifest, waypoint_manifest)

    lock, repository_root, source_hashes = _verify_source_hashes(
        root,
        dataset_manifest,
        verify_round3_lock=verify_round3_lock,
    )
    _validate_source_metadata(
        root,
        dataset_manifest,
        dataset_schema,
        run_config,
        waypoint_manifest,
        waypoint_schema,
        lock,
        repository_root,
        verify_round3_lock,
    )

    episodes = load_dataset_v1(root, split=None, verify_checksums=True)
    canonical_batches = load_waypoint_samples(
        root, split=None, verify_checksums=True
    )
    assignments = {
        str(key): str(value)
        for key, value in dataset_manifest["split_assignments"].items()
    }
    validate_waypoint_samples(
        canonical_batches,
        expected_episode_splits=assignments,
        source_episodes=episodes,
    )
    _validate_fixed_splits(
        episodes,
        canonical_batches,
        dataset_manifest,
        waypoint_manifest,
        assignments,
        lock,
        verify_round3_lock,
    )

    materialized = tuple(
        int(value) for value in waypoint_manifest["materialized_k_values"]
    )
    if horizon_value not in materialized:
        raise ValueError(
            "horizon {} is not materialized in the locked dataset".format(
                horizon_value
            )
        )
    expected_k_counts = _load_k_counts(
        root / "waypoint_k_statistics.csv"
    )
    if verify_round3_lock:
        locked_k_count = int(
            lock["waypoint_valid_samples"][str(horizon_value)]
        )
        observed_k_count = sum(
            expected_k_counts[(name, horizon_value)]
            for name in SPLIT_NAMES
        )
        if observed_k_count != locked_k_count:
            raise ValueError("locked waypoint K count changed")

    max_speed = _positive_float(
        "run_config.base_environment_config.max_speed",
        run_config["base_environment_config"]["max_speed"],
    )
    delta_t = _positive_float(
        "run_config.base_environment_config.delta_t",
        run_config["base_environment_config"]["delta_t"],
    )
    max_step_distance = max_speed * delta_t
    normalization_distance = horizon_value * max_step_distance

    n_max, state_dim, goal_dim = _schema_dimensions(
        waypoint_manifest, waypoint_schema
    )
    canonical_horizon_batches = _select_horizon(
        canonical_batches, horizon_value
    )
    desired_result = build_waypoint_sample_dataset(
        episodes,
        assignments,
        k_values=(horizon_value,),
        goal_source="desired_goal",
        max_num_buoys=n_max,
    )
    desired_batches = dict(desired_result.batches)
    validate_waypoint_samples(
        desired_batches,
        expected_episode_splits=assignments,
        source_episodes=episodes,
    )
    _assert_anchor_equivalence(
        canonical_horizon_batches, desired_batches
    )
    selected_batches = (
        canonical_horizon_batches
        if source == CANONICAL_GOAL_SOURCE
        else desired_batches
    )

    quality_by_episode = _load_quality_metadata(
        root / "baseline_episode_metrics.csv", episodes, assignments
    )
    by_episode = {str(value.episode_id): value for value in episodes}
    splits: Dict[str, Round4SpatialSplit] = {}
    reachability_max = 0.0
    reconstruction_max = 0.0
    for split_name in SPLIT_NAMES:
        batch = selected_batches[split_name]
        expected_count = expected_k_counts[(split_name, horizon_value)]
        if batch.sample_count != expected_count:
            raise ValueError(
                "K statistics disagree for split {!r}".format(split_name)
            )
        split, diagnostics = _build_split(
            split_name=split_name,
            batch=batch,
            episodes_by_id=by_episode,
            quality_by_episode=quality_by_episode,
            horizon=horizon_value,
            goal_source=source,
            max_speed_mps=max_speed,
            delta_t_seconds=delta_t,
            normalization_distance_m=normalization_distance,
            reachability_tolerance_m=tolerance,
        )
        splits[split_name] = split
        reachability_max = max(
            reachability_max,
            diagnostics["maximum_normalized_displacement_norm"],
        )
        reconstruction_max = max(
            reconstruction_max,
            diagnostics["raw_reconstruction_max_abs_error_m"],
        )

    goal_audits = _goal_source_audits(
        canonical_horizon_batches, desired_batches
    )
    state_only_dim = state_dim + n_max
    goal_conditioned_dim = state_dim + goal_dim + n_max
    adapter_schema = {
        "schema_version": ROUND4_DATA_SCHEMA_VERSION,
        "input_recipes": {
            "goal_conditioned": [
                "masked state_t",
                "masked goal",
                "valid_node_mask_t",
            ],
            "state_only": [
                "masked state_t",
                "valid_node_mask_t",
            ],
        },
        "dimensions": {
            "state_dim": state_dim,
            "goal_dim": goal_dim,
            "max_num_buoys": n_max,
            "goal_conditioned_input_dim": goal_conditioned_dim,
            "state_only_input_dim": state_only_dim,
            "output_dim": OUTPUT_DIM,
        },
        "target": {
            "raw": "q_future_m - q_current_m",
            "normalized": (
                "raw_delta_xy_m / "
                "(horizon * max_speed_mps * delta_t_seconds)"
            ),
        },
        "predictor_leakage_denylist": sorted(
            PREDICTOR_LEAKAGE_DENYLIST
        ),
    }
    schema = MappingProxyType(
        {
            "dataset_schema": dataset_schema,
            "waypoint_sample_schema": waypoint_schema,
            "round4_adapter_schema": adapter_schema,
        }
    )
    provenance = MappingProxyType(
        {
            "source_read_only": True,
            "source_run_id": str(run_config["run_id"]),
            "round3_lock_verified": bool(verify_round3_lock),
            "episode_checksums_verified": True,
            "waypoint_checksums_verified": True,
            "authoritative_episode_split_preserved": True,
            "goal_materialization": (
                "stored_canonical_waypoint_npz"
                if source == CANONICAL_GOAL_SOURCE
                else "in_memory_from_authoritative_episode_desired_goal"
            ),
            "hindsight_goal_conditioning": (
                source == CANONICAL_GOAL_SOURCE
            ),
            "maximum_normalized_displacement_norm": reachability_max,
            "raw_reconstruction_max_abs_error_m": reconstruction_max,
            "reachability_tolerance_m": tolerance,
            "fixed_split_episode_counts": {
                split_name: len(
                    set(splits[split_name].episode_id.tolist())
                )
                for split_name in SPLIT_NAMES
            },
            "fixed_split_sample_counts": {
                split_name: len(splits[split_name])
                for split_name in SPLIT_NAMES
            },
            "training_performed": False,
        }
    )
    dataset = Round4SpatialDataset(
        schema_version=ROUND4_DATA_SCHEMA_VERSION,
        source_dir=root,
        source_run_id=str(run_config["run_id"]),
        horizon=horizon_value,
        goal_source=source,
        max_speed_mps=max_speed,
        delta_t_seconds=delta_t,
        max_step_distance_m=max_step_distance,
        horizon_normalization_distance_m=normalization_distance,
        state_dim=state_dim,
        goal_dim=goal_dim,
        max_num_buoys=n_max,
        state_only_input_dim=state_only_dim,
        goal_conditioned_input_dim=goal_conditioned_dim,
        output_dim=OUTPUT_DIM,
        splits=MappingProxyType(dict(splits)),
        schema=schema,
        provenance=provenance,
        source_hashes=MappingProxyType(source_hashes),
        goal_source_audits=MappingProxyType(goal_audits),
    )
    return dataset


def load_round4_spatial_split(
    source_dir: Any,
    horizon: int,
    split: str,
    goal_source: str = CANONICAL_GOAL_SOURCE,
    **kwargs: Any
) -> Round4SpatialSplit:
    """Convenience wrapper returning one view after full-dataset auditing."""

    return load_round4_spatial_dataset(
        source_dir,
        horizon,
        goal_source=goal_source,
        **kwargs
    ).split(split)


def _build_split(
    *,
    split_name: str,
    batch: Any,
    episodes_by_id: Mapping[str, Any],
    quality_by_episode: Mapping[str, str],
    horizon: int,
    goal_source: str,
    max_speed_mps: float,
    delta_t_seconds: float,
    normalization_distance_m: float,
    reachability_tolerance_m: float,
) -> Tuple[Round4SpatialSplit, Dict[str, float]]:
    assert_predictor_fields_safe(
        GOAL_CONDITIONED_SOURCE_FIELDS, goal_conditioned=True
    )
    assert_predictor_fields_safe(
        STATE_ONLY_SOURCE_FIELDS, goal_conditioned=False
    )
    count = batch.sample_count
    q_current = np.empty((count, OUTPUT_DIM), dtype=np.float64)
    q_future = np.empty((count, OUTPUT_DIM), dtype=np.float64)
    current_velocity = np.empty((count, OUTPUT_DIM), dtype=np.float64)

    for episode_id in np.unique(batch.episode_id):
        key = str(episode_id)
        if key not in episodes_by_id:
            raise ValueError("waypoint sample references a missing episode")
        episode = episodes_by_id[key]
        rows = np.flatnonzero(batch.episode_id == episode_id)
        starts = np.asarray(batch.t[rows], dtype=np.int64)
        target_indices = np.asarray(
            batch.target_transition_index[rows], dtype=np.int64
        )
        if np.any(starts < 0) or np.any(
            target_indices >= int(episode.episode_length)
        ):
            raise ValueError("spatial target index crosses an episode")
        q_current[rows] = np.asarray(
            episode.uav_positions_m[starts], dtype=np.float64
        )
        q_future[rows] = np.asarray(
            episode.next_uav_positions_m[target_indices], dtype=np.float64
        )
        velocity_indices = _velocity_indices(episode)
        normalized_velocity = np.asarray(
            episode.observations[starts][:, velocity_indices],
            dtype=np.float64,
        )
        current_velocity[rows] = normalized_velocity * max_speed_mps
        _validate_current_velocity(
            episode,
            starts,
            current_velocity[rows],
            delta_t_seconds,
            max_speed_mps,
            reachability_tolerance_m,
        )
        _validate_normalized_positions(
            episode,
            starts,
            target_indices,
            batch,
            rows,
            q_current[rows],
            q_future[rows],
        )

    raw_delta = q_future - q_current
    distances = np.linalg.norm(raw_delta, axis=1)
    if np.any(
        distances > normalization_distance_m + reachability_tolerance_m
    ):
        raise ValueError("spatial target violates horizon reachability")
    normalized64 = raw_delta / normalization_distance_m
    normalized = normalized64.astype(np.float32)
    reconstruction_error = float(
        np.max(
            np.abs(
                normalized.astype(np.float64)
                * normalization_distance_m
                - raw_delta
            )
        )
    )

    state_mask = np.asarray(batch.state_mask_t, dtype=np.bool_)
    goal_mask = np.asarray(batch.goal_mask, dtype=np.bool_)
    node_mask = np.asarray(batch.valid_node_mask_t, dtype=np.bool_)
    masked_state = np.where(
        state_mask, np.asarray(batch.state_t, dtype=np.float32), 0.0
    ).astype(np.float32, copy=False)
    masked_goal = np.where(
        goal_mask, np.asarray(batch.goal, dtype=np.float32), 0.0
    ).astype(np.float32, copy=False)
    node_features = node_mask.astype(np.float32)
    state_only_inputs = np.concatenate(
        (masked_state, node_features), axis=1
    ).astype(np.float32, copy=False)
    goal_conditioned_inputs = np.concatenate(
        (masked_state, masked_goal, node_features), axis=1
    ).astype(np.float32, copy=False)

    quality = np.asarray(
        [quality_by_episode[str(value)] for value in batch.episode_id]
    )
    result = Round4SpatialSplit(
        split_name=split_name,
        horizon=horizon,
        goal_source=goal_source,
        goal_conditioned_inputs=goal_conditioned_inputs,
        state_only_inputs=state_only_inputs,
        normalized_delta_xy=normalized,
        raw_delta_xy_m=raw_delta,
        q_current_m=q_current,
        q_future_m=q_future,
        current_velocity_mps=current_velocity,
        masked_state_t=masked_state,
        state_mask_t=state_mask.copy(),
        masked_goal=masked_goal,
        goal_mask=goal_mask.copy(),
        valid_node_mask_t=node_mask.copy(),
        sample_id=batch.sample_id.copy(),
        episode_id=batch.episode_id.copy(),
        scenario_id=batch.scenario_id.copy(),
        seed=batch.seed.copy(),
        policy_id=batch.policy_id.copy(),
        arrival_trace_id=batch.arrival_trace_id.copy(),
        num_buoys=batch.num_buoys.copy(),
        quality_class=quality,
        t=batch.t.copy(),
        target_t=batch.target_t.copy(),
        target_transition_index=batch.target_transition_index.copy(),
        episode_length=batch.episode_length.copy(),
        endpoint_terminated=batch.endpoint_terminated.copy(),
        endpoint_truncated=batch.endpoint_truncated.copy(),
    )
    return result, {
        "maximum_normalized_displacement_norm": float(
            np.max(np.linalg.norm(normalized64, axis=1))
        ),
        "raw_reconstruction_max_abs_error_m": reconstruction_error,
    }


def _select_horizon(
    batches: Mapping[str, Any], horizon: int
) -> Dict[str, Any]:
    """Rebuild one fixed-K view without changing episode assignments."""

    selected: Dict[str, Any] = {}
    for split_name in SPLIT_NAMES:
        batch = batches[split_name]
        indices = np.flatnonzero(batch.k == horizon)
        arrays = {
            name: np.asarray(array)[indices].copy()
            for name, array in batch.arrays().items()
        }
        selected[split_name] = type(batch)(
            split_name=split_name,
            max_num_buoys=batch.max_num_buoys,
            **arrays
        )
    return selected


def _assert_anchor_equivalence(
    canonical: Mapping[str, Any], desired: Mapping[str, Any]
) -> None:
    excluded = {"goal", "goal_mask", "goal_source"}
    for split_name in SPLIT_NAMES:
        left = canonical[split_name]
        right = desired[split_name]
        if left.sample_count != right.sample_count:
            raise ValueError("goal materializations have different anchors")
        for name in left.arrays():
            if name in excluded:
                continue
            if not np.array_equal(
                getattr(left, name), getattr(right, name)
            ):
                raise ValueError(
                    "desired_goal changed non-goal field {!r}".format(name)
                )
        if set(right.goal_source.tolist()) != {"desired_goal"}:
            raise ValueError("desired goal labels were silently mixed")


def _goal_source_audits(
    canonical: Mapping[str, Any], desired: Mapping[str, Any]
) -> Dict[str, GoalSourceAudit]:
    result: Dict[str, GoalSourceAudit] = {}
    for source, batches in (
        (CANONICAL_GOAL_SOURCE, canonical),
        ("desired_goal", desired),
    ):
        goals = np.concatenate(
            [batches[name].goal for name in SPLIT_NAMES], axis=0
        )
        masks = np.concatenate(
            [batches[name].goal_mask for name in SPLIT_NAMES], axis=0
        )
        episode_ids = np.concatenate(
            [batches[name].episode_id for name in SPLIT_NAMES], axis=0
        )
        canonical_goals = np.concatenate(
            [canonical[name].goal for name in SPLIT_NAMES], axis=0
        )
        canonical_masks = np.concatenate(
            [canonical[name].goal_mask for name in SPLIT_NAMES], axis=0
        )
        goal_keys = {_array_key(row) for row in goals}
        mask_keys = {_array_key(row) for row in masks}
        pair_keys = {
            (_array_key(goal), _array_key(mask))
            for goal, mask in zip(goals, masks)
        }
        episode_pairs: Dict[str, Tuple[bytes, bytes]] = {}
        for episode_id, goal, mask in zip(episode_ids, goals, masks):
            key = (_array_key(goal), _array_key(mask))
            prior = episode_pairs.setdefault(str(episode_id), key)
            if prior != key:
                raise ValueError("episode goal changes across anchors")
        changed = np.any(goals != canonical_goals, axis=1) | np.any(
            masks != canonical_masks, axis=1
        )
        if source == CANONICAL_GOAL_SOURCE:
            limitation = (
                "Hindsight goal contains terminal outcome information; "
                "results are not causal online forecasts."
            )
            primary = True
            materialization = "stored_canonical_waypoint_npz"
        else:
            limitation = (
                "Desired goal has no spatial position target and only the "
                "variation stored by the source episodes is retained; no "
                "synthetic variation is permitted."
            )
            primary = False
            materialization = (
                "in_memory_from_authoritative_episode_desired_goal"
            )
        result[source] = GoalSourceAudit(
            goal_source=source,
            materialization=materialization,
            sample_count=int(goals.shape[0]),
            unique_goal_rows=len(goal_keys),
            unique_goal_mask_rows=len(mask_keys),
            unique_goal_and_mask_pairs=len(pair_keys),
            unique_episode_goal_and_mask_pairs=len(set(episode_pairs.values())),
            position_mask_true_rows=int(
                np.count_nonzero(np.all(masks[:, :2], axis=1))
            ),
            changed_from_canonical_rows=int(np.count_nonzero(changed)),
            is_hindsight_goal=(source == CANONICAL_GOAL_SOURCE),
            primary_experiment_eligible=primary,
            limitation=limitation,
        )
    return result


def _velocity_indices(episode: Any) -> Tuple[int, int]:
    schema = episode.observation_schema
    names = schema.get("base_feature_names")
    if not isinstance(names, list):
        raise ValueError("observation schema lacks base feature names")
    required = (
        "normalized_uav_velocity_x",
        "normalized_uav_velocity_y",
    )
    try:
        return (names.index(required[0]), names.index(required[1]))
    except ValueError as error:
        raise ValueError(
            "observation schema lacks normalized UAV velocity"
        ) from error


def _validate_current_velocity(
    episode: Any,
    starts: np.ndarray,
    velocity: np.ndarray,
    delta_t_seconds: float,
    max_speed_mps: float,
    tolerance_m: float,
) -> None:
    if np.any(
        np.linalg.norm(velocity, axis=1)
        > max_speed_mps + tolerance_m / delta_t_seconds
    ):
        raise ValueError("current velocity exceeds configured max_speed")
    expected = np.zeros_like(velocity)
    positive = starts > 0
    if np.any(positive):
        current = np.asarray(
            episode.uav_positions_m[starts[positive]], dtype=np.float64
        )
        previous = np.asarray(
            episode.uav_positions_m[starts[positive] - 1],
            dtype=np.float64,
        )
        expected[positive] = (current - previous) / delta_t_seconds
    if not np.allclose(velocity, expected, rtol=1.0e-5, atol=2.0e-6):
        raise ValueError("current velocity disagrees with position history")


def _validate_normalized_positions(
    episode: Any,
    starts: np.ndarray,
    target_indices: np.ndarray,
    batch: Any,
    rows: np.ndarray,
    q_current: np.ndarray,
    q_future: np.ndarray,
) -> None:
    current_normalized = np.column_stack(
        (
            2.0 * q_current[:, 0] / float(episode.area_width) - 1.0,
            2.0 * q_current[:, 1] / float(episode.area_height) - 1.0,
        )
    )
    future_normalized = np.column_stack(
        (
            2.0 * q_future[:, 0] / float(episode.area_width) - 1.0,
            2.0 * q_future[:, 1] / float(episode.area_height) - 1.0,
        )
    )
    if not np.allclose(
        batch.state_t[rows, :2],
        current_normalized,
        rtol=2.0e-6,
        atol=2.0e-6,
    ):
        raise ValueError("q_current disagrees with waypoint state position")
    if not np.allclose(
        batch.waypoint_target_t_plus_k[rows, :2],
        future_normalized,
        rtol=2.0e-6,
        atol=2.0e-6,
    ):
        raise ValueError("q_future disagrees with waypoint target position")
    if np.any(batch.t[rows] != starts) or np.any(
        batch.target_transition_index[rows] != target_indices
    ):
        raise ValueError("row indexing changed during spatial reconstruction")


def _schema_dimensions(
    manifest: Mapping[str, Any], schema: Mapping[str, Any]
) -> Tuple[int, int, int]:
    dimensions = schema.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise ValueError("waypoint schema lacks dimensions")
    n_max = _positive_integer(
        "max_num_buoys", dimensions.get("max_num_buoys")
    )
    state_dim = _positive_integer("state_dim", dimensions.get("state_dim"))
    goal_dim = _positive_integer(
        "waypoint_dim", dimensions.get("waypoint_dim")
    )
    expected = {
        "max_num_buoys": n_max,
        "state_dim": state_dim,
        "waypoint_dim": goal_dim,
    }
    for name, value in expected.items():
        if int(manifest.get(name, -1)) != value:
            raise ValueError("waypoint manifest dimension mismatch")
    layout = schema.get("state_layout")
    if not isinstance(layout, list) or not layout:
        raise ValueError("waypoint schema lacks state layout")
    intervals = sorted(
        (int(item["start"]), int(item["stop"])) for item in layout
    )
    cursor = 0
    for start, stop in intervals:
        if start != cursor or stop <= start:
            raise ValueError("state layout is not contiguous")
        cursor = stop
    if cursor != state_dim:
        raise ValueError("state layout does not cover state_dim")
    return n_max, state_dim, goal_dim


def _validate_fixed_splits(
    episodes: Sequence[Any],
    batches: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    waypoint_manifest: Mapping[str, Any],
    assignments: Mapping[str, str],
    lock: Optional[Mapping[str, Any]],
    verify_lock: bool,
) -> None:
    episode_ids = {str(value.episode_id) for value in episodes}
    if set(assignments) != episode_ids:
        raise ValueError("dataset split assignment coverage is incomplete")
    waypoint_assignments = {
        str(key): str(value)
        for key, value in waypoint_manifest[
            "source_episode_split_assignments"
        ].items()
    }
    if waypoint_assignments != dict(assignments):
        raise ValueError("waypoint and episode splits disagree")
    manifest_entry_ids = set()
    for entry in dataset_manifest["episodes"]:
        episode_id = str(entry["episode_id"])
        manifest_entry_ids.add(episode_id)
        if str(entry["split"]) != assignments[episode_id]:
            raise ValueError("episode manifest split was changed")
    if manifest_entry_ids != episode_ids:
        raise ValueError("episode manifest coverage is incomplete")

    split_episode_sets = {}
    split_transition_counts = {}
    for split_name in SPLIT_NAMES:
        split_ids = {
            value for value, name in assignments.items() if name == split_name
        }
        split_episode_sets[split_name] = split_ids
        split_transition_counts[split_name] = sum(
            int(value.episode_length)
            for value in episodes
            if str(value.episode_id) in split_ids
        )
        batch_ids = set(map(str, batches[split_name].episode_id.tolist()))
        if batch_ids != split_ids:
            raise ValueError("waypoint split episode coverage changed")
        if np.any(batches[split_name].split != split_name):
            raise ValueError("waypoint split label changed")
    for index, left_name in enumerate(SPLIT_NAMES):
        for right_name in SPLIT_NAMES[index + 1 :]:
            if split_episode_sets[left_name] & split_episode_sets[right_name]:
                raise ValueError("episode leakage across source splits")

    if verify_lock:
        if lock is None:
            raise ValueError("Round-3 lock was not loaded")
        acceptance = lock["acceptance"]
        names = {
            "train": "train",
            "validation": "validation",
            "test": "test",
        }
        for split_name, lock_name in names.items():
            if len(split_episode_sets[split_name]) != int(
                acceptance["pilot_{}_episodes".format(lock_name)]
            ):
                raise ValueError("locked episode split count changed")
            if split_transition_counts[split_name] != int(
                acceptance["pilot_{}_transitions".format(lock_name)]
            ):
                raise ValueError("locked transition split count changed")


def _load_quality_metadata(
    path: Path,
    episodes: Sequence[Any],
    assignments: Mapping[str, str],
) -> Dict[str, str]:
    values: Dict[str, str] = {}
    episode_by_id = {str(value.episode_id): value for value in episodes}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            episode_id = str(row["episode_id"])
            if episode_id in values:
                raise ValueError("duplicate episode quality row")
            if episode_id not in episode_by_id:
                raise ValueError("quality metadata references unknown episode")
            episode = episode_by_id[episode_id]
            if str(row["split"]) != assignments[episode_id]:
                raise ValueError("quality metadata split changed")
            if str(row["policy_id"]) != str(episode.policy_id):
                raise ValueError("quality metadata policy changed")
            if int(row["num_buoys"]) != int(episode.num_buoys):
                raise ValueError("quality metadata buoy scale changed")
            label = str(row["quality_class"])
            if label not in {"high", "mid", "low"}:
                raise ValueError("unknown trajectory quality class")
            values[episode_id] = label
    if set(values) != set(episode_by_id):
        raise ValueError("quality metadata does not cover every episode")
    return values


def _load_k_counts(path: Path) -> Dict[Tuple[str, int], int]:
    result: Dict[Tuple[str, int], int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (str(row["split"]), int(row["k"]))
            if key in result:
                raise ValueError("duplicate waypoint K statistic")
            result[key] = int(row["valid_sample_count"])
            if int(row["invalid_padding_count"]) != 0:
                raise ValueError("locked K statistics contain padding anchors")
            if int(row["invalid_internal_terminated_count"]) != 0:
                raise ValueError("locked K statistics cross termination")
            if int(row["invalid_internal_truncated_count"]) != 0:
                raise ValueError("locked K statistics cross truncation")
    return result


def _validate_source_metadata(
    root: Path,
    dataset_manifest: Mapping[str, Any],
    dataset_schema: Mapping[str, Any],
    run_config: Mapping[str, Any],
    waypoint_manifest: Mapping[str, Any],
    waypoint_schema: Mapping[str, Any],
    lock: Optional[Mapping[str, Any]],
    repository_root: Path,
    verify_lock: bool,
) -> None:
    if str(run_config.get("run_id")) != CANONICAL_ROUND3_RUN_ID:
        raise ValueError("source run is not the Round-3 canonical pilot")
    for mapping, label in (
        (dataset_manifest, "dataset manifest"),
        (dataset_schema, "dataset schema"),
        (run_config, "run config"),
        (waypoint_manifest, "waypoint manifest"),
        (waypoint_schema, "waypoint schema"),
    ):
        if mapping.get("training_performed") is not False:
            raise ValueError("{} must record no training".format(label))
    if dataset_manifest.get("dataset_schema_version") != "v1":
        raise ValueError("Round-4 requires dataset schema v1")
    if dataset_schema.get("dataset_schema_version") != "v1":
        raise ValueError("dataset schema file is not v1")
    if waypoint_manifest.get("goal_source") != CANONICAL_GOAL_SOURCE:
        raise ValueError("canonical waypoint goal source was changed")
    if waypoint_manifest.get("waypoint_sample_schema_version") != (
        "waypoint_schema_v1"
    ):
        raise ValueError("Round-4 requires waypoint schema v1")
    if waypoint_schema.get("waypoint_sample_schema_version") != (
        "waypoint_schema_v1"
    ):
        raise ValueError("waypoint schema file is not v1")
    digest = hashlib.sha256(
        json.dumps(
            dataset_manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if digest.lower() != str(
        waypoint_manifest["source_dataset_manifest_sha256"]
    ).lower():
        raise ValueError("waypoint source-manifest digest mismatch")
    if int(run_config["episodes"]) != int(
        dataset_manifest["episode_count"]
    ):
        raise ValueError("run config episode count changed")
    if int(run_config["transitions"]) != int(
        dataset_manifest["transition_count"]
    ):
        raise ValueError("run config transition count changed")
    if verify_lock:
        if lock is None or lock.get("status") != "completed":
            raise ValueError("Round-3 lock is not completed")
        relative = root.relative_to(repository_root).as_posix()
        if relative != str(lock["canonical_outputs"]["pilot_data"]):
            raise ValueError("source path is not the locked canonical pilot")


def _verify_source_hashes(
    root: Path,
    dataset_manifest: Mapping[str, Any],
    *,
    verify_round3_lock: bool,
) -> Tuple[Optional[Mapping[str, Any]], Path, Dict[str, str]]:
    round_dir = root.parent.parent
    repository_root = round_dir.parent.resolve()
    lock_path = round_dir / "ROUND_LOCK.json"
    lock: Optional[Mapping[str, Any]] = None
    result: Dict[str, str] = {}
    if verify_round3_lock:
        lock = _read_json(lock_path)
        expected = lock.get("canonical_output_sha256")
        if not isinstance(expected, Mapping):
            raise ValueError("Round-3 lock lacks canonical output hashes")
        prefix = root.relative_to(repository_root).as_posix() + "/"
        covered = set()
        for relative, expected_hash in expected.items():
            relative_string = str(relative)
            if not relative_string.startswith(prefix):
                continue
            path = _safe_child(repository_root, relative_string)
            actual = _sha256(path)
            if actual.lower() != str(expected_hash).lower():
                raise ValueError(
                    "locked SHA-256 mismatch: {}".format(relative_string)
                )
            result[relative_string] = actual
            covered.add(relative_string)
        required = {prefix + name for name in _REQUIRED_TOP_LEVEL_FILES}
        if not required.issubset(covered):
            raise ValueError("Round-3 lock does not cover required artifacts")
        result[lock_path.relative_to(repository_root).as_posix()] = _sha256(
            lock_path
        )
    else:
        for name in _REQUIRED_TOP_LEVEL_FILES:
            result[name] = _sha256(root / name)

    # load_dataset_v1 recomputes every checksum below.  Recording the manifest
    # values here exposes the complete verified provenance without a third
    # redundant file read after that loader succeeds.
    for entry in dataset_manifest.get("episodes", []):
        for path_key, hash_key in (
            ("npz_file", "npz_sha256"),
            ("metadata_file", "metadata_sha256"),
        ):
            result[str(entry[path_key])] = str(entry[hash_key]).lower()
    return lock, repository_root, result


def _validate_manifest_paths(
    root: Path,
    dataset_manifest: Mapping[str, Any],
    waypoint_manifest: Mapping[str, Any],
) -> None:
    episodes = dataset_manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("dataset manifest contains no episodes")
    for entry in episodes:
        if not isinstance(entry, Mapping):
            raise ValueError("invalid episode manifest entry")
        _safe_child(root, str(entry["npz_file"]))
        _safe_child(root, str(entry["metadata_file"]))
    sample_files = waypoint_manifest.get("sample_files")
    if not isinstance(sample_files, Mapping):
        raise ValueError("waypoint manifest lacks sample files")
    for split_name in SPLIT_NAMES:
        _safe_child(root, str(sample_files[split_name]["path"]))
    for name in ("schema_file", "k_statistics_file"):
        _safe_child(root, str(waypoint_manifest[name]["path"]))


def _safe_child(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("artifact path escapes its source root") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _read_json(path: Path) -> Dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError("non-finite JSON constant: {}".format(value))

    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object: {}".format(path))
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("{} must be a positive integer".format(name))
    integer = int(value)
    if integer < 1 or integer != value:
        raise ValueError("{} must be a positive integer".format(name))
    return integer


def _positive_float(name: str, value: Any) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError("{} must be finite and positive".format(name))
    return number


def _array_key(value: np.ndarray) -> bytes:
    array = np.ascontiguousarray(value)
    return (
        str(array.dtype).encode("ascii")
        + repr(array.shape).encode("ascii")
        + array.tobytes()
    )


__all__ = [
    "CANONICAL_GOAL_SOURCE",
    "CANONICAL_ROUND3_RUN_ID",
    "GOAL_CONDITIONED_SOURCE_FIELDS",
    "GoalSourceAudit",
    "OUTPUT_DIM",
    "PREDICTOR_LEAKAGE_DENYLIST",
    "ROUND4_DATA_SCHEMA_VERSION",
    "Round4SpatialDataset",
    "Round4SpatialSplit",
    "STATE_ONLY_SOURCE_FIELDS",
    "assert_predictor_fields_safe",
    "load_round4_spatial_dataset",
    "load_round4_spatial_split",
]
