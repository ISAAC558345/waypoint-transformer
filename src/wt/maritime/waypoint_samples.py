"""Schema-v1 supervised Waypoint sample construction and safe persistence.

This module materializes labels only.  It contains no model, trainable
parameter, optimizer, loss, backward pass, or training loop.

For an anchor transition ``t`` and horizon ``K``, the exact future state is
``next_observations[t + K - 1]``.  Samples never cross an episode, split,
padding entry, or an internal terminated/truncated transition.  A done flag on
the target transition itself is valid because it leads to the endpoint state
``s_(t+K)``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


WAYPOINT_SAMPLE_SCHEMA_VERSION = "waypoint_schema_v1"
TRAJECTORY_SCHEMA_VERSION = "v1"
GOAL_SPEC_VERSION = "v1"
DEFAULT_GOAL_SOURCE = "achieved_terminal_goal"
GOAL_SOURCES = ("achieved_terminal_goal", "desired_goal")
SUPPORTED_K_VALUES = (1, 5, 10, 20, 30)
SPLIT_NAMES = ("train", "validation", "test")

OBSERVATION_BASE_DIM = 9
OBSERVATION_NODE_DIM = 7
OBSERVATION_POSITION_SLICE = slice(0, 2)
OBSERVATION_ENERGY_INDEX = 4
NODE_QUEUE_INDEX = 4
NODE_COMPLETION_INDEX = 6

K_STATISTICS_COLUMNS = (
    "split",
    "k",
    "episode_count",
    "transition_count",
    "candidate_anchor_count",
    "valid_sample_count",
    "valid_sample_ratio",
    "invalid_short_horizon_count",
    "invalid_padding_count",
    "invalid_internal_terminated_count",
    "invalid_internal_truncated_count",
    "endpoint_terminated_count",
    "endpoint_truncated_count",
)


def waypoint_dim(max_num_buoys: int) -> int:
    """Return ``W = 3 + 2*N_max`` for waypoint schema v1."""
    value = _positive_integer("max_num_buoys", max_num_buoys)
    return 3 + 2 * value


def padded_observation_dim(max_num_buoys: int) -> int:
    """Return ``D_obs,max = 9 + 7*N_max``."""
    value = _positive_integer("max_num_buoys", max_num_buoys)
    return OBSERVATION_BASE_DIM + OBSERVATION_NODE_DIM * value


def waypoint_state_dim(max_num_buoys: int) -> int:
    """Return the input width ``W + D_obs,max + N_max``."""
    value = _positive_integer("max_num_buoys", max_num_buoys)
    return waypoint_dim(value) + padded_observation_dim(value) + value


@dataclass
class WaypointSampleBatch:
    """One split's fixed-width, pickle-free supervised sample arrays."""

    split_name: str
    max_num_buoys: int
    sample_id: np.ndarray
    split: np.ndarray
    episode_id: np.ndarray
    scenario_id: np.ndarray
    seed: np.ndarray
    policy_id: np.ndarray
    arrival_trace_id: np.ndarray
    num_buoys: np.ndarray
    episode_length: np.ndarray
    t: np.ndarray
    k: np.ndarray
    target_t: np.ndarray
    target_transition_index: np.ndarray
    state_t: np.ndarray
    state_mask_t: np.ndarray
    valid_node_mask_t: np.ndarray
    goal: np.ndarray
    goal_mask: np.ndarray
    goal_source: np.ndarray
    waypoint_target_t_plus_k: np.ndarray
    waypoint_mask_t_plus_k: np.ndarray
    valid_node_mask_t_plus_k: np.ndarray
    terminated_between_t_and_t_plus_k: np.ndarray
    truncated_between_t_and_t_plus_k: np.ndarray
    endpoint_terminated: np.ndarray
    endpoint_truncated: np.ndarray
    trajectory_schema_version: np.ndarray
    goal_spec_version: np.ndarray
    schema_version: np.ndarray

    def __post_init__(self) -> None:
        self.split_name = _split_name(self.split_name)
        self.max_num_buoys = _positive_integer(
            "max_num_buoys", self.max_num_buoys
        )

        unicode_names = (
            "sample_id",
            "split",
            "episode_id",
            "scenario_id",
            "policy_id",
            "arrival_trace_id",
            "goal_source",
            "trajectory_schema_version",
            "goal_spec_version",
            "schema_version",
        )
        integer_names = (
            "seed",
            "num_buoys",
            "episode_length",
            "t",
            "k",
            "target_t",
            "target_transition_index",
        )
        boolean_names = (
            "state_mask_t",
            "valid_node_mask_t",
            "goal_mask",
            "waypoint_mask_t_plus_k",
            "valid_node_mask_t_plus_k",
            "terminated_between_t_and_t_plus_k",
            "truncated_between_t_and_t_plus_k",
            "endpoint_terminated",
            "endpoint_truncated",
        )
        float_names = ("state_t", "goal", "waypoint_target_t_plus_k")

        for name in unicode_names:
            setattr(self, name, _unicode_array(name, getattr(self, name)))
        for name in integer_names:
            setattr(
                self,
                name,
                _typed_array(name, getattr(self, name), np.int64),
            )
        for name in boolean_names:
            setattr(
                self,
                name,
                _typed_array(name, getattr(self, name), np.bool_),
            )
        for name in float_names:
            value = _typed_array(name, getattr(self, name), np.float32)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains NaN or Inf")
            setattr(self, name, value)
        self._validate()

    @property
    def sample_count(self) -> int:
        return int(self.sample_id.shape[0])

    @property
    def state_dim(self) -> int:
        return waypoint_state_dim(self.max_num_buoys)

    @property
    def waypoint_dim(self) -> int:
        return waypoint_dim(self.max_num_buoys)

    @property
    def padded_observation_dim(self) -> int:
        return padded_observation_dim(self.max_num_buoys)

    def arrays(self) -> Dict[str, np.ndarray]:
        """Return every persisted array in stable field-name order."""
        return {
            name: getattr(self, name)
            for name in _PERSISTED_ARRAY_FIELDS
        }

    def _validate(self) -> None:
        count = self.sample_count
        state_width = self.state_dim
        waypoint_width = self.waypoint_dim
        node_width = self.max_num_buoys

        vector_fields = (
            "sample_id",
            "split",
            "episode_id",
            "scenario_id",
            "seed",
            "policy_id",
            "arrival_trace_id",
            "num_buoys",
            "episode_length",
            "t",
            "k",
            "target_t",
            "target_transition_index",
            "goal_source",
            "terminated_between_t_and_t_plus_k",
            "truncated_between_t_and_t_plus_k",
            "endpoint_terminated",
            "endpoint_truncated",
            "trajectory_schema_version",
            "goal_spec_version",
            "schema_version",
        )
        matrix_shapes = {
            "state_t": (count, state_width),
            "state_mask_t": (count, state_width),
            "valid_node_mask_t": (count, node_width),
            "goal": (count, waypoint_width),
            "goal_mask": (count, waypoint_width),
            "waypoint_target_t_plus_k": (count, waypoint_width),
            "waypoint_mask_t_plus_k": (count, waypoint_width),
            "valid_node_mask_t_plus_k": (count, node_width),
        }
        for name in vector_fields:
            value = getattr(self, name)
            if value.shape != (count,):
                raise ValueError(
                    f"{name} must have shape {(count,)}, got {value.shape}"
                )
        for name, expected in matrix_shapes.items():
            value = getattr(self, name)
            if value.shape != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {value.shape}"
                )

        if count == 0:
            return
        if len(set(self.sample_id.tolist())) != count:
            raise ValueError("sample_id values must be unique within a split")
        if np.any(self.split != self.split_name):
            raise ValueError("split array contradicts split_name")
        if np.any(self.trajectory_schema_version != TRAJECTORY_SCHEMA_VERSION):
            raise ValueError("samples must come from trajectory schema v1")
        if np.any(self.goal_spec_version != GOAL_SPEC_VERSION):
            raise ValueError("samples must use goal spec v1")
        if np.any(self.schema_version != WAYPOINT_SAMPLE_SCHEMA_VERSION):
            raise ValueError("unsupported waypoint sample schema version")
        if np.any(~np.isin(self.goal_source, GOAL_SOURCES)):
            raise ValueError(f"goal_source must be one of {GOAL_SOURCES}")
        if len(set(self.goal_source.tolist())) != 1:
            raise ValueError("a saved batch must use one goal_source")
        if np.any(~np.isin(self.k, SUPPORTED_K_VALUES)):
            raise ValueError(
                f"K values must be selected from {SUPPORTED_K_VALUES}"
            )
        if np.any(self.num_buoys < 1) or np.any(
            self.num_buoys > self.max_num_buoys
        ):
            raise ValueError("num_buoys is outside [1, max_num_buoys]")
        canonical_nodes = (
            np.arange(self.max_num_buoys, dtype=np.int64)[None, :]
            < self.num_buoys[:, None]
        )
        if not np.array_equal(self.valid_node_mask_t, canonical_nodes):
            raise ValueError(
                "valid_node_mask_t must mark exactly the real buoy prefix"
            )
        if not np.array_equal(
            self.valid_node_mask_t_plus_k, canonical_nodes
        ):
            raise ValueError(
                "valid_node_mask_t_plus_k must mark exactly the real buoy "
                "prefix"
            )
        if np.any(self.episode_length < 1):
            raise ValueError("episode_length must be positive")
        if np.any(self.t < 0):
            raise ValueError("t must be nonnegative")
        if not np.array_equal(self.target_t, self.t + self.k):
            raise ValueError("target_t must equal t + K")
        if not np.array_equal(
            self.target_transition_index, self.target_t - 1
        ):
            raise ValueError(
                "target_transition_index must equal target_t - 1"
            )
        if np.any(self.target_t > self.episode_length):
            raise ValueError("target_t crosses the source episode boundary")
        if np.any(self.terminated_between_t_and_t_plus_k):
            raise ValueError("saved samples may not cross internal termination")
        if np.any(self.truncated_between_t_and_t_plus_k):
            raise ValueError("saved samples may not cross internal truncation")
        if np.any(
            np.logical_and(self.endpoint_terminated, self.endpoint_truncated)
        ):
            raise ValueError(
                "an endpoint cannot be both terminated and truncated"
            )

        expected_source_waypoint_masks = _waypoint_masks(
            self.valid_node_mask_t
        )
        if not np.array_equal(
            self.state_mask_t[:, :waypoint_width],
            expected_source_waypoint_masks,
        ):
            raise ValueError("state waypoint-feature mask is inconsistent")
        obs_start = waypoint_width
        obs_stop = obs_start + self.padded_observation_dim
        expected_observation_masks = _observation_masks(
            self.valid_node_mask_t
        )
        if not np.array_equal(
            self.state_mask_t[:, obs_start:obs_stop],
            expected_observation_masks,
        ):
            raise ValueError("padded observation mask is inconsistent")
        if not np.all(self.state_mask_t[:, obs_stop:]):
            raise ValueError(
                "explicit valid-node-mask input slots must themselves be valid"
            )
        if not np.array_equal(
            self.state_t[:, obs_stop:],
            self.valid_node_mask_t.astype(np.float32),
        ):
            raise ValueError(
                "state_t tail must contain the explicit valid node mask"
            )
        expected_target_masks = _waypoint_masks(
            self.valid_node_mask_t_plus_k
        )
        if not np.array_equal(
            self.waypoint_mask_t_plus_k, expected_target_masks
        ):
            raise ValueError("target waypoint mask is inconsistent")
        if np.any(self.state_t[~self.state_mask_t] != 0.0):
            raise ValueError("masked state padding must be exactly zero")
        if np.any(
            self.waypoint_target_t_plus_k[
                ~self.waypoint_mask_t_plus_k
            ]
            != 0.0
        ):
            raise ValueError("masked waypoint target padding must be zero")
        if np.any(self.goal[~self.goal_mask] != 0.0):
            raise ValueError("masked goal padding must be exactly zero")
        _validate_waypoint_numeric_ranges(
            "waypoint_target_t_plus_k",
            self.waypoint_target_t_plus_k,
            self.waypoint_mask_t_plus_k,
            self.max_num_buoys,
        )
        _validate_waypoint_numeric_ranges(
            "goal", self.goal, self.goal_mask, self.max_num_buoys
        )

        desired = self.goal_source == "desired_goal"
        if np.any(desired):
            desired_masks = _desired_goal_masks(
                self.num_buoys[desired], self.max_num_buoys
            )
            if not np.array_equal(self.goal_mask[desired], desired_masks):
                raise ValueError("desired_goal mask is inconsistent")
        achieved = self.goal_source == "achieved_terminal_goal"
        if np.any(achieved):
            achieved_masks = _canonical_waypoint_masks(
                self.num_buoys[achieved], self.max_num_buoys
            )
            if not np.array_equal(self.goal_mask[achieved], achieved_masks):
                raise ValueError("achieved terminal goal mask is inconsistent")


@dataclass(frozen=True)
class WaypointSampleBuildResult:
    """In-memory batches plus deterministic K-boundary audit rows."""

    batches: Mapping[str, WaypointSampleBatch]
    k_statistics: Tuple[Mapping[str, Any], ...]
    max_num_buoys: int
    k_values: Tuple[int, ...]
    goal_source: str
    split_assignments: Mapping[str, str]


_PERSISTED_ARRAY_FIELDS = (
    "sample_id",
    "split",
    "episode_id",
    "scenario_id",
    "seed",
    "policy_id",
    "arrival_trace_id",
    "num_buoys",
    "episode_length",
    "t",
    "k",
    "target_t",
    "target_transition_index",
    "state_t",
    "state_mask_t",
    "valid_node_mask_t",
    "goal",
    "goal_mask",
    "goal_source",
    "waypoint_target_t_plus_k",
    "waypoint_mask_t_plus_k",
    "valid_node_mask_t_plus_k",
    "terminated_between_t_and_t_plus_k",
    "truncated_between_t_and_t_plus_k",
    "endpoint_terminated",
    "endpoint_truncated",
    "trajectory_schema_version",
    "goal_spec_version",
    "schema_version",
)


def build_waypoint_sample_dataset(
    episodes: Sequence[Any],
    split_assignments: Mapping[str, str],
    k_values: Sequence[int] = SUPPORTED_K_VALUES,
    goal_source: str = DEFAULT_GOAL_SOURCE,
    max_num_buoys: Optional[int] = None,
) -> WaypointSampleBuildResult:
    """Build deterministic, split-isolated supervised samples.

    ``episodes`` must contain only ``MaritimeEpisodeV1``-compatible objects.
    The function deliberately uses duck typing so the builder remains usable
    while schema-v1 storage evolves, but every required field, shape, schema
    version, goal definition, and terminal value is validated strictly.
    """
    values = _validate_episode_sequence(episodes)
    assignments = _validate_split_assignments(values, split_assignments)
    ks = _validate_k_values(k_values)
    source = _goal_source(goal_source)
    observed_max = max(int(episode.num_buoys) for episode in values)
    if max_num_buoys is None:
        n_max = observed_max
    else:
        n_max = _positive_integer("max_num_buoys", max_num_buoys)
        if n_max < observed_max:
            raise ValueError(
                "max_num_buoys cannot be smaller than an episode scale"
            )

    records: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name in SPLIT_NAMES
    }
    counters: Dict[Tuple[str, int], Dict[str, int]] = {}
    split_episode_counts: Dict[str, int] = {name: 0 for name in SPLIT_NAMES}
    split_transition_counts: Dict[str, int] = {
        name: 0 for name in SPLIT_NAMES
    }

    ordered_episodes = sorted(
        values,
        key=lambda episode: (
            SPLIT_NAMES.index(assignments[episode.episode_id]),
            str(episode.episode_id),
        ),
    )
    for episode in ordered_episodes:
        split_name = assignments[episode.episode_id]
        _validate_v1_episode(episode)
        split_episode_counts[split_name] += 1
        split_transition_counts[split_name] += int(episode.episode_length)
        episode_goal, episode_goal_mask = _episode_goal(
            episode, source, n_max
        )
        for k_value in ks:
            counter = counters.setdefault(
                (split_name, k_value),
                {
                    "candidate": 0,
                    "valid": 0,
                    "short": 0,
                    "padding": 0,
                    "internal_terminated": 0,
                    "internal_truncated": 0,
                    "endpoint_terminated": 0,
                    "endpoint_truncated": 0,
                },
            )
            for start in range(int(episode.episode_length)):
                counter["candidate"] += 1
                outcome = _anchor_outcome(episode, start, k_value)
                if outcome["short"]:
                    counter["short"] += 1
                    continue
                if outcome["padding"]:
                    counter["padding"] += 1
                    continue
                if outcome["internal_terminated"]:
                    counter["internal_terminated"] += 1
                if outcome["internal_truncated"]:
                    counter["internal_truncated"] += 1
                if (
                    outcome["internal_terminated"]
                    or outcome["internal_truncated"]
                ):
                    continue

                target_transition = start + k_value - 1
                endpoint_terminated = bool(
                    episode.terminated[target_transition]
                )
                endpoint_truncated = bool(
                    episode.truncated[target_transition]
                )
                counter["endpoint_terminated"] += int(endpoint_terminated)
                counter["endpoint_truncated"] += int(endpoint_truncated)
                counter["valid"] += 1
                records[split_name].append(
                    _sample_record(
                        episode=episode,
                        split_name=split_name,
                        start=start,
                        k_value=k_value,
                        max_num_buoys=n_max,
                        goal_source=source,
                        goal=episode_goal,
                        goal_mask=episode_goal_mask,
                        endpoint_terminated=endpoint_terminated,
                        endpoint_truncated=endpoint_truncated,
                    )
                )

    batches = {
        split_name: _records_to_batch(
            records[split_name], split_name, n_max
        )
        for split_name in SPLIT_NAMES
    }
    statistics: List[Mapping[str, Any]] = []
    for split_name in SPLIT_NAMES:
        for k_value in ks:
            counter = counters.get(
                (split_name, k_value),
                {
                    "candidate": 0,
                    "valid": 0,
                    "short": 0,
                    "padding": 0,
                    "internal_terminated": 0,
                    "internal_truncated": 0,
                    "endpoint_terminated": 0,
                    "endpoint_truncated": 0,
                },
            )
            candidate_count = counter["candidate"]
            statistics.append(
                {
                    "split": split_name,
                    "k": k_value,
                    "episode_count": split_episode_counts[split_name],
                    "transition_count": split_transition_counts[split_name],
                    "candidate_anchor_count": candidate_count,
                    "valid_sample_count": counter["valid"],
                    "valid_sample_ratio": (
                        float(counter["valid"]) / float(candidate_count)
                        if candidate_count
                        else 0.0
                    ),
                    "invalid_short_horizon_count": counter["short"],
                    "invalid_padding_count": counter["padding"],
                    "invalid_internal_terminated_count": counter[
                        "internal_terminated"
                    ],
                    "invalid_internal_truncated_count": counter[
                        "internal_truncated"
                    ],
                    "endpoint_terminated_count": counter[
                        "endpoint_terminated"
                    ],
                    "endpoint_truncated_count": counter[
                        "endpoint_truncated"
                    ],
                }
            )
    return WaypointSampleBuildResult(
        batches=batches,
        k_statistics=tuple(statistics),
        max_num_buoys=n_max,
        k_values=ks,
        goal_source=source,
        split_assignments=dict(assignments),
    )


def build_waypoint_samples(
    episodes: Sequence[Any],
    split_assignments: Mapping[str, str],
    k_values: Sequence[int] = SUPPORTED_K_VALUES,
    goal_source: str = DEFAULT_GOAL_SOURCE,
    max_num_buoys: Optional[int] = None,
) -> Dict[str, WaypointSampleBatch]:
    """Convenience API returning only ``train/validation/test`` batches."""
    result = build_waypoint_sample_dataset(
        episodes=episodes,
        split_assignments=split_assignments,
        k_values=k_values,
        goal_source=goal_source,
        max_num_buoys=max_num_buoys,
    )
    return dict(result.batches)


def validate_waypoint_samples(
    batches: Mapping[str, WaypointSampleBatch],
    *,
    expected_episode_splits: Optional[Mapping[str, str]] = None,
    source_episodes: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Validate arrays, split isolation, unique identities, masks, and values."""
    normalized = _validate_batches_mapping(batches)
    all_ids: List[str] = []
    episode_to_split: Dict[str, str] = {}
    for split_name in SPLIT_NAMES:
        batch = normalized[split_name]
        batch._validate()
        all_ids.extend(batch.sample_id.tolist())
        for episode_id in np.unique(batch.episode_id):
            previous = episode_to_split.setdefault(
                str(episode_id), split_name
            )
            if previous != split_name:
                raise ValueError(
                    f"episode {episode_id!r} leaks across splits"
                )
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("sample_id values must be globally unique")
    if expected_episode_splits is not None:
        expected = {
            str(key): _split_name(value)
            for key, value in expected_episode_splits.items()
        }
        for episode_id, observed_split in episode_to_split.items():
            if episode_id not in expected:
                raise ValueError(
                    f"sample references unknown episode {episode_id!r}"
                )
            if expected[episode_id] != observed_split:
                raise ValueError(
                    f"sample split for {episode_id!r} contradicts source data"
                )
    sample_count = sum(batch.sample_count for batch in normalized.values())
    finite = all(
        np.all(np.isfinite(array))
        for batch in normalized.values()
        for array in (
            batch.state_t,
            batch.goal,
            batch.waypoint_target_t_plus_k,
        )
    )
    if not finite:
        raise ValueError("waypoint sample dataset contains NaN or Inf")
    if source_episodes is not None:
        _validate_samples_against_source_episodes(
            normalized, source_episodes, expected_episode_splits
        )
    return {
        "schema_version": WAYPOINT_SAMPLE_SCHEMA_VERSION,
        "sample_count": int(sample_count),
        "split_sample_counts": {
            name: normalized[name].sample_count for name in SPLIT_NAMES
        },
        "episode_count": len(episode_to_split),
        "split_leakage_count": 0,
        "internal_done_violation_count": 0,
        "padding_target_violation_count": 0,
        "all_finite": True,
        "source_target_exact": source_episodes is not None,
        "training_performed": False,
    }


def save_waypoint_samples(
    output_dir: Any,
    build_result: Any,
    *,
    source_dataset_manifest: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Safely save three NPZ batches, schema JSON, manifest JSON, and K CSV.

    ``build_result`` may be a :class:`WaypointSampleBuildResult` or a mapping
    of split names to batches.  A mapping-only call is accepted, but has no
    candidate-anchor audit context and therefore writes zero-width statistics
    only if every split is empty; callers should prefer the build-result form.
    Existing target artifacts are never overwritten.
    """
    if isinstance(build_result, WaypointSampleBuildResult):
        result = build_result
        batches = _validate_batches_mapping(result.batches)
        statistics = list(result.k_statistics)
        k_values = result.k_values
        goal_source = result.goal_source
        n_max = result.max_num_buoys
    elif isinstance(build_result, Mapping):
        batches = _validate_batches_mapping(build_result)
        n_values = {batch.max_num_buoys for batch in batches.values()}
        if len(n_values) != 1:
            raise ValueError("all split batches must share max_num_buoys")
        n_max = next(iter(n_values))
        k_values = tuple(
            sorted(
                {
                    int(value)
                    for batch in batches.values()
                    for value in batch.k.tolist()
                }
            )
        )
        if not k_values:
            k_values = SUPPORTED_K_VALUES
        sources = {
            str(value)
            for batch in batches.values()
            for value in batch.goal_source.tolist()
        }
        if len(sources) > 1:
            raise ValueError("all split batches must share one goal_source")
        goal_source = next(iter(sources), DEFAULT_GOAL_SOURCE)
        statistics = _statistics_from_batches(batches, k_values)
        result = WaypointSampleBuildResult(
            batches=batches,
            k_statistics=tuple(statistics),
            max_num_buoys=n_max,
            k_values=k_values,
            goal_source=goal_source,
            split_assignments={
                str(episode_id): split_name
                for split_name, batch in batches.items()
                for episode_id in np.unique(batch.episode_id).tolist()
            },
        )
    else:
        raise TypeError(
            "build_result must be WaypointSampleBuildResult or split mapping"
        )

    validation = validate_waypoint_samples(
        batches,
        expected_episode_splits=result.split_assignments,
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        split_name: root / f"waypoint_samples_{split_name}.npz"
        for split_name in SPLIT_NAMES
    }
    schema_path = root / "waypoint_sample_schema.json"
    manifest_path = root / "waypoint_sample_manifest.json"
    statistics_path = root / "waypoint_k_statistics.csv"
    all_paths = list(paths.values()) + [
        schema_path,
        manifest_path,
        statistics_path,
    ]
    existing = [path for path in all_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite waypoint artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    for split_name, path in paths.items():
        arrays = batches[split_name].arrays()
        _validate_npz_arrays(arrays)
        np.savez_compressed(str(path), **arrays)
    _write_k_statistics(statistics_path, statistics)
    schema = build_waypoint_sample_schema(n_max)
    _write_json(schema_path, schema)

    created_at = datetime.now(timezone.utc).isoformat()
    sample_files = {
        split_name: {
            "path": paths[split_name].name,
            "sha256": _sha256(paths[split_name]),
            "sample_count": batches[split_name].sample_count,
            "array_fields": {
                name: {
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                }
                for name, array in batches[split_name].arrays().items()
            },
        }
        for split_name in SPLIT_NAMES
    }
    source_manifest_digest = None
    if source_dataset_manifest is not None:
        source_manifest_digest = hashlib.sha256(
            json.dumps(
                _json_compatible(dict(source_dataset_manifest)),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    manifest = {
        "waypoint_sample_schema_version": WAYPOINT_SAMPLE_SCHEMA_VERSION,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "goal_spec_version": GOAL_SPEC_VERSION,
        "created_at_utc": created_at,
        "training_performed": False,
        "goal_source": goal_source,
        "supported_k_values": list(SUPPORTED_K_VALUES),
        "materialized_k_values": list(k_values),
        "max_num_buoys": n_max,
        "waypoint_dim": waypoint_dim(n_max),
        "padded_observation_dim": padded_observation_dim(n_max),
        "state_dim": waypoint_state_dim(n_max),
        "sample_count": validation["sample_count"],
        "split_sample_counts": validation["split_sample_counts"],
        "source_episode_split_assignments": {
            str(episode_id): _split_name(split_name)
            for episode_id, split_name in sorted(
                result.split_assignments.items()
            )
        },
        "sample_files": sample_files,
        "schema_file": {
            "path": schema_path.name,
            "sha256": _sha256(schema_path),
        },
        "k_statistics_file": {
            "path": statistics_path.name,
            "sha256": _sha256(statistics_path),
            "row_count": len(statistics),
        },
        "source_dataset_manifest_sha256": source_manifest_digest,
        "metadata": (
            _json_compatible(dict(metadata)) if metadata is not None else {}
        ),
    }
    _write_json(manifest_path, manifest)
    return manifest


def build_and_save_waypoint_samples(
    output_dir: Any,
    episodes: Sequence[Any],
    split_assignments: Mapping[str, str],
    *,
    k_values: Sequence[int] = SUPPORTED_K_VALUES,
    goal_source: str = DEFAULT_GOAL_SOURCE,
    max_num_buoys: Optional[int] = None,
    source_dataset_manifest: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build, validate, and persist the complete schema-v1 artifact set."""
    result = build_waypoint_sample_dataset(
        episodes=episodes,
        split_assignments=split_assignments,
        k_values=k_values,
        goal_source=goal_source,
        max_num_buoys=max_num_buoys,
    )
    return save_waypoint_samples(
        output_dir,
        result,
        source_dataset_manifest=source_dataset_manifest,
        metadata=metadata,
    )


def load_waypoint_samples(
    output_dir: Any,
    *,
    split: Optional[str] = None,
    verify_checksums: bool = True,
) -> Any:
    """Load saved samples with ``allow_pickle=False`` and strict descriptors.

    With ``split=None`` the return value is a three-split mapping.  Otherwise
    the selected :class:`WaypointSampleBatch` is returned.
    """
    root = Path(output_dir)
    manifest = _read_json(root / "waypoint_sample_manifest.json")
    schema = _read_json(root / "waypoint_sample_schema.json")
    if (
        manifest.get("waypoint_sample_schema_version")
        != WAYPOINT_SAMPLE_SCHEMA_VERSION
        or schema.get("waypoint_sample_schema_version")
        != WAYPOINT_SAMPLE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported waypoint sample schema version")
    if manifest.get("trajectory_schema_version") != TRAJECTORY_SCHEMA_VERSION:
        raise ValueError("waypoint samples did not come from trajectory v1")
    if manifest.get("goal_spec_version") != GOAL_SPEC_VERSION:
        raise ValueError("waypoint samples do not use goal spec v1")
    if manifest.get("training_performed") is not False:
        raise ValueError("waypoint sample manifest must record no training")
    if verify_checksums:
        schema_entry = manifest["schema_file"]
        if _sha256(root / schema_entry["path"]) != schema_entry["sha256"]:
            raise ValueError("waypoint sample schema checksum mismatch")
        statistics_entry = manifest["k_statistics_file"]
        if (
            _sha256(root / statistics_entry["path"])
            != statistics_entry["sha256"]
        ):
            raise ValueError("waypoint K statistics checksum mismatch")
    requested = SPLIT_NAMES if split is None else (_split_name(split),)
    batches: Dict[str, WaypointSampleBatch] = {}
    for split_name in requested:
        entry = manifest["sample_files"][split_name]
        path = root / entry["path"]
        if verify_checksums and _sha256(path) != entry["sha256"]:
            raise ValueError(f"checksum mismatch for {path}")
        with np.load(str(path), allow_pickle=False) as archive:
            if set(archive.files) != set(_PERSISTED_ARRAY_FIELDS):
                raise ValueError(
                    f"unexpected NPZ fields for split {split_name!r}"
                )
            arrays = {name: archive[name].copy() for name in archive.files}
        _validate_loaded_descriptors(arrays, entry["array_fields"])
        _validate_npz_arrays(arrays)
        batches[split_name] = WaypointSampleBatch(
            split_name=split_name,
            max_num_buoys=int(manifest["max_num_buoys"]),
            **arrays,
        )
        if batches[split_name].sample_count != int(entry["sample_count"]):
            raise ValueError("manifest sample_count disagrees with NPZ")
    if split is not None:
        return batches[_split_name(split)]
    validation = validate_waypoint_samples(
        batches,
        expected_episode_splits=manifest.get(
            "source_episode_split_assignments", {}
        ),
    )
    if validation["sample_count"] != int(manifest["sample_count"]):
        raise ValueError("manifest total sample_count disagrees with NPZ files")
    return batches


def validate_saved_waypoint_samples(output_dir: Any) -> Dict[str, Any]:
    """Checksum, reload, and strictly revalidate a saved artifact set."""
    batches = load_waypoint_samples(
        output_dir, split=None, verify_checksums=True
    )
    result = validate_waypoint_samples(batches)
    result["roundtrip_exact"] = True
    return result


def build_waypoint_sample_schema(max_num_buoys: int) -> Dict[str, Any]:
    """Return the machine-readable waypoint schema for a configured ``N_max``."""
    n_max = _positive_integer("max_num_buoys", max_num_buoys)
    w_dim = waypoint_dim(n_max)
    obs_dim = padded_observation_dim(n_max)
    state_width = waypoint_state_dim(n_max)
    return {
        "waypoint_sample_schema_version": WAYPOINT_SAMPLE_SCHEMA_VERSION,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "goal_spec_version": GOAL_SPEC_VERSION,
        "training_performed": False,
        "supported_k_values": list(SUPPORTED_K_VALUES),
        "temporal_indexing": {
            "source": "observations[t]",
            "target": "next_observations[t+K-1]",
            "target_state_index": "t+K",
            "endpoint_done_allowed": True,
            "internal_done_allowed": False,
            "cross_episode_allowed": False,
            "cross_split_allowed": False,
        },
        "dimensions": {
            "max_num_buoys": n_max,
            "waypoint_dim": w_dim,
            "padded_observation_dim": obs_dim,
            "state_dim": state_width,
        },
        "observation_layout_v1": {
            "base_dim": OBSERVATION_BASE_DIM,
            "per_node_dim": OBSERVATION_NODE_DIM,
            "normalized_position_indices": [0, 1],
            "remaining_energy_ratio_index": OBSERVATION_ENERGY_INDEX,
            "node_normalized_queue_offset": NODE_QUEUE_INDEX,
            "node_service_completion_offset": NODE_COMPLETION_INDEX,
        },
        "waypoint_layout": [
            {
                "name": "uav_position_normalized",
                "start": 0,
                "stop": 2,
            },
            {
                "name": "normalized_queue",
                "start": 2,
                "stop": 2 + n_max,
            },
            {
                "name": "service_completion_ratio",
                "start": 2 + n_max,
                "stop": 2 + 2 * n_max,
            },
            {
                "name": "remaining_energy_ratio",
                "start": 2 + 2 * n_max,
                "stop": w_dim,
            },
        ],
        "state_layout": [
            {
                "name": "current_waypoint",
                "start": 0,
                "stop": w_dim,
            },
            {
                "name": "source_observation_padded",
                "start": w_dim,
                "stop": w_dim + obs_dim,
            },
            {
                "name": "valid_node_mask_as_float",
                "start": w_dim + obs_dim,
                "stop": state_width,
            },
        ],
        "goal_sources": {
            "default": DEFAULT_GOAL_SOURCE,
            "achieved_terminal_goal": {
                "position": "final achieved normalized UAV position",
                "queue": "final achieved per-buoy queue ratio",
                "completion": "final achieved per-buoy completion ratio",
                "energy": "final achieved energy ratio",
            },
            "desired_goal": {
                "position": "not specified; value 0 and mask false",
                "queue": (
                    "maximum_allowed_residual_queue_ratio broadcast to "
                    "valid nodes"
                ),
                "completion": (
                    "target_task_completion_ratio broadcast to valid nodes"
                ),
                "energy": "minimum_terminal_energy_ratio",
            },
        },
        "mask_semantics": {
            "value_true": "real supervised/input feature",
            "value_false": "padding or unspecified desired-goal position",
            "padded_numeric_value": 0.0,
            "loss_requirement": (
                "future training must apply waypoint_mask before reduction"
            ),
        },
        "array_fields": _array_field_schema(state_width, w_dim, n_max),
        "storage": {
            "files": [
                "waypoint_samples_train.npz",
                "waypoint_samples_validation.npz",
                "waypoint_samples_test.npz",
                "waypoint_sample_manifest.json",
                "waypoint_sample_schema.json",
                "waypoint_k_statistics.csv",
            ],
            "npz_allow_pickle": False,
            "unicode_metadata_arrays": True,
            "object_dtype_allowed": False,
            "checksum": "SHA-256",
        },
    }


def _sample_record(
    *,
    episode: Any,
    split_name: str,
    start: int,
    k_value: int,
    max_num_buoys: int,
    goal_source: str,
    goal: np.ndarray,
    goal_mask: np.ndarray,
    endpoint_terminated: bool,
    endpoint_truncated: bool,
) -> Dict[str, Any]:
    target_transition = start + k_value - 1
    source_observation = np.asarray(
        episode.observations[start], dtype=np.float32
    )
    target_observation = np.asarray(
        episode.next_observations[target_transition], dtype=np.float32
    )
    source_nodes = np.asarray(
        episode.valid_node_mask[start], dtype=np.bool_
    )
    target_nodes = np.asarray(
        episode.next_valid_node_mask[target_transition], dtype=np.bool_
    )
    source_waypoint, source_waypoint_mask = _extract_waypoint(
        source_observation,
        source_nodes,
        int(episode.num_buoys),
        max_num_buoys,
    )
    target_waypoint, target_waypoint_mask = _extract_waypoint(
        target_observation,
        target_nodes,
        int(episode.num_buoys),
        max_num_buoys,
    )
    padded_observation, observation_mask = _pad_observation(
        source_observation,
        source_nodes,
        int(episode.num_buoys),
        max_num_buoys,
    )
    state = np.concatenate(
        (
            source_waypoint,
            padded_observation,
            _pad_nodes(source_nodes, max_num_buoys).astype(np.float32),
        )
    ).astype(np.float32, copy=False)
    state_mask = np.concatenate(
        (
            source_waypoint_mask,
            observation_mask,
            np.ones((max_num_buoys,), dtype=np.bool_),
        )
    )
    sample_id = (
        f"{episode.episode_id}|split={split_name}|t={start}|K={k_value}"
    )
    return {
        "sample_id": sample_id,
        "split": split_name,
        "episode_id": str(episode.episode_id),
        "scenario_id": str(episode.scenario_id),
        "seed": int(episode.seed),
        "policy_id": str(episode.policy_id),
        "arrival_trace_id": str(episode.arrival_trace_id),
        "num_buoys": int(episode.num_buoys),
        "episode_length": int(episode.episode_length),
        "t": int(start),
        "k": int(k_value),
        "target_t": int(start + k_value),
        "target_transition_index": int(target_transition),
        "state_t": state,
        "state_mask_t": state_mask,
        "valid_node_mask_t": _pad_nodes(
            source_nodes, max_num_buoys
        ),
        "goal": goal.copy(),
        "goal_mask": goal_mask.copy(),
        "goal_source": goal_source,
        "waypoint_target_t_plus_k": target_waypoint,
        "waypoint_mask_t_plus_k": target_waypoint_mask,
        "valid_node_mask_t_plus_k": _pad_nodes(
            target_nodes, max_num_buoys
        ),
        "terminated_between_t_and_t_plus_k": False,
        "truncated_between_t_and_t_plus_k": False,
        "endpoint_terminated": endpoint_terminated,
        "endpoint_truncated": endpoint_truncated,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "goal_spec_version": GOAL_SPEC_VERSION,
        "schema_version": WAYPOINT_SAMPLE_SCHEMA_VERSION,
    }


def _validate_samples_against_source_episodes(
    batches: Mapping[str, WaypointSampleBatch],
    source_episodes: Sequence[Any],
    expected_episode_splits: Optional[Mapping[str, str]],
) -> None:
    """Recompute every state/goal/label from authoritative source episodes."""
    episodes = _validate_episode_sequence(source_episodes)
    by_id = {str(episode.episode_id): episode for episode in episodes}
    assignments = (
        {
            str(episode_id): _split_name(split_name)
            for episode_id, split_name in expected_episode_splits.items()
        }
        if expected_episode_splits is not None
        else {
            str(episode_id): split_name
            for split_name, batch in batches.items()
            for episode_id in np.unique(batch.episode_id).tolist()
        }
    )
    for split_name in SPLIT_NAMES:
        batch = batches[split_name]
        for row in range(batch.sample_count):
            episode_id = str(batch.episode_id[row])
            if episode_id not in by_id:
                raise ValueError(
                    f"sample references missing source episode {episode_id!r}"
                )
            episode = by_id[episode_id]
            if assignments.get(episode_id) != split_name:
                raise ValueError(
                    f"sample split for {episode_id!r} is not authoritative"
                )
            scalar_expectations = {
                "scenario_id": str(episode.scenario_id),
                "seed": int(episode.seed),
                "policy_id": str(episode.policy_id),
                "arrival_trace_id": str(episode.arrival_trace_id),
                "num_buoys": int(episode.num_buoys),
                "episode_length": int(episode.episode_length),
            }
            for name, expected in scalar_expectations.items():
                actual = getattr(batch, name)[row]
                if actual != expected:
                    raise ValueError(
                        f"sample {batch.sample_id[row]!r} has incorrect "
                        f"{name}"
                    )
            start = int(batch.t[row])
            k_value = int(batch.k[row])
            outcome = _anchor_outcome(episode, start, k_value)
            if any(outcome.values()):
                raise ValueError(
                    f"sample {batch.sample_id[row]!r} crosses a source "
                    "boundary"
                )
            target_transition = start + k_value - 1
            source_goal, source_goal_mask = _episode_goal(
                episode,
                str(batch.goal_source[row]),
                batch.max_num_buoys,
            )
            expected_record = _sample_record(
                episode=episode,
                split_name=split_name,
                start=start,
                k_value=k_value,
                max_num_buoys=batch.max_num_buoys,
                goal_source=str(batch.goal_source[row]),
                goal=source_goal,
                goal_mask=source_goal_mask,
                endpoint_terminated=bool(
                    episode.terminated[target_transition]
                ),
                endpoint_truncated=bool(
                    episode.truncated[target_transition]
                ),
            )
            exact_array_fields = (
                "state_t",
                "state_mask_t",
                "valid_node_mask_t",
                "goal",
                "goal_mask",
                "waypoint_target_t_plus_k",
                "waypoint_mask_t_plus_k",
                "valid_node_mask_t_plus_k",
            )
            for name in exact_array_fields:
                if not np.array_equal(
                    getattr(batch, name)[row], expected_record[name]
                ):
                    raise ValueError(
                        f"sample {batch.sample_id[row]!r} has incorrect "
                        f"{name}"
                    )
            exact_scalar_fields = (
                "sample_id",
                "target_t",
                "target_transition_index",
                "endpoint_terminated",
                "endpoint_truncated",
            )
            for name in exact_scalar_fields:
                if getattr(batch, name)[row] != expected_record[name]:
                    raise ValueError(
                        f"sample {batch.sample_id[row]!r} has incorrect "
                        f"{name}"
                    )


def _records_to_batch(
    records: Sequence[Mapping[str, Any]],
    split_name: str,
    max_num_buoys: int,
) -> WaypointSampleBatch:
    ordered = sorted(
        records,
        key=lambda row: (
            str(row["episode_id"]),
            int(row["k"]),
            int(row["t"]),
        ),
    )
    count = len(ordered)
    state_width = waypoint_state_dim(max_num_buoys)
    w_dim = waypoint_dim(max_num_buoys)
    unicode_names = (
        "sample_id",
        "split",
        "episode_id",
        "scenario_id",
        "policy_id",
        "arrival_trace_id",
        "goal_source",
        "trajectory_schema_version",
        "goal_spec_version",
        "schema_version",
    )
    integer_names = (
        "seed",
        "num_buoys",
        "episode_length",
        "t",
        "k",
        "target_t",
        "target_transition_index",
    )
    bool_vector_names = (
        "terminated_between_t_and_t_plus_k",
        "truncated_between_t_and_t_plus_k",
        "endpoint_terminated",
        "endpoint_truncated",
    )
    arrays: Dict[str, np.ndarray] = {}
    for name in unicode_names:
        arrays[name] = _unicode_array(
            name, [row[name] for row in ordered]
        )
    for name in integer_names:
        arrays[name] = np.asarray(
            [row[name] for row in ordered], dtype=np.int64
        )
    for name in bool_vector_names:
        arrays[name] = np.asarray(
            [row[name] for row in ordered], dtype=np.bool_
        )

    shaped_fields = {
        "state_t": (state_width, np.float32),
        "state_mask_t": (state_width, np.bool_),
        "valid_node_mask_t": (max_num_buoys, np.bool_),
        "goal": (w_dim, np.float32),
        "goal_mask": (w_dim, np.bool_),
        "waypoint_target_t_plus_k": (w_dim, np.float32),
        "waypoint_mask_t_plus_k": (w_dim, np.bool_),
        "valid_node_mask_t_plus_k": (max_num_buoys, np.bool_),
    }
    for name, (width, dtype) in shaped_fields.items():
        if count:
            arrays[name] = np.stack(
                [np.asarray(row[name], dtype=dtype) for row in ordered],
                axis=0,
            )
        else:
            arrays[name] = np.empty((0, width), dtype=dtype)
    return WaypointSampleBatch(
        split_name=split_name,
        max_num_buoys=max_num_buoys,
        **arrays,
    )


def _validate_episode_sequence(episodes: Sequence[Any]) -> List[Any]:
    if isinstance(episodes, (str, bytes)) or not isinstance(
        episodes, Sequence
    ):
        raise TypeError("episodes must be a sequence")
    values = list(episodes)
    if not values:
        raise ValueError("episodes must not be empty")
    ids: List[str] = []
    for episode in values:
        _validate_v1_episode(episode)
        ids.append(str(episode.episode_id))
    if len(ids) != len(set(ids)):
        raise ValueError("episode_id values must be globally unique")
    return values


def _validate_v1_episode(episode: Any) -> None:
    required = (
        "episode_id",
        "scenario_id",
        "seed",
        "policy_id",
        "arrival_trace_id",
        "num_buoys",
        "episode_length",
        "observations",
        "next_observations",
        "valid_transition_mask",
        "valid_node_mask",
        "next_valid_node_mask",
        "terminated",
        "truncated",
        "desired_goal",
        "achieved_goal_global",
        "achieved_goal_per_buoy_queue_ratio",
        "achieved_goal_per_buoy_completion",
        "schema_version",
        "goal_spec_version",
    )
    missing = [name for name in required if not hasattr(episode, name)]
    if missing:
        raise TypeError(
            "MaritimeEpisodeV1-compatible object is missing fields: "
            + ", ".join(missing)
        )
    if str(episode.schema_version) != TRAJECTORY_SCHEMA_VERSION:
        raise ValueError(
            "waypoint samples require trajectory schema_version='v1'"
        )
    if str(episode.goal_spec_version) != GOAL_SPEC_VERSION:
        raise ValueError("waypoint samples require goal_spec_version='v1'")
    for name in (
        "episode_id",
        "scenario_id",
        "policy_id",
        "arrival_trace_id",
    ):
        if not str(getattr(episode, name)):
            raise ValueError(f"{name} must be nonempty")
    _nonnegative_integer("seed", episode.seed)
    num_buoys = _positive_integer("num_buoys", episode.num_buoys)
    length = _positive_integer("episode_length", episode.episode_length)
    expected_obs_dim = (
        OBSERVATION_BASE_DIM + OBSERVATION_NODE_DIM * num_buoys
    )
    observation = np.asarray(episode.observations)
    next_observation = np.asarray(episode.next_observations)
    if observation.shape != (length, expected_obs_dim):
        raise ValueError(
            "observations must have v1 shape "
            f"{(length, expected_obs_dim)}, got {observation.shape}"
        )
    if next_observation.shape != (length, expected_obs_dim):
        raise ValueError(
            "next_observations must have v1 shape "
            f"{(length, expected_obs_dim)}, got {next_observation.shape}"
        )
    if not np.all(np.isfinite(observation)) or not np.all(
        np.isfinite(next_observation)
    ):
        raise ValueError("observations contain NaN or Inf")
    one_dimensional = (
        "valid_transition_mask",
        "terminated",
        "truncated",
    )
    for name in one_dimensional:
        value = np.asarray(getattr(episode, name))
        if value.shape != (length,):
            raise ValueError(
                f"{name} must have shape {(length,)}, got {value.shape}"
            )
    if np.any(
        np.logical_and(
            np.asarray(episode.terminated, dtype=np.bool_),
            np.asarray(episode.truncated, dtype=np.bool_),
        )
    ):
        raise ValueError("a transition cannot be terminated and truncated")
    for name in ("valid_node_mask", "next_valid_node_mask"):
        value = np.asarray(getattr(episode, name))
        if value.shape != (length, num_buoys):
            raise ValueError(
                f"{name} must have shape {(length, num_buoys)}, "
                f"got {value.shape}"
            )
        if not np.all(value.astype(np.bool_)):
            raise ValueError(
                "unpadded v1 episode storage requires every real node valid"
            )
    desired_goal = np.asarray(episode.desired_goal)
    if desired_goal.shape != (3,) or not np.all(np.isfinite(desired_goal)):
        raise ValueError("desired_goal must be finite shape [3]")
    achieved_global = np.asarray(episode.achieved_goal_global)
    if achieved_global.shape != (5,) or not np.all(
        np.isfinite(achieved_global)
    ):
        raise ValueError("achieved_goal_global must be finite shape [5]")
    for name in (
        "achieved_goal_per_buoy_queue_ratio",
        "achieved_goal_per_buoy_completion",
    ):
        value = np.asarray(getattr(episode, name))
        if value.shape != (num_buoys,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite shape [{num_buoys}]")
    bounded = (
        desired_goal,
        achieved_global[:3],
        np.asarray(episode.achieved_goal_per_buoy_queue_ratio),
        np.asarray(episode.achieved_goal_per_buoy_completion),
    )
    if any(np.any(value < -1.0e-6) or np.any(value > 1.0 + 1.0e-6)
           for value in bounded):
        raise ValueError("goal ratios must lie in [0, 1]")

    terminal_waypoint, _ = _extract_waypoint(
        np.asarray(episode.next_observations[-1], dtype=np.float32),
        np.asarray(episode.next_valid_node_mask[-1], dtype=np.bool_),
        num_buoys,
        num_buoys,
    )
    expected_goal = np.concatenate(
        (
            achieved_global[3:5],
            np.asarray(
                episode.achieved_goal_per_buoy_queue_ratio,
                dtype=np.float32,
            ),
            np.asarray(
                episode.achieved_goal_per_buoy_completion,
                dtype=np.float32,
            ),
            achieved_global[2:3],
        )
    ).astype(np.float32)
    if not np.allclose(
        terminal_waypoint, expected_goal, rtol=1.0e-5, atol=1.0e-5
    ):
        raise ValueError(
            "achieved goal fields must be recomputable from terminal "
            "next_observation"
        )


def _validate_split_assignments(
    episodes: Sequence[Any], assignments: Mapping[str, str]
) -> Dict[str, str]:
    if not isinstance(assignments, Mapping):
        raise TypeError("split_assignments must be a mapping")
    normalized = {
        str(episode_id): _split_name(split)
        for episode_id, split in assignments.items()
    }
    expected = {str(episode.episode_id) for episode in episodes}
    if set(normalized) != expected:
        raise ValueError(
            "split assignment coverage mismatch: "
            f"missing={sorted(expected-set(normalized))}, "
            f"extra={sorted(set(normalized)-expected)}"
        )
    comparison_groups: Dict[Tuple[str, int], str] = {}
    for episode in episodes:
        group = (str(episode.scenario_id), int(episode.seed))
        split_name = normalized[str(episode.episode_id)]
        prior = comparison_groups.setdefault(group, split_name)
        if prior != split_name:
            raise ValueError(
                "(scenario_id, seed) comparison group leaks across splits"
            )
    return normalized


def _episode_goal(
    episode: Any, goal_source: str, max_num_buoys: int
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(episode.num_buoys)
    w_dim = waypoint_dim(max_num_buoys)
    goal = np.zeros((w_dim,), dtype=np.float32)
    if goal_source == "desired_goal":
        desired = np.asarray(episode.desired_goal, dtype=np.float32)
        completion_target = float(desired[0])
        residual_queue_maximum = float(desired[1])
        minimum_energy = float(desired[2])
        goal[2 : 2 + n] = residual_queue_maximum
        goal[2 + max_num_buoys : 2 + max_num_buoys + n] = (
            completion_target
        )
        goal[-1] = minimum_energy
        mask = _desired_goal_masks(
            np.asarray([n], dtype=np.int64), max_num_buoys
        )[0]
        return goal, mask

    achieved_global = np.asarray(
        episode.achieved_goal_global, dtype=np.float32
    )
    goal[:2] = achieved_global[3:5]
    goal[2 : 2 + n] = np.asarray(
        episode.achieved_goal_per_buoy_queue_ratio, dtype=np.float32
    )
    goal[2 + max_num_buoys : 2 + max_num_buoys + n] = np.asarray(
        episode.achieved_goal_per_buoy_completion, dtype=np.float32
    )
    goal[-1] = achieved_global[2]
    mask = _canonical_waypoint_masks(
        np.asarray([n], dtype=np.int64), max_num_buoys
    )[0]
    return goal, mask


def _extract_waypoint(
    observation: np.ndarray,
    node_mask: np.ndarray,
    num_buoys: int,
    max_num_buoys: int,
) -> Tuple[np.ndarray, np.ndarray]:
    expected = OBSERVATION_BASE_DIM + OBSERVATION_NODE_DIM * num_buoys
    if observation.shape != (expected,):
        raise ValueError(
            f"v1 observation must have shape {(expected,)}, "
            f"got {observation.shape}"
        )
    nodes = np.asarray(
        observation[OBSERVATION_BASE_DIM:], dtype=np.float32
    ).reshape(num_buoys, OBSERVATION_NODE_DIM)
    padded_mask = _pad_nodes(node_mask, max_num_buoys)
    result = np.zeros((waypoint_dim(max_num_buoys),), dtype=np.float32)
    result[:2] = observation[OBSERVATION_POSITION_SLICE]
    result[2 : 2 + num_buoys] = nodes[:, NODE_QUEUE_INDEX]
    completion_start = 2 + max_num_buoys
    result[
        completion_start : completion_start + num_buoys
    ] = nodes[:, NODE_COMPLETION_INDEX]
    result[-1] = observation[OBSERVATION_ENERGY_INDEX]
    mask = _waypoint_masks(padded_mask.reshape(1, -1))[0]
    result[~mask] = 0.0
    return result, mask


def _pad_observation(
    observation: np.ndarray,
    node_mask: np.ndarray,
    num_buoys: int,
    max_num_buoys: int,
) -> Tuple[np.ndarray, np.ndarray]:
    nodes = np.asarray(
        observation[OBSERVATION_BASE_DIM:], dtype=np.float32
    ).reshape(num_buoys, OBSERVATION_NODE_DIM)
    padded = np.zeros(
        (padded_observation_dim(max_num_buoys),), dtype=np.float32
    )
    padded[:OBSERVATION_BASE_DIM] = observation[:OBSERVATION_BASE_DIM]
    node_block = padded[OBSERVATION_BASE_DIM:].reshape(
        max_num_buoys, OBSERVATION_NODE_DIM
    )
    node_block[:num_buoys] = nodes
    padded_nodes = _pad_nodes(node_mask, max_num_buoys)
    node_block[~padded_nodes] = 0.0
    mask = _observation_masks(padded_nodes.reshape(1, -1))[0]
    return padded, mask


def _pad_nodes(node_mask: np.ndarray, max_num_buoys: int) -> np.ndarray:
    source = np.asarray(node_mask, dtype=np.bool_)
    if source.ndim != 1 or source.shape[0] > max_num_buoys:
        raise ValueError("node mask width is invalid")
    result = np.zeros((max_num_buoys,), dtype=np.bool_)
    result[: source.shape[0]] = source
    return result


def _waypoint_masks(node_masks: np.ndarray) -> np.ndarray:
    values = np.asarray(node_masks, dtype=np.bool_)
    if values.ndim != 2:
        raise ValueError("node_masks must have shape [B, N_max]")
    batch, n_max = values.shape
    return np.concatenate(
        (
            np.ones((batch, 2), dtype=np.bool_),
            values,
            values,
            np.ones((batch, 1), dtype=np.bool_),
        ),
        axis=1,
    )


def _canonical_waypoint_masks(
    num_buoys: np.ndarray, max_num_buoys: int
) -> np.ndarray:
    counts = np.asarray(num_buoys, dtype=np.int64)
    nodes = np.arange(max_num_buoys)[None, :] < counts[:, None]
    return _waypoint_masks(nodes)


def _desired_goal_masks(
    num_buoys: np.ndarray, max_num_buoys: int
) -> np.ndarray:
    counts = np.asarray(num_buoys, dtype=np.int64)
    nodes = np.arange(max_num_buoys)[None, :] < counts[:, None]
    return np.concatenate(
        (
            np.zeros((counts.shape[0], 2), dtype=np.bool_),
            nodes,
            nodes,
            np.ones((counts.shape[0], 1), dtype=np.bool_),
        ),
        axis=1,
    )


def _observation_masks(node_masks: np.ndarray) -> np.ndarray:
    values = np.asarray(node_masks, dtype=np.bool_)
    if values.ndim != 2:
        raise ValueError("node_masks must have shape [B, N_max]")
    batch = values.shape[0]
    node_features = np.repeat(
        values[:, :, None], OBSERVATION_NODE_DIM, axis=2
    ).reshape(batch, -1)
    return np.concatenate(
        (
            np.ones((batch, OBSERVATION_BASE_DIM), dtype=np.bool_),
            node_features,
        ),
        axis=1,
    )


def _validate_waypoint_numeric_ranges(
    name: str,
    values: np.ndarray,
    masks: np.ndarray,
    max_num_buoys: int,
) -> None:
    """Validate normalized waypoint fields without reading masked padding."""
    tolerance = 1.0e-6
    positions = values[:, :2]
    position_masks = masks[:, :2]
    if np.any(positions[position_masks] < -1.0 - tolerance) or np.any(
        positions[position_masks] > 1.0 + tolerance
    ):
        raise ValueError(f"{name} normalized position is outside [-1, 1]")
    nonnegative = values[:, 2:]
    nonnegative_masks = masks[:, 2:]
    if np.any(
        nonnegative[nonnegative_masks] < -tolerance
    ) or np.any(nonnegative[nonnegative_masks] > 1.0 + tolerance):
        raise ValueError(
            f"{name} queue/completion/energy values are outside [0, 1]"
        )
    expected_width = waypoint_dim(max_num_buoys)
    if values.shape[1] != expected_width:
        raise ValueError(f"{name} width contradicts max_num_buoys")


def _anchor_outcome(
    episode: Any, start: int, k_value: int
) -> Dict[str, bool]:
    target = start + k_value - 1
    if target >= int(episode.episode_length):
        return {
            "short": True,
            "padding": False,
            "internal_terminated": False,
            "internal_truncated": False,
        }
    valid_window = np.asarray(
        episode.valid_transition_mask[start : target + 1],
        dtype=np.bool_,
    )
    if valid_window.shape != (k_value,) or not np.all(valid_window):
        return {
            "short": False,
            "padding": True,
            "internal_terminated": False,
            "internal_truncated": False,
        }
    return {
        "short": False,
        "padding": False,
        "internal_terminated": bool(
            np.any(episode.terminated[start:target])
        ),
        "internal_truncated": bool(
            np.any(episode.truncated[start:target])
        ),
    }


def _statistics_from_batches(
    batches: Mapping[str, WaypointSampleBatch],
    k_values: Sequence[int],
) -> List[Mapping[str, Any]]:
    rows: List[Mapping[str, Any]] = []
    for split_name in SPLIT_NAMES:
        batch = batches[split_name]
        for k_value in k_values:
            selected = batch.k == k_value
            count = int(np.sum(selected))
            rows.append(
                {
                    "split": split_name,
                    "k": int(k_value),
                    "episode_count": int(
                        len(set(batch.episode_id[selected].tolist()))
                    ),
                    "transition_count": "",
                    "candidate_anchor_count": "",
                    "valid_sample_count": count,
                    "valid_sample_ratio": "",
                    "invalid_short_horizon_count": "",
                    "invalid_padding_count": "",
                    "invalid_internal_terminated_count": "",
                    "invalid_internal_truncated_count": "",
                    "endpoint_terminated_count": int(
                        np.sum(batch.endpoint_terminated[selected])
                    ),
                    "endpoint_truncated_count": int(
                        np.sum(batch.endpoint_truncated[selected])
                    ),
                }
            )
    return rows


def _write_k_statistics(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(K_STATISTICS_COLUMNS)
        )
        writer.writeheader()
        for row in rows:
            missing = [
                column for column in K_STATISTICS_COLUMNS
                if column not in row
            ]
            if missing:
                raise ValueError(
                    f"K statistics row is missing columns: {missing}"
                )
            writer.writerow(
                {column: row[column] for column in K_STATISTICS_COLUMNS}
            )


def _validate_batches_mapping(
    batches: Mapping[str, WaypointSampleBatch]
) -> Dict[str, WaypointSampleBatch]:
    if not isinstance(batches, Mapping):
        raise TypeError("batches must map split names to sample batches")
    if set(batches) != set(SPLIT_NAMES):
        raise ValueError(
            f"batches must contain exactly the splits {SPLIT_NAMES}"
        )
    normalized: Dict[str, WaypointSampleBatch] = {}
    max_values = set()
    for split_name in SPLIT_NAMES:
        batch = batches[split_name]
        if not isinstance(batch, WaypointSampleBatch):
            raise TypeError("every split value must be WaypointSampleBatch")
        if batch.split_name != split_name:
            raise ValueError("batch split_name contradicts mapping key")
        normalized[split_name] = batch
        max_values.add(batch.max_num_buoys)
    if len(max_values) != 1:
        raise ValueError("all batches must share max_num_buoys")
    return normalized


def _validate_npz_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"{name} uses forbidden object dtype")
        if array.dtype.kind == "S":
            raise ValueError(
                f"{name} must use Unicode rather than byte-string dtype"
            )


def _validate_loaded_descriptors(
    arrays: Mapping[str, np.ndarray],
    descriptors: Mapping[str, Mapping[str, Any]],
) -> None:
    if set(arrays) != set(descriptors):
        raise ValueError("manifest array descriptors disagree with NPZ")
    for name, array in arrays.items():
        descriptor = descriptors[name]
        if str(array.dtype) != descriptor.get("dtype"):
            raise ValueError(f"dtype mismatch for field {name!r}")
        if list(array.shape) != descriptor.get("shape"):
            raise ValueError(f"shape mismatch for field {name!r}")


def _array_field_schema(
    state_width: int, waypoint_width: int, node_width: int
) -> Dict[str, Any]:
    unicode_fields = {
        "sample_id",
        "split",
        "episode_id",
        "scenario_id",
        "policy_id",
        "arrival_trace_id",
        "goal_source",
        "trajectory_schema_version",
        "goal_spec_version",
        "schema_version",
    }
    integer_fields = {
        "seed",
        "num_buoys",
        "episode_length",
        "t",
        "k",
        "target_t",
        "target_transition_index",
    }
    bool_vectors = {
        "terminated_between_t_and_t_plus_k",
        "truncated_between_t_and_t_plus_k",
        "endpoint_terminated",
        "endpoint_truncated",
    }
    shapes = {
        "state_t": ["B", state_width],
        "state_mask_t": ["B", state_width],
        "valid_node_mask_t": ["B", node_width],
        "goal": ["B", waypoint_width],
        "goal_mask": ["B", waypoint_width],
        "waypoint_target_t_plus_k": ["B", waypoint_width],
        "waypoint_mask_t_plus_k": ["B", waypoint_width],
        "valid_node_mask_t_plus_k": ["B", node_width],
    }
    result: Dict[str, Any] = {}
    for name in _PERSISTED_ARRAY_FIELDS:
        if name in unicode_fields:
            result[name] = {
                "dtype": "fixed-width Unicode",
                "shape": ["B"],
            }
        elif name in integer_fields:
            result[name] = {"dtype": "int64", "shape": ["B"]}
        elif name in bool_vectors:
            result[name] = {"dtype": "bool", "shape": ["B"]}
        elif name in shapes:
            dtype = (
                "float32"
                if name
                in ("state_t", "goal", "waypoint_target_t_plus_k")
                else "bool"
            )
            result[name] = {"dtype": dtype, "shape": shapes[name]}
        else:
            raise RuntimeError(f"missing schema descriptor for {name}")
    return result


def _validate_k_values(values: Sequence[int]) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not values:
        raise ValueError("k_values must be a non-empty sequence")
    result = tuple(_positive_integer("K", value) for value in values)
    if len(result) != len(set(result)):
        raise ValueError("k_values must not contain duplicates")
    invalid = [value for value in result if value not in SUPPORTED_K_VALUES]
    if invalid:
        raise ValueError(
            f"K values must be selected from {SUPPORTED_K_VALUES}: {invalid}"
        )
    return tuple(sorted(result))


def _goal_source(value: Any) -> str:
    result = str(value)
    if result not in GOAL_SOURCES:
        raise ValueError(f"goal_source must be one of {GOAL_SOURCES}")
    return result


def _split_name(value: Any) -> str:
    result = str(value)
    if result not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}")
    return result


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _unicode_array(name: str, value: Any) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    strings = [str(item) for item in raw.tolist()]
    width = max((len(item) for item in strings), default=1)
    result = np.asarray(strings, dtype=f"<U{max(width, 1)}")
    return result


def _typed_array(name: str, value: Any, dtype: Any) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.hasobject:
        # Construction accepts Python lists, but persisted object arrays are
        # rejected before this helper by the safe loader.
        try:
            return np.asarray(raw.tolist(), dtype=dtype)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} cannot be converted safely") from error
    try:
        return np.asarray(raw, dtype=dtype)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} cannot be converted safely") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            _json_compatible(dict(value)),
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("JSON value contains NaN or Inf")
        return number
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON value contains NaN or Inf")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot encode {type(value).__name__} in JSON")


__all__ = [
    "DEFAULT_GOAL_SOURCE",
    "GOAL_SOURCES",
    "K_STATISTICS_COLUMNS",
    "OBSERVATION_BASE_DIM",
    "OBSERVATION_NODE_DIM",
    "SUPPORTED_K_VALUES",
    "WAYPOINT_SAMPLE_SCHEMA_VERSION",
    "WaypointSampleBatch",
    "WaypointSampleBuildResult",
    "build_and_save_waypoint_samples",
    "build_waypoint_sample_dataset",
    "build_waypoint_sample_schema",
    "build_waypoint_samples",
    "load_waypoint_samples",
    "padded_observation_dim",
    "save_waypoint_samples",
    "validate_saved_waypoint_samples",
    "validate_waypoint_samples",
    "waypoint_dim",
    "waypoint_state_dim",
]
