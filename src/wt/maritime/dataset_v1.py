"""Safe, exact persistence for schema-v1 maritime queue trajectories.

The loader in this module accepts only :class:`MaritimeEpisodeV1`.  Round-02
``1.0``/goal-v0 artifacts continue to be handled by ``dataset_io.py`` and are
rejected here, preventing silent semantic mixing.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .trajectory_v1 import (
    ACHIEVED_GOAL_GLOBAL_NAMES_V1,
    DESIRED_GOAL_NAMES_V1,
    GOAL_SPEC_VERSION_V1,
    MaritimeEpisodeV1,
    REWARD_COMPONENT_NAMES_V1,
    TRAJECTORY_SCHEMA_VERSION_V1,
    trajectories_v1_equal,
)


DATASET_SCHEMA_VERSION_V1 = "v1"
SPLIT_NAMES_V1 = ("train", "validation", "test")
DEFAULT_SPLIT_RATIOS_V1 = (0.60, 0.20, 0.20)


def deterministic_episode_split_v1(
    episodes: Sequence[MaritimeEpisodeV1],
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS_V1,
    seed: int = 0,
) -> Dict[str, str]:
    """Deterministically split indivisible ``(scenario_id, seed)`` groups.

    Grouping all policies from the same scenario seed prevents shared arrival
    traces and initial states from leaking across dataset partitions.
    """
    values = _validate_episode_sequence(episodes)
    normalized_ratios = _validate_ratios(ratios)
    split_seed = _nonnegative_integer("seed", seed)

    by_scenario: Dict[
        str, Dict[Tuple[str, int], List[MaritimeEpisodeV1]]
    ] = {}
    for episode in values:
        group = (episode.scenario_id, episode.seed)
        by_scenario.setdefault(episode.scenario_id, {}).setdefault(
            group, []
        ).append(episode)

    assignments: Dict[str, str] = {}
    for scenario_id in sorted(by_scenario):
        groups = by_scenario[scenario_id]
        ordered = sorted(
            groups,
            key=lambda key: (
                _stable_group_digest(split_seed, key[0], key[1]),
                key[0],
                key[1],
            ),
        )
        counts = _apportion_group_counts(len(ordered), normalized_ratios)
        cursor = 0
        for split, count in zip(SPLIT_NAMES_V1, counts):
            for key in ordered[cursor : cursor + count]:
                for episode in sorted(
                    groups[key], key=lambda item: item.episode_id
                ):
                    assignments[episode.episode_id] = split
            cursor += count
    if len(assignments) != len(values):
        raise RuntimeError("internal split assignment did not cover all episodes")
    _validate_split_assignments(values, assignments)
    return assignments


def split_composition_v1(
    episodes: Sequence[MaritimeEpisodeV1],
    split_assignments: Mapping[str, str],
) -> Dict[str, Dict[str, Any]]:
    """Return deterministic policy/scenario/scale composition per split."""
    values = _validate_episode_sequence(episodes)
    assignments = _normalize_split_assignments(values, split_assignments)
    result: Dict[str, Dict[str, Any]] = {}
    for split in SPLIT_NAMES_V1:
        selected = [
            episode
            for episode in values
            if assignments[episode.episode_id] == split
        ]
        result[split] = {
            "episode_count": len(selected),
            "transition_count": int(
                sum(episode.episode_length for episode in selected)
            ),
            "comparison_group_count": len(
                {(item.scenario_id, item.seed) for item in selected}
            ),
            "episode_ids": sorted(item.episode_id for item in selected),
            "policy_composition": _count_values(
                item.policy_id for item in selected
            ),
            "scenario_composition": _count_values(
                item.scenario_id for item in selected
            ),
            "num_buoys_composition": _count_values(
                str(item.num_buoys) for item in selected
            ),
            "seed_composition": _count_values(
                str(item.seed) for item in selected
            ),
            "arrival_trace_ids": sorted(
                {item.arrival_trace_id for item in selected}
            ),
        }
    return result


def validate_dataset_v1(
    episodes: Sequence[MaritimeEpisodeV1],
    split_assignments: Optional[Mapping[str, str]] = None,
    raise_on_error: bool = True,
) -> Dict[str, Any]:
    """Audit exact transition chains, shared traces, and split isolation."""
    errors: List[str] = []
    warnings: List[str] = []
    try:
        values = _validate_episode_sequence(episodes)
    except (TypeError, ValueError) as error:
        if raise_on_error:
            raise
        return {
            "passed": False,
            "schema_version": DATASET_SCHEMA_VERSION_V1,
            "episode_count": 0,
            "transition_count": 0,
            "errors": [str(error)],
            "warnings": [],
            "training_performed": False,
        }

    for episode in values:
        try:
            _revalidate_episode(episode)
        except (TypeError, ValueError) as error:
            errors.append(f"{episode.episode_id}: {error}")
            continue
        if episode.episode_length > 1:
            chain_pairs = (
                ("next_observations", "observations"),
                ("next_uav_positions_m", "uav_positions_m"),
                ("next_rates_bps", "rates_bps"),
                ("next_remaining_energy", "remaining_energy"),
                ("next_raw_queue_bits", "raw_queue_bits"),
                (
                    "next_service_completion_ratio",
                    "service_completion_ratio",
                ),
                ("next_valid_node_mask", "valid_node_mask"),
                ("next_cumulative_distance_m", "cumulative_distance_m"),
                (
                    "next_cumulative_energy_consumed",
                    "cumulative_energy_consumed",
                ),
                (
                    "next_cumulative_payload_bits",
                    "cumulative_payload_bits",
                ),
                ("next_task_completion_ratio", "task_completion_ratio"),
            )
            for next_name, current_name in chain_pairs:
                if not np.array_equal(
                    getattr(episode, next_name)[:-1],
                    getattr(episode, current_name)[1:],
                ):
                    errors.append(
                        f"{episode.episode_id}: {current_name} transition "
                        "chain breaks"
                    )
        if episode.buoy_positions_m.shape[0] > 1 and not np.all(
            episode.buoy_positions_m
            == episode.buoy_positions_m[0:1]
        ):
            errors.append(
                f"{episode.episode_id}: static buoy positions changed"
            )

    # Fairness is assessed per variable-scale case.  Dataset splitting is
    # deliberately coarser at (scenario_id, seed), so both buoy scales for one
    # seed remain in the same partition.
    groups: Dict[
        Tuple[str, int, int], List[MaritimeEpisodeV1]
    ] = {}
    for episode in values:
        groups.setdefault(
            (episode.scenario_id, episode.seed, episode.num_buoys), []
        ).append(episode)
    for group_key, members in sorted(groups.items()):
        trace_ids = {item.arrival_trace_id for item in members}
        if len(trace_ids) != 1:
            errors.append(
                f"{group_key!r}: policies do not share arrival_trace_id"
            )
        reference = sorted(members, key=lambda item: item.episode_id)[0]
        for episode in members:
            if not np.array_equal(
                episode.uav_positions_m[0], reference.uav_positions_m[0]
            ):
                errors.append(
                    f"{group_key!r}: policies do not share initial UAV state"
                )
            if not np.array_equal(
                episode.buoy_positions_m[0],
                reference.buoy_positions_m[0],
            ):
                errors.append(
                    f"{group_key!r}: policies do not share buoy initialization"
                )
            if not np.array_equal(
                episode.raw_queue_bits[0], reference.raw_queue_bits[0]
            ):
                errors.append(
                    f"{group_key!r}: policies do not share initial queues"
                )
            common_length = min(
                episode.episode_length, reference.episode_length
            )
            if not np.array_equal(
                episode.raw_arrival_bits[:common_length],
                reference.raw_arrival_bits[:common_length],
            ):
                errors.append(
                    f"{group_key!r}: policies do not share arrival-trace prefix"
                )

    assignments: Optional[Dict[str, str]] = None
    if split_assignments is not None:
        try:
            assignments = _normalize_split_assignments(
                values, split_assignments
            )
            _validate_split_assignments(values, assignments)
        except (TypeError, ValueError) as error:
            errors.append(str(error))

    report: Dict[str, Any] = {
        "passed": not errors,
        "schema_version": DATASET_SCHEMA_VERSION_V1,
        "goal_spec_version": GOAL_SPEC_VERSION_V1,
        "episode_count": len(values),
        "transition_count": int(
            sum(item.episode_length for item in values)
        ),
        "comparison_group_count": len(groups),
        "arrival_trace_count": len(
            {item.arrival_trace_id for item in values}
        ),
        "all_finite": all(
            all(np.all(np.isfinite(array)) for array in item.arrays().values())
            for item in values
        ),
        "errors": errors,
        "warnings": warnings,
        "training_performed": False,
    }
    if assignments is not None and not errors:
        report["split_composition"] = split_composition_v1(
            values, assignments
        )
    if errors and raise_on_error:
        raise ValueError("schema-v1 dataset validation failed: " + "; ".join(errors))
    return report


def save_dataset_v1(
    output_dir: Any,
    episodes: Sequence[MaritimeEpisodeV1],
    split_assignments: Optional[Mapping[str, str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    overwrite: bool = False,
    split_ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS_V1,
    split_seed: int = 0,
) -> Dict[str, Any]:
    """Persist each episode as NPZ+JSON with checksums and no pickle."""
    values = _validate_episode_sequence(episodes)
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping or None")
    if metadata is not None and metadata.get("training_performed") not in (
        None,
        False,
    ):
        raise ValueError("schema-v1 dataset metadata must record no training")
    ratios = _validate_ratios(split_ratios)
    assignments = (
        deterministic_episode_split_v1(values, ratios, split_seed)
        if split_assignments is None
        else _normalize_split_assignments(values, split_assignments)
    )
    validation = validate_dataset_v1(values, assignments)

    root = Path(output_dir)
    manifest_path = root / "dataset_manifest.json"
    schema_path = root / "dataset_schema.json"
    if not overwrite and (manifest_path.exists() or schema_path.exists()):
        raise FileExistsError(
            f"dataset metadata already exists in {str(root)!r}"
        )
    root.mkdir(parents=True, exist_ok=True)
    episode_dir = root / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)

    schema = build_dataset_schema_v1(values)
    _write_json(schema_path, schema)

    entries: List[Dict[str, Any]] = []
    for index, episode in enumerate(
        sorted(values, key=lambda item: item.episode_id)
    ):
        stem = f"{index:06d}_{_safe_stem(episode.episode_id)}"
        npz_path = episode_dir / f"{stem}.npz"
        json_path = episode_dir / f"{stem}.json"
        if not overwrite and (npz_path.exists() or json_path.exists()):
            raise FileExistsError(
                f"episode output already exists for {episode.episode_id!r}"
            )
        np.savez_compressed(str(npz_path), **episode.arrays())
        episode_metadata = _episode_metadata(
            episode, assignments[episode.episode_id]
        )
        _write_json(json_path, episode_metadata)
        entries.append(
            {
                **episode.identity,
                "arrival_trace_id": episode.arrival_trace_id,
                "split": assignments[episode.episode_id],
                "episode_length": episode.episode_length,
                "episode_return": episode.episode_return,
                "terminal_reason": episode.terminal_reason,
                "npz_file": npz_path.relative_to(root).as_posix(),
                "metadata_file": json_path.relative_to(root).as_posix(),
                "npz_sha256": _sha256(npz_path),
                "metadata_sha256": _sha256(json_path),
            }
        )

    created_at = datetime.now(timezone.utc).isoformat()
    manifest: Dict[str, Any] = {
        "created_at_utc": created_at,
        "dataset_schema_version": DATASET_SCHEMA_VERSION_V1,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION_V1,
        "goal_spec_version": GOAL_SPEC_VERSION_V1,
        "episode_storage": "one compressed NPZ plus one JSON per episode",
        "pickle_used": False,
        "split_group_key": ["scenario_id", "seed"],
        "split_ratios": {
            name: ratio for name, ratio in zip(SPLIT_NAMES_V1, ratios)
        },
        "split_seed": _nonnegative_integer("split_seed", split_seed),
        "split_assignments": {
            episode_id: assignments[episode_id]
            for episode_id in sorted(assignments)
        },
        "split_composition": split_composition_v1(values, assignments),
        "episode_count": len(values),
        "transition_count": int(
            sum(item.episode_length for item in values)
        ),
        "policies": sorted({item.policy_id for item in values}),
        "scenario_ids": sorted({item.scenario_id for item in values}),
        "seeds": sorted({item.seed for item in values}),
        "num_buoys_scales": sorted({item.num_buoys for item in values}),
        "arrival_trace_ids": sorted(
            {item.arrival_trace_id for item in values}
        ),
        "episodes": entries,
        "availability": {
            "queues": True,
            "arrivals": True,
            "link_capacity": True,
            "actual_transmitted_payload": True,
            "service_completion_ratio": True,
            "desired_goal": True,
            "achieved_goal": True,
            "valid_transition_mask": True,
            "valid_node_mask": True,
        },
        "validation": validation,
        "training_performed": False,
        "user_metadata": _json_compatible(
            dict(metadata) if metadata is not None else {}
        ),
    }
    _write_json(manifest_path, manifest)
    return manifest


def load_dataset_v1(
    output_dir: Any,
    split: Optional[str] = None,
    verify_checksums: bool = True,
) -> List[MaritimeEpisodeV1]:
    """Load and validate schema-v1 episodes, optionally selecting a split."""
    root = Path(output_dir)
    manifest = _read_json(root / "dataset_manifest.json")
    schema = _read_json(root / "dataset_schema.json")
    _require_v1_dataset_metadata(manifest, "dataset manifest")
    _require_v1_dataset_metadata(schema, "dataset schema")
    if manifest.get("training_performed") is not False:
        raise ValueError("dataset manifest must record training_performed=false")
    if split is not None and split not in SPLIT_NAMES_V1:
        raise ValueError(f"split must be one of {SPLIT_NAMES_V1}")

    episodes: List[MaritimeEpisodeV1] = []
    for entry in manifest.get("episodes", []):
        if not isinstance(entry, Mapping):
            raise ValueError("manifest episode entries must be objects")
        if split is not None and entry.get("split") != split:
            continue
        npz_path = root / str(entry["npz_file"])
        metadata_path = root / str(entry["metadata_file"])
        if verify_checksums:
            if _sha256(npz_path) != entry.get("npz_sha256"):
                raise ValueError(f"checksum mismatch for {npz_path}")
            if _sha256(metadata_path) != entry.get("metadata_sha256"):
                raise ValueError(f"checksum mismatch for {metadata_path}")
        episode = load_episode_v1(npz_path, metadata_path)
        if episode.episode_id != entry.get("episode_id"):
            raise ValueError("manifest and episode identity disagree")
        if episode.arrival_trace_id != entry.get("arrival_trace_id"):
            raise ValueError("manifest and episode arrival_trace_id disagree")
        episodes.append(episode)

    all_assignments = manifest.get("split_assignments")
    if not isinstance(all_assignments, Mapping):
        raise ValueError("manifest split_assignments must be an object")
    selected_assignments = {
        item.episode_id: all_assignments[item.episode_id]
        for item in episodes
    }
    if episodes:
        validate_dataset_v1(episodes, selected_assignments)
    elif split is None:
        raise ValueError("a complete schema-v1 dataset must contain episodes")
    if split is None:
        if len(episodes) != int(manifest.get("episode_count", -1)):
            raise ValueError("manifest episode_count does not match files")
        transition_count = int(
            sum(item.episode_length for item in episodes)
        )
        if transition_count != int(manifest.get("transition_count", -1)):
            raise ValueError("manifest transition_count does not match files")
    return episodes


def load_episode_v1(
    npz_path: Any, metadata_path: Optional[Any] = None
) -> MaritimeEpisodeV1:
    """Load one v1 episode with ``allow_pickle=False``."""
    array_path = Path(npz_path)
    json_path = (
        Path(metadata_path)
        if metadata_path is not None
        else array_path.with_suffix(".json")
    )
    metadata = _read_json(json_path)
    _require_v1_dataset_metadata(metadata, "episode metadata")
    if metadata.get("trajectory_schema_version") != (
        TRAJECTORY_SCHEMA_VERSION_V1
    ):
        raise ValueError("episode is not trajectory schema v1")
    with np.load(str(array_path), allow_pickle=False) as archive:
        expected_keys = set(metadata.get("array_fields", {}))
        actual_keys = set(archive.files)
        if actual_keys != expected_keys:
            raise ValueError(
                "NPZ array keys disagree with JSON metadata: "
                f"missing={sorted(expected_keys-actual_keys)}, "
                f"unexpected={sorted(actual_keys-expected_keys)}"
            )
        arrays = {name: archive[name].copy() for name in archive.files}
    for name, descriptor in metadata["array_fields"].items():
        array = arrays[name]
        if str(array.dtype) != descriptor.get("dtype"):
            raise ValueError(f"array dtype mismatch for {name!r}")
        if list(array.shape) != descriptor.get("shape"):
            raise ValueError(f"array shape mismatch for {name!r}")

    episode_meta = metadata.get("episode")
    identity = metadata.get("identity")
    config = metadata.get("config")
    if not isinstance(episode_meta, Mapping):
        raise ValueError("episode metadata lacks episode object")
    if not isinstance(identity, Mapping):
        raise ValueError("episode metadata lacks identity object")
    if not isinstance(config, Mapping):
        raise ValueError("episode metadata lacks config object")
    return MaritimeEpisodeV1(
        episode_id=identity["episode_id"],
        policy_id=identity["policy_id"],
        seed=identity["seed"],
        scenario_id=identity["scenario_id"],
        num_buoys=identity["num_buoys"],
        arrival_trace_id=episode_meta["arrival_trace_id"],
        terminal_reason=episode_meta["terminal_reason"],
        area_width=config["area_width"],
        area_height=config["area_height"],
        initial_energy=config["initial_energy"],
        queue_capacity_bits=config["queue_capacity_bits"],
        observation_schema=metadata["observation_schema"],
        episode_return=episode_meta["episode_return"],
        episode_length=episode_meta["episode_length"],
        goal_spec_version=metadata["goal_spec_version"],
        schema_version=metadata["trajectory_schema_version"],
        **arrays,
    )


def validate_saved_dataset_v1(output_dir: Any) -> Dict[str, Any]:
    """Verify checksums, reload, validate, and report exact roundtrip."""
    root = Path(output_dir)
    manifest = _read_json(root / "dataset_manifest.json")
    episodes = load_dataset_v1(root, verify_checksums=True)
    result = validate_dataset_v1(
        episodes, manifest.get("split_assignments")
    )
    result["roundtrip_exact"] = True
    result["checksums_verified"] = True
    return result


def build_dataset_schema_v1(
    episodes: Sequence[MaritimeEpisodeV1],
) -> Dict[str, Any]:
    """Describe v1 arrays, temporal alignment, goals, and units."""
    values = _validate_episode_sequence(episodes)
    observation_schemas: Dict[str, Any] = {}
    for episode in values:
        key = str(episode.num_buoys)
        previous = observation_schemas.setdefault(
            key, episode.observation_schema
        )
        if previous != episode.observation_schema:
            raise ValueError(
                f"inconsistent observation schema at num_buoys={key}"
            )
    example = values[0]
    array_fields: Dict[str, Any] = {}
    episode_level = {
        "desired_goal",
        "achieved_goal_global",
        "achieved_goal_per_buoy_queue_ratio",
        "achieved_goal_per_buoy_completion",
    }
    for name, array in example.arrays().items():
        descriptor: Dict[str, Any] = {
            "dtype": str(array.dtype),
            "rank": int(array.ndim),
            "leading_dimension": (
                "episode-level" if name in episode_level else "episode_length"
            ),
        }
        if name in episode_level:
            descriptor["shape_for_example_scale"] = list(array.shape)
        array_fields[name] = descriptor

    return {
        "dataset_schema_version": DATASET_SCHEMA_VERSION_V1,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION_V1,
        "goal_spec_version": GOAL_SPEC_VERSION_V1,
        "training_performed": False,
        "storage": {
            "unit": "complete variable-length unpadded episode",
            "array_format": "compressed NPZ",
            "metadata_format": "JSON",
            "pickle_allowed": False,
            "one_npz_and_json_per_episode": True,
            "cross_episode_indices_allowed": False,
        },
        "slot_order": {
            "name": "service_before_arrival",
            "payload_constraint": "0 <= D_t <= min(Q_t, C_t)",
            "queue_update": (
                "Q_(t+1)=clip(Q_t-D_t+A_t,0,Q_max)"
            ),
            "overflow": (
                "dropped_arrival_bits=max(Q_t-D_t+A_t-Q_max,0)"
            ),
        },
        "units": {
            "positions_and_distance": "metre",
            "rates": "bit/second",
            "queues_arrivals_capacity_payload_and_drops": "bit",
            "energy": "environment energy unit",
            "ratios": "dimensionless [0,1]",
        },
        "transition_time_alignment": {
            "observations": "s_t before action_t",
            "next_observations": "s_(t+1) after action_t",
            "raw_queue_bits": "Q_t before service",
            "transmitted_payload_bits": "D_t actually delivered",
            "raw_arrival_bits": "A_t admitted after service",
            "next_raw_queue_bits": "Q_(t+1)",
            "link_capacity_bits": "C_t potential allocated capacity",
            "rates_bps": "instantaneous rates at s_t",
            "next_rates_bps": "instantaneous rates at s_(t+1)",
            "valid_node_mask": "real node slots at s_t",
            "next_valid_node_mask": "real node slots at s_(t+1)",
        },
        "reward_component_order": list(REWARD_COMPONENT_NAMES_V1),
        "goals": {
            "desired_goal": {
                "shape": [3],
                "names": list(DESIRED_GOAL_NAMES_V1),
                "source": "task configuration fixed before rollout",
            },
            "achieved_goal_global": {
                "shape": [5],
                "names": list(ACHIEVED_GOAL_GLOBAL_NAMES_V1),
                "source": "final physical/task state",
            },
            "achieved_goal_per_buoy_queue_ratio": {
                "shape": ["num_buoys"],
                "normalization": "final_queue_bits/queue_capacity_bits",
            },
            "achieved_goal_per_buoy_completion": {
                "shape": ["num_buoys"],
                "source": "final service_completion_ratio",
            },
        },
        "array_fields": array_fields,
        "availability": {
            "queues": True,
            "arrivals": True,
            "link_capacity": True,
            "actual_transmitted_payload": True,
            "service_completion_ratio": True,
            "desired_goal": True,
            "achieved_goal": True,
            "node_masks": True,
        },
        "observation_schemas_by_num_buoys": observation_schemas,
    }


def assert_roundtrip_exact_v1(
    expected: Sequence[MaritimeEpisodeV1],
    actual: Sequence[MaritimeEpisodeV1],
) -> None:
    """Raise if two episode sets differ in any scalar, dtype, or array bit."""
    expected_values = _validate_episode_sequence(expected)
    actual_values = _validate_episode_sequence(actual)
    left = {item.episode_id: item for item in expected_values}
    right = {item.episode_id: item for item in actual_values}
    if set(left) != set(right):
        raise ValueError("roundtrip episode IDs differ")
    for episode_id in sorted(left):
        if not trajectories_v1_equal(left[episode_id], right[episode_id]):
            raise ValueError(f"roundtrip mismatch for episode {episode_id!r}")


def _episode_metadata(
    episode: MaritimeEpisodeV1, split: str
) -> Dict[str, Any]:
    arrays = episode.arrays()
    return {
        "dataset_schema_version": DATASET_SCHEMA_VERSION_V1,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION_V1,
        "goal_spec_version": GOAL_SPEC_VERSION_V1,
        "identity": episode.identity,
        "episode": {
            "arrival_trace_id": episode.arrival_trace_id,
            "terminal_reason": episode.terminal_reason,
            "split": split,
            "episode_length": episode.episode_length,
            "episode_return": episode.episode_return,
            "state_dim": episode.state_dim,
            "action_dim": episode.action_dim,
            "training_performed": False,
        },
        "config": {
            "area_width": episode.area_width,
            "area_height": episode.area_height,
            "initial_energy": episode.initial_energy,
            "queue_capacity_bits": episode.queue_capacity_bits,
        },
        "observation_schema": episode.observation_schema,
        "array_fields": {
            name: {"dtype": str(array.dtype), "shape": list(array.shape)}
            for name, array in arrays.items()
        },
    }


def _validate_episode_sequence(
    episodes: Sequence[MaritimeEpisodeV1],
) -> List[MaritimeEpisodeV1]:
    if isinstance(episodes, (str, bytes)) or not isinstance(episodes, Sequence):
        raise TypeError("episodes must be a sequence of MaritimeEpisodeV1")
    values = list(episodes)
    if not values:
        raise ValueError("episodes must not be empty")
    for episode in values:
        if not isinstance(episode, MaritimeEpisodeV1):
            version = getattr(episode, "schema_version", None)
            if version in ("1.0", "v0"):
                raise TypeError(
                    "v0/round-02 trajectories cannot be mixed into dataset v1"
                )
            raise TypeError("every episode must be MaritimeEpisodeV1")
        if episode.schema_version != "v1" or episode.goal_spec_version != "v1":
            raise ValueError("dataset v1 accepts only schema/goal version 'v1'")
    ids = [item.episode_id for item in values]
    if len(ids) != len(set(ids)):
        raise ValueError("episode_id values must be globally unique")
    return values


def _revalidate_episode(episode: MaritimeEpisodeV1) -> None:
    MaritimeEpisodeV1(
        episode_id=episode.episode_id,
        policy_id=episode.policy_id,
        seed=episode.seed,
        scenario_id=episode.scenario_id,
        num_buoys=episode.num_buoys,
        arrival_trace_id=episode.arrival_trace_id,
        terminal_reason=episode.terminal_reason,
        area_width=episode.area_width,
        area_height=episode.area_height,
        initial_energy=episode.initial_energy,
        queue_capacity_bits=episode.queue_capacity_bits,
        observation_schema=episode.observation_schema,
        episode_return=episode.episode_return,
        episode_length=episode.episode_length,
        goal_spec_version=episode.goal_spec_version,
        schema_version=episode.schema_version,
        **episode.arrays(),
    )


def _normalize_split_assignments(
    episodes: Sequence[MaritimeEpisodeV1],
    assignments: Mapping[str, str],
) -> Dict[str, str]:
    if not isinstance(assignments, Mapping):
        raise TypeError("split_assignments must be a mapping")
    normalized = {str(key): str(value) for key, value in assignments.items()}
    expected = {item.episode_id for item in episodes}
    if set(normalized) != expected:
        missing = sorted(expected - set(normalized))
        extra = sorted(set(normalized) - expected)
        raise ValueError(
            f"split assignment coverage mismatch: missing={missing}, extra={extra}"
        )
    invalid = {
        key: value
        for key, value in normalized.items()
        if value not in SPLIT_NAMES_V1
    }
    if invalid:
        raise ValueError(f"invalid split names: {invalid}")
    return normalized


def _validate_split_assignments(
    episodes: Sequence[MaritimeEpisodeV1],
    assignments: Mapping[str, str],
) -> None:
    normalized = _normalize_split_assignments(episodes, assignments)
    groups: Dict[Tuple[str, int], str] = {}
    for episode in episodes:
        key = (episode.scenario_id, episode.seed)
        split = normalized[episode.episode_id]
        previous = groups.setdefault(key, split)
        if previous != split:
            raise ValueError(
                "(scenario_id, seed) comparison group leaks across splits: "
                f"{key!r} appears in {previous!r} and {split!r}"
            )


def _validate_ratios(
    ratios: Sequence[float],
) -> Tuple[float, float, float]:
    if isinstance(ratios, (str, bytes)) or len(ratios) != 3:
        raise ValueError("ratios must contain train, validation, and test")
    values = tuple(float(value) for value in ratios)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("split ratios must be finite and nonnegative")
    total = sum(values)
    if total <= 0.0:
        raise ValueError("at least one split ratio must be positive")
    normalized = tuple(value / total for value in values)
    return normalized  # type: ignore[return-value]


def _apportion_group_counts(
    count: int, ratios: Tuple[float, float, float]
) -> Tuple[int, int, int]:
    if count < 1:
        return (0, 0, 0)
    raw = [ratio * count for ratio in ratios]
    result = [int(math.floor(value)) for value in raw]
    remainder = count - sum(result)
    order = sorted(
        range(3), key=lambda index: (-(raw[index] - result[index]), index)
    )
    for index in order[:remainder]:
        result[index] += 1
    positive = [index for index, ratio in enumerate(ratios) if ratio > 0.0]
    if count >= len(positive):
        for empty_index in positive:
            if result[empty_index]:
                continue
            donors = sorted(
                (
                    index
                    for index in positive
                    if result[index] > 1
                ),
                key=lambda index: (-result[index], index),
            )
            if donors:
                result[donors[0]] -= 1
                result[empty_index] += 1
    if sum(result) != count:
        raise RuntimeError("internal split apportionment error")
    return (result[0], result[1], result[2])


def _require_v1_dataset_metadata(
    metadata: Mapping[str, Any], label: str
) -> None:
    version = metadata.get("dataset_schema_version")
    if version != DATASET_SCHEMA_VERSION_V1:
        if version in ("1.0", "v0"):
            raise ValueError(
                f"{label} is v0/round-02 data and cannot be loaded as v1"
            )
        raise ValueError(f"{label} has unsupported dataset schema version")
    goal = metadata.get("goal_spec_version")
    if goal is not None and goal != GOAL_SPEC_VERSION_V1:
        raise ValueError(f"{label} has unsupported goal spec version")


def _stable_group_digest(seed: int, scenario_id: str, group_seed: int) -> str:
    payload = f"{seed}\x1f{scenario_id}\x1f{group_seed}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _count_values(values: Iterable[str]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return {key: result[key] for key in sorted(result)}


def _safe_stem(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return (result or "episode")[:100]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("JSON metadata contains NaN or Inf")
        return numeric
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON metadata contains NaN or Inf")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"JSON metadata cannot encode {type(value).__name__}")


def _nonnegative_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


# Friendly aliases inside this explicitly v1-named module.
DATASET_SCHEMA_VERSION = DATASET_SCHEMA_VERSION_V1
DEFAULT_SPLIT_RATIOS = DEFAULT_SPLIT_RATIOS_V1
SPLIT_NAMES = SPLIT_NAMES_V1
assert_roundtrip_exact = assert_roundtrip_exact_v1
build_dataset_schema = build_dataset_schema_v1
deterministic_episode_split = deterministic_episode_split_v1
load_dataset = load_dataset_v1
load_episode = load_episode_v1
save_dataset = save_dataset_v1
split_composition = split_composition_v1
validate_dataset = validate_dataset_v1
validate_saved_dataset = validate_saved_dataset_v1


__all__ = [
    "DATASET_SCHEMA_VERSION",
    "DATASET_SCHEMA_VERSION_V1",
    "DEFAULT_SPLIT_RATIOS",
    "DEFAULT_SPLIT_RATIOS_V1",
    "SPLIT_NAMES",
    "SPLIT_NAMES_V1",
    "assert_roundtrip_exact",
    "assert_roundtrip_exact_v1",
    "build_dataset_schema",
    "build_dataset_schema_v1",
    "deterministic_episode_split",
    "deterministic_episode_split_v1",
    "load_dataset",
    "load_dataset_v1",
    "load_episode",
    "load_episode_v1",
    "save_dataset",
    "save_dataset_v1",
    "split_composition",
    "split_composition_v1",
    "validate_dataset",
    "validate_dataset_v1",
    "validate_saved_dataset",
    "validate_saved_dataset_v1",
]
