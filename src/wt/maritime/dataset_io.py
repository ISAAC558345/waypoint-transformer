"""Exact per-episode persistence, grouped splitting, and dataset validation.

Each variable-scale episode is stored as one compressed NPZ plus one JSON
metadata file.  No object arrays or pickle are used.  Dataset partitions are
assigned at ``(scenario_id, seed)`` group granularity, so policies evaluated
from the same environment initialization can never leak across splits.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .trajectory import (
    ACHIEVED_GOAL_NAMES,
    EpisodeTrajectory,
    GOAL_SPEC_VERSION,
    TRAJECTORY_SCHEMA_VERSION,
    default_observation_schema,
    trajectories_equal,
)


DATASET_SCHEMA_VERSION = "1.0"
SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_SPLIT_RATIOS = (0.70, 0.15, 0.15)


def deterministic_episode_split(
    episodes: Sequence[EpisodeTrajectory],
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    seed: int = 0,
) -> Dict[str, str]:
    """Assign complete comparison groups to deterministic dataset splits.

    The indivisible unit is ``(scenario_id, seed)`` rather than an episode.
    Consequently all policies that saw the same scenario initialization remain
    together.  Groups are stratified by scenario before deterministic
    allocation, which records and preserves variable-scale coverage as far as
    the available number of seeds allows.
    """
    validated = _validate_episode_sequence(episodes)
    split_ratios = _validate_ratios(ratios)
    split_seed = _nonnegative_integer("seed", seed)

    by_scenario: Dict[str, Dict[Tuple[str, int], List[EpisodeTrajectory]]] = {}
    for episode in validated:
        scenario_groups = by_scenario.setdefault(episode.scenario_id, {})
        key = (episode.scenario_id, episode.seed)
        scenario_groups.setdefault(key, []).append(episode)

    assignments: Dict[str, str] = {}
    for scenario_id in sorted(by_scenario):
        groups = by_scenario[scenario_id]
        ordered_keys = sorted(
            groups,
            key=lambda key: (
                _stable_group_digest(split_seed, key[0], key[1]),
                key[0],
                key[1],
            ),
        )
        split_group_counts = _apportion_group_counts(
            len(ordered_keys), split_ratios
        )
        cursor = 0
        for split_name, group_count in zip(SPLIT_NAMES, split_group_counts):
            for key in ordered_keys[cursor : cursor + group_count]:
                for episode in sorted(
                    groups[key], key=lambda value: value.episode_id
                ):
                    assignments[episode.episode_id] = split_name
            cursor += group_count
    if len(assignments) != len(validated):
        raise RuntimeError("split assignment did not cover every episode")
    _validate_split_assignments(validated, assignments)
    return assignments


def split_composition(
    episodes: Sequence[EpisodeTrajectory],
    split_assignments: Mapping[str, str],
) -> Dict[str, Dict[str, Any]]:
    """Summarize episodes, transitions, policies, scenarios, scales, and seeds."""
    validated = _validate_episode_sequence(episodes)
    assignments = _normalize_split_assignments(validated, split_assignments)
    result: Dict[str, Dict[str, Any]] = {}
    for split_name in SPLIT_NAMES:
        selected = [
            episode
            for episode in validated
            if assignments[episode.episode_id] == split_name
        ]
        result[split_name] = {
            "episode_count": len(selected),
            "transition_count": int(
                sum(episode.episode_length for episode in selected)
            ),
            "policy_composition": _count_values(
                episode.policy_id for episode in selected
            ),
            "scenario_composition": _count_values(
                episode.scenario_id for episode in selected
            ),
            "num_buoys_composition": _count_values(
                str(episode.num_buoys) for episode in selected
            ),
            "seed_composition": _count_values(
                str(episode.seed) for episode in selected
            ),
            "episode_ids": sorted(
                episode.episode_id for episode in selected
            ),
        }
    return result


def validate_dataset(
    episodes: Sequence[EpisodeTrajectory],
    split_assignments: Optional[Mapping[str, str]] = None,
    raise_on_error: bool = True,
) -> Dict[str, Any]:
    """Audit finiteness, transition continuity, boundaries, and split leakage."""
    errors: List[str] = []
    warnings: List[str] = []
    try:
        validated = _validate_episode_sequence(episodes)
    except (TypeError, ValueError) as error:
        if raise_on_error:
            raise
        return {
            "passed": False,
            "episode_count": 0,
            "transition_count": 0,
            "errors": [str(error)],
            "warnings": [],
        }

    for episode in validated:
        try:
            # Reconstructing from copied fields reruns every trajectory invariant
            # and detects mutation after initial construction.
            _revalidate_episode(episode)
        except (TypeError, ValueError) as error:
            errors.append(f"{episode.episode_id}: {error}")
            continue
        if episode.episode_length > 1:
            if not np.array_equal(
                episode.next_observations[:-1], episode.observations[1:]
            ):
                errors.append(
                    f"{episode.episode_id}: next_observation does not match "
                    "the following observation"
                )
            if not np.array_equal(
                episode.next_uav_positions_m[:-1],
                episode.uav_positions_m[1:],
            ):
                errors.append(
                    f"{episode.episode_id}: physical UAV position chain breaks"
                )
            if not np.array_equal(
                episode.next_remaining_energy[:-1],
                episode.remaining_energy[1:],
            ):
                errors.append(
                    f"{episode.episode_id}: remaining-energy chain breaks"
                )
            for current_name, next_name in (
                ("uav_velocities_mps", "next_uav_velocities_mps"),
                ("buoy_positions_m", "next_buoy_positions_m"),
                (
                    "horizontal_distances_m",
                    "next_horizontal_distances_m",
                ),
                ("distances_3d_m", "next_distances_3d_m"),
                ("channel_gains", "next_channel_gains"),
                ("snr_linear", "next_snr_linear"),
                ("rates_bps", "next_rates_bps"),
                ("total_rate_bps", "next_total_rate_bps"),
            ):
                if not np.array_equal(
                    getattr(episode, next_name)[:-1],
                    getattr(episode, current_name)[1:],
                ):
                    errors.append(
                        f"{episode.episode_id}: {current_name} physical "
                        "transition chain breaks"
                    )
        if not (bool(episode.terminated[-1]) or bool(episode.truncated[-1])):
            errors.append(
                f"{episode.episode_id}: complete episode must end with a "
                "terminated or truncated flag"
            )

    assignments: Optional[Dict[str, str]] = None
    if split_assignments is not None:
        try:
            assignments = _normalize_split_assignments(
                validated, split_assignments
            )
            _validate_split_assignments(validated, assignments)
        except (TypeError, ValueError) as error:
            errors.append(str(error))

    report: Dict[str, Any] = {
        "passed": not errors,
        "episode_count": len(validated),
        "transition_count": int(
            sum(episode.episode_length for episode in validated)
        ),
        "errors": errors,
        "warnings": warnings,
    }
    if assignments is not None and not errors:
        report["split_composition"] = split_composition(
            validated, assignments
        )
    if errors and raise_on_error:
        raise ValueError("dataset validation failed: " + "; ".join(errors))
    return report


def save_dataset(
    output_dir: Any,
    episodes: Sequence[EpisodeTrajectory],
    split_assignments: Optional[Mapping[str, str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Save compressed per-episode files and JSON schema/manifest metadata."""
    validated = _validate_episode_sequence(episodes)
    assignments = (
        deterministic_episode_split(validated)
        if split_assignments is None
        else _normalize_split_assignments(validated, split_assignments)
    )
    validation = validate_dataset(validated, assignments)
    root = Path(output_dir)
    manifest_path = root / "dataset_manifest.json"
    schema_path = root / "dataset_schema.json"
    if not overwrite and (manifest_path.exists() or schema_path.exists()):
        raise FileExistsError(
            f"dataset metadata already exists in {str(root)!r}; choose a new "
            "output directory or pass overwrite=True explicitly"
        )
    root.mkdir(parents=True, exist_ok=True)
    episode_dir = root / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)

    schema = build_dataset_schema(validated)
    _write_json(schema_path, schema)

    entries: List[Dict[str, Any]] = []
    for index, episode in enumerate(
        sorted(validated, key=lambda value: value.episode_id)
    ):
        stem = f"{index:06d}_{_safe_stem(episode.episode_id)}"
        npz_path = episode_dir / f"{stem}.npz"
        episode_metadata_path = episode_dir / f"{stem}.json"
        if not overwrite and (npz_path.exists() or episode_metadata_path.exists()):
            raise FileExistsError(
                f"episode output already exists for {episode.episode_id!r}"
            )
        np.savez_compressed(str(npz_path), **episode.arrays())
        episode_metadata = _episode_metadata(
            episode, assignments[episode.episode_id]
        )
        _write_json(episode_metadata_path, episode_metadata)
        entries.append(
            {
                "episode_id": episode.episode_id,
                "policy_id": episode.policy_id,
                "seed": episode.seed,
                "scenario_id": episode.scenario_id,
                "num_buoys": episode.num_buoys,
                "split": assignments[episode.episode_id],
                "episode_length": episode.episode_length,
                "episode_return": episode.episode_return,
                "terminal_reason": episode.terminal_reason,
                "npz_file": npz_path.relative_to(root).as_posix(),
                "metadata_file": episode_metadata_path.relative_to(
                    root
                ).as_posix(),
                "npz_sha256": _sha256(npz_path),
                "metadata_sha256": _sha256(episode_metadata_path),
            }
        )

    manifest: Dict[str, Any] = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "goal_spec_version": GOAL_SPEC_VERSION,
        "episode_storage": "one compressed NPZ plus one JSON per episode",
        "split_group_key": ["scenario_id", "seed"],
        "split_assignments": {
            episode_id: assignments[episode_id]
            for episode_id in sorted(assignments)
        },
        "split_composition": split_composition(validated, assignments),
        "episode_count": len(validated),
        "transition_count": int(
            sum(episode.episode_length for episode in validated)
        ),
        "episodes": entries,
        "availability": {
            "queues": False,
            "actual_transmitted_payload": False,
            "desired_goal": False,
            "achieved_goal": True,
            "action_mask": bool(
                any(episode.action_mask_available for episode in validated)
            ),
        },
        "validation": validation,
        "user_metadata": _json_compatible(
            dict(metadata) if metadata is not None else {}
        ),
    }
    _write_json(manifest_path, manifest)
    return manifest


def load_dataset(
    output_dir: Any,
    split: Optional[str] = None,
    verify_checksums: bool = True,
) -> List[EpisodeTrajectory]:
    """Load episodes exactly, optionally filtering one manifest split."""
    root = Path(output_dir)
    manifest_path = root / "dataset_manifest.json"
    manifest = _read_json(manifest_path)
    schema = _read_json(root / "dataset_schema.json")
    if manifest.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported dataset schema version")
    if schema.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("dataset schema and loader versions disagree")
    if (
        schema.get("trajectory_schema_version")
        != TRAJECTORY_SCHEMA_VERSION
    ):
        raise ValueError("unsupported trajectory schema version")
    if split is not None and split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}")

    episodes: List[EpisodeTrajectory] = []
    for entry in manifest.get("episodes", []):
        if split is not None and entry.get("split") != split:
            continue
        npz_path = root / entry["npz_file"]
        metadata_path = root / entry["metadata_file"]
        if verify_checksums:
            if _sha256(npz_path) != entry["npz_sha256"]:
                raise ValueError(f"checksum mismatch for {npz_path}")
            if _sha256(metadata_path) != entry["metadata_sha256"]:
                raise ValueError(f"checksum mismatch for {metadata_path}")
        episode = load_episode(npz_path, metadata_path)
        if episode.episode_id != entry["episode_id"]:
            raise ValueError("manifest and episode metadata IDs disagree")
        episodes.append(episode)

    assignment_mapping = manifest.get("split_assignments", {})
    selected_assignments = {
        episode.episode_id: assignment_mapping[episode.episode_id]
        for episode in episodes
    }
    validate_dataset(episodes, selected_assignments)
    if split is None:
        if len(episodes) != int(manifest.get("episode_count", -1)):
            raise ValueError("manifest episode_count does not match loaded data")
        transition_count = int(
            sum(episode.episode_length for episode in episodes)
        )
        if transition_count != int(manifest.get("transition_count", -1)):
            raise ValueError(
                "manifest transition_count does not match loaded data"
            )
    return episodes


def load_episode(npz_path: Any, metadata_path: Optional[Any] = None) -> EpisodeTrajectory:
    """Load one safe NPZ/JSON episode without enabling pickle."""
    array_path = Path(npz_path)
    meta_path = (
        Path(metadata_path)
        if metadata_path is not None
        else array_path.with_suffix(".json")
    )
    metadata = _read_json(meta_path)
    with np.load(str(array_path), allow_pickle=False) as archive:
        actual_keys = set(archive.files)
        expected_keys = set(metadata.get("array_fields", {}))
        if actual_keys != expected_keys:
            raise ValueError(
                f"NPZ array keys disagree with metadata: "
                f"missing={sorted(expected_keys-actual_keys)}, "
                f"unexpected={sorted(actual_keys-expected_keys)}"
            )
        arrays = {name: archive[name].copy() for name in archive.files}
    for name, descriptor in metadata["array_fields"].items():
        array = arrays[name]
        if str(array.dtype) != descriptor["dtype"]:
            raise ValueError(f"array dtype mismatch for {name!r}")
        if list(array.shape) != descriptor["shape"]:
            raise ValueError(f"array shape mismatch for {name!r}")

    scalar = metadata["episode"]
    return EpisodeTrajectory(
        episode_id=scalar["episode_id"],
        policy_id=scalar["policy_id"],
        seed=scalar["seed"],
        scenario_id=scalar["scenario_id"],
        num_buoys=scalar["num_buoys"],
        action_mask_available=scalar["action_mask_available"],
        queue_values_available=scalar["queue_values_available"],
        transmitted_payload_available=scalar[
            "transmitted_payload_available"
        ],
        desired_goal_available=scalar["desired_goal_available"],
        achieved_goal_available=scalar["achieved_goal_available"],
        episode_return=scalar["episode_return"],
        episode_length=scalar["episode_length"],
        terminal_reason=scalar["terminal_reason"],
        observation_schema=metadata["observation_schema"],
        goal_spec_version=scalar["goal_spec_version"],
        schema_version=scalar["schema_version"],
        **arrays,
    )


def validate_saved_dataset(output_dir: Any) -> Dict[str, Any]:
    """Reload all artifacts, verify checksums, and rerun dataset validation."""
    root = Path(output_dir)
    manifest = _read_json(root / "dataset_manifest.json")
    episodes = load_dataset(root, verify_checksums=True)
    assignments = manifest["split_assignments"]
    result = validate_dataset(episodes, assignments)
    result["roundtrip_exact"] = True
    return result


def build_dataset_schema(
    episodes: Sequence[EpisodeTrajectory],
) -> Dict[str, Any]:
    """Describe array semantics, availability, and per-scale observations."""
    validated = _validate_episode_sequence(episodes)
    observation_schemas: Dict[str, Any] = {}
    for episode in validated:
        key = str(episode.num_buoys)
        existing = observation_schemas.get(key)
        if existing is not None and existing != episode.observation_schema:
            raise ValueError(
                f"inconsistent observation schema for num_buoys={key}"
            )
        observation_schemas[key] = episode.observation_schema
    example = validated[0]
    descriptors = {
        name: {
            "dtype": str(array.dtype),
            "rank": int(array.ndim),
            "leading_dimension": "episode_length",
        }
        for name, array in example.arrays().items()
        if name not in ("desired_goal", "achieved_goal")
    }
    descriptors["desired_goal"] = {
        "dtype": "float64",
        "shape": [0],
        "available": False,
        "reason": "the current environment has no episode task goal",
    }
    descriptors["achieved_goal"] = {
        "dtype": "float64",
        "shape": [3],
        "available": True,
        "names": list(ACHIEVED_GOAL_NAMES),
        "source": "final physical state only",
    }
    return {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "goal_spec_version": GOAL_SPEC_VERSION,
        "storage": {
            "unit": "complete variable-length episode",
            "arrays": "compressed NPZ without pickle",
            "metadata": "JSON",
            "temporal_padding": False,
            "cross_episode_indices_allowed": False,
        },
        "transition_identity": {
            "time_steps": "stored explicitly as int64 [T]",
            "episode_level_fields_broadcast_to_every_transition": [
                "episode_id",
                "policy_id",
                "seed",
                "scenario_id",
                "num_buoys",
            ],
        },
        "transition_time_alignment": {
            "observations": "normalized state s_t before action_t",
            "next_observations": "normalized state s_(t+1) after action_t",
            "actions_rewards_done": (
                "action_t, reward_t, terminated_t, and truncated_t describe "
                "the transition s_t -> s_(t+1)"
            ),
            "raw_state_before": (
                "uav_positions_m, uav_velocities_mps, buoy_positions_m, "
                "remaining_energy, horizontal_distances_m, distances_3d_m, "
                "channel_gains, snr_linear, rates_bps, and total_rate_bps "
                "come from info_before and are raw counterparts of s_t"
            ),
            "raw_state_after": (
                "the matching next_* arrays come from post-action info_after "
                "and are raw counterparts of s_(t+1)"
            ),
            "step_link_capacity_source": (
                "next_rates_bps and next_total_rate_bps are the post-action "
                "potential link rates used by the environment reward and by "
                "the cumulative link-capacity proxy; they are not actual "
                "delivered payload"
            ),
            "transition_physics": (
                "actual_energy_consumed, energy_cost, reward components, "
                "action-processing flags, boundary flags, and constraint "
                "flags describe action_t applied from s_t to s_(t+1)"
            ),
            "masks": (
                "valid_transition_mask selects real transitions; "
                "valid_node_mask selects real node slots at s_t; action_mask "
                "has zero width when unavailable"
            ),
        },
        "array_fields": descriptors,
        "reward_component_order": [
            "rate_reward",
            "movement_penalty",
            "boundary_penalty",
            "energy_depletion_penalty",
        ],
        "unavailable_semantics": {
            "queue_values": {
                "shape": "[T, 0]",
                "available": False,
                "reason": "MaritimeEnv does not model data queues",
            },
            "transmitted_payload_bits": {
                "shape": "[T, 0]",
                "available": False,
                "reason": "rate is exposed but actual delivered payload is not",
            },
            "action_mask": {
                "shape_when_unavailable": "[T, 0]",
                "availability_is_per_episode": True,
            },
        },
        "observation_schemas_by_num_buoys": observation_schemas,
    }


def assert_roundtrip_exact(
    expected: Sequence[EpisodeTrajectory],
    actual: Sequence[EpisodeTrajectory],
) -> None:
    """Raise if saved and loaded episode sequences differ at any bit or dtype."""
    expected_by_id = {episode.episode_id: episode for episode in expected}
    actual_by_id = {episode.episode_id: episode for episode in actual}
    if set(expected_by_id) != set(actual_by_id):
        raise ValueError("roundtrip episode IDs differ")
    for episode_id in sorted(expected_by_id):
        if not trajectories_equal(
            expected_by_id[episode_id], actual_by_id[episode_id]
        ):
            raise ValueError(f"roundtrip mismatch for episode {episode_id!r}")


def _episode_metadata(
    episode: EpisodeTrajectory, split: str
) -> Dict[str, Any]:
    arrays = episode.arrays()
    return {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "episode": {
            "episode_id": episode.episode_id,
            "policy_id": episode.policy_id,
            "seed": episode.seed,
            "scenario_id": episode.scenario_id,
            "num_buoys": episode.num_buoys,
            "state_dim": episode.state_dim,
            "action_dim": episode.action_dim,
            "split": split,
            "episode_return": episode.episode_return,
            "episode_length": episode.episode_length,
            "terminal_reason": episode.terminal_reason,
            "action_mask_available": episode.action_mask_available,
            "queue_values_available": episode.queue_values_available,
            "transmitted_payload_available": (
                episode.transmitted_payload_available
            ),
            "desired_goal_available": episode.desired_goal_available,
            "achieved_goal_available": episode.achieved_goal_available,
            "goal_spec_version": episode.goal_spec_version,
            "schema_version": episode.schema_version,
        },
        "observation_schema": episode.observation_schema,
        "array_fields": {
            name: {"dtype": str(array.dtype), "shape": list(array.shape)}
            for name, array in arrays.items()
        },
    }


def _validate_episode_sequence(
    episodes: Sequence[EpisodeTrajectory],
) -> List[EpisodeTrajectory]:
    if isinstance(episodes, (str, bytes)) or not isinstance(episodes, Sequence):
        raise TypeError("episodes must be a sequence of EpisodeTrajectory objects")
    values = list(episodes)
    if not values:
        raise ValueError("episodes must not be empty")
    for episode in values:
        if not isinstance(episode, EpisodeTrajectory):
            raise TypeError("every episode must be an EpisodeTrajectory")
    ids = [episode.episode_id for episode in values]
    if len(ids) != len(set(ids)):
        raise ValueError("episode_id values must be globally unique")
    return values


def _revalidate_episode(episode: EpisodeTrajectory) -> None:
    scalar_names = (
        "episode_id",
        "policy_id",
        "seed",
        "scenario_id",
        "num_buoys",
        "action_mask_available",
        "queue_values_available",
        "transmitted_payload_available",
        "desired_goal_available",
        "achieved_goal_available",
        "episode_return",
        "episode_length",
        "terminal_reason",
        "observation_schema",
        "goal_spec_version",
        "schema_version",
    )
    kwargs = {name: getattr(episode, name) for name in scalar_names}
    kwargs.update(episode.arrays())
    EpisodeTrajectory(**kwargs)


def _normalize_split_assignments(
    episodes: Sequence[EpisodeTrajectory],
    assignments: Mapping[str, str],
) -> Dict[str, str]:
    if not isinstance(assignments, Mapping):
        raise TypeError("split_assignments must map episode_id to split name")
    normalized = {str(key): str(value) for key, value in assignments.items()}
    expected_ids = {episode.episode_id for episode in episodes}
    if set(normalized) != expected_ids:
        missing = sorted(expected_ids - set(normalized))
        extra = sorted(set(normalized) - expected_ids)
        raise ValueError(
            f"split assignment coverage mismatch: missing={missing}, extra={extra}"
        )
    invalid = {
        episode_id: split
        for episode_id, split in normalized.items()
        if split not in SPLIT_NAMES
    }
    if invalid:
        raise ValueError(f"invalid split names: {invalid}")
    return normalized


def _validate_split_assignments(
    episodes: Sequence[EpisodeTrajectory],
    assignments: Mapping[str, str],
) -> None:
    normalized = _normalize_split_assignments(episodes, assignments)
    group_splits: Dict[Tuple[str, int], str] = {}
    for episode in episodes:
        key = (episode.scenario_id, episode.seed)
        split = normalized[episode.episode_id]
        previous = group_splits.setdefault(key, split)
        if previous != split:
            raise ValueError(
                "(scenario_id, seed) comparison group leaks across splits: "
                f"{key!r} appears in {previous!r} and {split!r}"
            )


def _validate_ratios(ratios: Sequence[float]) -> Tuple[float, float, float]:
    if isinstance(ratios, (str, bytes)) or len(ratios) != 3:
        raise ValueError("ratios must contain train, validation, and test values")
    values = tuple(float(value) for value in ratios)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("split ratios must be finite and nonnegative")
    total = sum(values)
    if total <= 0.0:
        raise ValueError("at least one split ratio must be positive")
    return tuple(value / total for value in values)  # type: ignore[return-value]


def _apportion_group_counts(
    group_count: int, ratios: Tuple[float, float, float]
) -> Tuple[int, int, int]:
    if group_count < 1:
        return (0, 0, 0)
    raw = [ratio * group_count for ratio in ratios]
    counts = [int(math.floor(value)) for value in raw]
    remainder = group_count - sum(counts)
    order = sorted(
        range(3), key=lambda index: (-(raw[index] - counts[index]), index)
    )
    for index in order[:remainder]:
        counts[index] += 1

    positive = [index for index, ratio in enumerate(ratios) if ratio > 0.0]
    if group_count >= len(positive):
        for empty_index in positive:
            if counts[empty_index] > 0:
                continue
            donors = sorted(
                (
                    index
                    for index in positive
                    if counts[index] > 1
                ),
                key=lambda index: (-counts[index], index),
            )
            if donors:
                counts[donors[0]] -= 1
                counts[empty_index] += 1
    if sum(counts) != group_count:
        raise RuntimeError("internal split apportionment error")
    return (counts[0], counts[1], counts[2])


def _stable_group_digest(seed: int, scenario_id: str, group_seed: int) -> str:
    payload = f"{seed}\x1f{scenario_id}\x1f{group_seed}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _count_values(values: Iterable[str]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return {key: result[key] for key in sorted(result)}


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not stem:
        stem = "episode"
    return stem[:100]


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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("JSON metadata contains NaN or Inf")
        return numeric
    if isinstance(value, (np.bool_,)):
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


__all__ = [
    "DATASET_SCHEMA_VERSION",
    "DEFAULT_SPLIT_RATIOS",
    "SPLIT_NAMES",
    "assert_roundtrip_exact",
    "build_dataset_schema",
    "deterministic_episode_split",
    "load_dataset",
    "load_episode",
    "save_dataset",
    "split_composition",
    "validate_dataset",
    "validate_saved_dataset",
]
