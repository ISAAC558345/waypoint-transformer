"""Locked Round-3 to causal-policy data adapter for Round-5.

The adapter deliberately separates transition targets from model inputs.  In
particular, the action label is reconstructed from the recorded physical
positions and is never used while constructing a policy token.  One sequence
is materialized for every transition, histories are left padded, and no
history may cross an episode boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..maritime.dataset_v1 import SPLIT_NAMES_V1, load_dataset_v1
from ..round4.round4_data import (
    CANONICAL_GOAL_SOURCE,
    load_round4_spatial_dataset,
)


ROUND5_TRANSITION_SCHEMA_VERSION = "round05_transition_data_v1"
ROUND5_SEQUENCE_SCHEMA_VERSION = "round05_causal_sequence_v1"
ACTION_NORMALIZATION_VERSION = "physical_displacement_over_max_step_v1"
ORACLE_WAYPOINT_VERSION = "terminal_clamped_horizon_delta_v1"
CANONICAL_COLLECTION_POLICIES = (
    "max_queue",
    "nearest_buoy",
    "queue_rate_energy",
    "random_valid",
)

# Accepted names intentionally include the formal experiment variant names.
_CONDITION_ALIASES = {
    "predicted_waypoint": "predicted_waypoint",
    "waypoint": "predicted_waypoint",
    "waypoint_transformer": "predicted_waypoint",
    "wt_transformer": "predicted_waypoint",
    "mlp_waypoint": "predicted_waypoint",
    "direct_goal": "direct_goal",
    "direct_goal_transformer": "direct_goal",
    "state_only": "state_only",
    "state_only_transformer": "state_only",
    "mlp_state_only": "state_only",
    "oracle_waypoint": "oracle_waypoint",
    "oracle_waypoint_transformer": "oracle_waypoint",
}


@dataclass(frozen=True)
class Round5TransitionSplit:
    """Every recorded transition in one authoritative episode split."""

    split_name: str
    states: np.ndarray
    state_masks: np.ndarray
    goals: np.ndarray
    goal_masks: np.ndarray
    node_masks: np.ndarray
    actions: np.ndarray
    actions_m: np.ndarray
    current_positions_m: np.ndarray
    next_positions_m: np.ndarray
    remaining_time: np.ndarray
    oracle_waypoints: np.ndarray
    oracle_waypoints_m: np.ndarray
    oracle_effective_horizon: np.ndarray
    oracle_target_t: np.ndarray
    oracle_endpoint_terminated: np.ndarray
    oracle_endpoint_truncated: np.ndarray
    sample_id: np.ndarray
    episode_id: np.ndarray
    scenario_id: np.ndarray
    seed: np.ndarray
    policy_name: np.ndarray
    arrival_trace_id: np.ndarray
    num_buoys: np.ndarray
    quality_class: np.ndarray
    t: np.ndarray
    episode_length: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray

    def __post_init__(self) -> None:
        if self.split_name not in SPLIT_NAMES_V1:
            raise ValueError("invalid Round-5 split name")
        count = int(self.states.shape[0])
        if count < 1 or self.states.ndim != 2:
            raise ValueError("states must be a non-empty rank-two matrix")
        matrix_names = (
            "state_masks",
            "goals",
            "goal_masks",
            "node_masks",
            "actions",
            "actions_m",
            "current_positions_m",
            "next_positions_m",
            "remaining_time",
            "oracle_waypoints",
            "oracle_waypoints_m",
        )
        for name in matrix_names:
            value = getattr(self, name)
            if value.ndim != 2 or int(value.shape[0]) != count:
                raise ValueError(f"{name} must be a rank-two sample matrix")
            if value.dtype.kind == "O":
                raise ValueError(f"object dtype is forbidden: {name}")
        if self.state_masks.shape != self.states.shape:
            raise ValueError("state mask shape disagrees with states")
        if self.goal_masks.shape != self.goals.shape:
            raise ValueError("goal mask shape disagrees with goals")
        action_dim = int(self.actions.shape[1])
        if action_dim < 1:
            raise ValueError("physical action dimension must be positive")
        for name in (
            "actions_m",
            "current_positions_m",
            "next_positions_m",
            "oracle_waypoints",
            "oracle_waypoints_m",
        ):
            if getattr(self, name).shape[1] != action_dim:
                raise ValueError(f"{name} has inconsistent spatial width")
        if self.remaining_time.shape != (count, 1):
            raise ValueError("remaining_time must have shape [N,1]")

        vector_names = (
            "oracle_effective_horizon",
            "oracle_target_t",
            "oracle_endpoint_terminated",
            "oracle_endpoint_truncated",
            "sample_id",
            "episode_id",
            "scenario_id",
            "seed",
            "policy_name",
            "arrival_trace_id",
            "num_buoys",
            "quality_class",
            "t",
            "episode_length",
            "terminated",
            "truncated",
        )
        for name in vector_names:
            value = getattr(self, name)
            if value.shape != (count,):
                raise ValueError(f"{name} must have shape [N]")
            if value.dtype.kind == "O":
                raise ValueError(f"object dtype is forbidden: {name}")

        numeric_names = (
            "states",
            "goals",
            "actions",
            "actions_m",
            "current_positions_m",
            "next_positions_m",
            "remaining_time",
            "oracle_waypoints",
            "oracle_waypoints_m",
        )
        for name in numeric_names:
            if not np.all(np.isfinite(getattr(self, name))):
                raise ValueError(f"{name} contains NaN or Inf")
        if np.any(self.states[~self.state_masks] != 0.0):
            raise ValueError("masked state padding must be zero")
        if np.any(self.goals[~self.goal_masks] != 0.0):
            raise ValueError("masked goal padding must be zero")
        if np.any(self.remaining_time <= 0.0) or np.any(
            self.remaining_time > 1.0 + 1.0e-7
        ):
            raise ValueError("remaining_time must lie in (0,1]")
        if np.any(self.oracle_effective_horizon < 1):
            raise ValueError("oracle effective horizon must be positive")
        if np.any(self.oracle_target_t != self.t + self.oracle_effective_horizon):
            raise ValueError("oracle target index is inconsistent")
        if np.any(self.oracle_target_t > self.episode_length):
            raise ValueError("oracle target crossed an episode boundary")
        if len(set(self.sample_id.tolist())) != count:
            raise ValueError("transition sample IDs are not unique")
        reconstructed = self.next_positions_m - self.current_positions_m
        if not np.allclose(reconstructed, self.actions_m, rtol=0.0, atol=1e-10):
            raise ValueError("action labels are not physical position deltas")

        # Every episode must be complete, contiguous, and represented exactly
        # once per transition inside this split.
        for episode in np.unique(self.episode_id):
            rows = np.flatnonzero(self.episode_id == episode)
            ordered = rows[np.argsort(self.t[rows], kind="stable")]
            length = int(self.episode_length[ordered[0]])
            if ordered.size != length:
                raise ValueError("transition coverage is incomplete for an episode")
            if not np.array_equal(
                self.t[ordered], np.arange(length, dtype=self.t.dtype)
            ):
                raise ValueError("episode timesteps are not contiguous")
            if np.any(self.episode_length[ordered] != length):
                raise ValueError("episode_length changed within an episode")

    def __len__(self) -> int:
        return int(self.states.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[1])

    @property
    def goal_dim(self) -> int:
        return int(self.goals.shape[1])

    @property
    def node_mask_dim(self) -> int:
        return int(self.node_masks.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    def evaluation_metadata(self) -> Mapping[str, np.ndarray]:
        """Return grouping fields that are never policy predictor features."""

        return MappingProxyType(
            {
                "split": np.full((len(self),), self.split_name, dtype="<U10"),
                "sample_id": self.sample_id,
                "episode_id": self.episode_id,
                "scenario_id": self.scenario_id,
                "seed": self.seed,
                "policy_name": self.policy_name,
                "policy_id": self.policy_name,
                "arrival_trace_id": self.arrival_trace_id,
                "num_buoys": self.num_buoys,
                "quality_class": self.quality_class,
                "t": self.t,
                "episode_length": self.episode_length,
                "terminated": self.terminated,
                "truncated": self.truncated,
                "current_positions_m": self.current_positions_m,
                "next_positions_m": self.next_positions_m,
                "actions_m": self.actions_m,
                "oracle_effective_horizon": self.oracle_effective_horizon,
                "oracle_target_t": self.oracle_target_t,
                "oracle_endpoint_terminated": self.oracle_endpoint_terminated,
                "oracle_endpoint_truncated": self.oracle_endpoint_truncated,
            }
        )


@dataclass(frozen=True)
class Round5TransitionDataset:
    """Schema-derived, fixed-split transition dataset for Round-5."""

    schema_version: str
    source_dir: Path
    source_run_id: str
    state_dim: int
    goal_dim: int
    node_mask_dim: int
    action_dim: int
    episode_horizon: int
    waypoint_horizon: int
    maximum_step_distance_m: float
    waypoint_normalization_distance_m: float
    splits: Mapping[str, Round5TransitionSplit]
    schema: Mapping[str, Any]
    provenance: Mapping[str, Any]
    source_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.schema_version != ROUND5_TRANSITION_SCHEMA_VERSION:
            raise ValueError("unexpected Round-5 transition schema")
        if set(self.splits) != set(SPLIT_NAMES_V1):
            raise ValueError("all three locked splits are required")
        if self.maximum_step_distance_m <= 0.0:
            raise ValueError("maximum step distance must be positive")
        if self.waypoint_horizon < 1 or self.episode_horizon < 1:
            raise ValueError("horizons must be positive")
        expected_dims = (
            self.state_dim,
            self.goal_dim,
            self.node_mask_dim,
            self.action_dim,
        )
        episode_sets: List[set[str]] = []
        sample_ids: List[str] = []
        for split_name in SPLIT_NAMES_V1:
            split = self.splits[split_name]
            if split.split_name != split_name:
                raise ValueError("split mapping and object disagree")
            observed = (
                split.state_dim,
                split.goal_dim,
                split.node_mask_dim,
                split.action_dim,
            )
            if observed != expected_dims:
                raise ValueError("schema dimensions changed across splits")
            episode_sets.append(set(map(str, split.episode_id.tolist())))
            sample_ids.extend(map(str, split.sample_id.tolist()))
        for index, left in enumerate(episode_sets):
            for right in episode_sets[index + 1 :]:
                if left & right:
                    raise ValueError("episode leakage across Round-5 splits")
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("transition sample IDs leak across splits")

    @property
    def train(self) -> Round5TransitionSplit:
        return self.splits["train"]

    @property
    def validation(self) -> Round5TransitionSplit:
        return self.splits["validation"]

    @property
    def test(self) -> Round5TransitionSplit:
        return self.splits["test"]

    def split(self, split_name: str) -> Round5TransitionSplit:
        if split_name not in SPLIT_NAMES_V1:
            raise ValueError(f"split must be one of {SPLIT_NAMES_V1}")
        return self.splits[split_name]

    def summary(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_run_id": self.source_run_id,
            "sample_counts": {
                name: len(self.splits[name]) for name in SPLIT_NAMES_V1
            },
            "episode_counts": {
                name: len(set(self.splits[name].episode_id.tolist()))
                for name in SPLIT_NAMES_V1
            },
            "state_dim": self.state_dim,
            "goal_dim": self.goal_dim,
            "node_mask_dim": self.node_mask_dim,
            "action_dim": self.action_dim,
            "episode_horizon": self.episode_horizon,
            "waypoint_horizon": self.waypoint_horizon,
            "maximum_step_distance_m": self.maximum_step_distance_m,
            "waypoint_normalization_distance_m": (
                self.waypoint_normalization_distance_m
            ),
        }


@dataclass(frozen=True)
class Round5SequenceSplit:
    """Mapping-first causal histories, one sequence per transition."""

    split_name: str
    context_length: int
    condition_source: str
    states: np.ndarray
    node_masks: np.ndarray
    conditions: np.ndarray
    remaining_time: np.ndarray
    attention_mask: np.ndarray
    actions: np.ndarray
    loss_mask: np.ndarray
    metadata: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if self.split_name not in SPLIT_NAMES_V1:
            raise ValueError("invalid sequence split")
        if self.context_length < 1:
            raise ValueError("context_length must be positive")
        count = int(self.states.shape[0])
        expected_prefix = (count, self.context_length)
        for name in (
            "states",
            "node_masks",
            "conditions",
            "remaining_time",
            "actions",
        ):
            value = getattr(self, name)
            if value.ndim != 3 or value.shape[:2] != expected_prefix:
                raise ValueError(f"{name} must have shape [N,H,D]")
            if value.dtype.kind == "O" or not np.all(np.isfinite(value)):
                raise ValueError(f"invalid sequence tensor: {name}")
        for name in ("attention_mask", "loss_mask"):
            value = getattr(self, name)
            if value.shape != expected_prefix or value.dtype != np.bool_:
                raise ValueError(f"{name} must be bool [N,H]")
        if np.any(self.loss_mask & ~self.attention_mask):
            raise ValueError("padding timesteps may not contribute to loss")
        if not np.all(np.sum(self.loss_mask, axis=1) == 1):
            raise ValueError("each sequence must supervise exactly its final token")
        if not np.all(self.loss_mask[:, -1]):
            raise ValueError("the final sequence token must be the supervised anchor")
        if np.any(self.attention_mask[:, 1:] < self.attention_mask[:, :-1]):
            raise ValueError("sequence padding must be on the left")
        if np.any(self.states[~self.attention_mask] != 0.0):
            raise ValueError("padded state tokens must be zero")
        if np.any(self.node_masks[~self.attention_mask] != 0.0):
            raise ValueError("padded node-mask tokens must be zero")
        if self.conditions.shape[2] and np.any(
            self.conditions[~self.attention_mask] != 0.0
        ):
            raise ValueError("padded condition tokens must be zero")
        if np.any(self.remaining_time[~self.attention_mask] != 0.0):
            raise ValueError("padded remaining-time tokens must be zero")
        if set(("policy_name", "episode_id", "t")) - set(self.metadata):
            raise ValueError("sequence metadata lacks required grouping fields")
        for name, values in self.metadata.items():
            if np.asarray(values).shape != (count,):
                raise ValueError(f"metadata {name!r} must have shape [N]")

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("sequence index must be an integer")
        resolved = int(index)
        if resolved < 0:
            resolved += len(self)
        if resolved < 0 or resolved >= len(self):
            raise IndexError(index)
        return {
            "states": self.states[resolved],
            "node_masks": self.node_masks[resolved],
            "conditions": self.conditions[resolved],
            "remaining_time": self.remaining_time[resolved],
            "attention_mask": self.attention_mask[resolved],
            "actions": self.actions[resolved],
            "loss_mask": self.loss_mask[resolved],
            "metadata": {
                name: _python_scalar(values[resolved])
                for name, values in self.metadata.items()
            },
        }

    @property
    def policy_names(self) -> np.ndarray:
        return np.asarray(self.metadata["policy_name"])

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[2])

    @property
    def node_mask_dim(self) -> int:
        return int(self.node_masks.shape[2])

    @property
    def condition_dim(self) -> int:
        return int(self.conditions.shape[2])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[2])

    def as_mapping(self) -> Mapping[str, np.ndarray]:
        """Return only canonical numeric batch keys (never action inputs)."""

        return MappingProxyType(
            {
                "states": self.states,
                "node_masks": self.node_masks,
                "conditions": self.conditions,
                "remaining_time": self.remaining_time,
                "attention_mask": self.attention_mask,
                "actions": self.actions,
                "loss_mask": self.loss_mask,
            }
        )

    def summary(self) -> Dict[str, Any]:
        valid_tokens = int(np.sum(self.attention_mask))
        return {
            "schema_version": ROUND5_SEQUENCE_SCHEMA_VERSION,
            "split": self.split_name,
            "context_length": self.context_length,
            "condition_source": self.condition_source,
            "sequence_count": len(self),
            "state_dim": self.state_dim,
            "node_mask_dim": self.node_mask_dim,
            "condition_dim": self.condition_dim,
            "action_dim": self.action_dim,
            "valid_token_count": valid_tokens,
            "padding_token_count": int(
                self.attention_mask.size - valid_tokens
            ),
            "loss_token_count": int(np.sum(self.loss_mask)),
            "history_action_input": False,
        }


class PolicyBalancedBatchSampler:
    """Deterministic equal-policy full-batch sampler.

    Sampling is without replacement until a policy stream is exhausted, then
    that stream is deterministically reshuffled.  Thus shorter policy streams
    are cycled without changing the equal count per batch.
    """

    def __init__(
        self,
        policy_names: Sequence[Any],
        batch_size: int,
        *,
        seed: int = 0,
        epoch: int = 0,
        num_batches: Optional[int] = None,
        expected_policies: Optional[Sequence[str]] = CANONICAL_COLLECTION_POLICIES,
    ) -> None:
        labels = np.asarray(policy_names)
        if labels.ndim != 1 or labels.size < 1 or labels.dtype.kind == "O":
            raise ValueError("policy_names must be a non-empty string vector")
        if isinstance(batch_size, bool) or int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        policies = tuple(sorted(map(str, np.unique(labels).tolist())))
        if expected_policies is not None:
            expected = tuple(sorted(map(str, expected_policies)))
            if policies != expected:
                raise ValueError(
                    f"policy set changed: expected {expected}, observed {policies}"
                )
        if int(batch_size) % len(policies):
            raise ValueError("batch_size must be divisible by the policy count")
        if isinstance(seed, bool) or int(seed) < 0:
            raise ValueError("seed must be a nonnegative integer")
        if isinstance(epoch, bool) or int(epoch) < 0:
            raise ValueError("epoch must be a nonnegative integer")
        batches = (
            int(math.ceil(labels.size / int(batch_size)))
            if num_batches is None
            else int(num_batches)
        )
        if batches < 1:
            raise ValueError("num_batches must be positive")
        self.policy_names = labels.astype(str, copy=True)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.num_batches = batches
        self.policies = policies
        self.samples_per_policy = self.batch_size // len(self.policies)
        self._indices = {
            name: np.flatnonzero(self.policy_names == name).astype(np.int64)
            for name in self.policies
        }
        if any(values.size == 0 for values in self._indices.values()):
            raise ValueError("each policy must contain at least one sequence")

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or int(epoch) < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(_sampler_seed(self.seed, self.epoch))
        streams = {
            name: rng.permutation(values).astype(np.int64, copy=False)
            for name, values in self._indices.items()
        }
        cursors = {name: 0 for name in self.policies}
        for _ in range(self.num_batches):
            pieces: List[np.ndarray] = []
            for name in self.policies:
                need = self.samples_per_policy
                chosen: List[np.ndarray] = []
                while need:
                    stream = streams[name]
                    cursor = cursors[name]
                    available = int(stream.size - cursor)
                    take = min(need, available)
                    if take:
                        chosen.append(stream[cursor : cursor + take])
                        cursors[name] = cursor + take
                        need -= take
                    if cursors[name] == stream.size:
                        streams[name] = rng.permutation(
                            self._indices[name]
                        ).astype(np.int64, copy=False)
                        cursors[name] = 0
                pieces.append(np.concatenate(chosen))
            batch = np.concatenate(pieces)
            batch = batch[rng.permutation(batch.size)]
            yield batch.astype(np.int64, copy=False).tolist()

    def audit_epoch(self) -> Dict[str, Any]:
        counts = {name: 0 for name in self.policies}
        digest = hashlib.sha256()
        for batch in self:
            array = np.asarray(batch, dtype=np.int64)
            digest.update(array.tobytes(order="C"))
            for name in self.policies:
                counts[name] += int(np.sum(self.policy_names[array] == name))
        return {
            "sampler": "policy_balanced",
            "seed": self.seed,
            "epoch": self.epoch,
            "batch_size": self.batch_size,
            "num_batches": self.num_batches,
            "samples_per_policy_per_batch": self.samples_per_policy,
            "policy_counts": counts,
            "index_stream_sha256": digest.hexdigest(),
        }


def load_round5_transition_dataset(
    source_dir: Any,
    *,
    waypoint_horizon: int = 30,
    goal_source: str = CANONICAL_GOAL_SOURCE,
    verify_round3_lock: bool = True,
    reachability_tolerance_m: float = 1.0e-6,
) -> Round5TransitionDataset:
    """Load every locked transition and derive safe Round-5 labels.

    Round-4's H=1 adapter is reused as the audited schema/padding view because
    its H=1 anchors are exactly the complete set of stored transitions.  Raw
    episodes are independently loaded to reconstruct the physical action and
    terminal-clamped diagnostic waypoint.
    """

    root = Path(source_dir).expanduser().resolve()
    horizon = _positive_integer("waypoint_horizon", waypoint_horizon)
    tolerance = float(reachability_tolerance_m)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("reachability_tolerance_m must be finite and nonnegative")
    if goal_source != CANONICAL_GOAL_SOURCE:
        raise ValueError("formal Round-5 requires the locked canonical goal source")

    anchor_dataset = load_round4_spatial_dataset(
        root,
        horizon=1,
        goal_source=goal_source,
        verify_round3_lock=verify_round3_lock,
        reachability_tolerance_m=tolerance,
    )
    episodes = load_dataset_v1(root, split=None, verify_checksums=True)
    by_episode = {str(item.episode_id): item for item in episodes}
    manifest = _read_json(root / "dataset_manifest.json")
    run_config = _read_json(root / "run_config.json")
    assignments = {
        str(key): str(value)
        for key, value in manifest["split_assignments"].items()
    }
    if set(assignments) != set(by_episode):
        raise ValueError("authoritative split assignment coverage changed")
    episode_horizon = _positive_integer(
        "episode_horizon", run_config.get("episode_horizon")
    )
    maximum_step = float(anchor_dataset.max_step_distance_m)
    waypoint_scale = horizon * maximum_step

    coordinate_dims = {
        int(item.uav_positions_m.shape[1]) for item in episodes
    }
    if len(coordinate_dims) != 1:
        raise ValueError("physical coordinate dimension changed across episodes")
    action_dim = coordinate_dims.pop()
    if action_dim != int(anchor_dataset.output_dim):
        raise ValueError("Round-4 spatial output disagrees with physical action dim")

    splits: Dict[str, Round5TransitionSplit] = {}
    maximum_action_norm = 0.0
    for split_name in SPLIT_NAMES_V1:
        anchor = anchor_dataset.split(split_name)
        expected_ids = {
            episode_id
            for episode_id, assigned in assignments.items()
            if assigned == split_name
        }
        if set(map(str, anchor.episode_id.tolist())) != expected_ids:
            raise ValueError("H=1 anchors do not preserve episode split coverage")
        split = _build_transition_split(
            split_name=split_name,
            anchor=anchor,
            episodes_by_id=by_episode,
            action_dim=action_dim,
            maximum_step_distance_m=maximum_step,
            episode_horizon=episode_horizon,
            waypoint_horizon=horizon,
            waypoint_normalization_distance_m=waypoint_scale,
            tolerance_m=tolerance,
        )
        splits[split_name] = split
        maximum_action_norm = max(
            maximum_action_norm,
            float(np.max(np.linalg.norm(split.actions, axis=1))),
        )

    state_dim = int(anchor_dataset.state_dim)
    goal_dim = int(anchor_dataset.goal_dim)
    node_dim = int(anchor_dataset.max_num_buoys)
    schema = MappingProxyType(
        {
            "schema_version": ROUND5_TRANSITION_SCHEMA_VERSION,
            "dimensions": {
                "state_dim": state_dim,
                "goal_dim": goal_dim,
                "node_mask_dim": node_dim,
                "action_dim": action_dim,
            },
            "predictor_fields": {
                "state": "masked state_t",
                "node_mask": "valid_node_mask_t padded by locked schema",
                "remaining_time": (
                    "(configured_episode_horizon - t) / configured_episode_horizon"
                ),
                "predicted_waypoint": "provided only by frozen Round-4 adapter",
            },
            "target": {
                "raw": "next_uav_positions_m - uav_positions_m",
                "normalized": "raw / maximum_step_distance_m",
                "normalization_version": ACTION_NORMALIZATION_VERSION,
                "stored_episode_actions_used": False,
            },
            "oracle_waypoint": {
                "role": "non_deployable_diagnostic_only",
                "raw": "q_(min(t+H,T)) - q_t",
                "normalized": "raw / (H * maximum_step_distance_m)",
                "terminal_handling": "clamp to valid final next-state endpoint",
                "version": ORACLE_WAYPOINT_VERSION,
            },
            "history_action_input": False,
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
            "transition_anchor": "audited_round4_H1_all_transitions",
            "action_source": (
                "recorded next_uav_positions_m minus recorded uav_positions_m"
            ),
            "stored_action_array_used_as_target": False,
            "stored_action_array_used_as_input": False,
            "future_action_input": False,
            "actual_episode_length_used_as_predictor": False,
            "remaining_time_uses_known_configured_horizon": True,
            "oracle_waypoint_non_deployable": True,
            "maximum_normalized_action_norm": maximum_action_norm,
            "fixed_split_transition_counts": {
                name: len(splits[name]) for name in SPLIT_NAMES_V1
            },
            "training_performed": False,
        }
    )
    return Round5TransitionDataset(
        schema_version=ROUND5_TRANSITION_SCHEMA_VERSION,
        source_dir=root,
        source_run_id=str(run_config["run_id"]),
        state_dim=state_dim,
        goal_dim=goal_dim,
        node_mask_dim=node_dim,
        action_dim=action_dim,
        episode_horizon=episode_horizon,
        waypoint_horizon=horizon,
        maximum_step_distance_m=maximum_step,
        waypoint_normalization_distance_m=waypoint_scale,
        splits=MappingProxyType(splits),
        schema=schema,
        provenance=provenance,
        source_hashes=MappingProxyType(dict(anchor_dataset.source_hashes)),
    )


def build_round5_sequence_split(
    transitions: Round5TransitionSplit,
    *,
    context_length: int,
    condition_source: str,
    waypoint_cache_split: Optional[Any] = None,
) -> Round5SequenceSplit:
    """Build deterministic, left-padded episode-safe causal histories."""

    if not isinstance(transitions, Round5TransitionSplit):
        raise TypeError("transitions must be Round5TransitionSplit")
    context = _positive_integer("context_length", context_length)
    canonical = _canonical_condition_source(condition_source)
    if canonical == "predicted_waypoint":
        condition_matrix = _validated_predicted_waypoints(
            transitions, waypoint_cache_split
        )
    elif canonical == "direct_goal":
        if waypoint_cache_split is not None:
            raise ValueError("direct-goal control must not receive waypoint cache")
        condition_matrix = transitions.goals
    elif canonical == "oracle_waypoint":
        if waypoint_cache_split is not None:
            raise ValueError("oracle diagnostic must not receive predicted cache")
        condition_matrix = transitions.oracle_waypoints
    else:
        if waypoint_cache_split is not None:
            raise ValueError("state-only control must not receive waypoint cache")
        condition_matrix = np.empty((len(transitions), 0), dtype=np.float32)

    count = len(transitions)
    states = np.zeros(
        (count, context, transitions.state_dim), dtype=np.float32
    )
    nodes = np.zeros(
        (count, context, transitions.node_mask_dim), dtype=np.float32
    )
    conditions = np.zeros(
        (count, context, int(condition_matrix.shape[1])), dtype=np.float32
    )
    remaining = np.zeros((count, context, 1), dtype=np.float32)
    actions = np.zeros(
        (count, context, transitions.action_dim), dtype=np.float32
    )
    attention = np.zeros((count, context), dtype=np.bool_)
    loss = np.zeros((count, context), dtype=np.bool_)

    row_lookup: Dict[Tuple[str, int], int] = {}
    for row, (episode_id, timestep) in enumerate(
        zip(transitions.episode_id.tolist(), transitions.t.tolist())
    ):
        key = (str(episode_id), int(timestep))
        if key in row_lookup:
            raise ValueError("duplicate episode/timestep transition key")
        row_lookup[key] = row

    for anchor in range(count):
        episode_id = str(transitions.episode_id[anchor])
        timestep = int(transitions.t[anchor])
        start = max(0, timestep - context + 1)
        source_rows = [
            row_lookup[(episode_id, historical_t)]
            for historical_t in range(start, timestep + 1)
        ]
        width = len(source_rows)
        destination = slice(context - width, context)
        states[anchor, destination] = transitions.states[source_rows]
        nodes[anchor, destination] = transitions.node_masks[source_rows].astype(
            np.float32
        )
        if conditions.shape[2]:
            conditions[anchor, destination] = condition_matrix[source_rows]
        remaining[anchor, destination] = transitions.remaining_time[source_rows]
        # Actions are labels only.  The policy model receives no action tensor.
        actions[anchor, destination] = transitions.actions[source_rows]
        attention[anchor, destination] = True
        loss[anchor, -1] = True

    metadata = MappingProxyType(
        {
            "sample_id": transitions.sample_id.copy(),
            "episode_id": transitions.episode_id.copy(),
            "scenario_id": transitions.scenario_id.copy(),
            "seed": transitions.seed.copy(),
            "policy_name": transitions.policy_name.copy(),
            "policy_id": transitions.policy_name.copy(),
            "arrival_trace_id": transitions.arrival_trace_id.copy(),
            "num_buoys": transitions.num_buoys.copy(),
            "quality_class": transitions.quality_class.copy(),
            "t": transitions.t.copy(),
            "episode_length": transitions.episode_length.copy(),
            "terminated": transitions.terminated.copy(),
            "truncated": transitions.truncated.copy(),
        }
    )
    return Round5SequenceSplit(
        split_name=transitions.split_name,
        context_length=context,
        condition_source=canonical,
        states=states,
        node_masks=nodes,
        conditions=conditions,
        remaining_time=remaining,
        attention_mask=attention,
        actions=actions,
        loss_mask=loss,
        metadata=metadata,
    )


def build_round5_sequence_datasets(
    dataset: Round5TransitionDataset,
    *,
    context_length: int,
    condition_source: str,
    waypoint_cache: Optional[Any] = None,
) -> Mapping[str, Round5SequenceSplit]:
    """Build all fixed splits with identical sequence semantics."""

    if not isinstance(dataset, Round5TransitionDataset):
        raise TypeError("dataset must be Round5TransitionDataset")
    cache_splits = getattr(waypoint_cache, "splits", None)
    result: Dict[str, Round5SequenceSplit] = {}
    for split_name in SPLIT_NAMES_V1:
        cache_split = None
        if cache_splits is not None:
            cache_split = cache_splits[split_name]
        elif waypoint_cache is not None:
            raise TypeError("waypoint_cache must expose a split mapping")
        result[split_name] = build_round5_sequence_split(
            dataset.split(split_name),
            context_length=context_length,
            condition_source=condition_source,
            waypoint_cache_split=cache_split,
        )
    return MappingProxyType(result)


def _build_transition_split(
    *,
    split_name: str,
    anchor: Any,
    episodes_by_id: Mapping[str, Any],
    action_dim: int,
    maximum_step_distance_m: float,
    episode_horizon: int,
    waypoint_horizon: int,
    waypoint_normalization_distance_m: float,
    tolerance_m: float,
) -> Round5TransitionSplit:
    count = len(anchor)
    raw_actions = np.empty((count, action_dim), dtype=np.float64)
    current_positions = np.empty((count, action_dim), dtype=np.float64)
    next_positions = np.empty((count, action_dim), dtype=np.float64)
    remaining = np.empty((count, 1), dtype=np.float32)
    oracle_raw = np.empty((count, action_dim), dtype=np.float64)
    effective = np.empty((count,), dtype=np.int64)
    target_t = np.empty((count,), dtype=np.int64)
    endpoint_terminated = np.empty((count,), dtype=np.bool_)
    endpoint_truncated = np.empty((count,), dtype=np.bool_)
    terminated = np.empty((count,), dtype=np.bool_)
    truncated = np.empty((count,), dtype=np.bool_)

    for episode_id in np.unique(anchor.episode_id):
        key = str(episode_id)
        if key not in episodes_by_id:
            raise ValueError("transition anchor references a missing episode")
        episode = episodes_by_id[key]
        rows = np.flatnonzero(anchor.episode_id == episode_id)
        times = np.asarray(anchor.t[rows], dtype=np.int64)
        if np.any(times < 0) or np.any(times >= int(episode.episode_length)):
            raise ValueError("transition timestep lies outside its episode")
        if np.any(times >= episode_horizon):
            raise ValueError("transition exceeds the configured episode horizon")
        current_positions[rows] = np.asarray(
            episode.uav_positions_m[times], dtype=np.float64
        )
        next_positions[rows] = np.asarray(
            episode.next_uav_positions_m[times], dtype=np.float64
        )
        raw_actions[rows] = next_positions[rows] - current_positions[rows]
        remaining[rows, 0] = (
            (episode_horizon - times).astype(np.float64) / episode_horizon
        ).astype(np.float32)

        local_effective = np.minimum(
            waypoint_horizon, int(episode.episode_length) - times
        ).astype(np.int64)
        endpoint_indices = times + local_effective - 1
        oracle_endpoint = np.asarray(
            episode.next_uav_positions_m[endpoint_indices], dtype=np.float64
        )
        oracle_raw[rows] = oracle_endpoint - current_positions[rows]
        effective[rows] = local_effective
        target_t[rows] = times + local_effective
        endpoint_terminated[rows] = np.asarray(
            episode.terminated[endpoint_indices], dtype=np.bool_
        )
        endpoint_truncated[rows] = np.asarray(
            episode.truncated[endpoint_indices], dtype=np.bool_
        )
        terminated[rows] = np.asarray(episode.terminated[times], dtype=np.bool_)
        truncated[rows] = np.asarray(episode.truncated[times], dtype=np.bool_)

    action_norms_m = np.linalg.norm(raw_actions, axis=1)
    if np.any(action_norms_m > maximum_step_distance_m + tolerance_m):
        raise ValueError("executed physical action exceeds maximum step distance")
    oracle_norms_m = np.linalg.norm(oracle_raw, axis=1)
    reachable = effective.astype(np.float64) * maximum_step_distance_m
    if np.any(oracle_norms_m > reachable + tolerance_m):
        raise ValueError("terminal-clamped oracle waypoint is unreachable")
    normalized_actions = (
        raw_actions / maximum_step_distance_m
    ).astype(np.float32)
    oracle = (
        oracle_raw / waypoint_normalization_distance_m
    ).astype(np.float32)
    if not np.allclose(
        normalized_actions.astype(np.float64) * maximum_step_distance_m,
        raw_actions,
        rtol=2.0e-6,
        atol=2.0e-6,
    ):
        raise ValueError("physical action normalization is not reversible")

    # H=1 must cover every stored transition, including its terminal action.
    if int(sum(item.episode_length for item in episodes_by_id.values()
               if str(item.episode_id) in set(map(str, anchor.episode_id.tolist())))) != count:
        raise ValueError("H=1 transition anchor count is incomplete")

    return Round5TransitionSplit(
        split_name=split_name,
        states=np.asarray(anchor.masked_state_t, dtype=np.float32).copy(),
        state_masks=np.asarray(anchor.state_mask_t, dtype=np.bool_).copy(),
        goals=np.asarray(anchor.masked_goal, dtype=np.float32).copy(),
        goal_masks=np.asarray(anchor.goal_mask, dtype=np.bool_).copy(),
        node_masks=np.asarray(anchor.valid_node_mask_t, dtype=np.bool_).copy(),
        actions=normalized_actions,
        actions_m=raw_actions,
        current_positions_m=current_positions,
        next_positions_m=next_positions,
        remaining_time=remaining,
        oracle_waypoints=oracle,
        oracle_waypoints_m=oracle_raw,
        oracle_effective_horizon=effective,
        oracle_target_t=target_t,
        oracle_endpoint_terminated=endpoint_terminated,
        oracle_endpoint_truncated=endpoint_truncated,
        sample_id=np.asarray(anchor.sample_id).astype(str),
        episode_id=np.asarray(anchor.episode_id).astype(str),
        scenario_id=np.asarray(anchor.scenario_id).astype(str),
        seed=np.asarray(anchor.seed, dtype=np.int64).copy(),
        policy_name=np.asarray(anchor.policy_id).astype(str),
        arrival_trace_id=np.asarray(anchor.arrival_trace_id).astype(str),
        num_buoys=np.asarray(anchor.num_buoys, dtype=np.int64).copy(),
        quality_class=np.asarray(anchor.quality_class).astype(str),
        t=np.asarray(anchor.t, dtype=np.int64).copy(),
        episode_length=np.asarray(anchor.episode_length, dtype=np.int64).copy(),
        terminated=terminated,
        truncated=truncated,
    )


def _validated_predicted_waypoints(
    transitions: Round5TransitionSplit, cache_split: Any
) -> np.ndarray:
    if cache_split is None:
        raise ValueError("predicted-waypoint condition requires a frozen cache")
    split_name = getattr(cache_split, "split_name", None)
    if split_name != transitions.split_name:
        raise ValueError("waypoint cache split does not match transitions")
    episode_id = np.asarray(getattr(cache_split, "episode_id", None))
    timestep = np.asarray(getattr(cache_split, "t", None))
    if not np.array_equal(episode_id.astype(str), transitions.episode_id):
        raise ValueError("waypoint cache episode order does not match transitions")
    if not np.array_equal(timestep.astype(np.int64), transitions.t):
        raise ValueError("waypoint cache timestep order does not match transitions")
    values = np.asarray(
        getattr(cache_split, "predicted_waypoints", None), dtype=np.float32
    )
    if values.ndim != 2 or values.shape[0] != len(transitions):
        raise ValueError("waypoint cache predictions have an invalid shape")
    if values.shape[1] != transitions.action_dim:
        raise ValueError("waypoint condition dimension disagrees with action space")
    if not np.all(np.isfinite(values)):
        raise ValueError("waypoint cache contains NaN or Inf")
    return values


def _canonical_condition_source(value: str) -> str:
    key = str(value).strip().lower()
    if key not in _CONDITION_ALIASES:
        raise ValueError(
            "condition_source must select predicted waypoint, direct goal, "
            "state only, or oracle waypoint"
        )
    return _CONDITION_ALIASES[key]


def _sampler_seed(seed: int, epoch: int) -> int:
    digest = hashlib.sha256(
        f"round05-policy-balanced\x1f{seed}\x1f{epoch}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


# Short aliases are convenient for runner/trainer code while preserving the
# explicit Round-5 names above in manifests and tests.
build_causal_sequence_split = build_round5_sequence_split
build_causal_sequence_datasets = build_round5_sequence_datasets
load_transition_dataset = load_round5_transition_dataset


__all__ = [
    "ACTION_NORMALIZATION_VERSION",
    "CANONICAL_COLLECTION_POLICIES",
    "ORACLE_WAYPOINT_VERSION",
    "PolicyBalancedBatchSampler",
    "ROUND5_SEQUENCE_SCHEMA_VERSION",
    "ROUND5_TRANSITION_SCHEMA_VERSION",
    "Round5SequenceSplit",
    "Round5TransitionDataset",
    "Round5TransitionSplit",
    "build_causal_sequence_datasets",
    "build_causal_sequence_split",
    "build_round5_sequence_datasets",
    "build_round5_sequence_split",
    "load_round5_transition_dataset",
    "load_transition_dataset",
]
