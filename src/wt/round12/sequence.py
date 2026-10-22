"""Episode-safe, left-padded causal sequences for Round-12 policies."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .contracts import CONTEXT_LENGTH
from .data import Round12TransitionSplit


ROUND12_SEQUENCE_SCHEMA = "round12_causal_sequence_v1"


@dataclass(frozen=True)
class Round12SequenceSplit:
    """One causal history per anchor transition, with action labels only."""

    split_name: str
    context_length: int
    states: np.ndarray
    node_masks: np.ndarray
    predicted_phi: np.ndarray
    oracle_phi: np.ndarray
    remaining_time: np.ndarray
    actions: np.ndarray
    attention_mask: np.ndarray
    loss_mask: np.ndarray
    metadata: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if self.context_length != CONTEXT_LENGTH:
            raise ValueError("Round-12 sequence context must be Hc=10")
        count = int(self.states.shape[0])
        prefix = (count, self.context_length)
        for name in ("states", "node_masks", "predicted_phi", "oracle_phi", "remaining_time", "actions"):
            value = getattr(self, name)
            if value.ndim != 3 or value.shape[:2] != prefix:
                raise ValueError(f"{name} must have shape [N,H,D]")
            if value.dtype.kind == "O" or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite and non-object")
        if self.actions.shape[2] != 2:
            raise ValueError("Round-12 action width must equal two")
        for name in ("attention_mask", "loss_mask"):
            value = getattr(self, name)
            if value.shape != prefix or value.dtype != np.bool_:
                raise ValueError(f"{name} must be boolean [N,H]")
        if np.any(self.loss_mask & ~self.attention_mask):
            raise ValueError("padded timesteps must never contribute to action loss")
        if not np.all(self.loss_mask[:, -1]) or not np.all(self.loss_mask.sum(axis=1) == 1):
            raise ValueError("each causal sequence must supervise its final physical timestep only")
        if np.any(self.attention_mask[:, 1:] < self.attention_mask[:, :-1]):
            raise ValueError("causal sequences must be left padded")
        for name in ("states", "node_masks", "predicted_phi", "oracle_phi", "remaining_time", "actions"):
            if np.any(getattr(self, name)[~self.attention_mask] != 0.0):
                raise ValueError(f"padded {name} values must be zero")
        required = {
            "episode_id", "scenario_id", "behavior_generator", "quality", "difficulty",
            "num_buoys", "timestep", "split", "terminated", "truncated",
        }
        if not required.issubset(self.metadata):
            raise ValueError("Round-12 sequence metadata is incomplete")
        for name, value in self.metadata.items():
            if np.asarray(value).shape != (count,):
                raise ValueError(f"metadata {name!r} must have shape [N]")

    def __len__(self) -> int:
        return int(self.states.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[-1])

    @property
    def max_buoys(self) -> int:
        return int(self.node_masks.shape[-1])

    @property
    def phi_dim(self) -> int:
        return int(self.predicted_phi.shape[-1])

    def take(self, indices: np.ndarray) -> "Round12SequenceSplit":
        chosen = np.asarray(indices, dtype=np.int64)
        if chosen.ndim != 1 or chosen.size < 1:
            raise ValueError("sequence take requires nonempty one-dimensional indices")
        return Round12SequenceSplit(
            split_name=self.split_name,
            context_length=self.context_length,
            states=self.states[chosen].copy(),
            node_masks=self.node_masks[chosen].copy(),
            predicted_phi=self.predicted_phi[chosen].copy(),
            oracle_phi=self.oracle_phi[chosen].copy(),
            remaining_time=self.remaining_time[chosen].copy(),
            actions=self.actions[chosen].copy(),
            attention_mask=self.attention_mask[chosen].copy(),
            loss_mask=self.loss_mask[chosen].copy(),
            metadata=MappingProxyType({name: np.asarray(value)[chosen].copy() for name, value in self.metadata.items()}),
        )

    def numeric_arrays(self) -> Mapping[str, np.ndarray]:
        """Canonical tensor arrays; action labels are never policy inputs."""
        return MappingProxyType({
            "states": self.states,
            "node_masks": self.node_masks,
            "predicted_phi": self.predicted_phi,
            "oracle_phi": self.oracle_phi,
            "remaining_time": self.remaining_time,
            "attention_mask": self.attention_mask,
            "actions": self.actions,
            "loss_mask": self.loss_mask,
        })


def causal_attention_mask(attention_mask: np.ndarray) -> np.ndarray:
    """Return the exact [B,H,H] causal/key-padding allowance mask."""
    valid = np.asarray(attention_mask, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("attention_mask must have shape [B,H]")
    horizon = valid.shape[1]
    lower = np.tril(np.ones((horizon, horizon), dtype=bool))
    return valid[:, :, None] & valid[:, None, :] & lower[None, :, :]


def build_causal_sequence_split(
    transitions: Round12TransitionSplit,
    *,
    context_length: int = CONTEXT_LENGTH,
) -> Round12SequenceSplit:
    """Build histories without crossing episodes, split boundaries, or terminals."""
    if not isinstance(transitions, Round12TransitionSplit):
        raise TypeError("transitions must be Round12TransitionSplit")
    if context_length != CONTEXT_LENGTH:
        raise ValueError("Round-12 policy context is fixed at Hc=10")
    count, state_dim, node_dim, phi_dim = (
        len(transitions), transitions.state_dim, transitions.max_buoys, transitions.phi_dim
    )
    states = np.zeros((count, context_length, state_dim), dtype=np.float32)
    nodes = np.zeros((count, context_length, node_dim), dtype=np.float32)
    predicted = np.zeros((count, context_length, phi_dim), dtype=np.float32)
    oracle = np.zeros((count, context_length, phi_dim), dtype=np.float32)
    remaining = np.zeros((count, context_length, 1), dtype=np.float32)
    actions = np.zeros((count, context_length, 2), dtype=np.float32)
    attention = np.zeros((count, context_length), dtype=np.bool_)
    loss = np.zeros((count, context_length), dtype=np.bool_)

    by_episode: dict[str, list[int]] = {}
    for row, episode in enumerate(transitions.episode_id.tolist()):
        by_episode.setdefault(str(episode), []).append(row)
    for episode, rows in by_episode.items():
        rows.sort(key=lambda row: int(transitions.timestep[row]))
        history: list[int] = []
        previous_t: int | None = None
        for row in rows:
            timestep = int(transitions.timestep[row])
            if previous_t is not None and timestep != previous_t + 1:
                raise ValueError(f"episode {episode!r} has non-contiguous permitted H10 rows")
            previous_t = timestep
            history.append(row)
            source_rows = history[-context_length:]
            width = len(source_rows)
            destination = slice(context_length - width, context_length)
            states[row, destination] = transitions.states[source_rows]
            nodes[row, destination] = transitions.node_masks[source_rows]
            predicted[row, destination] = transitions.predicted_phi[source_rows]
            oracle[row, destination] = transitions.oracle_phi[source_rows]
            remaining[row, destination] = transitions.remaining_time[source_rows]
            # Labels are saved for loss/evaluation only.  No action tensor is
            # handed to a policy forward call.
            actions[row, destination] = transitions.actions[source_rows]
            attention[row, destination] = True
            loss[row, -1] = True
            if bool(transitions.terminated[row]) or bool(transitions.truncated[row]):
                history = []
                previous_t = None

    metadata = MappingProxyType({
        "episode_id": transitions.episode_id.copy(),
        "scenario_id": transitions.scenario_id.copy(),
        "behavior_generator": transitions.behavior_generator.copy(),
        "quality": transitions.quality.copy(),
        "difficulty": transitions.difficulty.copy(),
        "num_buoys": transitions.num_buoys.copy(),
        "timestep": transitions.timestep.copy(),
        "split": np.full(count, transitions.split_name),
        "terminated": transitions.terminated.copy(),
        "truncated": transitions.truncated.copy(),
    })
    return Round12SequenceSplit(
        split_name=transitions.split_name,
        context_length=context_length,
        states=states,
        node_masks=nodes,
        predicted_phi=predicted,
        oracle_phi=oracle,
        remaining_time=remaining,
        actions=actions,
        attention_mask=attention,
        loss_mask=loss,
        metadata=metadata,
    )


__all__ = [
    "ROUND12_SEQUENCE_SCHEMA",
    "Round12SequenceSplit",
    "build_causal_sequence_split",
    "causal_attention_mask",
]
