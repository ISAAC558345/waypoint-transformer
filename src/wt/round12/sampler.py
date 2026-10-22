"""Deterministic scenario- and behaviour-balanced sampling for Round-12."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np

from .sequence import Round12SequenceSplit


AUDIT_FIELDS = ("quality", "difficulty", "behavior_generator", "num_buoys")


def _shares(values: Sequence[Any]) -> dict[str, float]:
    count = Counter(str(value) for value in values)
    total = sum(count.values())
    return {key: value / total for key, value in sorted(count.items())} if total else {}


class ScenarioBehaviorBalancedSampler:
    """Cycle uniformly through observed (scenario, behaviour) strata.

    Equal stratum selection gives each scenario and each behaviour generator
    equal influence whenever the registered dataset has the normal complete
    scenario-by-behaviour grid.  Within a stratum, rows are reshuffled and
    cycled deterministically.  Quality/difficulty/scale metadata is audited,
    never used as a policy input.
    """

    def __init__(self, sequence: Round12SequenceSplit, *, seed: int = 0) -> None:
        if not isinstance(sequence, Round12SequenceSplit):
            raise TypeError("sampler requires Round12SequenceSplit")
        self.sequence = sequence
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        scenarios = np.asarray(sequence.metadata["scenario_id"]).astype(str)
        behaviours = np.asarray(sequence.metadata["behavior_generator"]).astype(str)
        buckets: dict[tuple[str, str], np.ndarray] = {}
        grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, key in enumerate(zip(scenarios.tolist(), behaviours.tolist())):
            grouped[key].append(index)
        for key, values in grouped.items():
            buckets[key] = np.asarray(values, dtype=np.int64)
        if not buckets:
            raise ValueError("sampler requires at least one stratum")
        self._buckets = buckets
        self._keys = tuple(sorted(buckets))
        self._orders = {key: self.rng.permutation(value) for key, value in buckets.items()}
        self._positions = {key: 0 for key in buckets}
        self._cursor = 0
        self.last_indices = np.empty((0,), dtype=np.int64)

    def _next_from(self, key: tuple[str, str]) -> int:
        order = self._orders[key]
        position = self._positions[key]
        if position >= len(order):
            order = self.rng.permutation(self._buckets[key])
            self._orders[key] = order
            position = 0
        self._positions[key] = position + 1
        return int(order[position])

    def sample_indices(self, batch_size: int) -> np.ndarray:
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        result = np.empty((int(batch_size),), dtype=np.int64)
        for position in range(len(result)):
            key = self._keys[self._cursor % len(self._keys)]
            self._cursor += 1
            result[position] = self._next_from(key)
        self.last_indices = result
        return result.copy()

    def audit(self, indices: Sequence[int] | None = None) -> dict[str, Any]:
        chosen = self.last_indices if indices is None else np.asarray(indices, dtype=np.int64)
        metadata = self.sequence.metadata
        sampled = {
            field: _shares(np.asarray(metadata[field])[chosen].tolist())
            for field in AUDIT_FIELDS
        } if chosen.size else {field: {} for field in AUDIT_FIELDS}
        source = {
            field: _shares(np.asarray(metadata[field]).tolist())
            for field in AUDIT_FIELDS
        }
        scenario = _shares(np.asarray(metadata["scenario_id"])[chosen].tolist()) if chosen.size else {}
        behaviour = _shares(np.asarray(metadata["behavior_generator"])[chosen].tolist()) if chosen.size else {}
        return {
            "sampler": "scenario_behavior_balanced_v1",
            "sample_count": int(chosen.size),
            "stratum_count": len(self._keys),
            "sampled_shares": sampled,
            "source_shares": source,
            "scenario_share": scenario,
            "behavior_generator_share": behaviour,
            "quality_share": sampled["quality"],
            "difficulty_share": sampled["difficulty"],
            "buoy_scale_share": sampled["num_buoys"],
        }

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "seed": self.seed,
            "cursor": self._cursor,
            "positions": dict(self._positions),
            "orders": {f"{key[0]}\x1f{key[1]}": value.copy() for key, value in self._orders.items()},
            "rng_state": deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("seed", self.seed)) != self.seed:
            raise ValueError("sampler seed differs from checkpoint")
        self._cursor = int(state["cursor"])
        positions = state["positions"]
        orders = state["orders"]
        for key in self._keys:
            serialized = f"{key[0]}\x1f{key[1]}"
            if serialized not in orders or key not in positions:
                raise ValueError("checkpoint sampler strata differ from current data")
            order = np.asarray(orders[serialized], dtype=np.int64)
            if set(order.tolist()) != set(self._buckets[key].tolist()):
                raise ValueError("checkpoint sampler bucket differs from current data")
            self._orders[key] = order
            self._positions[key] = int(positions[key])
        self.rng.bit_generator.state = deepcopy(state["rng_state"])


__all__ = ["AUDIT_FIELDS", "ScenarioBehaviorBalancedSampler"]
