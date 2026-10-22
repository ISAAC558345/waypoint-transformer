"""Non-learning baselines for the maritime communication environment.

Every policy implements the same small interface::

    policy.reset(seed=episode_seed)
    action = policy.act(observation, info)

The policies consume only the current observation and current ``info`` mapping.
They never inspect future states, contain no trainable parameters, and return a
finite normalized action inside the unit ball.  The current environment uses
the first two action coordinates for planar motion; any later extra action
coordinates are conservatively set to zero because their semantics are not yet
defined.

Queue, urgency, and node-mask fields are optional.  The Stage-1A environment
does not expose queues or urgency, so the documented fallbacks are intentional:

* ``nearest_buoy`` targets the nearest currently valid physical buoy;
* ``max_queue`` hovers because no queue ranking is available;
* ``urgency_rate_energy`` becomes a current-rate/one-step-movement-proxy rule.

No environment channel or energy formula is copied into this module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Type

import numpy as np


RANDOM_VALID = "random_valid"
NEAREST_BUOY = "nearest_buoy"
MAX_QUEUE = "max_queue"
URGENCY_RATE_ENERGY = "urgency_rate_energy"
QUEUE_RATE_ENERGY = "queue_rate_energy"

_POLICY_NAMES = (
    RANDOM_VALID,
    NEAREST_BUOY,
    MAX_QUEUE,
    URGENCY_RATE_ENERGY,
)
_ROUND3_POLICY_NAMES = (
    RANDOM_VALID,
    NEAREST_BUOY,
    MAX_QUEUE,
    QUEUE_RATE_ENERGY,
)
_POSITION_DIM = 2
_VALID_MASK_KEYS = (
    "valid_node_mask",
    "buoy_valid_mask",
    "active_buoy_mask",
)
_QUEUE_KEYS = (
    "raw_queue_bits",
    "queue_values",
    "remaining_queue",
    "remaining_queues",
    "queue_lengths",
    "queue_backlogs",
    "buoy_queues",
)
_URGENCY_KEYS = (
    "urgency_values",
    "urgency",
    "urgencies",
    "urgency_scores",
    "buoy_urgency",
)
_ENERGY_PROXY_KEYS = (
    "one_step_movement_energy_proxy",
    "estimated_one_step_energy_costs",
)
_GEOMETRY_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class _PolicyContext:
    """Validated current-state values shared by deterministic policies."""

    action_dim: int
    uav_position_m: np.ndarray
    buoy_positions_m: np.ndarray
    horizontal_distances_m: np.ndarray
    valid_node_mask: np.ndarray
    mask_sources: Tuple[str, ...]

    @property
    def num_slots(self) -> int:
        """Return the number of physical or padded node slots."""
        return int(self.buoy_positions_m.shape[0])


class BaselinePolicy(ABC):
    """Abstract stateless/non-learning policy interface.

    ``reset`` records an optional episode seed.  Deterministic policies do not
    otherwise use it; :class:`RandomValidPolicy` reconstructs its private
    ``numpy.random.Generator`` when a seed is supplied.
    """

    policy_id = "baseline"
    deterministic = True
    formula = "defined by subclass"
    fallback_note = "none"

    def __init__(self, seed: Optional[int] = 0) -> None:
        self._seed = _validate_seed(seed)
        self._last_decision: Dict[str, Any] = {
            "status": "not_called",
            "selected_buoy_index": None,
            "effective_formula": self.formula,
        }

    @property
    def name(self) -> str:
        """Return the canonical policy identifier."""
        return self.policy_id

    @property
    def seed(self) -> Optional[int]:
        """Return the most recently supplied explicit seed."""
        return self._seed

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return JSON-compatible policy and most-recent-decision metadata."""
        return {
            "policy_id": self.policy_id,
            "family": "non_learning",
            "deterministic": bool(self.deterministic),
            "trainable": False,
            "uses_optimizer": False,
            "uses_future_information": False,
            "action_constraint": "finite vector with L2 norm <= 1",
            "formula": self.formula,
            "fallback_note": self.fallback_note,
            "optional_mask_fields": list(_VALID_MASK_KEYS),
            "optional_queue_fields": list(_QUEUE_KEYS),
            "optional_urgency_fields": list(_URGENCY_KEYS),
            "last_decision": dict(self._last_decision),
        }

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset episode-local state.

        Passing ``None`` preserves the current seed/RNG stream, matching the
        environment's optional-seed convention.  Passing an integer records a
        new seed and, for the random policy, restarts its private stream.
        """
        if seed is not None:
            self._seed = _validate_seed(seed)
        self._last_decision = {
            "status": "reset",
            "selected_buoy_index": None,
            "effective_formula": self.formula,
        }

    @abstractmethod
    def act(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> np.ndarray:
        """Return one finite legal action for the current state."""

    def _record_decision(
        self,
        *,
        status: str,
        selected_buoy_index: Optional[int],
        effective_formula: str,
        **details: Any,
    ) -> None:
        record: Dict[str, Any] = {
            "status": status,
            "selected_buoy_index": selected_buoy_index,
            "effective_formula": effective_formula,
        }
        record.update(details)
        self._last_decision = record


class RandomValidPolicy(BaselinePolicy):
    """Sample uniformly from the currently feasible normalized action set."""

    policy_id = RANDOM_VALID
    deterministic = False
    formula = (
        "a ~ Uniform({a: ||a||_2 <= 1 and the current planar step "
        "remains inside the environment bounds})"
    )
    fallback_note = (
        "Uses current position/bounds when exposed; otherwise falls back to "
        "the legacy unit-ball sampler. It never uses node, queue, urgency, "
        "or future fields."
    )

    def __init__(self, seed: Optional[int] = 0) -> None:
        super().__init__(seed=seed)
        self._rng = np.random.default_rng(self._seed)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset metadata and reconstruct the private RNG for explicit seeds."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(self._seed)

    def act(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> np.ndarray:
        """Draw one reproducible finite action from the feasible unit ball."""
        _validate_observation(observation, info)
        action_dim = _extract_action_dim(info)
        motion_bounds = _extract_optional_motion_bounds(info)
        rejected_boundary_samples = 0
        while True:
            direction = self._rng.normal(size=action_dim)
            norm = float(np.linalg.norm(direction))
            if not math.isfinite(norm) or norm <= _GEOMETRY_TOLERANCE:
                continue
            radius = float(self._rng.random() ** (1.0 / action_dim))
            action = _validate_legal_action(
                direction * (radius / norm), action_dim
            )
            if motion_bounds is None or _motion_stays_in_bounds(
                action, motion_bounds
            ):
                legal_action = action
                break
            rejected_boundary_samples += 1
            if rejected_boundary_samples >= 10000:
                raise RuntimeError(
                    "failed to sample a boundary-safe random action"
                )
        self._record_decision(
            status="sampled",
            selected_buoy_index=None,
            effective_formula=self.formula,
            action_dim=action_dim,
            sampled_radius=float(np.linalg.norm(legal_action)),
            boundary_constraints_available=motion_bounds is not None,
            rejected_boundary_samples=rejected_boundary_samples,
        )
        return legal_action


class NearestBuoyPolicy(BaselinePolicy):
    """Move toward the nearest valid physical buoy.

    When a current queue field exists, zero-backlog nodes are treated as
    completed and ignored.  Because Stage 1A has no queue field, all physically
    valid buoys remain eligible there.  Ties are resolved by the smallest array
    index.
    """

    policy_id = NEAREST_BUOY
    formula = "argmin_i current_horizontal_distance_i over eligible nodes"
    fallback_note = (
        "Without queue data, selects the nearest valid physical buoy; with no "
        "eligible node, returns the zero/hover action."
    )

    def act(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> np.ndarray:
        """Return a unit-disk direction toward the nearest eligible buoy."""
        context = _extract_context(observation, info)
        queue, queue_source = _extract_optional_nonnegative_vector(
            info, _QUEUE_KEYS, context.num_slots, "queue"
        )
        eligible = context.valid_node_mask.copy()
        if queue is not None:
            eligible &= queue > 0.0

        selected = _masked_argmin(context.horizontal_distances_m, eligible)
        if selected is None:
            action = _hover_action(context.action_dim)
            status = "hover_no_eligible_node"
        else:
            action = _movement_action(context, selected)
            status = "selected_nearest"
        self._record_decision(
            status=status,
            selected_buoy_index=selected,
            effective_formula=self.formula,
            queue_available=queue is not None,
            queue_source=queue_source,
            mask_sources=list(context.mask_sources),
            eligible_node_count=int(np.count_nonzero(eligible)),
        )
        return action


class MaxQueuePolicy(BaselinePolicy):
    """Move toward the valid node with the largest current queue.

    The Stage-1A environment does not expose a queue, so this policy must and
    does hover rather than fabricating a backlog.  Equal queues use stable
    lowest-index tie breaking.
    """

    policy_id = MAX_QUEUE
    formula = "argmax_i current_queue_i over valid nodes"
    fallback_note = (
        "If queue information is unavailable or every valid queue is empty, "
        "returns the zero/hover action."
    )

    def act(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> np.ndarray:
        """Return a direction to the maximum-queue node, or safely hover."""
        context = _extract_context(observation, info)
        queue, queue_source = _extract_optional_nonnegative_vector(
            info, _QUEUE_KEYS, context.num_slots, "queue"
        )
        if queue is None:
            self._record_decision(
                status="hover_queue_unavailable",
                selected_buoy_index=None,
                effective_formula="hover; queue unavailable",
                queue_available=False,
                queue_source=None,
                mask_sources=list(context.mask_sources),
                eligible_node_count=int(
                    np.count_nonzero(context.valid_node_mask)
                ),
            )
            return _hover_action(context.action_dim)

        eligible = context.valid_node_mask & (queue > 0.0)
        selected = _masked_priority_argmax(
            queue, context.horizontal_distances_m, eligible
        )
        if selected is None:
            action = _hover_action(context.action_dim)
            status = "hover_no_nonempty_queue"
        else:
            action = _movement_action(context, selected)
            status = "selected_max_queue"
        self._record_decision(
            status=status,
            selected_buoy_index=selected,
            effective_formula=self.formula,
            queue_available=True,
            queue_source=queue_source,
            mask_sources=list(context.mask_sources),
            eligible_node_count=int(np.count_nonzero(eligible)),
        )
        return action


class QueueRateEnergyPolicy(BaselinePolicy):
    """Prioritize real queue, current rate, and current movement-energy proxy.

    For each currently valid nonempty buoy ``i``:

    ``priority_i = normalized_queue_i * normalized_rate_i /
    (normalized_movement_energy_i + epsilon)``.

    No urgency, deadline, AoI, future channel, future position, or future
    arrival is used. Equal priorities prefer the nearer buoy and then its
    stable array index.
    """

    policy_id = QUEUE_RATE_ENERGY
    formula = (
        "normalized_queue * normalized_current_rate / "
        "(normalized_current_movement_energy + epsilon)"
    )
    fallback_note = (
        "Requires a real current queue. If every valid queue is empty or queue "
        "data is unavailable, returns the zero/hover action."
    )

    def __init__(
        self, seed: Optional[int] = 0, epsilon: float = 1.0e-8
    ) -> None:
        super().__init__(seed=seed)
        if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
            raise ValueError("epsilon must be a positive finite real number")
        self.epsilon = float(epsilon)
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be a positive finite real number")

    @property
    def metadata(self) -> Dict[str, Any]:
        metadata = super().metadata
        metadata["epsilon"] = self.epsilon
        metadata["urgency_used"] = False
        metadata["energy_quantity"] = (
            "current normalized one-step movement-energy proxy supplied by "
            "the environment"
        )
        return metadata

    def act(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> np.ndarray:
        context = _extract_context(observation, info)
        queue, queue_source = _extract_optional_nonnegative_vector(
            info, _QUEUE_KEYS, context.num_slots, "queue"
        )
        if queue is None:
            self._record_decision(
                status="hover_queue_unavailable",
                selected_buoy_index=None,
                effective_formula="hover; queue unavailable",
                queue_available=False,
                eligible_node_count=0,
            )
            return _hover_action(context.action_dim)

        rates = _required_nonnegative_vector(
            info, "rates_bps", context.num_slots
        )
        eligible = context.valid_node_mask & (queue > 0.0)
        if not np.any(eligible):
            self._record_decision(
                status="hover_no_nonempty_queue",
                selected_buoy_index=None,
                effective_formula="hover; no valid nonempty queue",
                queue_available=True,
                queue_source=queue_source,
                eligible_node_count=0,
            )
            return _hover_action(context.action_dim)

        movement_energy, proxy_source = _one_step_movement_proxy(
            context, info
        )
        queue_factor = _normalize_over_mask(queue, eligible)
        rate_factor = _normalize_over_mask(rates, eligible)
        priority = queue_factor * rate_factor / (
            movement_energy + self.epsilon
        )
        selected = _masked_priority_argmax(
            priority, context.horizontal_distances_m, eligible
        )
        if selected is None:
            raise RuntimeError("queue-rate-energy ranking found no valid node")
        action = _movement_action(context, selected)
        self._record_decision(
            status="selected_queue_rate_energy",
            selected_buoy_index=selected,
            effective_formula=self.formula,
            queue_available=True,
            queue_source=queue_source,
            mask_sources=list(context.mask_sources),
            energy_proxy_source=proxy_source,
            eligible_node_count=int(np.count_nonzero(eligible)),
            selected_priority=float(priority[selected]),
        )
        return action


class UrgencyRateEnergyPolicy(BaselinePolicy):
    """Rank current nodes by available urgency, queue, rate, and motion cost.

    For eligible node ``i`` the implemented score is

    ``priority_i = U_i * Q_i * R_i / (M_i + epsilon)``.

    Each available nonnegative factor is normalized by its largest eligible
    value.  Missing queue or urgency factors are omitted, never fabricated.
    ``R_i`` is the current ``info["rates_bps"][i]``.  ``M_i`` is either an
    explicitly supplied current one-step proxy, the fraction of a supplied
    ``max_step_distance_m`` needed to reach the node, or finally the norm of
    the legal movement action (a dimensionless fallback).  It is deliberately
    *not* a reimplementation of the environment energy equation.
    """

    policy_id = URGENCY_RATE_ENERGY
    formula = "normalized(U)*normalized(Q)*normalized(R)/(M + epsilon)"
    fallback_note = (
        "Missing urgency omits U; missing queue omits Q. With both unavailable "
        "the effective rule is current rate divided by a one-step normalized "
        "movement proxy. No physical energy is inferred."
    )

    def __init__(
        self, seed: Optional[int] = 0, epsilon: float = 1.0e-8
    ) -> None:
        super().__init__(seed=seed)
        if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
            raise ValueError("epsilon must be a positive finite real number")
        self.epsilon = float(epsilon)
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be a positive finite real number")

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return base metadata augmented with the numerical stabilizer."""
        metadata = super().metadata
        metadata["epsilon"] = self.epsilon
        metadata["energy_quantity"] = (
            "normalized one-step movement proxy, not environment energy"
        )
        metadata["optional_energy_proxy_fields"] = list(_ENERGY_PROXY_KEYS)
        return metadata

    def act(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> np.ndarray:
        """Return a deterministic direction selected by the available score."""
        context = _extract_context(observation, info)
        queue, queue_source = _extract_optional_nonnegative_vector(
            info, _QUEUE_KEYS, context.num_slots, "queue"
        )
        urgency, urgency_source = _extract_optional_nonnegative_vector(
            info, _URGENCY_KEYS, context.num_slots, "urgency"
        )
        rates = _required_nonnegative_vector(
            info, "rates_bps", context.num_slots
        )

        eligible = context.valid_node_mask.copy()
        if queue is not None:
            eligible &= queue > 0.0
        if not np.any(eligible):
            self._record_decision(
                status="hover_no_eligible_node",
                selected_buoy_index=None,
                effective_formula="hover; no eligible node",
                queue_available=queue is not None,
                urgency_available=urgency is not None,
                queue_source=queue_source,
                urgency_source=urgency_source,
                mask_sources=list(context.mask_sources),
                energy_proxy_source=None,
                eligible_node_count=0,
            )
            return _hover_action(context.action_dim)

        rate_factor = _normalize_over_mask(rates, eligible)
        priority = rate_factor.copy()
        formula_factors = ["normalized(current_rate)"]
        if queue is not None:
            priority *= _normalize_over_mask(queue, eligible)
            formula_factors.insert(0, "normalized(current_queue)")
        if urgency is not None:
            priority *= _normalize_over_mask(urgency, eligible)
            formula_factors.insert(0, "normalized(current_urgency)")

        movement_proxy, proxy_source = _one_step_movement_proxy(
            context, info
        )
        priority /= movement_proxy + self.epsilon
        priority[~eligible] = -np.inf
        selected = int(np.argmax(priority))
        if not math.isfinite(float(priority[selected])):
            raise RuntimeError("baseline priority calculation is non-finite")

        effective_formula = (
            " * ".join(formula_factors)
            + f" / ({proxy_source} + epsilon)"
        )
        action = _movement_action(context, selected)
        self._record_decision(
            status="selected_priority",
            selected_buoy_index=selected,
            effective_formula=effective_formula,
            queue_available=queue is not None,
            urgency_available=urgency is not None,
            queue_source=queue_source,
            urgency_source=urgency_source,
            mask_sources=list(context.mask_sources),
            energy_proxy_source=proxy_source,
            eligible_node_count=int(np.count_nonzero(eligible)),
            selected_priority=float(priority[selected]),
        )
        return action


def available_baseline_policies(
    task_semantics_version: str = "v0",
) -> Tuple[str, ...]:
    """Return stable policy names for the requested versioned semantics."""
    if task_semantics_version == "v0":
        return _POLICY_NAMES
    if task_semantics_version == "v1":
        return _ROUND3_POLICY_NAMES
    raise ValueError("task_semantics_version must be 'v0' or 'v1'")


def create_baseline_policy(
    name: str, seed: Optional[int] = 0, **kwargs: Any
) -> BaselinePolicy:
    """Construct a non-learning baseline by canonical name.

    Parameters
    ----------
    name:
        One of :func:`available_baseline_policies`.
    seed:
        Explicit private RNG seed.  Deterministic policies record but do not
        otherwise consume it.
    **kwargs:
        Policy-specific settings.  Currently only
        ``urgency_rate_energy`` accepts ``epsilon``.
    """
    if not isinstance(name, str):
        raise TypeError("baseline policy name must be a string")
    canonical_name = name.strip().lower()
    policy_types: Dict[str, Type[BaselinePolicy]] = {
        RANDOM_VALID: RandomValidPolicy,
        NEAREST_BUOY: NearestBuoyPolicy,
        MAX_QUEUE: MaxQueuePolicy,
        URGENCY_RATE_ENERGY: UrgencyRateEnergyPolicy,
        QUEUE_RATE_ENERGY: QueueRateEnergyPolicy,
    }
    if canonical_name not in policy_types:
        supported = ", ".join((*_POLICY_NAMES, QUEUE_RATE_ENERGY))
        raise ValueError(
            f"unknown baseline policy {name!r}; expected one of: {supported}"
        )
    return policy_types[canonical_name](seed=seed, **kwargs)


def baseline_policy_metadata(
    task_semantics_version: str = "v0",
) -> Dict[str, Dict[str, Any]]:
    """Return static JSON-compatible metadata for all canonical baselines."""
    return {
        name: create_baseline_policy(name, seed=0).metadata
        for name in available_baseline_policies(task_semantics_version)
    }


def _validate_seed(seed: Optional[int]) -> Optional[int]:
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError(f"seed must be a nonnegative integer or None, got {seed!r}")
    numeric_seed = int(seed)
    if numeric_seed < 0:
        raise ValueError(f"seed must be nonnegative, got {seed!r}")
    return numeric_seed


def _validate_observation(
    observation: np.ndarray, info: Mapping[str, Any]
) -> np.ndarray:
    if not isinstance(info, Mapping):
        raise TypeError("info must be a mapping")
    try:
        untyped = np.asarray(observation)
    except (TypeError, ValueError) as error:
        raise ValueError("observation must be a finite one-dimensional array") from error
    if np.iscomplexobj(untyped):
        raise ValueError("observation must contain only real values")
    try:
        values = np.asarray(observation, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("observation must be a finite one-dimensional array") from error
    if values.ndim != 1 or values.size < 1:
        raise ValueError(
            "observation must be a nonempty one-dimensional array, "
            f"got shape {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("observation must contain only finite values")

    if "state_dim" in info:
        state_dim = _positive_integer(info["state_dim"], "info['state_dim']")
        if values.shape != (state_dim,):
            raise ValueError(
                f"observation shape must be ({state_dim},), got {values.shape}"
            )
    return values.copy()


def _extract_action_dim(info: Mapping[str, Any]) -> int:
    if not isinstance(info, Mapping):
        raise TypeError("info must be a mapping")
    if "action_dim" not in info:
        raise KeyError("info must contain 'action_dim'")
    return _positive_integer(info["action_dim"], "info['action_dim']")


def _extract_optional_motion_bounds(
    info: Mapping[str, Any],
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Return current planar motion bounds when the environment exposes them."""
    if "area_bounds_m" not in info or "max_step_distance_m" not in info:
        return None
    position = _required_matrix_or_vector(
        info, "uav_position_m", (_POSITION_DIM,)
    )
    bounds = _required_matrix_or_vector(
        info, "area_bounds_m", (_POSITION_DIM,)
    )
    if np.any(bounds <= 0.0):
        raise ValueError("info['area_bounds_m'] must contain positive values")
    if np.any(position < -_GEOMETRY_TOLERANCE) or np.any(
        position > bounds + _GEOMETRY_TOLERANCE
    ):
        raise ValueError("info['uav_position_m'] lies outside area_bounds_m")
    raw_step = info["max_step_distance_m"]
    if isinstance(raw_step, bool) or not isinstance(raw_step, Real):
        raise ValueError(
            "info['max_step_distance_m'] must be a positive finite real"
        )
    max_step_distance = float(raw_step)
    if not math.isfinite(max_step_distance) or max_step_distance <= 0.0:
        raise ValueError(
            "info['max_step_distance_m'] must be a positive finite real"
        )
    return position, bounds, max_step_distance


def _motion_stays_in_bounds(
    action: np.ndarray,
    motion_bounds: Tuple[np.ndarray, np.ndarray, float],
) -> bool:
    position, bounds, max_step_distance = motion_bounds
    candidate = (
        position
        + np.asarray(action[:_POSITION_DIM], dtype=np.float64)
        * max_step_distance
    )
    return bool(
        np.all(candidate >= -_GEOMETRY_TOLERANCE)
        and np.all(candidate <= bounds + _GEOMETRY_TOLERANCE)
    )


def _extract_context(
    observation: np.ndarray, info: Mapping[str, Any]
) -> _PolicyContext:
    _validate_observation(observation, info)
    action_dim = _extract_action_dim(info)
    if action_dim < _POSITION_DIM:
        raise ValueError(
            "physical buoy-targeting policies require action_dim >= 2"
        )

    uav_position = _required_matrix_or_vector(
        info, "uav_position_m", (_POSITION_DIM,)
    )
    if "buoy_positions_m" not in info:
        raise KeyError("info must contain 'buoy_positions_m'")
    try:
        untyped_positions = np.asarray(info["buoy_positions_m"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            "info['buoy_positions_m'] must be a finite K-by-2 array"
        ) from error
    if np.iscomplexobj(untyped_positions):
        raise ValueError("info['buoy_positions_m'] must contain real values")
    try:
        buoy_positions = np.asarray(
            info["buoy_positions_m"], dtype=np.float64
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "info['buoy_positions_m'] must be a finite K-by-2 array"
        ) from error
    if (
        buoy_positions.ndim != 2
        or buoy_positions.shape[1] != _POSITION_DIM
        or buoy_positions.shape[0] < 1
    ):
        raise ValueError(
            "info['buoy_positions_m'] must have shape (K, 2) with K >= 1, "
            f"got {buoy_positions.shape}"
        )
    if not np.all(np.isfinite(buoy_positions)):
        raise ValueError("info['buoy_positions_m'] contains NaN or Inf")
    num_slots = int(buoy_positions.shape[0])
    distances = _required_nonnegative_vector(
        info, "horizontal_distances_m", num_slots
    )
    valid_mask, mask_sources = _extract_valid_mask(info, num_slots)

    if "num_buoys" in info:
        reported_count = _positive_integer(
            info["num_buoys"], "info['num_buoys']"
        )
        if reported_count != num_slots:
            if not mask_sources or int(np.count_nonzero(valid_mask)) != reported_count:
                raise ValueError(
                    "info['num_buoys'] differs from the number of position "
                    "slots; a consistent valid-node mask is required for padding"
                )

    return _PolicyContext(
        action_dim=action_dim,
        uav_position_m=uav_position,
        buoy_positions_m=buoy_positions.copy(),
        horizontal_distances_m=distances,
        valid_node_mask=valid_mask,
        mask_sources=mask_sources,
    )


def _extract_valid_mask(
    info: Mapping[str, Any], size: int
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    valid = np.ones(size, dtype=bool)
    sources = []
    for key in _VALID_MASK_KEYS:
        if key not in info:
            continue
        values = np.asarray(info[key])
        if values.shape != (size,):
            raise ValueError(
                f"info[{key!r}] must have shape ({size},), got {values.shape}"
            )
        if values.dtype == np.bool_:
            mask = values.astype(bool, copy=True)
        else:
            try:
                numeric = np.asarray(values, dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"info[{key!r}] must be a boolean or binary mask"
                ) from error
            if not np.all(np.isfinite(numeric)) or not np.all(
                (numeric == 0.0) | (numeric == 1.0)
            ):
                raise ValueError(
                    f"info[{key!r}] must contain only boolean/0/1 values"
                )
            mask = numeric.astype(bool)
        valid &= mask
        sources.append(key)
    return valid, tuple(sources)


def _extract_optional_nonnegative_vector(
    info: Mapping[str, Any],
    keys: Sequence[str],
    size: int,
    quantity: str,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    present = [key for key in keys if key in info]
    if not present:
        return None, None
    source = present[0]
    values = _nonnegative_vector(info[source], size, f"info[{source!r}]")
    for alias in present[1:]:
        alias_values = _nonnegative_vector(
            info[alias], size, f"info[{alias!r}]"
        )
        if not np.array_equal(values, alias_values):
            raise ValueError(
                f"conflicting {quantity} fields {source!r} and {alias!r}"
            )
    return values, source


def _required_nonnegative_vector(
    info: Mapping[str, Any], key: str, size: int
) -> np.ndarray:
    if key not in info:
        raise KeyError(f"info must contain {key!r}")
    return _nonnegative_vector(info[key], size, f"info[{key!r}]")


def _nonnegative_vector(value: Any, size: int, name: str) -> np.ndarray:
    try:
        untyped = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite nonnegative vector") from error
    if np.iscomplexobj(untyped):
        raise ValueError(f"{name} must contain only real values")
    try:
        values = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite nonnegative vector") from error
    if values.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {values.shape}")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    return values.copy()


def _required_matrix_or_vector(
    info: Mapping[str, Any], key: str, shape: Tuple[int, ...]
) -> np.ndarray:
    if key not in info:
        raise KeyError(f"info must contain {key!r}")
    try:
        untyped = np.asarray(info[key])
    except (TypeError, ValueError) as error:
        raise ValueError(f"info[{key!r}] must have shape {shape}") from error
    if np.iscomplexobj(untyped):
        raise ValueError(f"info[{key!r}] must contain only real values")
    try:
        values = np.asarray(info[key], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"info[{key!r}] must have shape {shape}") from error
    if values.shape != shape:
        raise ValueError(
            f"info[{key!r}] must have shape {shape}, got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"info[{key!r}] contains NaN or Inf")
    return values.copy()


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    numeric = int(value)
    if numeric < 1:
        raise ValueError(f"{name} must be a positive integer")
    return numeric


def _movement_action(context: _PolicyContext, index: int) -> np.ndarray:
    displacement = (
        context.buoy_positions_m[index] - context.uav_position_m
    )
    distance = float(np.linalg.norm(displacement))
    action = np.zeros(context.action_dim, dtype=np.float64)
    if distance > _GEOMETRY_TOLERANCE:
        action[:_POSITION_DIM] = displacement / distance
    return _validate_legal_action(action, context.action_dim)


def _hover_action(action_dim: int) -> np.ndarray:
    return _validate_legal_action(
        np.zeros(action_dim, dtype=np.float64), action_dim
    )


def _validate_legal_action(action: np.ndarray, action_dim: int) -> np.ndarray:
    values = np.asarray(action, dtype=np.float64)
    if values.shape != (action_dim,):
        raise RuntimeError(
            f"baseline action must have shape ({action_dim},), got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise RuntimeError("baseline action contains NaN or Inf")
    norm = float(np.linalg.norm(values))
    if norm > 1.0 + 1.0e-12:
        raise RuntimeError(f"baseline action norm exceeds one: {norm}")
    if norm > 1.0:
        values = values / norm
    return values.astype(np.float64, copy=True)


def _masked_argmin(
    values: np.ndarray, mask: np.ndarray
) -> Optional[int]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None
    return int(indices[int(np.argmin(values[indices]))])


def _masked_argmax(
    values: np.ndarray, mask: np.ndarray
) -> Optional[int]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None
    return int(indices[int(np.argmax(values[indices]))])


def _masked_priority_argmax(
    values: np.ndarray,
    distances: np.ndarray,
    mask: np.ndarray,
) -> Optional[int]:
    """Maximize value, then minimize distance, then stable node index."""
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None
    maximum = float(np.max(values[indices]))
    tied = indices[
        np.isclose(
            values[indices],
            maximum,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    ]
    minimum_distance = float(np.min(distances[tied]))
    nearest = tied[
        np.isclose(
            distances[tied],
            minimum_distance,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    ]
    return int(np.min(nearest))


def _normalize_over_mask(
    values: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    if not np.any(mask):
        return result
    scale = float(np.max(values[mask]))
    if scale > 0.0:
        result[mask] = values[mask] / scale
    return result


def _one_step_movement_proxy(
    context: _PolicyContext, info: Mapping[str, Any]
) -> Tuple[np.ndarray, str]:
    supplied, source = _extract_optional_nonnegative_vector(
        info,
        _ENERGY_PROXY_KEYS,
        context.num_slots,
        "one-step movement energy proxy",
    )
    if supplied is not None:
        scale = float(np.max(supplied[context.valid_node_mask]))
        if scale > 0.0:
            return supplied / scale, f"normalized(info[{source!r}])"
        return supplied, f"info[{source!r}]"

    if "max_step_distance_m" in info:
        raw_scale = info["max_step_distance_m"]
        if isinstance(raw_scale, bool) or not isinstance(raw_scale, Real):
            raise ValueError(
                "info['max_step_distance_m'] must be a positive finite real"
            )
        max_step_distance = float(raw_scale)
        if not math.isfinite(max_step_distance) or max_step_distance <= 0.0:
            raise ValueError(
                "info['max_step_distance_m'] must be a positive finite real"
            )
        proxy = np.minimum(
            context.horizontal_distances_m, max_step_distance
        ) / max_step_distance
        return proxy, "normalized_one_step_movement_distance"

    proxy = np.zeros(context.num_slots, dtype=np.float64)
    for index in range(context.num_slots):
        proxy[index] = float(
            np.linalg.norm(_movement_action(context, index))
        )
    return proxy, "legal_action_norm_proxy"


__all__ = [
    "BaselinePolicy",
    "MAX_QUEUE",
    "MaxQueuePolicy",
    "NEAREST_BUOY",
    "NearestBuoyPolicy",
    "RANDOM_VALID",
    "RandomValidPolicy",
    "QUEUE_RATE_ENERGY",
    "QueueRateEnergyPolicy",
    "URGENCY_RATE_ENERGY",
    "UrgencyRateEnergyPolicy",
    "available_baseline_policies",
    "baseline_policy_metadata",
    "create_baseline_policy",
]
