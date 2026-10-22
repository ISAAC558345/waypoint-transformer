"""Pure current-state Task-Waypoint-to-motion mappings for Round-14.

These functions deliberately receive no trajectory, arrival, target, reward,
or future-outcome argument.  They map the frozen predictor's *current*
Task-Waypoint output and the already exposed current geometry/rates into the
existing normalized two-dimensional movement action space.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np


EPSILON = 1.0e-12
MappingName = Literal["A_service", "B_service_progress", "C_service_progress_rate"]
MAPPING_NAMES: tuple[MappingName, ...] = (
    "A_service",
    "B_service_progress",
    "C_service_progress_rate",
)


@dataclass(frozen=True)
class MappingResult:
    """A traceable movement intent produced entirely from current inputs."""

    mapping_name: str
    direction_xy: np.ndarray
    step_fraction: float
    node_weights: np.ndarray
    fallback: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "mapping_name": self.mapping_name,
            "direction_xy": self.direction_xy.tolist(),
            "step_fraction": float(self.step_fraction),
            "node_weights": self.node_weights.tolist(),
            "fallback": self.fallback,
        }


def _finite_vector(name: str, value: Sequence[float] | np.ndarray, width: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or (width is not None and result.size != width):
        expected = f"[{width}]" if width is not None else "one-dimensional"
        raise ValueError(f"{name} must be a finite {expected} vector")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def split_task_waypoint(phi_task: Sequence[float] | np.ndarray, max_buoys: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Split deployable ``Phi=[p_svc, DeltaD_srv, J_H]`` without labels."""
    if int(max_buoys) <= 0:
        raise ValueError("max_buoys must be positive")
    phi = _finite_vector("phi_task", phi_task, 2 * int(max_buoys) + 1)
    return phi[:max_buoys].copy(), phi[max_buoys : 2 * max_buoys].copy(), float(phi[-1])


def _masked_normalize(value: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    active = np.where(valid_mask, np.clip(value, 0.0, None), 0.0)
    total = float(active.sum())
    if total <= EPSILON:
        return np.zeros_like(active)
    return active / total


def _unit_vectors(uav_position_m: np.ndarray, buoy_positions_m: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    delta = buoy_positions_m - uav_position_m[None, :]
    distances = np.linalg.norm(delta, axis=1)
    result = np.zeros_like(delta)
    active = valid_mask & (distances > EPSILON)
    result[active] = delta[active] / distances[active, None]
    return result


def _validated_current_inputs(
    phi_task: Sequence[float] | np.ndarray,
    valid_node_mask: Sequence[bool] | np.ndarray,
    uav_position_m: Sequence[float] | np.ndarray,
    buoy_positions_m: Sequence[Sequence[float]] | np.ndarray,
    current_rates_bps: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = np.asarray(valid_node_mask, dtype=bool)
    if mask.ndim != 1 or mask.size < 1 or not bool(mask.any()):
        raise ValueError("valid_node_mask must be a nonempty vector with one valid buoy")
    service, progress, _ = split_task_waypoint(phi_task, int(mask.size))
    uav = _finite_vector("uav_position_m", uav_position_m, 2)
    buoys = np.asarray(buoy_positions_m, dtype=np.float64)
    if buoys.shape != (mask.size, 2) or not np.all(np.isfinite(buoys)):
        raise ValueError("buoy_positions_m must have shape [max_buoys,2] and be finite")
    rates = _finite_vector("current_rates_bps", current_rates_bps, int(mask.size))
    if np.any(rates < 0.0):
        raise ValueError("current_rates_bps must be non-negative")
    return service, progress, mask, uav, buoys, rates


def map_task_waypoint_to_motion(
    mapping_name: MappingName,
    phi_task: Sequence[float] | np.ndarray,
    valid_node_mask: Sequence[bool] | np.ndarray,
    uav_position_m: Sequence[float] | np.ndarray,
    buoy_positions_m: Sequence[Sequence[float]] | np.ndarray,
    current_rates_bps: Sequence[float] | np.ndarray,
    *,
    step_fraction: float = 1.0,
) -> MappingResult:
    """Map current Task-Waypoint heads and geometry into a local direction.

    A uses the masked service distribution; B multiplies it by positive
    predicted gross progress; C also multiplies it by a normalized *current*
    feasible rate.  The return head is intentionally absent here: it controls
    movement magnitude through :func:`return_magnitude`, never direction.
    """
    if mapping_name not in MAPPING_NAMES:
        raise ValueError(f"unknown mapping_name: {mapping_name!r}")
    if not np.isfinite(step_fraction) or not 0.0 <= float(step_fraction) <= 1.0:
        raise ValueError("step_fraction must be finite and within [0,1]")
    service, progress, mask, uav, buoys, rates = _validated_current_inputs(
        phi_task, valid_node_mask, uav_position_m, buoy_positions_m, current_rates_bps
    )
    base = _masked_normalize(service, mask)
    if mapping_name == "A_service":
        weights = base
    elif mapping_name == "B_service_progress":
        weighted = base * np.where(mask, np.clip(progress, 0.0, None), 0.0)
        weights = _masked_normalize(weighted, mask)
        if float(weights.sum()) <= EPSILON:
            weights = base
    else:
        rate_weights = _masked_normalize(rates, mask)
        weighted = base * np.where(mask, np.clip(progress, 0.0, None), 0.0) * rate_weights
        weights = _masked_normalize(weighted, mask)
        if float(weights.sum()) <= EPSILON:
            weights = _masked_normalize(base * rate_weights, mask)
        if float(weights.sum()) <= EPSILON:
            weights = base
    vector = np.sum(weights[:, None] * _unit_vectors(uav, buoys, mask), axis=0)
    norm = float(np.linalg.norm(vector))
    if norm <= EPSILON:
        return MappingResult(
            mapping_name=mapping_name,
            direction_xy=np.zeros(2, dtype=np.float64),
            step_fraction=float(step_fraction),
            node_weights=weights,
            fallback="zero_direction",
        )
    return MappingResult(
        mapping_name=mapping_name,
        direction_xy=(vector / norm).astype(np.float64, copy=False),
        step_fraction=float(step_fraction),
        node_weights=weights,
        fallback=None,
    )


def return_magnitude(phi_task: Sequence[float] | np.ndarray, max_buoys: int) -> float:
    """Fixed bounded return-head magnitude: sigmoid of predicted standardized J_H."""
    _, _, predicted_return = split_task_waypoint(phi_task, max_buoys)
    if predicted_return >= 0.0:
        value = 1.0 / (1.0 + np.exp(-predicted_return))
    else:
        exponent = np.exp(predicted_return)
        value = exponent / (1.0 + exponent)
    return float(value)


def guidance_action(result: MappingResult, *, magnitude: float | None = None) -> np.ndarray:
    """Return the unchanged normalized 2-D movement action, radially bounded."""
    fraction = result.step_fraction if magnitude is None else float(magnitude)
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("movement magnitude must be finite and within [0,1]")
    action = np.asarray(result.direction_xy, dtype=np.float64) * fraction
    norm = float(np.linalg.norm(action))
    if norm > 1.0 + EPSILON:
        action = action / norm
    if action.shape != (2,) or not np.all(np.isfinite(action)):
        raise RuntimeError("guidance action has invalid shape or values")
    return action


def current_observable_mapping_inputs(info: Mapping[str, object], phi_task: Sequence[float] | np.ndarray) -> dict[str, object]:
    """Expose exactly the current environment fields admitted to a mapping."""
    required = ("valid_node_mask", "uav_position_m", "buoy_positions_m", "rates_bps")
    missing = [key for key in required if key not in info]
    if missing:
        raise KeyError(f"current mapping information is missing: {missing}")
    return {
        "phi_task": np.asarray(phi_task, dtype=np.float64),
        "valid_node_mask": np.asarray(info["valid_node_mask"], dtype=bool),
        "uav_position_m": np.asarray(info["uav_position_m"], dtype=np.float64),
        "buoy_positions_m": np.asarray(info["buoy_positions_m"], dtype=np.float64),
        "current_rates_bps": np.asarray(info["rates_bps"], dtype=np.float64),
    }


__all__ = [
    "EPSILON",
    "MAPPING_NAMES",
    "MappingName",
    "MappingResult",
    "current_observable_mapping_inputs",
    "guidance_action",
    "map_task_waypoint_to_motion",
    "return_magnitude",
    "split_task_waypoint",
]
