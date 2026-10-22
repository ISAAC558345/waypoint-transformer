"""Replay-only diagnostics and bounded Round-5 interface smoke checks.

The replay path is intentionally incapable of stepping an environment: it
operates only on recorded states.  The separate interface smoke helper accepts
a disposable environment runner, executes at most five steps, and labels its
result so it cannot be mistaken for formal closed-loop evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .round5_evaluation import (
    assert_finite_tree,
    build_action_prediction_rows,
    radial_action_projection,
    write_finite_json,
    write_records_csv,
)


PathLike = Union[str, Path]


_EPISODE_KEYS = ("episode_id", "trajectory_id", "episode")
_TIMESTEP_KEYS = ("timestep", "t", "step", "transition_index")
_POLICY_KEYS = ("policy", "policy_id", "collection_policy", "heuristic_policy")
_BUOY_KEYS = ("num_buoys", "buoy_count", "n_buoys")


def select_representative_episodes(
    metadata: Sequence[Mapping[str, Any]],
    target_normalized: Sequence[Sequence[float]],
    predicted_normalized: Sequence[Sequence[float]],
    max_step_distance_m: float,
    *,
    representatives_per_policy: int = 1,
    representatives_per_buoy_scale: int = 1,
    high_error_count: int = 2,
    low_error_count: int = 2,
) -> List[Dict[str, Any]]:
    """Deterministically cover policy/buoy groups plus error extremes.

    Group representatives are episodes nearest to the median episode error,
    avoiding an implicit preference for only best or worst examples.  An
    episode selected for several reasons appears once with all reasons listed.
    """

    for name, value in (
        ("representatives_per_policy", representatives_per_policy),
        ("representatives_per_buoy_scale", representatives_per_buoy_scale),
        ("high_error_count", high_error_count),
        ("low_error_count", low_error_count),
    ):
        if int(value) < 0:
            raise ValueError(f"{name} must be non-negative")
    target = _actions(target_normalized, "target_normalized")
    prediction = _actions(predicted_normalized, "predicted_normalized")
    if target.shape != prediction.shape or len(metadata) != target.shape[0]:
        raise ValueError("metadata, target actions, and predictions must align")
    maximum = _positive(max_step_distance_m, "max_step_distance_m")
    per_row_error = np.linalg.norm((prediction - target) * maximum, axis=1)

    episodes: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(metadata):
        episode = str(_required_alias(row, _EPISODE_KEYS, "episode_id"))
        entry = episodes.setdefault(
            episode,
            {
                "episode_id": episode,
                "policy": str(_alias(row, _POLICY_KEYS, "unknown")),
                "num_buoys": str(_alias(row, _BUOY_KEYS, "unknown")),
                "errors": [],
            },
        )
        entry["errors"].append(float(per_row_error[index]))
    if not episodes:
        raise ValueError("representative selection requires at least one episode")
    summaries: List[Dict[str, Any]] = []
    for entry in episodes.values():
        errors = np.asarray(entry.pop("errors"), dtype=np.float64)
        summaries.append(
            {
                **entry,
                "transition_count": int(errors.size),
                "mean_action_error_m": float(np.mean(errors)),
                "max_action_error_m": float(np.max(errors)),
            }
        )
    summaries.sort(key=lambda row: str(row["episode_id"]))

    reasons: Dict[str, List[str]] = {}

    def add(row: Mapping[str, Any], reason: str) -> None:
        reason_list = reasons.setdefault(str(row["episode_id"]), [])
        if reason not in reason_list:
            reason_list.append(reason)

    for dimension, count in (
        ("policy", int(representatives_per_policy)),
        ("num_buoys", int(representatives_per_buoy_scale)),
    ):
        groups = _stable_unique([str(row[dimension]) for row in summaries])
        for group in groups:
            candidates = [row for row in summaries if str(row[dimension]) == group]
            group_median = float(np.median([row["mean_action_error_m"] for row in candidates]))
            ordered = sorted(
                candidates,
                key=lambda row: (
                    abs(float(row["mean_action_error_m"]) - group_median),
                    str(row["episode_id"]),
                ),
            )
            for row in ordered[:count]:
                add(row, f"representative_{dimension}:{group}")

    ascending = sorted(
        summaries, key=lambda row: (float(row["mean_action_error_m"]), str(row["episode_id"]))
    )
    for row in ascending[: int(low_error_count)]:
        add(row, "low_error")
    for row in list(reversed(ascending))[: int(high_error_count)]:
        add(row, "high_error")

    selected: List[Dict[str, Any]] = []
    summary_by_id = {str(row["episode_id"]): row for row in summaries}
    for episode in sorted(reasons):
        selected.append({**summary_by_id[episode], "selection_reasons": "|".join(reasons[episode])})
    assert_finite_tree(selected)
    return selected


def build_replay_diagnostics(
    metadata: Sequence[Mapping[str, Any]],
    target_normalized: Sequence[Sequence[float]],
    predicted_normalized: Sequence[Sequence[float]],
    max_step_distance_m: float,
    *,
    recorded_states: Optional[Sequence[Any]] = None,
    predicted_waypoints: Optional[Sequence[Sequence[float]]] = None,
    true_waypoints: Optional[Sequence[Sequence[float]]] = None,
    selected_episode_ids: Optional[Sequence[Any]] = None,
    representatives_per_policy: int = 1,
    representatives_per_buoy_scale: int = 1,
    high_error_count: int = 2,
    low_error_count: int = 2,
    near_zero_epsilon_m: float = 1e-6,
    base_fields: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build replay rows from recorded arrays without accepting an environment.

    The absence of any environment/step callback in this API is intentional and
    audited through ``environment_advanced=false`` in every output row.
    """

    target = _actions(target_normalized, "target_normalized")
    prediction = _actions(predicted_normalized, "predicted_normalized")
    if target.shape != prediction.shape or len(metadata) != target.shape[0]:
        raise ValueError("metadata, target actions, and predictions must align")
    if recorded_states is not None and len(recorded_states) != target.shape[0]:
        raise ValueError("recorded_states length does not match actions")
    predicted_wp = _optional_vectors(predicted_waypoints, target.shape[0], "predicted_waypoints")
    true_wp = _optional_vectors(true_waypoints, target.shape[0], "true_waypoints")
    if true_wp is not None and predicted_wp is None:
        raise ValueError("true_waypoints require predicted_waypoints")

    selections = select_representative_episodes(
        metadata,
        target,
        prediction,
        max_step_distance_m,
        representatives_per_policy=representatives_per_policy,
        representatives_per_buoy_scale=representatives_per_buoy_scale,
        high_error_count=high_error_count,
        low_error_count=low_error_count,
    )
    selection_reasons = {
        str(row["episode_id"]): str(row["selection_reasons"]) for row in selections
    }
    if selected_episode_ids is not None:
        requested = _stable_unique([str(value) for value in selected_episode_ids])
        known = {str(_required_alias(row, _EPISODE_KEYS, "episode_id")) for row in metadata}
        missing = [episode for episode in requested if episode not in known]
        if missing:
            raise ValueError(f"selected replay episodes are absent from metadata: {missing}")
        selection_reasons = {episode: "explicit" for episode in requested}
        selections = [
            {
                "episode_id": episode,
                "policy": "recorded",
                "num_buoys": "recorded",
                "transition_count": int(
                    sum(str(_required_alias(row, _EPISODE_KEYS, "episode_id")) == episode for row in metadata)
                ),
                "mean_action_error_m": 0.0,
                "max_action_error_m": 0.0,
                "selection_reasons": "explicit",
            }
            for episode in requested
        ]

    all_rows = build_action_prediction_rows(
        target,
        prediction,
        max_step_distance_m,
        metadata=metadata,
        near_zero_epsilon_m=near_zero_epsilon_m,
        base_fields=base_fields,
    )
    replay_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(all_rows):
        episode = str(_required_alias(row, _EPISODE_KEYS, "episode_id"))
        if episode not in selection_reasons:
            continue
        replay = {
            **row,
            "replay_mode": True,
            "recorded_state_used": True,
            "environment_advanced": False,
            "selection_reasons": selection_reasons[episode],
        }
        if recorded_states is not None:
            replay["recorded_state_sha256"] = stable_value_hash(recorded_states[index])
        if predicted_wp is not None:
            replay["predicted_waypoint_x"] = float(predicted_wp[index, 0])
            replay["predicted_waypoint_y"] = float(predicted_wp[index, 1])
        if true_wp is not None:
            replay["true_waypoint_x"] = float(true_wp[index, 0])
            replay["true_waypoint_y"] = float(true_wp[index, 1])
            replay["waypoint_error_l2"] = float(np.linalg.norm(predicted_wp[index] - true_wp[index]))
        assert_finite_tree(replay)
        replay_rows.append(replay)
    replay_rows.sort(key=_episode_timestep_key)
    result = {
        "schema_version": "round05_replay_diagnostics_v1",
        "replay_mode": True,
        "recorded_states_only": True,
        "environment_advanced": False,
        "formal_closed_loop_evaluation": False,
        "selected_episode_count": len(selection_reasons),
        "diagnostic_transition_count": len(replay_rows),
        "selected_episodes": selections,
        "rows": replay_rows,
    }
    assert_finite_tree(result)
    return result


def run_recorded_replay(
    records: Sequence[Mapping[str, Any]],
    waypoint_callback: Callable[[Any, Any, Any, Mapping[str, Any]], Sequence[float]],
    action_callback: Callable[[Tuple[Mapping[str, Any], ...]], Sequence[float]],
    max_step_distance_m: float,
    *,
    context_length: int,
    state_key: str = "state",
    goal_key: str = "goal",
    node_mask_key: str = "node_mask",
    remaining_time_key: str = "remaining_time",
    action_label_key: str = "true_action_normalized",
    true_waypoint_key: Optional[str] = None,
    **diagnostic_options: Any,
) -> Dict[str, Any]:
    """Infer actions over recorded histories; no environment object is accepted.

    ``action_callback`` receives tokens containing only state, node mask,
    predicted waypoint, and remaining time.  In particular, current and
    historical action labels are never included in its input.
    """

    context = int(context_length)
    if context <= 0:
        raise ValueError("context_length must be positive")
    if not records:
        raise ValueError("recorded replay requires at least one transition")
    ordered = sorted((dict(record) for record in records), key=_episode_timestep_key)
    metadata: List[Dict[str, Any]] = []
    states: List[Any] = []
    targets: List[np.ndarray] = []
    predictions: List[np.ndarray] = []
    predicted_waypoints: List[np.ndarray] = []
    true_waypoints: List[np.ndarray] = []
    histories: Dict[str, List[Dict[str, Any]]] = {}
    for record in ordered:
        episode = str(_required_alias(record, _EPISODE_KEYS, "episode_id"))
        if state_key not in record or action_label_key not in record:
            raise KeyError(f"replay record requires {state_key!r} and {action_label_key!r}")
        state = record[state_key]
        goal = record.get(goal_key)
        node_mask = record.get(node_mask_key)
        waypoint = _vector2(
            waypoint_callback(state, goal, node_mask, record), "waypoint_callback output"
        )
        token = {
            "state": state,
            "node_mask": node_mask,
            "predicted_waypoint": waypoint.copy(),
            "remaining_time": record.get(remaining_time_key, 0.0),
        }
        history = histories.setdefault(episode, [])
        history.append(token)
        model_history = tuple(history[-context:])
        prediction = _vector2(action_callback(model_history), "action_callback output")
        target = _vector2(record[action_label_key], action_label_key)
        metadata.append(_metadata_without_model_inputs(record))
        states.append(state)
        targets.append(target)
        predictions.append(prediction)
        predicted_waypoints.append(waypoint)
        if true_waypoint_key is not None:
            if true_waypoint_key not in record:
                raise KeyError(f"replay record misses {true_waypoint_key!r}")
            true_waypoints.append(_vector2(record[true_waypoint_key], true_waypoint_key))
    result = build_replay_diagnostics(
        metadata,
        np.stack(targets),
        np.stack(predictions),
        max_step_distance_m,
        recorded_states=states,
        predicted_waypoints=np.stack(predicted_waypoints),
        true_waypoints=np.stack(true_waypoints) if true_waypoints else None,
        **diagnostic_options,
    )
    result["waypoint_recomputed_per_transition"] = True
    result["action_history_input"] = False
    result["waypoint_callback_count"] = len(records)
    result["action_callback_count"] = len(records)
    assert_finite_tree(result)
    return result


def run_interface_smoke(
    env_runner: Any,
    state_callback: Callable[[Any, Mapping[str, Any], int], Any],
    waypoint_callback: Callable[[Any, Mapping[str, Any], int], Sequence[float]],
    action_callback: Callable[[Tuple[Mapping[str, Any], ...], Any, np.ndarray, int], Sequence[float]],
    max_step_distance_m: float,
    *,
    steps: int = 3,
    context_length: int = 10,
    reset_kwargs: Optional[Mapping[str, Any]] = None,
    environment_action_callback: Optional[
        Callable[[np.ndarray, np.ndarray, Any, np.ndarray, int], Any]
    ] = None,
) -> Dict[str, Any]:
    """Execute a strictly bounded 1--5 step interface integration smoke.

    The supplied runner is expected to be an isolated/disposable environment
    copy.  State and waypoint callbacks are invoked again before every action.
    Rewards are intentionally not accumulated or reported, preventing this
    engineering check from becoming a closed-loop performance comparison.
    """

    requested_steps = int(steps)
    if requested_steps < 1 or requested_steps > 5:
        raise ValueError("interface smoke steps must be in the inclusive range [1, 5]")
    context = int(context_length)
    if context <= 0:
        raise ValueError("context_length must be positive")
    maximum = _positive(max_step_distance_m, "max_step_distance_m")
    reset_fn = _runner_callable(env_runner, "reset")
    step_fn = _runner_callable(env_runner, "step")
    reset_result = reset_fn(**dict(reset_kwargs or {}))
    observation, info = _parse_reset(reset_result)
    history: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    terminated = False
    truncated = False
    state_calls = 0
    waypoint_calls = 0
    action_calls = 0
    for step_index in range(requested_steps):
        state = state_callback(observation, info, step_index)
        state_calls += 1
        waypoint = _vector2(
            waypoint_callback(state, info, step_index), "waypoint_callback output"
        )
        waypoint_calls += 1
        token = {
            "state": state,
            "predicted_waypoint": waypoint.copy(),
            "remaining_time": info.get("remaining_time", 0.0),
        }
        history.append(token)
        model_history = tuple(history[-context:])
        raw_normalized = _vector2(
            action_callback(model_history, state, waypoint.copy(), step_index),
            "action_callback output",
        )
        action_calls += 1
        projection = radial_action_projection(raw_normalized[None, :], maximum)
        projected_normalized = projection.projected_normalized[0]
        projected_m = projection.projected_m[0]
        env_action = (
            projected_m.copy()
            if environment_action_callback is None
            else environment_action_callback(
                projected_m.copy(), projected_normalized.copy(), state, waypoint.copy(), step_index
            )
        )
        next_observation, next_info, terminated, truncated = _parse_step(step_fn(env_action))
        row = {
            "step": step_index,
            "state_sha256": stable_value_hash(state),
            "waypoint_sha256": stable_value_hash(waypoint),
            "predicted_waypoint_x": float(waypoint[0]),
            "predicted_waypoint_y": float(waypoint[1]),
            "raw_action_normalized_x": float(raw_normalized[0]),
            "raw_action_normalized_y": float(raw_normalized[1]),
            "projected_action_x_m": float(projected_m[0]),
            "projected_action_y_m": float(projected_m[1]),
            "projection_activated": bool(projection.activation_mask[0]),
            "action_valid": bool(projection.projected_valid_mask[0]),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "state_recomputed": True,
            "waypoint_recomputed": True,
        }
        assert_finite_tree(row)
        step_rows.append(row)
        observation, info = next_observation, next_info
        if terminated or truncated:
            break
    executed = len(step_rows)
    result = {
        "schema_version": "round05_interface_smoke_v1",
        "interface_smoke": True,
        "closed_loop_interface_smoke_performed": True,
        "formal_closed_loop_evaluation": False,
        "closed_loop_evaluation_performed": False,
        "performance_ranking_performed": False,
        "steps_requested": requested_steps,
        "steps_executed": executed,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "state_callback_count": state_calls,
        "waypoint_callback_count": waypoint_calls,
        "action_callback_count": action_calls,
        "state_recomputed_each_step": state_calls == executed,
        "waypoint_recomputed_each_step": waypoint_calls == executed,
        "all_actions_valid": all(bool(row["action_valid"]) for row in step_rows),
        "rows": step_rows,
    }
    assert_finite_tree(result)
    return result


def write_replay_diagnostics(
    result: Mapping[str, Any], csv_path: PathLike, json_path: Optional[PathLike] = None
) -> Dict[str, Path]:
    """Persist replay rows and an optional strict JSON audit manifest."""

    if bool(result.get("environment_advanced", True)):
        raise ValueError("replay diagnostics must declare environment_advanced=false")
    rows = result.get("rows")
    if not isinstance(rows, Sequence) or not rows:
        raise ValueError("replay diagnostics contain no rows")
    paths = {"csv": write_records_csv(csv_path, rows)}
    if json_path is not None:
        paths["json"] = write_finite_json(json_path, result)
    return paths


def stable_value_hash(value: Any) -> str:
    """Return a deterministic hash for nested recorded state values."""

    canonical = _canonical_for_hash(value)
    payload = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _canonical_for_hash(value: Any) -> Any:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        if not np.all(np.isfinite(value)):
            raise ValueError("cannot hash an array containing NaN or Inf")
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if isinstance(value, Mapping):
        return {str(key): _canonical_for_hash(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_for_hash(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("cannot hash NaN or Inf")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return repr(value)


def _metadata_without_model_inputs(record: Mapping[str, Any]) -> Dict[str, Any]:
    excluded = {
        "state",
        "goal",
        "node_mask",
        "true_action_normalized",
        "action",
        "actions",
        "action_label",
    }
    result = {str(key): value for key, value in record.items() if key not in excluded}
    for key, value in list(result.items()):
        if isinstance(value, (Mapping, list, tuple, np.ndarray)):
            del result[key]
    assert_finite_tree(result)
    return result


def _runner_callable(runner: Any, name: str) -> Callable[..., Any]:
    callback = runner.get(name) if isinstance(runner, Mapping) else getattr(runner, name, None)
    if not callable(callback):
        raise TypeError(f"env_runner must provide callable {name}()")
    return callback


def _parse_reset(result: Any) -> Tuple[Any, Dict[str, Any]]:
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Mapping):
        return result[0], dict(result[1])
    return result, {}


def _parse_step(result: Any) -> Tuple[Any, Dict[str, Any], bool, bool]:
    if isinstance(result, Mapping):
        if "observation" not in result:
            raise ValueError("mapping step result requires an observation field")
        return (
            result["observation"],
            dict(result.get("info", {})),
            bool(result.get("terminated", result.get("done", False))),
            bool(result.get("truncated", False)),
        )
    if not isinstance(result, tuple):
        raise TypeError("env step must return a Gym-style tuple or mapping")
    if len(result) == 5:
        observation, _reward, terminated, truncated, info = result
        return observation, dict(info or {}), bool(terminated), bool(truncated)
    if len(result) == 4:
        observation, _reward, done, info = result
        info_dict = dict(info or {})
        truncated = bool(info_dict.get("TimeLimit.truncated", False))
        return observation, info_dict, bool(done) and not truncated, truncated
    raise ValueError("env step result must have four or five items")


def _actions(value: Any, name: str) -> np.ndarray:
    array = _to_numpy(value).astype(np.float64, copy=False)
    if array.ndim != 2 or array.shape[1] != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape [N, 2]")
    return array


def _optional_vectors(value: Any, count: int, name: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = _actions(value, name)
    if array.shape[0] != count:
        raise ValueError(f"{name} length does not match actions")
    return array


def _vector2(value: Any, name: str) -> np.ndarray:
    array = _to_numpy(value).astype(np.float64, copy=False).reshape(-1)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite two-dimensional vector")
    return array.copy()


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    return np.asarray(value)


def _episode_timestep_key(row: Mapping[str, Any]) -> Tuple[str, int, str]:
    episode = str(_required_alias(row, _EPISODE_KEYS, "episode_id"))
    timestep = _alias(row, _TIMESTEP_KEYS, 0)
    try:
        numeric_timestep = int(timestep)
    except (TypeError, ValueError):
        numeric_timestep = 0
    return episode, numeric_timestep, str(timestep)


def _required_alias(row: Mapping[str, Any], keys: Sequence[str], label: str) -> Any:
    value = _alias(row, keys, None)
    if value in (None, ""):
        raise KeyError(f"metadata row requires {label}")
    return value


def _alias(row: Mapping[str, Any], keys: Sequence[str], default: Any) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def _stable_unique(values: Sequence[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _positive(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


# Explicit aliases for callers that prefer callback-oriented naming.
run_interface_callback_smoke = run_interface_smoke
run_replay_mode = run_recorded_replay


__all__ = [
    "build_replay_diagnostics",
    "run_interface_callback_smoke",
    "run_interface_smoke",
    "run_recorded_replay",
    "run_replay_mode",
    "select_representative_episodes",
    "stable_value_hash",
    "write_replay_diagnostics",
]
