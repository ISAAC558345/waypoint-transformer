"""Unified runner for round-02 non-learning maritime baselines.

The runner compares every requested policy on an identical schedule of
environment seeds, buoy counts, horizon, and termination rules.  It collects
structured complete episodes, writes real environment metrics, audits K-step
availability, and generates figures.  It never constructs or trains a model.

Smoke example from the repository root::

    PYTHONPATH=src python -m wt.maritime.runner --smoke
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
import math
from numbers import Integral
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .baselines import (
    available_baseline_policies,
    baseline_policy_metadata,
    create_baseline_policy,
)
from .config import MaritimeEnvConfig
from .env import MaritimeEnv
from .metrics import (
    compute_episode_metrics,
    summarize_episode_metrics,
    write_baseline_summary_csv,
    write_episode_metrics_csv,
)
from .plotting import generate_required_figures
from .trajectory import EpisodeTrajectory, EpisodeTrajectoryBuilder


RUNNER_SCHEMA_VERSION = "round02_runner_v1"
DEFAULT_K_VALUES = (1, 5, 10, 20, 30)
SMOKE_SEEDS = (101, 202, 303)
SMOKE_BUOY_COUNTS = (3, 5)
SMOKE_HORIZON = 8
FORMAL_SEEDS = tuple(range(10))
FORMAL_BUOY_COUNTS = (3, 4, 6)
FORMAL_HORIZON = 20
_ACTION_TOLERANCE = 1.0e-12
_SAFE_ID_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class RunArtifacts:
    """In-memory episodes plus all persisted round-02 artifact paths."""

    run_id: str
    output_dir: Path
    figures_dir: Path
    episodes: Tuple[EpisodeTrajectory, ...]
    episode_metrics: Tuple[Mapping[str, Any], ...]
    summary_metrics: Tuple[Mapping[str, Any], ...]
    split_assignments: Mapping[str, str]
    artifact_paths: Mapping[str, Path]
    figure_paths: Mapping[str, Path]


def collect_baseline_episodes(
    *,
    env_config: Optional[MaritimeEnvConfig] = None,
    policy_names: Optional[Sequence[str]] = None,
    scenario_id: str = "maritime_static_buoys",
    episodes_per_policy: int = 3,
    seeds: Sequence[int] = SMOKE_SEEDS,
    buoy_counts: Optional[Sequence[int]] = None,
    max_steps: Optional[int] = None,
) -> List[EpisodeTrajectory]:
    """Collect comparable complete episodes without writing artifacts.

    ``episodes_per_policy`` applies independently to each buoy count.  The
    ordered seed list is cycled only when more episodes than seeds are
    requested.  Thus every policy receives precisely the same sequence of
    ``(num_buoys, seed, effective_horizon)`` cases.
    """

    base_config = env_config if env_config is not None else MaritimeEnvConfig()
    if not isinstance(base_config, MaritimeEnvConfig):
        raise TypeError("env_config must be MaritimeEnvConfig or None")
    canonical_policies = _validate_policy_names(policy_names)
    validated_scenario = _validate_identifier("scenario_id", scenario_id)
    episode_count = _positive_integer("episodes_per_policy", episodes_per_policy)
    validated_seeds = _validate_seeds(seeds)
    counts = _validate_buoy_counts(
        buoy_counts if buoy_counts is not None else (base_config.num_buoys,)
    )
    step_limit = (
        None if max_steps is None else _positive_integer("max_steps", max_steps)
    )

    episodes: List[EpisodeTrajectory] = []
    seed_schedule = tuple(
        validated_seeds[index % len(validated_seeds)]
        for index in range(episode_count)
    )
    for num_buoys in counts:
        case_config = replace(
            base_config,
            num_buoys=num_buoys,
        )
        for episode_index, seed in enumerate(seed_schedule):
            for policy_name in canonical_policies:
                episode_id = _episode_id(
                    validated_scenario,
                    policy_name,
                    num_buoys,
                    seed,
                    episode_index,
                )
                episodes.append(
                    _rollout_episode(
                        config=case_config,
                        policy_name=policy_name,
                        scenario_id=validated_scenario,
                        episode_id=episode_id,
                        seed=seed,
                        max_steps=step_limit,
                    )
                )
    _validate_comparison_schedule(episodes, canonical_policies)
    return episodes


def run_baseline_experiment(
    *,
    env_config: Optional[MaritimeEnvConfig] = None,
    policy_names: Optional[Sequence[str]] = None,
    scenario_id: str = "maritime_static_buoys",
    episodes_per_policy: int = 3,
    seeds: Sequence[int] = SMOKE_SEEDS,
    buoy_counts: Optional[Sequence[int]] = None,
    max_steps: Optional[int] = None,
    save_trajectories: bool = True,
    output_dir: Union[str, Path] = "round_02/outputs/manual_run",
    figures_dir: Union[str, Path] = "round_02/figures/manual_run",
    run_id: str = "manual_run",
    mode: str = "bounded",
    split_seed: int = 90210,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> RunArtifacts:
    """Collect, audit, summarize, and plot one bounded baseline experiment."""

    validated_run_id = _validate_identifier("run_id", run_id)
    validated_mode = _validate_mode(mode)
    if not isinstance(save_trajectories, bool):
        raise TypeError("save_trajectories must be bool")
    validated_split_seed = _nonnegative_integer("split_seed", split_seed)
    validated_k_values = tuple(
        _positive_integer("K", value) for value in k_values
    )
    if not validated_k_values:
        raise ValueError("k_values must not be empty")
    if len(set(validated_k_values)) != len(validated_k_values):
        raise ValueError("k_values must not contain duplicates")

    base_config = env_config if env_config is not None else MaritimeEnvConfig()
    episodes = collect_baseline_episodes(
        env_config=base_config,
        policy_names=policy_names,
        scenario_id=scenario_id,
        episodes_per_policy=episodes_per_policy,
        seeds=seeds,
        buoy_counts=buoy_counts,
        max_steps=max_steps,
    )
    metrics = [
        compute_episode_metrics(episode, delta_t=base_config.delta_t)
        for episode in episodes
    ]
    summaries = summarize_episode_metrics(metrics)

    output_path = Path(output_dir)
    figure_path = Path(figures_dir)
    _prepare_directory(output_path)
    _prepare_directory(figure_path)
    _require_outputs_absent(
        (
            output_path / "baseline_episode_metrics.csv",
            output_path / "baseline_summary.csv",
            output_path / "k_step_audit.csv",
            output_path / "run_config.json",
        )
    )
    artifact_paths: Dict[str, Path] = {}

    split_assignments = _split_episodes(
        episodes, split_seed=validated_split_seed
    )
    if save_trajectories:
        dataset_paths = _save_structured_dataset(
            output_path,
            episodes,
            split_assignments=split_assignments,
            metadata={
                "run_id": validated_run_id,
                "runner_schema_version": RUNNER_SCHEMA_VERSION,
                "mode": validated_mode,
                "training_performed": False,
                "policy_metadata": baseline_policy_metadata(),
                "metric_semantics": {
                    "cumulative_link_capacity_bits": (
                        "integral of potential total channel rate; not actual payload"
                    ),
                    "queues_and_task_completion": "unavailable",
                },
            },
        )
        artifact_paths.update(dataset_paths)

    artifact_paths["baseline_episode_metrics"] = write_episode_metrics_csv(
        metrics, output_path / "baseline_episode_metrics.csv"
    )
    artifact_paths["baseline_summary"] = write_baseline_summary_csv(
        summaries, output_path / "baseline_summary.csv"
    )

    audit_rows, audit_path = _write_k_step_audit(
        episodes,
        split_assignments=split_assignments,
        k_values=validated_k_values,
        destination=output_path / "k_step_audit.csv",
    )
    artifact_paths["k_step_audit"] = audit_path

    effective_counts = sorted({episode.num_buoys for episode in episodes})
    effective_horizons = sorted({episode.episode_length for episode in episodes})
    run_configuration = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "run_id": validated_run_id,
        "mode": validated_mode,
        "training_performed": False,
        "scenario_id": str(scenario_id),
        "policy_names": sorted({episode.policy_id for episode in episodes}),
        "episodes_per_policy_per_buoy_count": int(episodes_per_policy),
        "requested_seeds": list(_validate_seeds(seeds)),
        "buoy_counts": effective_counts,
        "maximum_steps": max_steps,
        "observed_episode_lengths": effective_horizons,
        "base_environment_config": asdict(base_config),
        "save_trajectories": save_trajectories,
        "split_seed": validated_split_seed,
        "k_values": list(validated_k_values),
        "episode_count": len(episodes),
        "transition_count": int(sum(episode.episode_length for episode in episodes)),
        "k_step_audit_row_count": len(audit_rows),
        "metric_availability": {
            "link_capacity": True,
            "energy": True,
            "position": True,
            "queues": False,
            "task_success": False,
            "task_completion": False,
            "actual_collected_payload": False,
            "service_fairness": False,
        },
    }
    artifact_paths["run_config"] = _write_json(
        output_path / "run_config.json", run_configuration
    )

    figure_paths = generate_required_figures(
        episodes,
        metrics,
        figure_path,
        delta_t=base_config.delta_t,
    )
    return RunArtifacts(
        run_id=validated_run_id,
        output_dir=output_path,
        figures_dir=figure_path,
        episodes=tuple(episodes),
        episode_metrics=tuple(metrics),
        summary_metrics=tuple(summaries),
        split_assignments=dict(split_assignments),
        artifact_paths=dict(artifact_paths),
        figure_paths=dict(figure_paths),
    )


def _rollout_episode(
    *,
    config: MaritimeEnvConfig,
    policy_name: str,
    scenario_id: str,
    episode_id: str,
    seed: int,
    max_steps: Optional[int],
) -> EpisodeTrajectory:
    environment = MaritimeEnv(config)
    observation, info = environment.reset(seed=seed)
    policy = create_baseline_policy(policy_name, seed=seed)
    policy.reset(seed=seed)
    builder = EpisodeTrajectoryBuilder(
        episode_id=episode_id,
        policy_id=policy.name,
        seed=seed,
        scenario_id=scenario_id,
        num_buoys=environment.num_buoys,
        initial_info=info,
    )

    completed_steps = 0
    terminal_reason: Optional[str] = None
    while True:
        _validate_observation_and_info(observation, info, environment)
        policy_info = _policy_info(info, environment)
        action = np.asarray(
            policy.act(observation.copy(), policy_info), dtype=np.float64
        )
        _validate_action(action, environment.action_dim)
        next_observation, reward, terminated, truncated, next_info = (
            environment.step(action)
        )
        _require_finite_array("next_observation", next_observation)
        if next_observation.shape != (environment.state_dim,):
            raise RuntimeError(
                "environment next observation shape does not match state_dim"
            )
        if not math.isfinite(float(reward)):
            raise RuntimeError("environment returned a non-finite reward")
        _validate_public_dimensions(next_info, environment)
        _validate_step_info(next_info, environment.num_buoys)
        completed_steps += 1
        collector_limit = bool(
            max_steps is not None
            and completed_steps >= max_steps
            and not terminated
            and not truncated
        )
        recorded_truncated = bool(truncated or collector_limit)
        if collector_limit:
            terminal_reason = "collector_limit"
        builder.append(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            terminated=terminated,
            truncated=recorded_truncated,
            info_before=info,
            info_after=next_info,
            valid_node_mask=np.ones(environment.num_buoys, dtype=np.bool_),
            action_mask=None,
        )
        observation = next_observation
        info = next_info
        if terminated or recorded_truncated:
            break

    return builder.finalize(terminal_reason=terminal_reason)


def _policy_info(info: Mapping[str, Any], environment: MaritimeEnv) -> Dict[str, Any]:
    copied: Dict[str, Any] = {}
    for key, value in info.items():
        copied[key] = value.copy() if isinstance(value, np.ndarray) else value
    copied["valid_node_mask"] = np.ones(
        environment.num_buoys, dtype=np.bool_
    )
    copied["max_step_distance_m"] = float(
        environment.config.max_speed * environment.config.delta_t
    )
    return copied


def _validate_action(action: np.ndarray, action_dim: int) -> None:
    if action.shape != (action_dim,):
        raise RuntimeError(
            f"policy action must have shape ({action_dim},), got {action.shape}"
        )
    _require_finite_array("policy action", action)
    norm = float(np.linalg.norm(action))
    if norm > 1.0 + _ACTION_TOLERANCE:
        raise RuntimeError(f"policy produced action outside unit ball: norm={norm}")


def _validate_step_info(
    info: Mapping[str, Any],
    num_buoys: int,
) -> None:
    _validate_current_info(info, num_buoys)
    for name in ("actual_energy_consumed", "applied_movement_distance_m"):
        if name not in info or not math.isfinite(float(info[name])):
            raise RuntimeError(f"environment info {name!r} is missing or non-finite")


def _validate_current_info(info: Mapping[str, Any], num_buoys: int) -> None:
    required_arrays = {
        "uav_position_m": (2,),
        "buoy_positions_m": (num_buoys, 2),
        "rates_bps": (num_buoys,),
    }
    for name, expected_shape in required_arrays.items():
        if name not in info:
            raise RuntimeError(f"environment info is missing {name!r}")
        array = np.asarray(info[name])
        if array.shape != expected_shape:
            raise RuntimeError(
                f"environment info {name!r} must have shape {expected_shape}"
            )
        _require_finite_array(f"info[{name!r}]", array)
    for name in ("remaining_energy", "total_rate_bps"):
        if name not in info or not math.isfinite(float(info[name])):
            raise RuntimeError(f"environment info {name!r} is missing or non-finite")


def _validate_observation_and_info(
    observation: np.ndarray,
    info: Mapping[str, Any],
    environment: MaritimeEnv,
) -> None:
    values = np.asarray(observation)
    if values.shape != (environment.state_dim,):
        raise RuntimeError(
            "environment observation shape does not match public state_dim"
        )
    _require_finite_array("observation", values)
    _validate_public_dimensions(info, environment)
    _validate_current_info(info, environment.num_buoys)


def _validate_public_dimensions(
    info: Mapping[str, Any], environment: MaritimeEnv
) -> None:
    expected_scalars = {
        "state_dim": environment.state_dim,
        "action_dim": environment.action_dim,
        "num_buoys": environment.num_buoys,
    }
    for key, expected in expected_scalars.items():
        if key not in info or int(info[key]) != int(expected):
            raise RuntimeError(
                f"environment info {key!r} does not match public attribute"
            )


def _save_structured_dataset(
    output_dir: Path,
    episodes: Sequence[EpisodeTrajectory],
    *,
    split_assignments: Mapping[str, str],
    metadata: Mapping[str, Any],
) -> Mapping[str, Path]:
    from .dataset_io import save_dataset

    manifest = save_dataset(
        output_dir,
        episodes,
        split_assignments=split_assignments,
        metadata=metadata,
    )
    paths: Dict[str, Path] = {
        "dataset_manifest": output_dir / "dataset_manifest.json",
        "dataset_schema": output_dir / "dataset_schema.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"{name} was not generated at {path}")
    if manifest is None:
        raise RuntimeError("save_dataset did not return a manifest")
    return paths


def _split_episodes(
    episodes: Sequence[EpisodeTrajectory], *, split_seed: int
) -> Mapping[str, str]:
    from .dataset_io import deterministic_episode_split

    return deterministic_episode_split(episodes, seed=split_seed)


def _write_k_step_audit(
    episodes: Sequence[EpisodeTrajectory],
    *,
    split_assignments: Mapping[str, str],
    k_values: Sequence[int],
    destination: Path,
) -> Tuple[Sequence[Mapping[str, Any]], Path]:
    from .k_step_audit import audit_k_steps, write_k_step_audit_csv

    rows = audit_k_steps(
        episodes,
        split_assignments=split_assignments,
        k_values=k_values,
    )
    written = write_k_step_audit_csv(rows, destination)
    path = Path(written) if written is not None else destination
    if not path.is_file():
        raise RuntimeError(f"K-step audit CSV was not generated at {path}")
    return rows, path


def _validate_comparison_schedule(
    episodes: Sequence[EpisodeTrajectory],
    policy_names: Sequence[str],
) -> None:
    expected: Optional[List[Tuple[int, int]]] = None
    for policy_name in policy_names:
        schedule = sorted(
            (
                int(episode.num_buoys),
                int(episode.seed),
            )
            for episode in episodes
            if episode.policy_id == policy_name
        )
        if expected is None:
            expected = schedule
        elif schedule != expected:
            raise RuntimeError(
                "baseline comparison schedules differ across policies"
            )


def _validate_policy_names(
    policy_names: Optional[Sequence[str]],
) -> Tuple[str, ...]:
    supported = available_baseline_policies()
    if policy_names is None:
        return tuple(supported)
    if isinstance(policy_names, str):
        requested = (policy_names,)
    else:
        requested = tuple(policy_names)
    if not requested:
        raise ValueError("policy_names must not be empty")
    canonical = tuple(str(name).strip().lower() for name in requested)
    unknown = sorted(set(canonical) - set(supported))
    if unknown:
        raise ValueError(
            f"unknown policies {unknown}; supported policies are {supported}"
        )
    if len(set(canonical)) != len(canonical):
        raise ValueError("policy_names must not contain duplicates")
    return canonical


def _validate_seeds(seeds: Sequence[int]) -> Tuple[int, ...]:
    if isinstance(seeds, (str, bytes)):
        raise TypeError("seeds must be a sequence of nonnegative integers")
    values = tuple(_nonnegative_integer("seed", value) for value in seeds)
    if not values:
        raise ValueError("seeds must not be empty")
    return values


def _validate_buoy_counts(values: Sequence[int]) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("buoy_counts must be a sequence of positive integers")
    counts = tuple(_positive_integer("num_buoys", value) for value in values)
    if not counts:
        raise ValueError("buoy_counts must not be empty")
    if len(set(counts)) != len(counts):
        raise ValueError("buoy_counts must not contain duplicates")
    return counts


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    numeric = int(value)
    if numeric < 1:
        raise ValueError(f"{name} must be a positive integer")
    return numeric


def _nonnegative_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a nonnegative integer")
    numeric = int(value)
    if numeric < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return numeric


def _validate_identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _validate_mode(value: str) -> str:
    mode = _validate_identifier("mode", value).lower()
    if mode not in {"smoke", "formal", "bounded"}:
        raise ValueError("mode must be smoke, formal, or bounded")
    return mode


def _episode_id(
    scenario_id: str,
    policy_name: str,
    num_buoys: int,
    seed: int,
    episode_index: int,
) -> str:
    safe_scenario = _SAFE_ID_PATTERN.sub("-", scenario_id).strip("-")
    return (
        f"{safe_scenario}__b{num_buoys}__e{episode_index:04d}"
        f"__s{seed}__{policy_name}"
    )


def _require_finite_array(name: str, values: Any) -> None:
    array = np.asarray(values)
    if not np.all(np.isfinite(array)):
        raise RuntimeError(f"{name} contains NaN or Inf")


def _prepare_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise RuntimeError(f"output path is not a directory: {path}")


def _require_outputs_absent(paths: Sequence[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing outputs: "
            + ", ".join(str(path) for path in existing)
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    return path


def _parse_integer_list(value: str, name: str) -> Tuple[int, ...]:
    pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not pieces:
        raise argparse.ArgumentTypeError(f"{name} must not be empty")
    try:
        return tuple(int(piece) for piece in pieces)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{name} must be a comma-separated integer list"
        ) from error


def _parse_policy_arguments(values: Optional[Sequence[str]]) -> Optional[Tuple[str, ...]]:
    if not values:
        return None
    names: List[str] = []
    for value in values:
        names.extend(piece.strip() for piece in value.split(",") if piece.strip())
    return tuple(names)


def _load_config(path: Optional[str]) -> Tuple[MaritimeEnvConfig, Dict[str, Any]]:
    if path is None:
        return MaritimeEnvConfig(), {}
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("configuration file must contain a JSON object")
    env_payload = payload.get("environment", payload.get("env_config", payload))
    if not isinstance(env_payload, dict):
        raise ValueError("environment configuration must be a JSON object")
    known_fields = set(MaritimeEnvConfig.__dataclass_fields__)
    env_values = {
        key: value for key, value in env_payload.items() if key in known_fields
    }
    unknown = sorted(set(env_payload) - known_fields)
    if unknown and env_payload is payload:
        runner_keys = {
            "scenario_id",
            "policy_names",
            "episodes",
            "seeds",
            "buoy_counts",
            "max_steps",
            "save_trajectories",
            "output_dir",
            "figures_dir",
            "run_id",
            "split_seed",
            "k_values",
        }
        unexpected = sorted(set(unknown) - runner_keys)
        if unexpected:
            raise ValueError(f"unknown configuration fields: {unexpected}")
    elif unknown:
        raise ValueError(f"unknown environment configuration fields: {unknown}")
    return MaritimeEnvConfig(**env_values), payload


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface without parsing global process state."""

    parser = argparse.ArgumentParser(
        description="Run and audit non-learning maritime baselines (no training)."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--smoke",
        action="store_true",
        help="Run 4 policies x 3 seeds x 2 buoy counts with horizon 8.",
    )
    mode_group.add_argument(
        "--formal",
        action="store_true",
        help="Run a bounded formal collection using explicit/default CLI limits.",
    )
    mode_group.add_argument(
        "--round3-smoke",
        action="store_true",
        help="Run the schema-v1 Round-3 shared-trace smoke pipeline.",
    )
    mode_group.add_argument(
        "--round3-pilot",
        action="store_true",
        help="Run the fixed 80-episode, horizon-128 Round-3 pilot.",
    )
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config.")
    parser.add_argument(
        "--policy",
        action="append",
        default=None,
        help="Policy name; repeat or pass comma-separated names (default: all four).",
    )
    parser.add_argument("--scenario-id", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--num-buoys", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--figures-dir", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--k-values", type=str, default=None)
    parser.add_argument(
        "--no-save-trajectories",
        action="store_true",
        help="Run metrics/figures only; the default saves the structured dataset.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the bounded CLI mode and print a concise artifact summary."""

    args = build_argument_parser().parse_args(argv)
    if args.round3_smoke or args.round3_pilot:
        from .round3_runner import main as round3_main

        round3_arguments = [
            "--smoke" if args.round3_smoke else "--pilot"
        ]
        if args.run_id is not None:
            round3_arguments.extend(("--run-id", args.run_id))
        if args.output_dir is not None:
            round3_arguments.extend(("--output-dir", args.output_dir))
        if args.figures_dir is not None:
            round3_arguments.extend(("--figures-dir", args.figures_dir))
        return int(round3_main(round3_arguments))
    env_config, file_config = _load_config(args.config)
    mode = "smoke" if args.smoke else "formal" if args.formal else "bounded"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = (
        args.run_id
        or file_config.get("run_id")
        or f"round02_{mode}_{timestamp}"
    )
    repository_root = Path(__file__).resolve().parents[3]
    output_dir = (
        args.output_dir
        or file_config.get("output_dir")
        or str(repository_root / "round_02" / "outputs" / run_id)
    )
    figures_dir = (
        args.figures_dir
        or file_config.get("figures_dir")
        or str(repository_root / "round_02" / "figures" / run_id)
    )

    if mode == "smoke":
        default_seeds = SMOKE_SEEDS
        default_counts = SMOKE_BUOY_COUNTS
        default_episodes = len(SMOKE_SEEDS)
        default_max_steps = SMOKE_HORIZON
    elif mode == "formal":
        default_seeds = FORMAL_SEEDS
        default_counts = FORMAL_BUOY_COUNTS
        default_episodes = len(FORMAL_SEEDS)
        default_max_steps = FORMAL_HORIZON
    else:
        default_seeds = SMOKE_SEEDS
        default_counts = (env_config.num_buoys,)
        default_episodes = len(default_seeds)
        default_max_steps = env_config.episode_horizon

    seeds = (
        _parse_integer_list(args.seeds, "seeds")
        if args.seeds is not None
        else tuple(file_config.get("seeds", default_seeds))
    )
    buoy_counts = (
        _parse_integer_list(args.num_buoys, "num_buoys")
        if args.num_buoys is not None
        else tuple(file_config.get("buoy_counts", default_counts))
    )
    policies = _parse_policy_arguments(args.policy)
    if policies is None and "policy_names" in file_config:
        policies = tuple(file_config["policy_names"])
    episodes = int(
        args.episodes
        if args.episodes is not None
        else file_config.get("episodes", default_episodes)
    )
    max_steps = int(
        args.max_steps
        if args.max_steps is not None
        else file_config.get("max_steps", default_max_steps)
    )
    scenario_id = str(
        args.scenario_id
        or file_config.get("scenario_id", "maritime_static_buoys")
    )
    split_seed = int(
        args.split_seed
        if args.split_seed is not None
        else file_config.get("split_seed", 90210)
    )
    k_values = (
        _parse_integer_list(args.k_values, "k_values")
        if args.k_values is not None
        else tuple(file_config.get("k_values", DEFAULT_K_VALUES))
    )
    save_trajectories = not args.no_save_trajectories
    if "save_trajectories" in file_config and not args.no_save_trajectories:
        save_trajectories = bool(file_config["save_trajectories"])

    result = run_baseline_experiment(
        env_config=env_config,
        policy_names=policies,
        scenario_id=scenario_id,
        episodes_per_policy=episodes,
        seeds=seeds,
        buoy_counts=buoy_counts,
        max_steps=max_steps,
        save_trajectories=save_trajectories,
        output_dir=output_dir,
        figures_dir=figures_dir,
        run_id=run_id,
        mode=mode,
        split_seed=split_seed,
        k_values=k_values,
    )
    transition_count = sum(
        episode.episode_length for episode in result.episodes
    )
    print(f"run_id: {result.run_id}")
    print(f"mode: {mode}")
    print(f"episodes: {len(result.episodes)}")
    print(f"transitions: {transition_count}")
    print(f"output_dir: {result.output_dir}")
    print(f"figures_dir: {result.figures_dir}")
    print(f"figures: {len(result.figure_paths)}")
    print("training_performed: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
