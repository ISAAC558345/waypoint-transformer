"""Round-3 shared-trace non-learning collection and Waypoint sample pipeline.

This module never imports a learning framework, constructs a model, creates an
optimizer, or performs training.  It is deliberately separate from the locked
round-02 runner and emits only into ``round_03`` directories.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .arrival import derive_scenario_trace_seed, materialize_arrival_trace
from .baselines import (
    QUEUE_RATE_ENERGY,
    available_baseline_policies,
    baseline_policy_metadata,
    create_baseline_policy,
)
from .config import MaritimeEnvConfig
from .env import MaritimeEnv
from .trajectory_v1 import MaritimeEpisodeBuilderV1, MaritimeEpisodeV1


ROUND3_RUNNER_SCHEMA_VERSION = "round3_runner_v1"
ROUND3_POLICY_NAMES = available_baseline_policies("v1")
ROUND3_K_VALUES = (1, 5, 10, 20, 30)
ROUND3_SMOKE_SEEDS = (101, 202, 303)
ROUND3_PILOT_SEEDS = tuple(range(10))
ROUND3_BUOY_COUNTS = (3, 5)
ROUND3_SMOKE_HORIZON = 8
ROUND3_PILOT_HORIZON = 128
ROUND3_SPLIT_RATIOS = (0.60, 0.20, 0.20)


@dataclass(frozen=True)
class Round3Collection:
    """In-memory episodes and immutable exogenous traces by scenario case."""

    episodes: Tuple[MaritimeEpisodeV1, ...]
    arrival_traces: Mapping[str, np.ndarray]
    case_metadata: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class Round3RunArtifacts:
    """Persisted Round-3 output locations and core in-memory results."""

    run_id: str
    output_dir: Path
    figures_dir: Path
    episodes: Tuple[MaritimeEpisodeV1, ...]
    split_assignments: Mapping[str, str]
    artifact_paths: Mapping[str, Path]
    figure_paths: Mapping[str, Path]


def collect_round3_episodes(
    *,
    base_config: Optional[MaritimeEnvConfig] = None,
    policy_names: Sequence[str] = ROUND3_POLICY_NAMES,
    scenario_id: str = "maritime_queue_collection_v1",
    seeds: Sequence[int] = ROUND3_SMOKE_SEEDS,
    buoy_counts: Sequence[int] = ROUND3_BUOY_COUNTS,
    horizon: int = ROUND3_SMOKE_HORIZON,
) -> Round3Collection:
    """Collect one fair episode for every policy/case combination.

    A complete arrival trace and initial-queue profile are generated once per
    ``(scenario_id, seed, num_buoys, horizon)`` case. Every policy receives a
    separate environment instance with identical exogenous inputs.
    """
    config = base_config or MaritimeEnvConfig()
    if not isinstance(config, MaritimeEnvConfig):
        raise TypeError("base_config must be MaritimeEnvConfig or None")
    names = _validate_policy_names(policy_names)
    validated_seeds = _validate_nonnegative_integers("seeds", seeds)
    counts = _validate_positive_integers("buoy_counts", buoy_counts)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not isinstance(scenario_id, str) or not scenario_id.strip():
        raise ValueError("scenario_id must be non-empty")

    episodes: List[MaritimeEpisodeV1] = []
    traces: Dict[str, np.ndarray] = {}
    cases: Dict[str, Mapping[str, Any]] = {}
    for num_buoys in counts:
        for scenario_seed in validated_seeds:
            case_id = _case_id(
                scenario_id, scenario_seed, num_buoys, horizon
            )
            case_config, trace, expected_trace_id = _build_case_config(
                config,
                scenario_id=scenario_id,
                scenario_seed=scenario_seed,
                num_buoys=num_buoys,
                horizon=horizon,
            )
            traces[case_id] = trace.copy()
            case_initial_records: List[Mapping[str, Any]] = []
            for policy_name in names:
                policy_seed = _stable_seed(
                    "policy",
                    scenario_id,
                    scenario_seed,
                    num_buoys,
                    policy_name,
                )
                episode, initial_record = _rollout_v1_episode(
                    config=case_config,
                    policy_name=policy_name,
                    policy_seed=policy_seed,
                    scenario_id=scenario_id,
                    scenario_seed=scenario_seed,
                    case_id=case_id,
                    expected_trace_id=expected_trace_id,
                )
                episodes.append(episode)
                case_initial_records.append(initial_record)
            _assert_case_fairness(
                case_id, case_initial_records, expected_trace_id
            )
            cases[case_id] = {
                "scenario_id": scenario_id,
                "scenario_seed": scenario_seed,
                "num_buoys": num_buoys,
                "horizon": horizon,
                "arrival_trace_id": expected_trace_id,
                "initial_queue_bits": list(
                    case_config.initial_queue_bits_by_buoy or ()
                ),
                "policy_names": list(names),
            }
    _assert_comparison_schedule(episodes, names)
    return Round3Collection(
        episodes=tuple(episodes),
        arrival_traces={key: value.copy() for key, value in traces.items()},
        case_metadata=dict(cases),
    )


def _build_case_config(
    base_config: MaritimeEnvConfig,
    *,
    scenario_id: str,
    scenario_seed: int,
    num_buoys: int,
    horizon: int,
) -> Tuple[MaritimeEnvConfig, np.ndarray, str]:
    queue_seed = _stable_seed(
        "initial_queue", scenario_id, scenario_seed, num_buoys, horizon
    )
    queue_rng = np.random.default_rng(queue_seed)
    queue_fractions = queue_rng.uniform(0.45, 0.90, size=num_buoys)
    # A deterministic small trend prevents accidental all-equal queues while
    # retaining genuine case-to-case randomness.
    queue_fractions += np.linspace(0.0, 0.04, num=num_buoys)
    queue_fractions = np.clip(queue_fractions, 0.0, 0.95)
    initial_queues = tuple(
        float(value * base_config.queue_capacity_bits)
        for value in queue_fractions
    )
    generator_config = replace(
        base_config,
        task_semantics_version="v1",
        num_buoys=num_buoys,
        episode_horizon=horizon,
        randomize_buoys=True,
        initial_queue_bits_by_buoy=initial_queues,
        arrival_mode="poisson",
        arrival_trace_bits=None,
        arrival_trace_id=None,
        # Keep a demanding completion target while retaining the documented
        # v1 residual-queue tolerance.  The previous 0.02 override classified
        # every otherwise useful pilot episode as sub-goal and removed all
        # high-quality coverage from the offline dataset.
        target_task_completion_ratio=max(
            0.985, base_config.target_task_completion_ratio
        ),
        maximum_allowed_residual_queue_ratio=(
            base_config.maximum_allowed_residual_queue_ratio
        ),
    )
    trace_seed = derive_scenario_trace_seed(
        scenario_id, scenario_seed, num_buoys, horizon
    )
    trace, _ = materialize_arrival_trace(
        generator_config, seed=trace_seed
    )
    trace_config = replace(
        generator_config,
        arrival_mode="trace",
        arrival_trace_bits=tuple(
            tuple(float(value) for value in row) for row in trace
        ),
        arrival_trace_id=None,
    )
    _, identifier = materialize_arrival_trace(trace_config)
    trace_config = replace(trace_config, arrival_trace_id=identifier)
    return trace_config, trace.copy(), identifier


def _rollout_v1_episode(
    *,
    config: MaritimeEnvConfig,
    policy_name: str,
    policy_seed: int,
    scenario_id: str,
    scenario_seed: int,
    case_id: str,
    expected_trace_id: str,
) -> Tuple[MaritimeEpisodeV1, Mapping[str, Any]]:
    environment = MaritimeEnv(config)
    observation, info = environment.reset(seed=scenario_seed)
    if info["arrival_trace_id"] != expected_trace_id:
        raise RuntimeError("environment did not consume the shared trace")
    policy = create_baseline_policy(policy_name, seed=policy_seed)
    policy.reset(seed=policy_seed)
    episode_id = (
        f"{case_id}__policy-{policy_name}"
    )
    builder = MaritimeEpisodeBuilderV1(
        episode_id=episode_id,
        policy_id=policy_name,
        seed=scenario_seed,
        scenario_id=scenario_id,
        num_buoys=environment.num_buoys,
        initial_info=info,
        observation_schema=environment.observation_schema,
        arrival_trace_id=expected_trace_id,
        desired_goal=info["desired_goal"],
        area_width=config.area_width,
        area_height=config.area_height,
        initial_energy=config.initial_energy,
        queue_capacity_bits=config.queue_capacity_bits,
    )
    initial_record = {
        "policy_id": policy_name,
        "arrival_trace_id": info["arrival_trace_id"],
        "observation": observation.copy(),
        "uav_position_m": info["uav_position_m"].copy(),
        "buoy_positions_m": info["buoy_positions_m"].copy(),
        "raw_queue_bits": info["raw_queue_bits"].copy(),
    }

    while True:
        policy_info = _current_policy_info(info, environment)
        action = np.asarray(
            policy.act(observation.copy(), policy_info), dtype=np.float64
        )
        if action.shape != (environment.action_dim,):
            raise RuntimeError("baseline returned an invalid action shape")
        if not np.all(np.isfinite(action)):
            raise RuntimeError("baseline returned NaN or Inf")
        if float(np.linalg.norm(action)) > 1.0 + 1.0e-12:
            raise RuntimeError("baseline returned action outside unit disk")
        selected = policy.metadata["last_decision"].get(
            "selected_buoy_index"
        )
        selected_index = -1 if selected is None else int(selected)
        next_observation, reward, terminated, truncated, next_info = (
            environment.step(action)
        )
        next_info = dict(next_info)
        next_info["policy_selected_buoy_index"] = selected_index
        builder.append(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            terminated=terminated,
            truncated=truncated,
            info_before=info,
            info_after=next_info,
            policy_selected_buoy_index=selected_index,
        )
        observation = next_observation
        info = next_info
        if terminated or truncated:
            break
    return builder.finalize(info["terminal_reason"]), initial_record


def _current_policy_info(
    info: Mapping[str, Any], environment: MaritimeEnv
) -> Dict[str, Any]:
    """Whitelist current values so a policy can never receive a future trace."""
    names = (
        "state_dim",
        "action_dim",
        "num_buoys",
        "uav_position_m",
        "area_bounds_m",
        "buoy_positions_m",
        "horizontal_distances_m",
        "rates_bps",
        "valid_node_mask",
        "raw_queue_bits",
        "one_step_movement_energy_proxy",
    )
    result: Dict[str, Any] = {}
    for name in names:
        if name not in info:
            raise RuntimeError(f"environment policy info is missing {name!r}")
        value = info[name]
        result[name] = value.copy() if isinstance(value, np.ndarray) else value
    result["max_step_distance_m"] = float(
        environment.config.max_speed * environment.config.delta_t
    )
    return result


def run_round3_experiment(
    *,
    mode: str,
    output_dir: Path,
    figures_dir: Path,
    run_id: str,
    base_config: Optional[MaritimeEnvConfig] = None,
) -> Round3RunArtifacts:
    """Run collection, strict v1 persistence, Waypoint construction and audit."""
    if mode not in ("smoke", "pilot"):
        raise ValueError("mode must be 'smoke' or 'pilot'")
    seeds = (
        ROUND3_SMOKE_SEEDS if mode == "smoke" else ROUND3_PILOT_SEEDS
    )
    horizon = (
        ROUND3_SMOKE_HORIZON
        if mode == "smoke"
        else ROUND3_PILOT_HORIZON
    )
    _prepare_new_directory(output_dir)
    _prepare_new_directory(figures_dir)
    collection = collect_round3_episodes(
        base_config=base_config,
        seeds=seeds,
        buoy_counts=ROUND3_BUOY_COUNTS,
        horizon=horizon,
    )
    episodes = list(collection.episodes)

    from .dataset_v1 import (
        deterministic_episode_split_v1,
        save_dataset_v1,
        validate_saved_dataset_v1,
    )
    from .waypoint_samples import (
        build_waypoint_sample_dataset,
        save_waypoint_samples,
    )

    split_assignments = deterministic_episode_split_v1(
        episodes,
        ratios=ROUND3_SPLIT_RATIOS,
        seed=90210,
    )
    manifest = save_dataset_v1(
        output_dir,
        episodes,
        split_assignments=split_assignments,
        metadata={
            "run_id": run_id,
            "runner_schema_version": ROUND3_RUNNER_SCHEMA_VERSION,
            "mode": mode,
            "training_performed": False,
            "policy_metadata": baseline_policy_metadata("v1"),
            "case_metadata": collection.case_metadata,
        },
    )
    validation = validate_saved_dataset_v1(output_dir)
    if not validation["passed"]:
        raise RuntimeError("saved v1 dataset validation failed")

    waypoint_result = build_waypoint_sample_dataset(
        episodes,
        split_assignments,
        k_values=ROUND3_K_VALUES,
        goal_source="achieved_terminal_goal",
        max_num_buoys=max(ROUND3_BUOY_COUNTS),
    )
    with (output_dir / "dataset_manifest.json").open(
        "r", encoding="utf-8"
    ) as stream:
        source_dataset_manifest = json.load(stream)
    waypoint_manifest = save_waypoint_samples(
        output_dir,
        waypoint_result,
        source_dataset_manifest=source_dataset_manifest,
        metadata={
            "run_id": run_id,
            "training_performed": False,
        },
    )
    batches = dict(waypoint_result.batches)
    waypoint_paths = {
        "waypoint_sample_manifest": (
            output_dir / "waypoint_sample_manifest.json"
        ),
        "waypoint_sample_schema": (
            output_dir / "waypoint_sample_schema.json"
        ),
        "waypoint_k_statistics": output_dir / "waypoint_k_statistics.csv",
        "waypoint_samples_train": output_dir / "waypoint_samples_train.npz",
        "waypoint_samples_validation": (
            output_dir / "waypoint_samples_validation.npz"
        ),
        "waypoint_samples_test": output_dir / "waypoint_samples_test.npz",
    }
    if int(waypoint_manifest["sample_count"]) < 1:
        raise RuntimeError("Waypoint construction produced no samples")

    # Analysis is imported lazily to keep environment imports lightweight.
    from .round3_analysis import run_round3_analysis

    analysis = run_round3_analysis(
        episodes=episodes,
        split_assignments=split_assignments,
        waypoint_batches=batches,
        output_dir=output_dir,
        figures_dir=figures_dir,
    )
    trace_path = _save_arrival_traces(
        output_dir / "arrival_traces.npz", collection.arrival_traces
    )
    run_config = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "schema_version": ROUND3_RUNNER_SCHEMA_VERSION,
        "dataset_schema_version": "v1",
        "goal_spec_version": "v1",
        "waypoint_schema_version": "waypoint_schema_v1",
        "run_id": run_id,
        "mode": mode,
        "scenario_id": "maritime_queue_collection_v1",
        "policy_names": list(ROUND3_POLICY_NAMES),
        "seeds": list(seeds),
        "buoy_counts": list(ROUND3_BUOY_COUNTS),
        "episode_horizon": horizon,
        "episodes": len(episodes),
        "transitions": int(sum(value.episode_length for value in episodes)),
        "k_values": list(ROUND3_K_VALUES),
        "split_ratios": list(ROUND3_SPLIT_RATIOS),
        "base_environment_config": asdict(
            base_config or MaritimeEnvConfig()
        ),
        "effective_case_configuration": {
            "task_semantics_version": "v1",
            "episode_horizon": horizon,
            "randomize_buoys": True,
            "arrival_mode": "shared materialized Poisson trace",
            "arrival_trace_file": "arrival_traces.npz",
            "queue_capacity_bits": float(
                (base_config or MaritimeEnvConfig()).queue_capacity_bits
            ),
            "poisson_arrival_packets_per_second": float(
                (
                    base_config or MaritimeEnvConfig()
                ).poisson_arrival_packets_per_second
            ),
            "arrival_packet_size_bits": float(
                (base_config or MaritimeEnvConfig()).arrival_packet_size_bits
            ),
            "service_order": "service_before_arrival",
            "service_allocation": (
                "fixed parallel orthogonal 1/K bandwidth shares already "
                "included in rates_bps"
            ),
            "target_task_completion_ratio": max(
                0.985,
                (
                    base_config or MaritimeEnvConfig()
                ).target_task_completion_ratio,
            ),
            "maximum_allowed_residual_queue_ratio": float(
                (
                    base_config or MaritimeEnvConfig()
                ).maximum_allowed_residual_queue_ratio
            ),
        },
        "case_metadata": collection.case_metadata,
        "training_performed": False,
        "learning_models_constructed": False,
        "dataset_validation": validation,
        "dataset_manifest_episode_count": manifest["episode_count"],
    }
    run_config_path = _write_json(output_dir / "run_config.json", run_config)
    repository_root = Path(__file__).resolve().parents[3]
    semantics_source = (
        repository_root
        / "report"
        / "round_03_environment_semantics_audit_20260730_220254.md"
    )
    semantics_destination = output_dir / "environment_semantics_audit.md"
    if semantics_destination.exists():
        raise FileExistsError(semantics_destination)
    shutil.copyfile(semantics_source, semantics_destination)
    schema_doc_source = repository_root / "docs" / "waypoint_sample_schema_v1.md"
    schema_doc_destination = output_dir / "waypoint_sample_schema_v1.md"
    shutil.copyfile(schema_doc_source, schema_doc_destination)

    artifact_paths: Dict[str, Path] = {
        "dataset_manifest": output_dir / "dataset_manifest.json",
        "dataset_schema": output_dir / "dataset_schema.json",
        "run_config": run_config_path,
        "arrival_traces": trace_path,
        "environment_semantics_audit": semantics_destination,
        "waypoint_sample_schema_document": schema_doc_destination,
    }
    artifact_paths.update(
        {str(key): Path(value) for key, value in waypoint_paths.items()}
    )
    analysis_artifacts = analysis.get("artifact_paths", {})
    artifact_paths.update(
        {str(key): Path(value) for key, value in analysis_artifacts.items()}
    )
    figure_paths = {
        str(key): Path(value)
        for key, value in analysis.get("figure_paths", {}).items()
    }
    return Round3RunArtifacts(
        run_id=run_id,
        output_dir=output_dir,
        figures_dir=figures_dir,
        episodes=tuple(episodes),
        split_assignments=dict(split_assignments),
        artifact_paths=artifact_paths,
        figure_paths=figure_paths,
    )


def _assert_case_fairness(
    case_id: str,
    records: Sequence[Mapping[str, Any]],
    expected_trace_id: str,
) -> None:
    if not records:
        raise RuntimeError("fairness check received no policies")
    reference = records[0]
    for record in records:
        if record["arrival_trace_id"] != expected_trace_id:
            raise RuntimeError(f"{case_id}: arrival trace mismatch")
        for name in (
            "observation",
            "uav_position_m",
            "buoy_positions_m",
            "raw_queue_bits",
        ):
            if not np.array_equal(record[name], reference[name]):
                raise RuntimeError(
                    f"{case_id}: policies received different initial {name}"
                )


def _assert_comparison_schedule(
    episodes: Sequence[MaritimeEpisodeV1], names: Sequence[str]
) -> None:
    expected: Optional[List[Tuple[int, int, str]]] = None
    for name in names:
        schedule = sorted(
            (
                episode.num_buoys,
                episode.seed,
                episode.arrival_trace_id,
            )
            for episode in episodes
            if episode.policy_id == name
        )
        if expected is None:
            expected = schedule
        elif schedule != expected:
            raise RuntimeError("Round-3 policy schedules are not identical")


def _case_id(
    scenario_id: str, seed: int, num_buoys: int, horizon: int
) -> str:
    safe = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in scenario_id
    )
    return f"{safe}__seed-{seed:05d}__buoys-{num_buoys}__h-{horizon}"


def _stable_seed(namespace: str, *values: Any) -> int:
    payload = "\x1f".join((namespace, *(str(value) for value in values)))
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little"
    )


def _validate_policy_names(values: Sequence[str]) -> Tuple[str, ...]:
    names = tuple(str(value).strip().lower() for value in values)
    if not names or len(names) != len(set(names)):
        raise ValueError("policy_names must be nonempty and unique")
    supported = set(ROUND3_POLICY_NAMES)
    if set(names) - supported:
        raise ValueError(f"unsupported v1 policies: {sorted(set(names)-supported)}")
    if "urgency_rate_energy" in names:
        raise ValueError(
            "urgency_rate_energy is a v0 compatibility name, not a v1 policy"
        )
    return names


def _validate_nonnegative_integers(
    name: str, values: Sequence[int]
) -> Tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or any(value < 0 for value in result):
        raise ValueError(f"{name} must contain nonnegative integers")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _validate_positive_integers(
    name: str, values: Sequence[int]
) -> Tuple[int, ...]:
    result = _validate_nonnegative_integers(name, values)
    if any(value < 1 for value in result):
        raise ValueError(f"{name} must contain positive integers")
    return result


def _prepare_new_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _save_arrival_traces(
    path: Path, traces: Mapping[str, np.ndarray]
) -> Path:
    if path.exists():
        raise FileExistsError(path)
    arrays = {
        f"trace_{index:04d}": np.asarray(traces[key], dtype=np.float64)
        for index, key in enumerate(sorted(traces))
    }
    np.savez_compressed(path, **arrays)
    index = {
        f"trace_{position:04d}": key
        for position, key in enumerate(sorted(traces))
    }
    _write_json(path.with_suffix(".json"), {"trace_index": index})
    return path


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            _json_value(dict(value)),
            stream,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
    return path


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot encode {type(value).__name__} in JSON")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Round-3 shared-trace non-learning pipeline (no training)."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--figures-dir", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    mode = "smoke" if args.smoke else "pilot"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = args.run_id or f"round03_{mode}_{timestamp}"
    repository_root = Path(__file__).resolve().parents[3]
    output_dir = Path(
        args.output_dir
        or repository_root / "round_03" / "outputs" / run_id
    )
    figures_dir = Path(
        args.figures_dir
        or repository_root / "round_03" / "figures" / run_id
    )
    result = run_round3_experiment(
        mode=mode,
        output_dir=output_dir,
        figures_dir=figures_dir,
        run_id=run_id,
    )
    print(f"run_id: {result.run_id}")
    print(f"mode: {mode}")
    print(f"episodes: {len(result.episodes)}")
    print(
        "transitions: "
        f"{sum(episode.episode_length for episode in result.episodes)}"
    )
    print(f"output_dir: {result.output_dir}")
    print(f"figures_dir: {result.figures_dir}")
    print(f"figures: {len(result.figure_paths)}")
    print("training_performed: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ROUND3_BUOY_COUNTS",
    "ROUND3_K_VALUES",
    "ROUND3_PILOT_HORIZON",
    "ROUND3_PILOT_SEEDS",
    "ROUND3_POLICY_NAMES",
    "ROUND3_SMOKE_HORIZON",
    "ROUND3_SMOKE_SEEDS",
    "Round3Collection",
    "Round3RunArtifacts",
    "collect_round3_episodes",
    "run_round3_experiment",
]
