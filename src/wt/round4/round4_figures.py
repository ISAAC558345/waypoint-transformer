"""Round-4 spatial Waypoint scientific figure generation.

The module is a CSV-only post-processing layer.  It never loads checkpoints,
constructs models, or starts training.  Each figure category is written to an
independent directory in both PNG and SVG formats using a white, two-
dimensional scientific style.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


PathLike = Union[str, Path]

ROUND4_FIGURE_CATEGORIES: Mapping[str, Tuple[str, str]] = {
    "loss": ("01_loss", "training_validation_loss"),
    "validation_rmse": ("02_validation_rmse", "validation_endpoint_rmse"),
    "normalized_metrics": ("03_normalized_metrics", "normalized_error_and_direction"),
    "goal_vs_state": ("04_goal_vs_state", "goal_conditioned_vs_state_only"),
    "network_vs_baseline": ("05_network_vs_baseline", "network_vs_nonlearning_baselines"),
    "dx_scatter": ("06_dx_scatter", "predicted_vs_true_delta_x"),
    "dy_scatter": ("07_dy_scatter", "predicted_vs_true_delta_y"),
    "magnitude": ("08_magnitude", "predicted_vs_true_displacement_magnitude"),
    "angular_histogram": ("09_angular_histogram", "direction_angular_error"),
    "projection_activation": ("10_projection_activation", "projection_activation_and_reachability"),
    "policy_rmse": ("11_policy_rmse", "test_rmse_by_policy"),
    "buoy_rmse": ("12_buoy_rmse", "test_rmse_by_buoy_count"),
    "representative_endpoints": ("13_representative_endpoints", "representative_endpoints_and_arrows"),
}


_VALIDATION_COLUMNS = {
    "horizon",
    "seed",
    "model_kind",
    "projection",
    "future_endpoint_rmse_m",
    "normalized_endpoint_error",
    "direction_cosine_similarity",
    "reachable_prediction_rate",
}
_BASELINE_COLUMNS = {
    "split",
    "horizon",
    "baseline",
    "future_endpoint_rmse_m",
}
_TEST_COLUMNS = {
    "group_dimension",
    "group_value",
    "horizon",
    "seed",
    "model_kind",
    "projection",
    "future_endpoint_rmse_m",
}
_PROJECTION_COLUMNS = {
    "split",
    "horizon",
    "model_kind",
    "group_dimension",
    "projection_activation_rate",
    "endpoint_error_change_m_mean",
}

_TABLE_CANDIDATES = {
    "validation": (
        "waypoint_validation_metrics.csv",
        "validation_metrics.csv",
        "validation_results.csv",
        "validation_summary.csv",
    ),
    "baseline": (
        "waypoint_baseline_metrics.csv",
        "baseline_metrics.csv",
        "baseline_results.csv",
        "baseline_summary.csv",
    ),
    "test": (
        "waypoint_test_metrics.csv",
        "test_metrics.csv",
        "test_group_metrics.csv",
        "test_results.csv",
        "test_group_summary.csv",
    ),
    "prediction_examples": (
        "waypoint_prediction_examples.csv",
        "prediction_examples.csv",
        "spatial_prediction_examples.csv",
    ),
    "training_summary": (
        "waypoint_training_summary.csv",
        "training_summary.csv",
        "training_run_summary.csv",
        "training_runs.csv",
    ),
    "projection": (
        "waypoint_projection_metrics.csv",
        "projection_metrics.csv",
        "projection_summary.csv",
    ),
}


def generate_round4_figures(
    output_dir: PathLike,
    figure_dir: PathLike,
    selected_horizon: Optional[int] = None,
    *,
    overwrite: bool = False,
) -> Dict[str, Path]:
    """Generate all 13 Round-4 figure categories from recorded CSV files.

    The returned mapping is flat and contains ``<category>_png`` and
    ``<category>_svg`` keys.  Core CSVs, required columns, non-empty validation
    data, prediction examples, and referenced training histories are strict
    contracts.  Optional stratified groups may be empty; their figures then
    contain an explicit availability notice instead of failing.
    """

    source = Path(output_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"Round-4 output directory does not exist: {source}")

    validation = _load_required_table(
        source,
        "validation",
        lambda fields: _VALIDATION_COLUMNS <= fields,
        _VALIDATION_COLUMNS,
    )
    baselines = _load_required_table(
        source,
        "baseline",
        lambda fields: _BASELINE_COLUMNS <= fields,
        _BASELINE_COLUMNS,
    )
    test_rows = _load_required_table(
        source,
        "test",
        lambda fields: _TEST_COLUMNS <= fields,
        _TEST_COLUMNS,
    )
    prediction_rows = _load_required_table(
        source,
        "prediction_examples",
        _is_prediction_table,
        {"policy_id", "num_buoys"},
    )
    training_summary = _load_required_table(
        source,
        "training_summary",
        lambda fields: "history_file" in fields,
        {"history_file"},
    )
    projection_rows = _load_required_table(
        source,
        "projection",
        lambda fields: _PROJECTION_COLUMNS <= fields,
        _PROJECTION_COLUMNS,
    )

    validation_overall = [
        row
        for row in validation
        if row.get("group_dimension", "overall").strip().lower() == "overall"
    ]
    if not validation_overall:
        raise ValueError("validation metrics contain no overall rows")
    horizons = sorted({_positive_int(row["horizon"], "validation horizon") for row in validation_overall})
    if not horizons:
        raise ValueError("validation metrics contain no horizons")
    chosen = None if selected_horizon is None else _positive_int(selected_horizon, "selected_horizon")
    if chosen is not None and chosen not in horizons:
        raise ValueError(
            f"selected_horizon={chosen} is absent from validation metrics {horizons}"
        )
    representative_horizon = chosen or _best_validation_horizon(validation_overall)

    example_columns = _resolve_prediction_columns(prediction_rows)
    histories = _load_training_histories(source, training_summary, chosen)
    paths = _prepare_output_paths(Path(figure_dir), overwrite=overwrite)

    plt = _load_pyplot()
    with plt.rc_context(_scientific_style()):
        _plot_loss_curves(plt, histories, paths_for(paths, "loss"))
        _plot_validation_rmse(plt, validation_overall, paths_for(paths, "validation_rmse"))
        _plot_normalized_metrics(plt, validation_overall, paths_for(paths, "normalized_metrics"))
        _plot_goal_vs_state(plt, validation_overall, paths_for(paths, "goal_vs_state"))
        _plot_network_vs_baseline(
            plt, validation_overall, baselines, paths_for(paths, "network_vs_baseline")
        )
        _plot_component_scatter(
            plt,
            prediction_rows,
            example_columns,
            component="x",
            horizon=chosen,
            destinations=paths_for(paths, "dx_scatter"),
        )
        _plot_component_scatter(
            plt,
            prediction_rows,
            example_columns,
            component="y",
            horizon=chosen,
            destinations=paths_for(paths, "dy_scatter"),
        )
        _plot_magnitude(
            plt,
            prediction_rows,
            example_columns,
            chosen,
            paths_for(paths, "magnitude"),
        )
        _plot_angular_histogram(
            plt,
            prediction_rows,
            example_columns,
            chosen,
            paths_for(paths, "angular_histogram"),
        )
        _plot_projection_activation(
            plt,
            projection_rows,
            paths_for(paths, "projection_activation"),
        )
        _plot_test_group_rmse(
            plt,
            test_rows,
            dimension_names=("policy_id", "policy"),
            horizon=representative_horizon,
            title="Selected-horizon test endpoint RMSE by collection policy",
            xlabel="Collection policy",
            destinations=paths_for(paths, "policy_rmse"),
        )
        _plot_test_group_rmse(
            plt,
            test_rows,
            dimension_names=("num_buoys", "buoy_count", "number_of_buoys"),
            horizon=representative_horizon,
            title="Selected-horizon test endpoint RMSE by buoy count",
            xlabel="Number of buoys",
            destinations=paths_for(paths, "buoy_rmse"),
        )
        _plot_representative_endpoints(
            plt,
            prediction_rows,
            example_columns,
            representative_horizon,
            paths_for(paths, "representative_endpoints"),
        )

    for path in paths.values():
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"failed to create Round-4 figure: {path}")
    return paths


def generate_per_horizon_loss_figures(
    output_dir: PathLike,
    figure_dir: PathLike,
    *,
    overwrite: bool = False,
) -> Dict[str, Path]:
    """Write independent train/validation loss figures for every horizon."""

    source = Path(output_dir)
    summaries = _load_required_table(
        source,
        "training_summary",
        lambda fields: {"history_file", "horizon"} <= fields,
        {"history_file", "horizon"},
    )
    histories = _load_training_histories(source, summaries, None)
    by_horizon: Dict[int, List[Mapping[str, Any]]] = {}
    for run in histories:
        horizon = _positive_int(
            run["summary"]["horizon"], "training-summary horizon"
        )
        by_horizon.setdefault(horizon, []).append(run)
    result: Dict[str, Path] = {}
    plt = _load_pyplot()
    with plt.rc_context(_scientific_style()):
        for horizon, runs in sorted(by_horizon.items()):
            directory = Path(figure_dir) / f"horizon_{horizon:03d}"
            directory.mkdir(parents=True, exist_ok=True)
            png = directory / "train_validation_loss.png"
            svg = directory / "train_validation_loss.svg"
            if not overwrite and (png.exists() or svg.exists()):
                raise FileExistsError(
                    f"refusing to overwrite horizon loss figure: {directory}"
                )
            _plot_loss_curves(plt, runs, (png, svg))
            result[f"horizon_{horizon:03d}_loss_png"] = png
            result[f"horizon_{horizon:03d}_loss_svg"] = svg
    return result


def paths_for(paths: Mapping[str, Path], category: str) -> Tuple[Path, Path]:
    return paths[f"{category}_png"], paths[f"{category}_svg"]


def _prepare_output_paths(root: Path, *, overwrite: bool = False) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for category, (directory_name, stem) in ROUND4_FIGURE_CATEGORIES.items():
        directory = root / directory_name
        directory.mkdir(parents=True, exist_ok=True)
        result[f"{category}_png"] = directory / f"{stem}.png"
        result[f"{category}_svg"] = directory / f"{stem}.svg"
    existing = [path for path in result.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing Round-4 figures: "
            + ", ".join(str(path) for path in existing)
        )
    return result


def _load_required_table(
    root: Path,
    kind: str,
    predicate: Callable[[set], bool],
    required_columns: Iterable[str],
) -> List[Dict[str, str]]:
    files = _locate_table_files(root, kind, predicate)
    if not files:
        raise FileNotFoundError(f"missing core Round-4 {kind} CSV under {root}")
    rows: List[Dict[str, str]] = []
    for path in files:
        table = _read_csv(path)
        missing = sorted(set(required_columns) - set(table[0]))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        for row in table:
            row["__source_file"] = str(path)
            rows.append(row)
    if not rows:
        raise ValueError(f"core Round-4 {kind} CSV is empty")
    return rows


def _locate_table_files(
    root: Path,
    kind: str,
    predicate: Callable[[set], bool],
) -> List[Path]:
    names = set(_TABLE_CANDIDATES[kind])
    direct = [root / name for name in _TABLE_CANDIDATES[kind] if (root / name).is_file()]
    if direct:
        return sorted(set(direct))

    named = [path for path in root.rglob("*.csv") if path.name in names]
    if named:
        depth = min(len(path.relative_to(root).parts) for path in named)
        return sorted(path for path in set(named) if len(path.relative_to(root).parts) == depth)

    matching = []
    for path in root.rglob("*.csv"):
        fields = _csv_fields(path)
        if predicate(fields):
            matching.append(path)
    if not matching:
        return []
    depth = min(len(path.relative_to(root).parts) for path in matching)
    return sorted(path for path in matching if len(path.relative_to(root).parts) == depth)


def _csv_fields(path: Path) -> set:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration:
            return set()
    return {str(value).strip() for value in header if str(value).strip()}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        fields = [str(name).strip() for name in reader.fieldnames]
        if any(not name for name in fields) or len(set(fields)) != len(fields):
            raise ValueError(f"CSV has invalid or duplicate columns: {path}")
        rows = []
        for raw in reader:
            row = {str(key).strip(): "" if value is None else str(value).strip() for key, value in raw.items()}
            if any(value != "" for value in row.values()):
                rows.append(row)
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    return rows


def _is_prediction_table(fields: set) -> bool:
    if not {"policy_id", "num_buoys"} <= fields:
        return False
    aliases = _PREDICTION_ALIASES
    return all(any(name in fields for name in aliases[key]) for key in ("true_x", "true_y", "pred_x", "pred_y", "projected_x", "projected_y", "current_x", "current_y"))


_PREDICTION_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "current_x": (
        "q_x_normalized", "current_x_normalized", "current_position_x_normalized",
        "q_current_x_m", "q_x_m", "current_x_m", "current_position_x_m", "current_uav_x_m",
        "q_current_x", "current_x",
    ),
    "current_y": (
        "q_y_normalized", "current_y_normalized", "current_position_y_normalized",
        "q_current_y_m", "q_y_m", "current_y_m", "current_position_y_m", "current_uav_y_m",
        "q_current_y", "current_y",
    ),
    "true_x": (
        "true_delta_x_normalized", "target_delta_x_normalized", "true_dx_normalized",
        "true_delta_x_m", "target_delta_x_m", "true_delta_x", "target_delta_x", "true_dx",
    ),
    "true_y": (
        "true_delta_y_normalized", "target_delta_y_normalized", "true_dy_normalized",
        "true_delta_y_m", "target_delta_y_m", "true_delta_y", "target_delta_y", "true_dy",
    ),
    "pred_x": (
        "predicted_delta_x_normalized", "pred_delta_x_normalized", "prediction_delta_x_normalized",
        "predicted_delta_x_m", "pred_delta_x_m", "predicted_delta_x", "pred_delta_x", "prediction_x",
    ),
    "pred_y": (
        "predicted_delta_y_normalized", "pred_delta_y_normalized", "prediction_delta_y_normalized",
        "predicted_delta_y_m", "pred_delta_y_m", "predicted_delta_y", "pred_delta_y", "prediction_y",
    ),
    "projected_x": (
        "projected_delta_x_normalized", "projection_delta_x_normalized",
        "projected_delta_x_m", "projection_delta_x_m", "projected_delta_x", "projection_delta_x",
    ),
    "projected_y": (
        "projected_delta_y_normalized", "projection_delta_y_normalized",
        "projected_delta_y_m", "projection_delta_y_m", "projected_delta_y", "projection_delta_y",
    ),
}


def _resolve_prediction_columns(rows: Sequence[Mapping[str, str]]) -> Dict[str, str]:
    fields = set(rows[0])
    result = {}
    for key, aliases in _PREDICTION_ALIASES.items():
        selected = next((name for name in aliases if name in fields), None)
        if selected is None:
            raise ValueError(f"prediction examples are missing the {key} coordinate")
        result[key] = selected
    return result


def _load_training_histories(
    output_root: Path,
    summaries: Sequence[Mapping[str, str]],
    selected_horizon: Optional[int],
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for summary in summaries:
        if selected_horizon is not None and summary.get("horizon", ""):
            if _positive_int(summary["horizon"], "training-summary horizon") != selected_horizon:
                continue
        history_path = _resolve_history_path(output_root, summary)
        history = _read_csv(history_path)
        required = {"epoch", "train_normalized_mse", "validation_normalized_mse"}
        missing = sorted(required - set(history[0]))
        if missing:
            raise ValueError(f"{history_path} is missing history columns: {missing}")
        result.append(
            {
                "summary": dict(summary),
                "history_path": history_path,
                "history": history,
            }
        )
    if not result:
        qualifier = "" if selected_horizon is None else f" for horizon {selected_horizon}"
        raise ValueError(f"no referenced training histories are available{qualifier}")
    return result


def _resolve_history_path(output_root: Path, summary: Mapping[str, str]) -> Path:
    value = str(summary.get("history_file", "")).strip()
    if not value:
        raise ValueError("training_summary history_file must be non-empty")
    raw = Path(value)
    source_parent = Path(summary["__source_file"]).parent
    repository_root = Path(__file__).resolve().parents[3]
    candidates = (
        [raw]
        if raw.is_absolute()
        else [repository_root / raw, output_root / raw, source_parent / raw]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = list(output_root.rglob(raw.name))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"training history referenced by summary was not found: {value}")


def _best_validation_horizon(rows: Sequence[Mapping[str, str]]) -> int:
    grouped: Dict[int, List[float]] = {}
    for row in rows:
        horizon = _positive_int(row["horizon"], "validation horizon")
        grouped.setdefault(horizon, []).append(_finite_float(row["future_endpoint_rmse_m"], "validation RMSE"))
    return min(grouped, key=lambda horizon: (float(np.mean(grouped[horizon])), horizon))


def _plot_loss_curves(plt: Any, runs: Sequence[Mapping[str, Any]], destinations: Tuple[Path, Path]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), constrained_layout=True)
    colors = _palette(plt, len(runs))
    for index, (run, color) in enumerate(zip(runs, colors)):
        summary = run["summary"]
        history = run["history"]
        epochs = _numeric(history, "epoch", "training epoch")
        train = _numeric(history, "train_normalized_mse", "training normalized MSE")
        validation = _numeric(history, "validation_normalized_mse", "validation normalized MSE")
        label = _run_label(summary, index)
        axes[0].plot(epochs, train, color=color, alpha=0.78, label=label)
        axes[1].plot(epochs, validation, color=color, alpha=0.78, label=label)
    axes[0].set_title("Training normalized displacement loss")
    axes[1].set_title("Validation normalized displacement loss")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("mean squared displacement norm")
        axis.set_yscale("log" if _all_positive_lines(axis) else "linear")
    _legend_if_useful(axes[1], max_entries=16)
    _save_both(plt, figure, destinations)


def _plot_validation_rmse(plt: Any, rows: Sequence[Mapping[str, str]], destinations: Tuple[Path, Path]) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 4.9), constrained_layout=True)
    _plot_grouped_horizon_metric(axis, rows, "future_endpoint_rmse_m")
    axis.set_xlabel("Waypoint horizon [step]")
    axis.set_ylabel("Future endpoint RMSE [m]")
    axis.set_title("Validation future-endpoint accuracy across horizons")
    _legend_if_useful(axis)
    _save_both(plt, figure, destinations)


def _plot_normalized_metrics(plt: Any, rows: Sequence[Mapping[str, str]], destinations: Tuple[Path, Path]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    _plot_grouped_horizon_metric(axes[0], rows, "normalized_endpoint_error")
    direction_rows = [
        row
        for row in rows
        if _is_finite_text(row.get("direction_cosine_similarity", ""))
    ]
    if direction_rows:
        _plot_grouped_horizon_metric(
            axes[1], direction_rows, "direction_cosine_similarity"
        )
    else:
        _availability_notice(
            axes[1], "No non-zero target/prediction pairs available"
        )
    axes[0].set_ylabel("Normalized endpoint error")
    axes[1].set_ylabel("Direction cosine similarity")
    axes[1].set_ylim(-1.05, 1.05)
    for axis in axes:
        axis.set_xlabel("Waypoint horizon [step]")
    axes[0].set_title("Scale-free endpoint error")
    axes[1].set_title("Direction agreement")
    _legend_if_useful(axes[1])
    _save_both(plt, figure, destinations)


def _plot_goal_vs_state(plt: Any, rows: Sequence[Mapping[str, str]], destinations: Tuple[Path, Path]) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 4.9), constrained_layout=True)
    raw_rows = [
        row
        for row in rows
        if _display_projection(row["projection"]) == "raw"
    ]
    comparison_rows = raw_rows or list(rows)
    kinds = sorted({row["model_kind"] for row in comparison_rows})
    colors = _palette(plt, len(kinds))
    for color, kind in zip(colors, kinds):
        selected = [row for row in comparison_rows if row["model_kind"] == kind]
        horizons, means, stds = _metric_by_horizon(selected, "future_endpoint_rmse_m")
        axis.errorbar(horizons, means, yerr=stds, marker="o", capsize=3, color=color, label=_display(kind))
    axis.set_xlabel("Waypoint horizon [step]")
    axis.set_ylabel("Future endpoint RMSE [m]")
    axis.set_title("Goal-conditioned versus state-only spatial waypoint models")
    _legend_if_useful(axis)
    _save_both(plt, figure, destinations)


def _plot_network_vs_baseline(
    plt: Any,
    validation: Sequence[Mapping[str, str]],
    baselines: Sequence[Mapping[str, str]],
    destinations: Tuple[Path, Path],
) -> None:
    figure, axis = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    series: List[Tuple[str, Sequence[Mapping[str, str]], str]] = []
    goal_validation = [
        row
        for row in validation
        if row["model_kind"].strip().lower() == "goal_conditioned"
    ]
    for key in sorted(
        {_model_projection_label(row) for row in goal_validation}
    ):
        series.append(
            (
                f"network: {key}",
                [
                    row
                    for row in goal_validation
                    if _model_projection_label(row) == key
                ],
                "future_endpoint_rmse_m",
            )
        )
    usable_baselines = [row for row in baselines if row["split"].strip().lower() in ("validation", "val")]
    if not usable_baselines:
        usable_baselines = list(baselines)
    nonlearning_names = {
        "zero_waypoint",
        "train_mean_waypoint",
        "constant_velocity",
    }
    usable_baselines = [
        row
        for row in usable_baselines
        if row["baseline"].strip().lower() in nonlearning_names
    ]
    for name in sorted({row["baseline"] for row in usable_baselines}):
        series.append((f"baseline: {_display(name)}", [row for row in usable_baselines if row["baseline"] == name], "future_endpoint_rmse_m"))
    colors = _palette(plt, len(series))
    for color, (label, selected, metric) in zip(colors, series):
        horizons, means, stds = _metric_by_horizon(selected, metric)
        axis.errorbar(horizons, means, yerr=stds, marker="o", capsize=3, color=color, label=label)
    axis.set_xlabel("Waypoint horizon [step]")
    axis.set_ylabel("Future endpoint RMSE [m]")
    axis.set_title("Spatial waypoint network versus non-learning displacement baselines")
    _legend_if_useful(axis, max_entries=18)
    _save_both(plt, figure, destinations)


def _plot_component_scatter(
    plt: Any,
    rows: Sequence[Mapping[str, str]],
    columns: Mapping[str, str],
    *,
    component: str,
    horizon: Optional[int],
    destinations: Tuple[Path, Path],
) -> None:
    selected = _filter_example_horizon(rows, horizon)
    scope = _example_scope(selected)
    true_values = _numeric(selected, columns[f"true_{component}"], f"true delta {component}")
    raw_values = _numeric(selected, columns[f"pred_{component}"], f"predicted delta {component}")
    projected_values = _numeric(selected, columns[f"projected_{component}"], f"projected delta {component}")
    figure, axis = plt.subplots(figsize=(6.2, 5.6), constrained_layout=True)
    axis.scatter(true_values, raw_values, s=18, alpha=0.32, color="#d95f02", label="Raw network")
    axis.scatter(true_values, projected_values, s=18, alpha=0.32, color="#1b9e77", label="Projected")
    lower, upper = _finite_limits(np.concatenate([true_values, raw_values, projected_values]))
    axis.plot([lower, upper], [lower, upper], color="#222222", linestyle="--", linewidth=1.1, label="Identity")
    unit = _coordinate_unit(columns[f"true_{component}"])
    axis.set_xlabel(f"True delta {component.upper()} [{unit}]")
    axis.set_ylabel(f"Predicted delta {component.upper()} [{unit}]")
    axis.set_title(
        f"Component-wise spatial waypoint agreement: delta {component.upper()}"
        f"\n{scope}"
    )
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    _legend_if_useful(axis)
    _save_both(plt, figure, destinations)


def _plot_magnitude(
    plt: Any,
    rows: Sequence[Mapping[str, str]],
    columns: Mapping[str, str],
    horizon: Optional[int],
    destinations: Tuple[Path, Path],
) -> None:
    selected = _filter_example_horizon(rows, horizon)
    scope = _example_scope(selected)
    true_xy, pred_xy, projected_xy = _example_deltas(selected, columns)
    true_mag = np.linalg.norm(true_xy, axis=1)
    pred_mag = np.linalg.norm(pred_xy, axis=1)
    projected_mag = np.linalg.norm(projected_xy, axis=1)
    figure, axis = plt.subplots(figsize=(6.3, 5.5), constrained_layout=True)
    axis.scatter(true_mag, pred_mag, s=18, alpha=0.34, color="#d95f02", label="Raw network")
    axis.scatter(true_mag, projected_mag, s=18, alpha=0.34, color="#1b9e77", label="Projected")
    lower, upper = _finite_limits(np.concatenate([true_mag, pred_mag, projected_mag]), include_zero=True)
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="#222222", linewidth=1.1, label="Identity")
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    unit = _coordinate_unit(columns["true_x"])
    axis.set_xlabel(f"True displacement magnitude [{unit}]")
    axis.set_ylabel(f"Predicted displacement magnitude [{unit}]")
    axis.set_title(f"Spatial waypoint displacement magnitude\n{scope}")
    _legend_if_useful(axis)
    _save_both(plt, figure, destinations)


def _plot_angular_histogram(
    plt: Any,
    rows: Sequence[Mapping[str, str]],
    columns: Mapping[str, str],
    horizon: Optional[int],
    destinations: Tuple[Path, Path],
) -> None:
    selected = _filter_example_horizon(rows, horizon)
    scope = _example_scope(selected)
    true_xy, pred_xy, projected_xy = _example_deltas(selected, columns)
    raw_angles = _angular_error_degrees(true_xy, pred_xy)
    projected_angles = _angular_error_degrees(true_xy, projected_xy)
    figure, axis = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    if raw_angles.size or projected_angles.size:
        combined = np.concatenate([raw_angles, projected_angles]) if raw_angles.size and projected_angles.size else (raw_angles if raw_angles.size else projected_angles)
        bins = np.linspace(0.0, max(10.0, min(180.0, float(np.max(combined)) + 5.0)), min(25, max(6, int(math.sqrt(combined.size)) + 2)))
        if raw_angles.size:
            axis.hist(raw_angles, bins=bins, alpha=0.48, color="#d95f02", label="Raw network")
        if projected_angles.size:
            axis.hist(projected_angles, bins=bins, alpha=0.48, color="#1b9e77", label="Projected")
        _legend_if_useful(axis)
    else:
        _availability_notice(axis, "No non-zero displacement pairs available for direction analysis")
    axis.set_xlabel("Angular error [degree]")
    axis.set_ylabel("Sample count")
    axis.set_title(f"Spatial waypoint direction-error distribution\n{scope}")
    _save_both(plt, figure, destinations)


def _plot_projection_activation(
    plt: Any,
    projection_rows: Sequence[Mapping[str, str]],
    destinations: Tuple[Path, Path],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    overall = [
        row
        for row in projection_rows
        if row["group_dimension"].strip().lower() == "overall"
        and row["split"].strip().lower() in ("validation", "val")
    ]
    if not overall:
        overall = [
            row
            for row in projection_rows
            if row["group_dimension"].strip().lower() == "overall"
        ]
    _plot_projection_horizon_metric(
        axes[0], overall, "projection_activation_rate"
    )
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xlabel("Waypoint horizon [step]")
    axes[0].set_ylabel("Projection activation rate")
    axes[0].set_title("Radial projection activation")
    _plot_projection_horizon_metric(
        axes[1], overall, "endpoint_error_change_m_mean"
    )
    axes[1].set_xlabel("Waypoint horizon [step]")
    axes[1].set_ylabel("Projected minus raw endpoint error [m]")
    axes[1].axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[1].set_title("Projection-induced endpoint-error change")
    _legend_if_useful(axes[0])
    _legend_if_useful(axes[1])
    _save_both(plt, figure, destinations)


def _plot_test_group_rmse(
    plt: Any,
    rows: Sequence[Mapping[str, str]],
    *,
    dimension_names: Sequence[str],
    horizon: int,
    title: str,
    xlabel: str,
    destinations: Tuple[Path, Path],
) -> None:
    accepted = {name.lower() for name in dimension_names}
    selected = [row for row in rows if row["group_dimension"].strip().lower() in accepted and _positive_int(row["horizon"], "test horizon") == horizon]
    figure, axis = plt.subplots(figsize=(8.4, 4.9), constrained_layout=True)
    if not selected:
        _availability_notice(axis, f"No {xlabel.lower()} stratified test rows are available for horizon {horizon}")
    else:
        groups = sorted({row["group_value"] for row in selected}, key=_natural_key)
        series = sorted({_model_projection_label(row) for row in selected})
        colors = _palette(plt, len(series))
        x = np.arange(len(groups), dtype=np.float64)
        offsets = np.linspace(-0.18, 0.18, len(series)) if len(series) > 1 else np.zeros(1)
        for offset, color, name in zip(offsets, colors, series):
            means, stds = [], []
            for group in groups:
                values = [_finite_float(row["future_endpoint_rmse_m"], "test RMSE") for row in selected if row["group_value"] == group and _model_projection_label(row) == name]
                means.append(float(np.mean(values)) if values else np.nan)
                stds.append(float(np.std(values)) if values else 0.0)
            axis.errorbar(x + offset, means, yerr=stds, marker="o", linestyle="none", capsize=3, color=color, label=name)
        axis.set_xticks(x)
        axis.set_xticklabels([_display(value) for value in groups], rotation=15, ha="right")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Future endpoint RMSE [m]")
        _legend_if_useful(axis)
    axis.set_title(f"{title} (H={horizon})")
    _save_both(plt, figure, destinations)


def _plot_representative_endpoints(
    plt: Any,
    rows: Sequence[Mapping[str, str]],
    columns: Mapping[str, str],
    horizon: int,
    destinations: Tuple[Path, Path],
) -> None:
    selected = _filter_example_horizon(rows, horizon)
    goal_conditioned = [
        row
        for row in selected
        if row.get("model_kind", "") == "goal_conditioned"
    ]
    if goal_conditioned:
        selected = goal_conditioned
    seeds = sorted(
        {row.get("seed", "") for row in selected}, key=_natural_key
    )
    if seeds:
        selected = [
            row for row in selected if row.get("seed", "") == seeds[0]
        ]
    scope = _example_scope(selected)
    if len(selected) > 12:
        indices = np.linspace(0, len(selected) - 1, 12, dtype=int)
        selected = [selected[index] for index in indices]
    current = np.column_stack((_numeric(selected, columns["current_x"], "current x"), _numeric(selected, columns["current_y"], "current y")))
    true_xy, pred_xy, projected_xy = _example_deltas(selected, columns)
    true_xy, pred_xy, projected_xy = _align_delta_units(selected, columns, true_xy, pred_xy, projected_xy)
    figure, axis = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
    for index in range(len(selected)):
        label_true = "True endpoint" if index == 0 else None
        label_raw = "Raw prediction" if index == 0 else None
        label_projected = "Projected prediction" if index == 0 else None
        axis.arrow(current[index, 0], current[index, 1], true_xy[index, 0], true_xy[index, 1], color="#1b9e77", alpha=0.72, width=0.0, head_width=_arrow_head_width(current, true_xy), length_includes_head=True, label=label_true)
        axis.arrow(current[index, 0], current[index, 1], pred_xy[index, 0], pred_xy[index, 1], color="#d95f02", alpha=0.62, width=0.0, head_width=_arrow_head_width(current, pred_xy), length_includes_head=True, label=label_raw)
        axis.arrow(current[index, 0], current[index, 1], projected_xy[index, 0], projected_xy[index, 1], color="#7570b3", alpha=0.62, width=0.0, head_width=_arrow_head_width(current, projected_xy), length_includes_head=True, label=label_projected)
    axis.scatter(current[:, 0], current[:, 1], marker="x", s=34, color="#222222", label="Current UAV position", zorder=4)
    all_points = np.vstack((current, current + true_xy, current + pred_xy, current + projected_xy))
    x_lower, x_upper = _finite_limits(all_points[:, 0])
    y_lower, y_upper = _finite_limits(all_points[:, 1])
    axis.set_xlim(x_lower, x_upper)
    axis.set_ylim(y_lower, y_upper)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("X coordinate [m]")
    axis.set_ylabel("Y coordinate [m]")
    axis.set_title(f"Representative current and future endpoints\n{scope}")
    _legend_if_useful(axis)
    _save_both(plt, figure, destinations)


def _plot_grouped_horizon_metric(axis: Any, rows: Sequence[Mapping[str, str]], metric: str) -> None:
    groups = sorted({_model_projection_label(row) for row in rows})
    colors = _palette_from_axis(axis, len(groups))
    for color, group in zip(colors, groups):
        selected = [row for row in rows if _model_projection_label(row) == group]
        horizons, means, stds = _metric_by_horizon(selected, metric)
        axis.errorbar(horizons, means, yerr=stds, marker="o", capsize=3, color=color, label=group)


def _plot_projection_horizon_metric(
    axis: Any, rows: Sequence[Mapping[str, str]], metric: str
) -> None:
    groups = sorted({row["model_kind"] for row in rows})
    if not groups:
        raise ValueError(f"no overall projection rows available for metric {metric}")
    colors = _palette_from_axis(axis, len(groups))
    for color, group in zip(colors, groups):
        selected = [row for row in rows if row["model_kind"] == group]
        horizons, means, stds = _metric_by_horizon(selected, metric)
        axis.errorbar(
            horizons,
            means,
            yerr=stds,
            marker="o",
            capsize=3,
            color=color,
            label=_display(group),
        )


def _metric_by_horizon(rows: Sequence[Mapping[str, str]], metric: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: Dict[int, List[float]] = {}
    for row in rows:
        horizon = _positive_int(row["horizon"], "horizon")
        grouped.setdefault(horizon, []).append(_finite_float(row[metric], metric))
    if not grouped:
        raise ValueError(f"no rows available for metric {metric}")
    horizons = np.asarray(sorted(grouped), dtype=np.float64)
    means = np.asarray([np.mean(grouped[int(value)]) for value in horizons], dtype=np.float64)
    stds = np.asarray([np.std(grouped[int(value)]) for value in horizons], dtype=np.float64)
    return horizons, means, stds


def _example_deltas(rows: Sequence[Mapping[str, str]], columns: Mapping[str, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_xy = np.column_stack((_numeric(rows, columns["true_x"], "true delta x"), _numeric(rows, columns["true_y"], "true delta y")))
    pred_xy = np.column_stack((_numeric(rows, columns["pred_x"], "predicted delta x"), _numeric(rows, columns["pred_y"], "predicted delta y")))
    projected_xy = np.column_stack((_numeric(rows, columns["projected_x"], "projected delta x"), _numeric(rows, columns["projected_y"], "projected delta y")))
    return true_xy, pred_xy, projected_xy


def _align_delta_units(
    rows: Sequence[Mapping[str, str]],
    columns: Mapping[str, str],
    true_xy: np.ndarray,
    pred_xy: np.ndarray,
    projected_xy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    current_is_m = "_m" in columns["current_x"] and "normalized" not in columns["current_x"]
    delta_is_normalized = "normalized" in columns["true_x"]
    if not current_is_m or not delta_is_normalized:
        return true_xy, pred_xy, projected_xy
    scale_column = next((name for name in ("normalization_distance_m", "horizon_distance_m", "maximum_reachable_distance_m") if name in rows[0]), None)
    if scale_column is not None:
        scales = _numeric(rows, scale_column, "normalization distance")[:, None]
    elif "maximum_step_distance_m" in rows[0] and "horizon" in rows[0]:
        scales = (_numeric(rows, "maximum_step_distance_m", "maximum step distance") * _numeric(rows, "horizon", "horizon"))[:, None]
    else:
        raise ValueError("representative endpoints mix metre positions with normalized deltas but provide no conversion distance")
    return true_xy * scales, pred_xy * scales, projected_xy * scales


def _angular_error_degrees(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    target_norm = np.linalg.norm(target, axis=1)
    prediction_norm = np.linalg.norm(prediction, axis=1)
    valid = (target_norm > 1.0e-12) & (prediction_norm > 1.0e-12)
    if not np.any(valid):
        return np.asarray([], dtype=np.float64)
    cosine = np.sum(target[valid] * prediction[valid], axis=1) / (target_norm[valid] * prediction_norm[valid])
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _filter_example_horizon(rows: Sequence[Mapping[str, str]], horizon: Optional[int]) -> List[Mapping[str, str]]:
    if horizon is None or "horizon" not in rows[0] or not rows[0].get("horizon", ""):
        return list(rows)
    selected = [row for row in rows if _positive_int(row["horizon"], "prediction-example horizon") == horizon]
    if not selected:
        raise ValueError(f"prediction examples contain no rows for horizon {horizon}")
    return selected


def _example_scope(rows: Sequence[Mapping[str, str]]) -> str:
    if not rows:
        return "no prediction examples"
    horizons = sorted(
        {
            _positive_int(row["horizon"], "prediction-example horizon")
            for row in rows
            if row.get("horizon", "")
        }
    )
    splits = sorted(
        {row.get("split", "") for row in rows if row.get("split", "")}
    )
    models = sorted(
        {
            _display(row.get("model_kind", ""))
            for row in rows
            if row.get("model_kind", "")
        }
    )
    seeds = sorted(
        {row.get("seed", "") for row in rows if row.get("seed", "")},
        key=_natural_key,
    )
    parts = []
    if horizons:
        parts.append("H=" + ",".join(str(value) for value in horizons))
    if splits:
        parts.append("split=" + ",".join(splits))
    if models:
        parts.append("model=" + ",".join(models))
    if seeds:
        parts.append("seed=" + ",".join(str(value) for value in seeds))
    return "; ".join(parts)


def _optional_horizon(row: Mapping[str, str]) -> Optional[int]:
    value = row.get("horizon", "")
    return None if value == "" else _positive_int(value, "prediction-example horizon")


def _numeric(rows: Sequence[Mapping[str, str]], column: str, name: str) -> np.ndarray:
    values = np.asarray([_finite_float(row[column], name) for row in rows], dtype=np.float64)
    if values.size == 0:
        raise ValueError(f"{name} contains no values")
    return values


def _model_projection_label(row: Mapping[str, str]) -> str:
    return f"{_display(row['model_kind'])} | {_display_projection(row['projection'])}"


def _display_projection(value: Any) -> str:
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "projected", "projection", "on"):
        return "projected"
    if text in ("0", "false", "no", "raw", "unprojected", "off"):
        return "raw"
    return _display(value)


def _run_label(summary: Mapping[str, str], index: int) -> str:
    parts = []
    if summary.get("horizon", ""):
        parts.append(f"H={summary['horizon']}")
    if summary.get("model_kind", ""):
        parts.append(_display(summary["model_kind"]))
    if summary.get("seed", ""):
        parts.append(f"seed={summary['seed']}")
    return " | ".join(parts) if parts else f"run {index + 1}"


def _save_both(plt: Any, figure: Any, destinations: Tuple[Path, Path]) -> None:
    png, svg = destinations
    figure.savefig(png, dpi=190, bbox_inches="tight", facecolor="white", transparent=False)
    figure.savefig(svg, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(figure)


def _load_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required for Round-4 figures") from error
    return plt


def _scientific_style() -> Dict[str, Any]:
    return {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.7,
        "font.size": 9.0,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 7.6,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "lines.linewidth": 1.4,
        "svg.fonttype": "none",
    }


def _palette(plt: Any, count: int) -> List[Any]:
    colormap = plt.get_cmap("tab20")
    return [colormap(index % 20) for index in range(count)]


def _palette_from_axis(axis: Any, count: int) -> List[Any]:
    colormap = axis.figure.get_cmap("tab10") if hasattr(axis.figure, "get_cmap") else None
    if colormap is None:
        import matplotlib.pyplot as plt
        return _palette(plt, count)
    return [colormap(index % 10) for index in range(count)]


def _legend_if_useful(axis: Any, max_entries: int = 12) -> None:
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles[:max_entries], labels[:max_entries], frameon=True, framealpha=0.92)


def _availability_notice(axis: Any, message: str) -> None:
    axis.text(0.5, 0.5, message, transform=axis.transAxes, ha="center", va="center", wrap=True, color="#555555")
    axis.set_xticks([])
    axis.set_yticks([])


def _all_positive_lines(axis: Any) -> bool:
    values = []
    for line in axis.lines:
        values.extend(np.asarray(line.get_ydata(), dtype=np.float64).tolist())
    return bool(values) and bool(np.all(np.asarray(values) > 0))


def _finite_limits(values: np.ndarray, *, include_zero: bool = False) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("plot limits require finite non-empty values")
    lower, upper = float(np.min(array)), float(np.max(array))
    if include_zero:
        lower = min(lower, 0.0)
    span = upper - lower
    margin = 0.05 * span if span > 0 else max(0.05, 0.05 * max(abs(lower), 1.0))
    return lower - margin, upper + margin


def _arrow_head_width(current: np.ndarray, displacement: np.ndarray) -> float:
    values = np.vstack((current, current + displacement))
    span = max(float(np.ptp(values[:, 0])), float(np.ptp(values[:, 1])), 1.0e-6)
    return 0.012 * span


def _coordinate_unit(column: str) -> str:
    return "m" if "_m" in column and "normalized" not in column else "normalized"


def _display(value: Any) -> str:
    return str(value).strip().replace("_", " ")


def _natural_key(value: Any) -> Tuple[int, Any]:
    text = str(value).strip()
    try:
        return 0, float(text)
    except ValueError:
        return 1, text.lower()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        numeric = int(value)
        original = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if numeric <= 0 or float(numeric) != original:
        raise ValueError(f"{name} must be a positive integer")
    return numeric


def _finite_float(value: Any, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _is_finite_text(value: Any) -> bool:
    text = str(value).strip()
    if not text:
        return False
    try:
        return math.isfinite(float(text))
    except (TypeError, ValueError):
        return False


__all__ = [
    "ROUND4_FIGURE_CATEGORIES",
    "generate_per_horizon_loss_figures",
    "generate_round4_figures",
]
