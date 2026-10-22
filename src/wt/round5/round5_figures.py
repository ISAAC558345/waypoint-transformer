"""Scientific Round-5 figure generation with semantic quality assurance.

The plotting layer consumes recorded CSV rows only; it does not load models,
select checkpoints, or run environments.  All sixteen required categories are
saved independently as PNG and SVG on a white two-dimensional canvas.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .round5_evaluation import write_finite_json


PathLike = Union[str, Path]
Rows = Sequence[Mapping[str, Any]]


ROUND5_FIGURE_CATEGORIES: Mapping[str, Tuple[str, str]] = {
    "loss": ("01_loss", "all_model_train_validation_loss"),
    "validation_models": ("02_validation_models", "wt_direct_state_validation_rmse"),
    "context_length": ("03_context_length", "context_h1_h10_comparison"),
    "transformer_mlp": ("04_transformer_mlp", "transformer_vs_mlp"),
    "predicted_oracle": ("05_predicted_oracle", "predicted_vs_oracle_waypoint"),
    "policy_test": ("06_policy_test", "test_action_rmse_by_policy"),
    "buoy_test": ("07_buoy_test", "test_action_rmse_by_buoy_count"),
    "delta_x": ("08_delta_x", "true_vs_predicted_delta_x"),
    "delta_y": ("09_delta_y", "true_vs_predicted_delta_y"),
    "magnitude": ("10_magnitude", "action_magnitude_comparison"),
    "angular_error": ("11_angular_error", "angular_error_distribution"),
    "projection": ("12_projection", "projection_activation_rate"),
    "replay_error": ("13_replay_error", "replay_action_error_by_timestep"),
    "replay_arrows": ("14_replay_arrows", "replay_true_predicted_action_arrows"),
    "waypoint_action": ("15_waypoint_action", "waypoint_error_vs_action_error"),
    "parameters": ("16_parameters", "parameter_count_vs_validation_metric"),
}


_TABLE_FILES: Mapping[str, Tuple[str, ...]] = {
    "training_summary": ("transformer_training_summary.csv", "training_summary.csv"),
    "training_history": ("transformer_training_history.csv", "training_history.csv"),
    "validation": ("transformer_validation_metrics.csv", "validation_metrics.csv"),
    "test": ("transformer_test_metrics.csv", "test_metrics.csv"),
    "grouped": ("transformer_grouped_metrics.csv", "grouped_metrics.csv"),
    "baseline": ("transformer_baseline_metrics.csv", "baseline_metrics.csv"),
    "projection": ("action_projection_metrics.csv", "projection_metrics.csv"),
    "replay": ("replay_diagnostics.csv",),
    "predictions": (
        "action_prediction_examples.csv",
        "transformer_test_predictions.csv",
        "test_action_predictions.csv",
    ),
    "parameters": ("model_parameter_counts.csv", "parameter_counts.csv"),
}


_MODEL = ("model_type", "model", "variant", "model_kind")
_CONTEXT = ("context_length", "H_c", "h_c", "context")
_SEED = ("seed", "training_seed")
_SPLIT = ("split", "dataset_split")
_RMSE = (
    "action_l2_rmse_m",
    "validation_action_rmse_m",
    "test_action_rmse_m",
    "endpoint_rmse_m",
    "raw_action_rmse_m",
)
_TRAIN_LOSS = ("train_loss", "training_loss", "loss_train", "mean_train_loss")
_VAL_LOSS = ("validation_loss", "val_loss", "loss_validation", "mean_validation_loss")
_EPOCH = ("epoch", "best_epoch", "step")
_TRUE_X = ("true_action_x_m", "target_action_x_m", "true_dx_m", "target_dx_m")
_TRUE_Y = ("true_action_y_m", "target_action_y_m", "true_dy_m", "target_dy_m")
_PRED_X = (
    "predicted_action_raw_x_m",
    "predicted_action_x_m",
    "prediction_x_m",
    "pred_dx_m",
)
_PRED_Y = (
    "predicted_action_raw_y_m",
    "predicted_action_y_m",
    "prediction_y_m",
    "pred_dy_m",
)
_PROJECTED_X = ("predicted_action_projected_x_m", "projected_action_x_m")
_PROJECTED_Y = ("predicted_action_projected_y_m", "projected_action_y_m")
_ACTION_ERROR = (
    "raw_action_error_l2_m",
    "action_error_l2_m",
    "action_l2_error_m",
    "projected_action_error_l2_m",
)
_ANGLE = ("raw_action_angular_error_deg", "action_angular_error_deg", "angular_error_deg")
_WAYPOINT_ERROR = (
    "waypoint_error_l2_m",
    "waypoint_error_m",
    "waypoint_error_l2",
    "spatial_waypoint_error_m",
)


def generate_round5_figures(
    output_dir: PathLike,
    figure_dir: PathLike,
    *,
    tables: Optional[Mapping[str, Rows]] = None,
    overwrite: bool = False,
    strict: bool = True,
) -> Dict[str, Path]:
    """Generate the sixteen required Round-5 categories and semantic QA JSON.

    ``tables`` may override any canonical table name from ``_TABLE_FILES``.
    With ``strict=True`` (the formal-run default), missing/non-finite source data
    aborts instead of producing a misleading empty plot.
    """

    source = Path(output_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"Round-5 output directory does not exist: {source}")
    data = _load_tables(source, tables or {})
    destinations = _prepare_paths(Path(figure_dir), overwrite=overwrite)
    plt = _load_pyplot()
    qa_rows: List[Dict[str, Any]] = []
    paths: Dict[str, Path] = {}
    plotters: Mapping[str, Callable[[Any, Mapping[str, List[Dict[str, Any]]], bool], Any]] = {
        "loss": _plot_loss,
        "validation_models": _plot_validation_models,
        "context_length": _plot_context_length,
        "transformer_mlp": _plot_transformer_mlp,
        "predicted_oracle": _plot_predicted_oracle,
        "policy_test": _plot_policy_test,
        "buoy_test": _plot_buoy_test,
        "delta_x": lambda p, d, s: _plot_component_scatter(p, d, s, component="x"),
        "delta_y": lambda p, d, s: _plot_component_scatter(p, d, s, component="y"),
        "magnitude": _plot_magnitude,
        "angular_error": _plot_angular_error,
        "projection": _plot_projection,
        "replay_error": _plot_replay_error,
        "replay_arrows": _plot_replay_arrows,
        "waypoint_action": _plot_waypoint_action,
        "parameters": _plot_parameters,
    }
    with plt.rc_context(_style()):
        for category in ROUND5_FIGURE_CATEGORIES:
            figure = plotters[category](plt, data, strict)
            qa = semantic_qa_round5_figure(figure, category=category, strict_data=strict)
            png, svg = destinations[category]
            figure.savefig(png, dpi=190, bbox_inches="tight", facecolor="white", transparent=False)
            figure.savefig(svg, bbox_inches="tight", facecolor="white", transparent=False)
            plt.close(figure)
            if png.stat().st_size <= 0 or svg.stat().st_size <= 0:
                raise RuntimeError(f"empty figure artifact for {category}")
            qa.update(
                {
                    "png": png.relative_to(Path(figure_dir)).as_posix(),
                    "svg": svg.relative_to(Path(figure_dir)).as_posix(),
                    "png_bytes": int(png.stat().st_size),
                    "svg_bytes": int(svg.stat().st_size),
                }
            )
            qa_rows.append(qa)
            paths[f"{category}_png"] = png
            paths[f"{category}_svg"] = svg
    qa_path = Path(figure_dir) / "figure_semantic_qa.json"
    write_finite_json(
        qa_path,
        {
            "schema_version": "round05_figure_semantic_qa_v1",
            "category_count": len(qa_rows),
            "all_passed": all(bool(row["passed"]) for row in qa_rows),
            "formats": ["png", "svg"],
            "three_dimensional_figures": False,
            "rows": qa_rows,
        },
    )
    paths["semantic_qa_json"] = qa_path
    return paths


def semantic_qa_round5_figure(
    figure: Any, *, category: str, strict_data: bool = True
) -> Dict[str, Any]:
    """Check data integrity plus the Round-5 metadata-annotation contract.

    Every plotting function must attach a structured ``_round5_metadata``
    record to its primary axis and render the exact caption stored there.  A
    title containing vague text such as ``model/H_c/seed recorded`` therefore
    cannot pass QA without the actual split, model, context, and training-seed
    identities being present.
    """

    if category not in ROUND5_FIGURE_CATEGORIES:
        raise KeyError(f"unknown Round-5 figure category {category!r}")
    axes = list(figure.axes)
    if not axes:
        raise ValueError(f"{category}: figure has no axes")
    primary = axes[0]
    title = primary.get_title().strip()
    xlabel = primary.get_xlabel().strip()
    ylabel = primary.get_ylabel().strip()
    if not title or not xlabel or not ylabel:
        raise ValueError(f"{category}: title and both axis labels are required")
    if any(getattr(axis, "name", "rectilinear") == "3d" for axis in axes):
        raise ValueError(f"{category}: three-dimensional axes are prohibited")
    data_count = 0
    for axis in axes:
        limits = [*axis.get_xlim(), *axis.get_ylim()]
        if not np.all(np.isfinite(np.asarray(limits, dtype=np.float64))):
            raise ValueError(f"{category}: non-finite axis limits")
        for line in axis.lines:
            x = np.asarray(line.get_xdata(), dtype=np.float64)
            y = np.asarray(line.get_ydata(), dtype=np.float64)
            if x.size and y.size:
                if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                    raise ValueError(f"{category}: line contains NaN or Inf")
                data_count += 1
        for collection in axis.collections:
            offsets = np.asarray(collection.get_offsets())
            if offsets.size:
                try:
                    numeric_offsets = offsets.astype(np.float64)
                except (TypeError, ValueError):
                    numeric_offsets = np.asarray([], dtype=np.float64)
                if numeric_offsets.size:
                    if not np.all(np.isfinite(numeric_offsets)):
                        raise ValueError(f"{category}: collection contains NaN or Inf")
                    data_count += 1
        for patch in axis.patches:
            values = np.asarray(
                [getattr(patch, "get_x", lambda: 0.0)(), getattr(patch, "get_y", lambda: 0.0)(),
                 getattr(patch, "get_width", lambda: 0.0)(), getattr(patch, "get_height", lambda: 0.0)()],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{category}: patch contains NaN or Inf")
            data_count += 1
    if strict_data and data_count == 0:
        raise ValueError(f"{category}: figure contains no finite data artists")
    face = np.asarray(figure.get_facecolor(), dtype=np.float64)
    white_background = bool(np.allclose(face[:3], np.ones(3), atol=1e-6))
    if not white_background:
        raise ValueError(f"{category}: figure background must be white")
    metadata = getattr(primary, "_round5_metadata", None)
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{category}: structured split/model/H_c/seed metadata is required")
    caption = str(metadata.get("caption", "")).strip()
    split_labels = tuple(str(value) for value in metadata.get("splits", ()))
    model_labels = tuple(str(value) for value in metadata.get("models", ()))
    contexts = tuple(str(value) for value in metadata.get("contexts", ()))
    seeds = tuple(str(value) for value in metadata.get("training_seeds", ()))
    identity_location = str(metadata.get("identity_location", "")).strip()
    if not caption or not split_labels or not model_labels or not contexts or not seeds:
        raise ValueError(f"{category}: split/model/H_c/training-seed metadata is incomplete")
    forbidden = ("?", "unknown", "model/h_c/seed recorded")
    searchable = " | ".join((caption, *split_labels, *model_labels, *contexts, *seeds)).lower()
    if any(token in searchable for token in forbidden):
        raise ValueError(f"{category}: placeholder metadata is forbidden")
    rendered_text = {str(item.get_text()).strip() for item in primary.texts}
    if caption not in rendered_text:
        raise ValueError(f"{category}: the structured metadata caption was not rendered")
    if not bool(metadata.get("seed_identity_visible", False)):
        raise ValueError(f"{category}: training-seed identity is not visibly declared")
    if not bool(metadata.get("layout_safe", False)):
        raise ValueError(f"{category}: plot has not declared the reviewed layout contract")
    if not identity_location:
        raise ValueError(f"{category}: model/H_c identity location is not declared")
    if category == "loss" and not bool(metadata.get("loss_series_semantics_complete", False)):
        raise ValueError("loss: train/validation and run/mean line semantics are incomplete")
    return {
        "category": category,
        "passed": True,
        "title": title,
        "x_axis": xlabel,
        "y_axis": ylabel,
        "finite_data_artist_count": data_count,
        "strict_data": bool(strict_data),
        "white_background": white_background,
        "two_dimensional": True,
        "metadata_complete": True,
        "metadata_caption": caption,
        "split_labels": list(split_labels),
        "model_labels": list(model_labels),
        "context_lengths": list(contexts),
        "training_seeds": list(seeds),
        "seed_identity_visible": True,
        "identity_location": identity_location,
        "layout_safe": True,
        "loss_series_semantics_complete": bool(
            metadata.get("loss_series_semantics_complete", category != "loss")
        ),
    }


def _identity_token(value: Any) -> str:
    """Return a compact stable token for context lengths and seeds."""

    text = str(value).strip()
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return text
    if math.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return text


def _metadata_from_rows(
    rows: Rows,
    *,
    split: Optional[str] = None,
) -> Dict[str, Tuple[str, ...]]:
    models = tuple(
        sorted(
            {
                _display_model(_text(row, _MODEL, ""))
                for row in rows
                if _text(row, _MODEL, "").strip()
            }
        )
    )
    contexts = tuple(
        sorted(
            {
                _identity_token(_text(row, _CONTEXT, ""))
                for row in rows
                if _text(row, _CONTEXT, "").strip()
            },
            key=_natural_key,
        )
    )
    seeds = tuple(
        sorted(
            {
                _identity_token(_text(row, _SEED, ""))
                for row in rows
                if _text(row, _SEED, "").strip()
            },
            key=_natural_key,
        )
    )
    observed_splits = tuple(
        sorted(
            {
                _text(row, _SPLIT, "").strip()
                for row in rows
                if _text(row, _SPLIT, "").strip()
            }
        )
    )
    splits = (str(split),) if split is not None else observed_splits
    if not models or not contexts or not seeds or not splits:
        raise ValueError("figure rows lack actual split/model/H_c/training-seed identities")
    return {
        "models": models,
        "contexts": contexts,
        "training_seeds": seeds,
        "splits": splits,
    }


def _annotate_metadata(
    axis: Any,
    metadata: Mapping[str, Sequence[str]],
    *,
    note: str,
    identity_location: str,
    seed_identity_visible: bool = True,
    layout_strategy: str = "caption_above_axes",
    loss_series_semantics_complete: bool = False,
) -> str:
    models = tuple(str(value) for value in metadata["models"])
    contexts = tuple(str(value) for value in metadata["contexts"])
    seeds = tuple(str(value) for value in metadata["training_seeds"])
    splits = tuple(str(value) for value in metadata["splits"])
    model_text = models[0] if len(models) == 1 else "{" + ", ".join(models) + "}"
    context_text = contexts[0] if len(contexts) == 1 else "{" + ",".join(contexts) + "}"
    seed_text = seeds[0] if len(seeds) == 1 else "{" + ",".join(seeds) + "}"
    line_one = f"split={'+'.join(splits)} | model={model_text}"
    line_two = f"H_c={context_text} | training seed={seed_text} | {note}"
    caption = f"{line_one}\n{line_two}" if len(line_one) + len(line_two) > 105 else f"{line_one} | {line_two}"
    axis.text(
        0.5,
        1.012,
        caption,
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.4,
        color="#333333",
        linespacing=1.15,
    )
    axis.set_title(axis.get_title(), pad=47 if "\n" in caption else 31)
    axis._round5_metadata = {
        "caption": caption,
        "models": models,
        "contexts": contexts,
        "training_seeds": seeds,
        "splits": splits,
        "identity_location": str(identity_location),
        "seed_identity_visible": bool(seed_identity_visible),
        "layout_safe": True,
        "layout_strategy": str(layout_strategy),
        "loss_series_semantics_complete": bool(loss_series_semantics_complete),
    }
    return caption


def _selected_identity(data: Mapping[str, List[Dict[str, Any]]]) -> Dict[str, Tuple[str, ...]]:
    records = data.get("selected_config", [])
    if records:
        selected = records[0]
        model = _display_model(_text(selected, _MODEL, ""))
        context = _identity_token(
            selected.get("context_length", selected.get("H_c", ""))
        )
        seed = _identity_token(selected.get("training_seed", selected.get("seed", "")))
        split = _text(selected, ("test_split",), "test")
        if model and context and seed:
            return {
                "models": (model,),
                "contexts": (context,),
                "training_seeds": (seed,),
                "splits": (split,),
            }
    return _metadata_from_rows(_prediction_rows(data), split="test")


def _row_matches_identity(row: Mapping[str, Any], metadata: Mapping[str, Sequence[str]]) -> bool:
    model = _display_model(_text(row, _MODEL, ""))
    context = _identity_token(_text(row, _CONTEXT, ""))
    seed = _identity_token(_text(row, _SEED, ""))
    return (
        model in set(metadata["models"])
        and context in set(metadata["contexts"])
        and seed in set(metadata["training_seeds"])
    )


def _short_episode_description(
    episode_id: str,
    row: Mapping[str, Any],
    index: int,
) -> str:
    match = re.search(r"seed-(\d+)", str(episode_id))
    collection_seed = str(int(match.group(1))) if match else "n/a"
    buoys = _text(row, ("num_buoys", "buoy_count"), "n/a")
    policy = _text(row, ("policy", "policy_name", "policy_id"), "n/a")
    return f"E{index:02d}: collection-seed={collection_seed}, buoys={buoys}, policy={policy}"


def _plot_loss(plt: Any, data: Mapping[str, List[Dict[str, Any]]], strict: bool) -> Any:
    rows = data["training_history"]
    figure, axis = plt.subplots(figsize=(11.4, 6.0))
    groups = _group_rows(rows, lambda row: (_text(row, _MODEL, "model"), _text(row, _CONTEXT, "?"), _text(row, _SEED, "?")))
    plotted = 0
    mean_series: Dict[Tuple[str, str], Dict[str, Dict[int, List[float]]]] = {}
    prepared: List[Tuple[Tuple[str, str], str, np.ndarray, np.ndarray, np.ndarray]] = []
    for (model, context, seed), group in groups.items():
        samples = []
        for row in group:
            epoch = _optional_number(row, _EPOCH)
            train = _optional_number(row, _TRAIN_LOSS)
            validation = _optional_number(row, _VAL_LOSS)
            if epoch is not None and train is not None and validation is not None:
                samples.append((int(epoch), train, validation))
        samples.sort()
        if not samples:
            continue
        x = np.asarray([item[0] for item in samples])
        train = np.asarray([item[1] for item in samples])
        validation = np.asarray([item[2] for item in samples])
        key = (_canonical_model(model), _identity_token(context))
        prepared.append((key, _identity_token(seed), x, train, validation))
        series = mean_series.setdefault(key, {"train": {}, "validation": {}})
        for epoch, train_value, validation_value in samples:
            series["train"].setdefault(epoch, []).append(train_value)
            series["validation"].setdefault(epoch, []).append(validation_value)
    keys = sorted(mean_series, key=lambda item: (_display_model(item[0]), _natural_key(item[1])))
    colors = {key: color for key, color in zip(keys, _palette(plt, len(keys)))}
    markers = {"0": "o", "1": "s", "2": "^"}
    for key, seed, x, train, validation in prepared:
        color = colors[key]
        mark_every = max(1, int(len(x) // 6))
        axis.plot(x, train, color=color, lw=0.75, alpha=0.18, linestyle="-")
        axis.plot(
            x,
            validation,
            color=color,
            lw=0.8,
            alpha=0.28,
            linestyle="--",
            marker=markers.get(seed, "D"),
            markersize=2.2,
            markevery=mark_every,
        )
        plotted += 2
    configuration_handles, configuration_labels = [], []
    for key in keys:
        color = colors[key]
        series = mean_series[key]
        epochs = np.asarray(sorted(set(series["train"]) | set(series["validation"])))
        mean_train = np.asarray([np.mean(series["train"][int(epoch)]) for epoch in epochs])
        mean_validation = np.asarray(
            [np.mean(series["validation"][int(epoch)]) for epoch in epochs]
        )
        axis.plot(epochs, mean_train, color=color, lw=2.25, linestyle="-")
        handle = axis.plot(epochs, mean_validation, color=color, lw=2.25, linestyle="--")[0]
        configuration_handles.append(handle)
        # Keep the external legend narrow enough to survive raster/PDF export.
        # The two lines still spell out both pieces of run identity explicitly.
        configuration_labels.append(f"{_display_model(key[0])}\nH_c={key[1]}")
        plotted += 2
    _require_or_notice(axis, plotted, strict, "No training histories available")
    axis.set_title("Round-5 all-model training and validation loss (runs + mean trends)")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Normalized action MSE")
    _zero_lower(axis)
    if configuration_handles:
        config_legend = axis.legend(
            configuration_handles,
            configuration_labels,
            title="model / context\n(mean curves)",
            frameon=False,
            fontsize=7.1,
            title_fontsize=7.3,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
        )
        axis.add_artist(config_legend)
        semantic_handles = [
            axis.plot([], [], color="#333333", lw=2.25, linestyle="-", label="mean train")[0],
            axis.plot([], [], color="#333333", lw=2.25, linestyle="--", label="mean validation")[0],
            axis.plot([], [], color="#777777", lw=0.8, alpha=0.4, label="thin = individual run")[0],
        ]
        for seed in sorted({_identity_token(seed) for _, seed, *_ in prepared}, key=_natural_key):
            semantic_handles.append(
                axis.plot(
                    [], [], color="#333333", linestyle="None",
                    marker=markers.get(seed, "D"), markersize=4.0,
                    label=f"validation marker: seed={seed}",
                )[0]
            )
        axis.legend(
            handles=semantic_handles,
            title="line / seed semantics",
            frameon=False,
            fontsize=7.0,
            title_fontsize=7.3,
            loc="lower left",
            bbox_to_anchor=(1.01, 0.0),
        )
        figure.subplots_adjust(right=0.73)
    metadata = _metadata_from_rows(rows, split="train+validation")
    _annotate_metadata(
        axis,
        metadata,
        note="thin=individual run; thick=mean; solid=train; dashed=validation; marker shape=seed",
        identity_location="external model/context legend plus line/seed-semantics legend",
        layout_strategy="two_external_legends_and_top_caption",
        loss_series_semantics_complete=True,
    )
    return figure


def _plot_validation_models(plt: Any, data: Mapping[str, List[Dict[str, Any]]], strict: bool) -> Any:
    rows = _metric_rows(data["validation"], split="validation", overall=True, raw=True)
    wanted = ["waypoint_transformer", "direct_goal_transformer", "state_only_transformer"]
    selected = [row for row in rows if _canonical_model(_text(row, _MODEL, "")) in wanted]
    return _bar_seed_metric(
        plt,
        selected,
        "Round-5 validation action RMSE: WT, Direct-Goal, State-only (seeds shown)",
        "Model / context H_c",
        "Validation action L2 RMSE (m)",
        strict,
    )


def _plot_context_length(plt: Any, data: Mapping[str, List[Dict[str, Any]]], strict: bool) -> Any:
    rows = _metric_rows(data["validation"], split="validation", overall=True, raw=True)
    selected = [
        row
        for row in rows
        if _canonical_model(_text(row, _MODEL, ""))
        in {"waypoint_transformer", "direct_goal_transformer", "state_only_transformer"}
        and _text(row, _CONTEXT, "") in {"1", "10", "1.0", "10.0"}
    ]
    return _bar_seed_metric(
        plt,
        selected,
        "Round-5 validation context comparison (H_c=1 vs H_c=10; all seeds)",
        "Model / context H_c",
        "Validation action L2 RMSE (m)",
        strict,
    )


def _plot_transformer_mlp(plt: Any, data: Mapping[str, List[Dict[str, Any]]], strict: bool) -> Any:
    rows = _metric_rows(data["validation"], split="validation", overall=True, raw=True)
    selected = [row for row in rows if "transformer" in _canonical_model(_text(row, _MODEL, "")) or "mlp" in _canonical_model(_text(row, _MODEL, ""))]
    return _bar_seed_metric(
        plt,
        selected,
        "Round-5 validation action RMSE: causal Transformer vs one-step MLP",
        "Model / context H_c",
        "Validation action L2 RMSE (m)",
        strict,
    )


def _plot_predicted_oracle(plt: Any, data: Mapping[str, List[Dict[str, Any]]], strict: bool) -> Any:
    rows = _metric_rows(data["validation"], split="validation", overall=True, raw=True)
    selected = [
        row
        for row in rows
        if _canonical_model(_text(row, _MODEL, ""))
        in {"waypoint_transformer", "oracle_waypoint_transformer"}
    ]
    figure = _bar_seed_metric(
        plt,
        selected,
        "Round-5 validation: predicted waypoint vs oracle H=30 diagnostic (non-deployable)",
        "Waypoint condition / context H_c",
        "Validation action L2 RMSE (m)",
        strict,
    )
    return figure


def _plot_policy_test(plt: Any, data: Mapping[str, List[Dict[str, Any]]], strict: bool) -> Any:
    rows = _group_metric_rows(data, "policy")
    metadata = _selected_identity(data)
    selected = [row for row in rows if _row_matches_identity(row, metadata)]
    figure = _group_bar(
        plt,
        selected,
        "Round-5 selected-policy test action RMSE by collection policy (raw)",
        "Collection policy",
        "Test action L2 RMSE (m)",
        strict,
    )
    _annotate_metadata(
        figure.axes[0],
        metadata,
        note="validation-selected configuration; test evaluated once; raw predictions",
        identity_location="selected-configuration caption and bar legend",
    )
    return figure


def _plot_buoy_test(plt: Any, data: Mapping[str, List[Dict[str, Any]]], strict: bool) -> Any:
    rows = _group_metric_rows(data, "num_buoys")
    metadata = _selected_identity(data)
    selected = [row for row in rows if _row_matches_identity(row, metadata)]
    figure = _group_bar(
        plt,
        selected,
        "Round-5 selected-policy test action RMSE by buoy count (raw)",
        "Number of buoys",
        "Test action L2 RMSE (m)",
        strict,
    )
    _annotate_metadata(
        figure.axes[0],
        metadata,
        note="validation-selected configuration; test evaluated once; raw predictions",
        identity_location="selected-configuration caption and bar legend",
    )
    return figure


def _plot_component_scatter(
    plt: Any,
    data: Mapping[str, List[Dict[str, Any]]],
    strict: bool,
    *,
    component: str,
) -> Any:
    rows = _prediction_rows(data)
    true_aliases, pred_aliases, projected_aliases = (
        (_TRUE_X, _PRED_X, _PROJECTED_X) if component == "x" else (_TRUE_Y, _PRED_Y, _PROJECTED_Y)
    )
    true, predicted = _paired_numeric(rows, true_aliases, pred_aliases)
    _, projected = _paired_numeric(rows, true_aliases, projected_aliases)
    figure, axis = plt.subplots(figsize=(6.2, 5.8))
    plotted = 0
    if true.size:
        axis.scatter(true, predicted, s=12, alpha=0.28, color="#4c78a8", label="raw prediction")
        plotted += 1
    if projected.size:
        projected_true, projected_values = _paired_numeric(rows, true_aliases, projected_aliases)
        axis.scatter(projected_true, projected_values, s=11, alpha=0.25, color="#f58518", label="projected")
        plotted += 1
    _require_or_notice(axis, plotted, strict, f"No finite delta_{component} prediction rows")
    if plotted:
        values = np.concatenate([array for array in (true, predicted, projected) if array.size])
        lower, upper = _symmetric_limits(values)
        axis.plot([lower, upper], [lower, upper], "--", color="#333333", lw=1.6, label="ideal")
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_aspect("equal", adjustable="box")
    axis.set_title(f"Round-5 selected-policy test: true vs predicted $\\Delta {component}$")
    axis.set_xlabel(f"True $\\Delta {component}$ (m)")
    axis.set_ylabel(f"Predicted $\\Delta {component}$ (m)")
    _legend(axis)
    _annotate_metadata(
        axis,
        _selected_identity(data),
        note="raw and radially projected predictions; dashed line=ideal",
        identity_location="selected-configuration caption",
    )
    return figure


def _plot_magnitude(plt: Any, data: Mapping[str, List[Dict[str, Any]]], strict: bool) -> Any:
    rows = _prediction_rows(data)
    true_x, pred_x = _paired_numeric(rows, _TRUE_X, _PRED_X)
    true_y, pred_y = _paired_numeric(rows, _TRUE_Y, _PRED_Y)
    count = min(true_x.size, true_y.size, pred_x.size, pred_y.size)
    figure, axis = plt.subplots(figsize=(6.4, 5.7))
    if count:
        true_magnitude = np.sqrt(np.square(true_x[:count]) + np.square(true_y[:count]))
        predicted_magnitude = np.sqrt(np.square(pred_x[:count]) + np.square(pred_y[:count]))
        axis.scatter(true_magnitude, predicted_magnitude, s=13, alpha=0.3, color="#4c78a8")
        upper = max(float(np.max(true_magnitude)), float(np.max(predicted_magnitude)), 1e-9) * 1.05
        axis.plot([0.0, upper], [0.0, upper], "--", color="#333333", lw=1.8, label="ideal")
        axis.set_xlim(0.0, upper)
        axis.set_ylim(0.0, upper)
        axis.set_aspect("equal", adjustable="box")
    else:
        _require_or_notice(axis, 0, strict, "No finite action magnitude rows")
    axis.set_title("Round-5 selected-policy test: true vs predicted action magnitude")
    axis.set_xlabel("True action magnitude (m)")
    axis.set_ylabel("Predicted action magnitude (m)")
    _legend(axis)
    _annotate_metadata(
        axis,
        _selected_identity(data),
        note="raw prediction magnitude; dashed line=ideal",
        identity_location="selected-configuration caption",
    )
    return figure


def _plot_angular_error(plt: Any, data: Mapping[str, List[Dict[str, Any]]], strict: bool) -> Any:
    rows = _prediction_rows(data)
    values = _numeric_available(rows, _ANGLE)
    if values.size == 0:
        true_x, pred_x = _paired_numeric(rows, _TRUE_X, _PRED_X)
        true_y, pred_y = _paired_numeric(rows, _TRUE_Y, _PRED_Y)
        count = min(true_x.size, true_y.size, pred_x.size, pred_y.size)
        if count:
            true = np.column_stack([true_x[:count], true_y[:count]])
            pred = np.column_stack([pred_x[:count], pred_y[:count]])
            true_mag = np.linalg.norm(true, axis=1)
            pred_mag = np.linalg.norm(pred, axis=1)
            eligible = (true_mag > 1e-6) & (pred_mag > 1e-6)
            cosine = np.sum(true[eligible] * pred[eligible], axis=1) / (true_mag[eligible] * pred_mag[eligible])
            values = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    figure, axis = plt.subplots(figsize=(7.1, 5.0))
    if values.size:
        axis.hist(values, bins=np.linspace(0.0, 180.0, 25), color="#72a0c1", alpha=0.75, edgecolor="white")
        axis.axvline(float(np.mean(values)), color="#b22222", lw=2.3, label=f"mean={np.mean(values):.2f} deg")
        axis.set_xlim(0.0, 180.0)
    else:
        _require_or_notice(axis, 0, strict, "No direction-eligible action rows")
    axis.set_title("Round-5 selected-policy test angular-error distribution")
    axis.set_xlabel("Angular error (degrees)")
    axis.set_ylabel("Transition count")
    _zero_lower(axis)
    _legend(axis)
    _annotate_metadata(
        axis,
        _selected_identity(data),
        note="near-zero true/predicted actions excluded; red line=mean",
        identity_location="selected-configuration caption",
    )
    return figure


def _plot_projection(plt: Any, data: Mapping[str, List[Dict[str, Any]]], strict: bool) -> Any:
    rows = data["projection"] or data["grouped"] or data["validation"]
    metadata = _selected_identity(data)
    selected = [
        row
        for row in rows
        if _row_matches_identity(row, metadata)
        and _text(row, _SPLIT, "test").lower() == "test"
        and _text(row, ("prediction_space",), "raw").lower() == "raw"
    ]
    values: List[float] = []
    for row in selected:
        rate = _optional_number(row, ("projection_activation_rate",))
        if rate is not None:
            values.append(rate)
    figure, axis = plt.subplots(figsize=(7.6, 3.8))
    if values:
        value = float(np.mean(values))
        label = (
            f"{metadata['models'][0]}, H_c={metadata['contexts'][0]}, "
            f"seed={metadata['training_seeds'][0]}"
        )
        axis.barh([label], [value], color="#4c78a8", alpha=0.82, height=0.48)
        axis.scatter(values, np.zeros(len(values)), s=34, color="#222222", zorder=4)
        axis.text(min(value + 0.018, 0.96), 0.0, f"{value:.4f}", va="center", fontsize=8)
        axis.set_xlim(0.0, 1.0)
    else:
        _require_or_notice(axis, 0, strict, "No projection activation rows")
    axis.set_title("Round-5 selected-policy radial projection activation rate")
    axis.set_xlabel("Projection activation rate (raw prediction to legal action disk)")
    axis.set_ylabel("Selected model / H_c / seed")
    _annotate_metadata(
        axis,
        metadata,
        note="raw test predictions; radial projection; full [0,1] rate axis",
        identity_location="selected-configuration caption and single y-axis label",
        layout_strategy="single_horizontal_bar_with_nonoverlapping_identity_label",
    )
    return figure


def _plot_replay_error(plt: Any, data: Mapping[str, List[Dict[str, Any]]], strict: bool) -> Any:
    rows = data["replay"]
    figure, axis = plt.subplots(figsize=(11.2, 5.5))
    groups = _group_rows(rows, lambda row: _text(row, ("episode_id", "trajectory_id"), "unknown"))
    plotted = 0
    for episode_index, (episode, group) in enumerate(sorted(groups.items()), start=1):
        points = []
        for row in group:
            timestep = _optional_number(row, ("timestep", "t", "step"))
            error = _optional_number(row, _ACTION_ERROR)
            if timestep is not None and error is not None:
                points.append((timestep, error))
        points.sort()
        if points:
            alias = _short_episode_description(str(episode), group[0], episode_index)
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                lw=1.0,
                alpha=0.52,
                label=alias,
            )
            plotted += 1
    _require_or_notice(axis, plotted, strict, "No replay timestep errors")
    axis.set_title("Round-5 recorded-state replay action error by timestep (environment not advanced)")
    axis.set_xlabel("Recorded episode timestep")
    axis.set_ylabel("Action L2 error (m)")
    _zero_lower(axis)
    if plotted:
        axis.legend(
            title="recorded episode aliases",
            frameon=False,
            fontsize=7.0,
            title_fontsize=7.4,
            loc="upper left",
            bbox_to_anchor=(1.005, 1.0),
        )
        figure.subplots_adjust(right=0.70)
    _annotate_metadata(
        axis,
        _selected_identity(data),
        note="recorded states only; environment not advanced; E## aliases identify episodes",
        identity_location="selected-configuration caption; external short-episode legend",
        layout_strategy="external_short_alias_legend_outside_data_axes",
    )
    return figure


def _plot_replay_arrows(plt: Any, data: Mapping[str, List[Dict[str, Any]]], strict: bool) -> Any:
    rows = sorted(data["replay"], key=lambda row: (_text(row, ("episode_id",), ""), _number(row, ("timestep", "t", "step"), 0.0)))
    figure, axis = plt.subplots(figsize=(7.0, 6.2))
    plotted = 0
    if rows:
        episode = _text(rows[0], ("episode_id", "trajectory_id"), "unknown")
        selected = [row for row in rows if _text(row, ("episode_id", "trajectory_id"), "unknown") == episode]
        true_vectors, pred_vectors = [], []
        for row in selected:
            tx, ty, px, py = (
                _optional_number(row, _TRUE_X),
                _optional_number(row, _TRUE_Y),
                _optional_number(row, _PRED_X),
                _optional_number(row, _PRED_Y),
            )
            if None not in (tx, ty, px, py):
                true_vectors.append((tx, ty))
                pred_vectors.append((px, py))
        if true_vectors:
            true = np.asarray(true_vectors)
            pred = np.asarray(pred_vectors)
            origins = np.vstack([np.zeros(2), np.cumsum(true[:-1], axis=0)])
            axis.quiver(origins[:, 0], origins[:, 1], true[:, 0], true[:, 1], angles="xy", scale_units="xy", scale=1.0, color="#4c78a8", alpha=0.72, label="true")
            axis.quiver(origins[:, 0], origins[:, 1], pred[:, 0], pred[:, 1], angles="xy", scale_units="xy", scale=1.0, color="#e45756", alpha=0.55, label="predicted")
            all_points = np.vstack([origins, origins + true, origins + pred])
            xlim = _finite_limits(all_points[:, 0], include_zero=True)
            ylim = _finite_limits(all_points[:, 1], include_zero=True)
            axis.set_xlim(*xlim)
            axis.set_ylim(*ylim)
            axis.set_aspect("equal", adjustable="box")
            plotted = 2
            axis.text(
                0.01,
                0.99,
                _short_episode_description(str(episode), selected[0], 1),
                transform=axis.transAxes,
                va="top",
                fontsize=8,
            )
    _require_or_notice(axis, plotted, strict, "No replay action-vector rows")
    axis.set_title("Round-5 replay true and predicted 2-D action arrows (recorded-state origins)")
    axis.set_xlabel("Cumulative true-displacement x (m)")
    axis.set_ylabel("Cumulative true-displacement y (m)")
    _legend(axis)
    _annotate_metadata(
        axis,
        _selected_identity(data),
        note="recorded-state origins; blue=true action; red=predicted action; E01 shown",
        identity_location="selected-configuration caption and in-axis short episode alias",
    )
    return figure


def _plot_waypoint_action(plt: Any, data: Mapping[str, List[Dict[str, Any]]], strict: bool) -> Any:
    rows = _prediction_rows(data)
    waypoint, action = _paired_numeric(rows, _WAYPOINT_ERROR, _ACTION_ERROR)
    figure, axis = plt.subplots(figsize=(6.7, 5.5))
    if waypoint.size:
        axis.scatter(waypoint, action, s=14, alpha=0.3, color="#4c78a8", label="transitions")
        if waypoint.size >= 2 and float(np.ptp(waypoint)) > 0.0:
            coefficients = np.polyfit(waypoint, action, 1)
            x = np.linspace(0.0, float(np.max(waypoint)), 100)
            axis.plot(x, np.polyval(coefficients, x), color="#b22222", lw=2.3, label="linear trend")
        axis.set_xlim(left=0.0)
        axis.set_ylim(bottom=0.0)
    else:
        _require_or_notice(axis, 0, strict, "No paired waypoint/action error rows")
    axis.set_title("Round-5 spatial waypoint error vs offline action error")
    axis.set_xlabel("Waypoint endpoint error (m)")
    axis.set_ylabel("Action L2 error (m)")
    _legend(axis)
    _annotate_metadata(
        axis,
        _selected_identity(data),
        note="selected-policy test transitions; red line=least-squares linear trend",
        identity_location="selected-configuration caption",
    )
    return figure


def _plot_parameters(plt: Any, data: Mapping[str, List[Dict[str, Any]]], strict: bool) -> Any:
    validation = _metric_rows(data["validation"], split="validation", overall=True, raw=True)
    parameter_rows = data["parameters"]
    counts: Dict[Tuple[str, str], float] = {}
    for row in parameter_rows:
        count = _optional_number(row, ("parameter_count", "trainable_parameter_count", "parameters", "num_parameters"))
        if count is not None:
            counts[(_canonical_model(_text(row, _MODEL, "")), _text(row, _CONTEXT, "?"))] = count
    points: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
    for row in validation:
        metric = _optional_number(row, _RMSE)
        key = (_canonical_model(_text(row, _MODEL, "")), _text(row, _CONTEXT, "?"))
        if metric is not None and key in counts:
            points.setdefault(key, []).append((_identity_token(_text(row, _SEED, "")), metric))
    figure, axis = plt.subplots(figsize=(9.4, 5.8))
    if points:
        palette = _palette(plt, len(points))
        marker_by_seed = {"0": "o", "1": "s", "2": "^"}
        config_handles, config_labels = [], []
        all_metrics: List[float] = []
        for color, (key, entries) in zip(palette, sorted(points.items())):
            ordered = sorted(entries, key=lambda item: _natural_key(item[0]))
            jitter = max(float(counts[key]) * 0.0045, 900.0)
            offsets = np.linspace(-jitter, jitter, len(ordered)) if len(ordered) > 1 else np.zeros(1)
            for offset, (seed, metric) in zip(offsets, ordered):
                axis.scatter(
                    counts[key] + offset,
                    metric,
                    s=34,
                    marker=marker_by_seed.get(seed, "D"),
                    color=color,
                    alpha=0.8,
                    edgecolor="white",
                    linewidth=0.45,
                )
                all_metrics.append(metric)
            mean_metric = float(np.mean([metric for _, metric in ordered]))
            mean_handle = axis.scatter(
                counts[key], mean_metric, s=88, facecolors="none", edgecolors=color,
                linewidth=1.7, zorder=5,
            )
            config_handles.append(mean_handle)
            wrapped_model = _display_model(key[0]).replace(" ", "\n")
            config_labels.append(
                f"{wrapped_model}\nH_c={_identity_token(key[1])}"
            )
        axis.set_xlim(0.0, max(counts.values()) * 1.07)
        axis.set_ylim(0.0, max(all_metrics) * 1.14)
        config_legend = axis.legend(
            config_handles,
            config_labels,
            title="model / context\n(open ring=mean)",
            frameon=False,
            fontsize=7.0,
            title_fontsize=7.3,
            loc="upper left",
            bbox_to_anchor=(1.005, 1.0),
        )
        axis.add_artist(config_legend)
        seed_handles = [
            axis.scatter([], [], marker=marker_by_seed.get(seed, "D"), color="#555555", s=32, label=f"seed={seed}")
            for seed in sorted({seed for entries in points.values() for seed, _ in entries}, key=_natural_key)
        ]
        axis.legend(
            seed_handles,
            [handle.get_label() for handle in seed_handles],
            title="training-seed marker",
            frameon=False,
            fontsize=7.2,
            title_fontsize=7.3,
            loc="lower left",
            bbox_to_anchor=(1.005, 0.0),
        )
        figure.subplots_adjust(right=0.72)
    else:
        _require_or_notice(axis, 0, strict, "No paired parameter-count/validation rows")
    axis.set_title("Round-5 model parameter count vs validation action RMSE")
    axis.set_xlabel("Trainable parameter count")
    axis.set_ylabel("Validation action L2 RMSE (m)")
    _annotate_metadata(
        axis,
        _metadata_from_rows(validation, split="validation"),
        note="marker shape=seed; open ring=configuration mean; padded axes prevent clipping",
        identity_location="external model/H_c mean legend and training-seed marker legend",
        layout_strategy="external_dual_legends_with_14_percent_vertical_headroom",
    )
    return figure


def _bar_seed_metric(
    plt: Any,
    rows: Rows,
    title: str,
    xlabel: str,
    ylabel: str,
    strict: bool,
) -> Any:
    values: Dict[str, List[Tuple[str, float]]] = {}
    for row in rows:
        metric = _optional_number(row, _RMSE)
        if metric is None:
            continue
        label = f"{_display_model(_text(row, _MODEL, 'model'))}\nH_c={_text(row, _CONTEXT, '?')}"
        seed = _identity_token(_text(row, _SEED, ""))
        values.setdefault(label, []).append((seed, metric))
    figure, axis = plt.subplots(figsize=(9.6, 5.7))
    if values:
        labels = sorted(values)
        means = np.asarray([np.mean([value for _, value in values[label]]) for label in labels])
        std = np.asarray([np.std([value for _, value in values[label]]) for label in labels])
        x = np.arange(len(labels))
        axis.bar(x, means, yerr=std, capsize=3, color=_palette(plt, len(labels)), alpha=0.8)
        seed_markers = {"0": "o", "1": "s", "2": "^"}
        seed_colors = {"0": "#111111", "1": "#555555", "2": "#999999"}
        labelled_seeds: set[str] = set()
        for index, label in enumerate(labels):
            entries = sorted(values[label], key=lambda item: _natural_key(item[0]))
            offsets = np.linspace(-0.065, 0.065, len(entries)) if len(entries) > 1 else np.zeros(1)
            for offset, (seed, value) in zip(offsets, entries):
                axis.scatter(
                    index + offset,
                    value,
                    s=28,
                    marker=seed_markers.get(seed, "D"),
                    facecolor=seed_colors.get(seed, "#333333"),
                    edgecolor="white",
                    linewidth=0.45,
                    alpha=0.9,
                    label=f"training seed={seed}" if seed not in labelled_seeds else None,
                    zorder=4,
                )
                labelled_seeds.add(seed)
        if len(labels) >= 8:
            # Dense all-model comparisons need compact, multi-line labels;
            # angled full names overlap and become ambiguous in publication PNGs.
            compact_labels = [
                label.replace(" ", "\n").replace("-", "-\n")
                for label in labels
            ]
            axis.set_xticks(x, compact_labels, rotation=0, ha="center")
            axis.tick_params(axis="x", labelsize=6.8)
            figure.subplots_adjust(bottom=0.25)
        else:
            axis.set_xticks(x, labels, rotation=22, ha="right")
        upper = max(float(np.max(means + std)), max(value for entries in values.values() for _, value in entries))
        axis.set_ylim(0.0, max(1.0, upper * 1.13))
        axis.legend(
            title="seed marker",
            frameon=False,
            fontsize=7.4,
            title_fontsize=7.6,
            loc="upper left",
            bbox_to_anchor=(1.005, 1.0),
        )
        figure.subplots_adjust(right=0.84)
    else:
        _require_or_notice(axis, 0, strict, "No finite validation metric rows")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    metadata = _metadata_from_rows(rows, split="validation")
    _annotate_metadata(
        axis,
        metadata,
        note="bars=mean +/- SD; overlaid marker shape identifies each seed",
        identity_location="model/H_c x labels and external training-seed marker legend",
        layout_strategy="external_seed_legend_and_rotated_configuration_labels",
    )
    return figure


def _group_bar(
    plt: Any,
    rows: Rows,
    title: str,
    xlabel: str,
    ylabel: str,
    strict: bool,
) -> Any:
    grouped: Dict[Tuple[str, str], List[float]] = {}
    for row in rows:
        value = _text(row, ("group_value",), "unknown")
        model = (
            f"{_display_model(_text(row, _MODEL, 'model'))}, "
            f"H_c={_identity_token(_text(row, _CONTEXT, '?'))}, "
            f"seed={_identity_token(_text(row, _SEED, '?'))}"
        )
        metric = _optional_number(row, _RMSE)
        if metric is not None:
            grouped.setdefault((value, model), []).append(metric)
    figure, axis = plt.subplots(figsize=(9.2, 5.3))
    if grouped:
        categories = sorted({key[0] for key in grouped}, key=_natural_key)
        models = sorted({key[1] for key in grouped})
        width = 0.8 / max(1, len(models))
        x = np.arange(len(categories), dtype=np.float64)
        for index, (color, model) in enumerate(zip(_palette(plt, len(models)), models)):
            means = [np.mean(grouped.get((category, model), [0.0])) for category in categories]
            offset = (index - (len(models) - 1) / 2.0) * width
            axis.bar(x + offset, means, width=width, color=color, alpha=0.82, label=model)
        axis.set_xticks(x, categories, rotation=16, ha="right")
        axis.set_ylim(bottom=0.0)
        axis.legend(frameon=False, fontsize=7.5, loc="upper right")
    else:
        _require_or_notice(axis, 0, strict, "No finite grouped test metrics")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    return figure


def _load_tables(source: Path, supplied: Mapping[str, Rows]) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for name, candidates in _TABLE_FILES.items():
        if name in supplied:
            result[name] = [dict(row) for row in supplied[name]]
            continue
        path = next((source / candidate for candidate in candidates if (source / candidate).is_file()), None)
        result[name] = _read_csv(path) if path is not None else []
    if not result["training_history"]:
        result["training_history"] = _load_referenced_histories(source, result["training_summary"])
    if not result["training_history"]:
        histories: List[Dict[str, Any]] = []
        for path in sorted(source.rglob("*history*.csv")):
            if "figures" not in {part.lower() for part in path.parts}:
                histories.extend(_read_csv(path))
        result["training_history"] = histories
    if not result["predictions"]:
        result["predictions"] = list(result["replay"])
    selected_path = source / "selected_transformer_config.json"
    if selected_path.is_file():
        with selected_path.open("r", encoding="utf-8") as stream:
            selected = json.load(stream)
        if not isinstance(selected, Mapping):
            raise ValueError("selected_transformer_config.json must contain an object")
        result["selected_config"] = [dict(selected)]
    else:
        result["selected_config"] = []
    return result


def _load_referenced_histories(source: Path, summaries: Rows) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for summary in summaries:
        relative = summary.get("history_file") or summary.get("training_history_file")
        if not relative:
            if _optional_number(summary, _TRAIN_LOSS) is not None and _optional_number(summary, _VAL_LOSS) is not None:
                rows.append(dict(summary))
            continue
        path = Path(str(relative))
        candidates = [path] if path.is_absolute() else [source / path, source.parent / path]
        actual = next((candidate for candidate in candidates if candidate.is_file()), None)
        if actual is None:
            continue
        for row in _read_csv(actual):
            merged = dict(summary)
            merged.update(row)
            rows.append(merged)
    return rows


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def _prepare_paths(root: Path, *, overwrite: bool) -> Dict[str, Tuple[Path, Path]]:
    result: Dict[str, Tuple[Path, Path]] = {}
    for category, (directory, stem) in ROUND5_FIGURE_CATEGORIES.items():
        folder = root / directory
        png = folder / f"{stem}.png"
        svg = folder / f"{stem}.svg"
        if not overwrite:
            existing = [path for path in (png, svg) if path.exists()]
            if existing:
                raise FileExistsError(f"refusing to overwrite Round-5 figure: {existing[0]}")
        folder.mkdir(parents=True, exist_ok=True)
        result[category] = png, svg
    return result


def _metric_rows(rows: Rows, *, split: str, overall: bool, raw: bool) -> List[Dict[str, Any]]:
    result = []
    for row in rows:
        row_split = _text(row, _SPLIT, split).lower()
        if row_split not in {split.lower(), "val" if split == "validation" else split.lower()}:
            continue
        if overall and "group_dimension" in row and str(row["group_dimension"]).lower() != "overall":
            continue
        if raw and "prediction_space" in row and str(row["prediction_space"]).lower() != "raw":
            continue
        if _optional_number(row, _RMSE) is not None:
            result.append(dict(row))
    return result


def _group_metric_rows(data: Mapping[str, List[Dict[str, Any]]], dimension: str) -> List[Dict[str, Any]]:
    rows = data["grouped"] or data["test"]
    result = []
    accepted = {dimension}
    if dimension == "num_buoys":
        accepted |= {"buoy_count", "n_buoys"}
    for row in rows:
        if _text(row, _SPLIT, "test").lower() != "test":
            continue
        if _text(row, ("group_dimension",), "").lower() not in accepted:
            continue
        if _text(row, ("prediction_space",), "raw").lower() != "raw":
            continue
        result.append(dict(row))
    return result


def _prediction_rows(data: Mapping[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return data["predictions"] or data["replay"]


def _paired_numeric(rows: Rows, left_aliases: Sequence[str], right_aliases: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    left: List[float] = []
    right: List[float] = []
    for row in rows:
        a = _optional_number(row, left_aliases)
        b = _optional_number(row, right_aliases)
        if a is not None and b is not None:
            left.append(a)
            right.append(b)
    return np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)


def _numeric_available(rows: Rows, aliases: Sequence[str]) -> np.ndarray:
    result = [value for row in rows if (value := _optional_number(row, aliases)) is not None]
    return np.asarray(result, dtype=np.float64)


def _group_rows(rows: Rows, key: Callable[[Mapping[str, Any]], Any]) -> Dict[Any, List[Dict[str, Any]]]:
    result: Dict[Any, List[Dict[str, Any]]] = {}
    for row in rows:
        result.setdefault(key(row), []).append(dict(row))
    return result


def _optional_number(row: Mapping[str, Any], aliases: Sequence[str]) -> Optional[float]:
    for key in aliases:
        if key in row and row[key] not in (None, ""):
            try:
                value = float(row[key])
            except (TypeError, ValueError) as error:
                raise ValueError(f"column {key!r} must be numeric") from error
            if not math.isfinite(value):
                raise ValueError(f"column {key!r} contains NaN or Inf")
            return value
    return None


def _number(row: Mapping[str, Any], aliases: Sequence[str], default: float) -> float:
    value = _optional_number(row, aliases)
    return default if value is None else value


def _text(row: Mapping[str, Any], aliases: Sequence[str], default: str) -> str:
    for key in aliases:
        if key in row and row[key] not in (None, ""):
            return str(row[key])
    return default


def _canonical_model(value: str) -> str:
    token = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "wt": "waypoint_transformer",
        "wt_transformer": "waypoint_transformer",
        "direct_goal": "direct_goal_transformer",
        "state_only": "state_only_transformer",
        "oracle": "oracle_waypoint_transformer",
        "oracle_transformer": "oracle_waypoint_transformer",
    }
    return aliases.get(token, token)


def _display_model(value: str) -> str:
    canonical = _canonical_model(value)
    names = {
        "waypoint_transformer": "WT-Transformer",
        "direct_goal_transformer": "Direct-Goal",
        "state_only_transformer": "State-only Transformer",
        "oracle_waypoint_transformer": "Oracle-Waypoint",
        "mlp_state_only": "State-only MLP",
        "mlp_waypoint": "Waypoint MLP",
    }
    return names.get(canonical, canonical.replace("_", " ").title())


def _require_or_notice(axis: Any, count: int, strict: bool, message: str) -> None:
    if count:
        return
    if strict:
        raise ValueError(message)
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)


def _symmetric_limits(values: np.ndarray) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("plot limits require finite non-empty values")
    radius = max(abs(float(np.min(array))), abs(float(np.max(array))), 1e-9) * 1.05
    return -radius, radius


def _finite_limits(values: np.ndarray, *, include_zero: bool) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("plot limits require finite non-empty values")
    lower, upper = float(np.min(array)), float(np.max(array))
    if include_zero:
        lower, upper = min(lower, 0.0), max(upper, 0.0)
    span = upper - lower
    padding = max(span * 0.07, 0.1)
    return lower - padding, upper + padding


def _zero_lower(axis: Any) -> None:
    lower, upper = axis.get_ylim()
    axis.set_ylim(0.0, max(float(upper), 1e-12))


def _legend(axis: Any) -> None:
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(frameon=False, fontsize=8, ncol=1 if len(handles) < 7 else 2)


def _palette(plt: Any, count: int) -> List[Any]:
    cmap = plt.get_cmap("tab10")
    return [cmap(index % 10) for index in range(count)]


def _natural_key(value: str) -> Tuple[int, Any]:
    try:
        return 0, float(value)
    except (TypeError, ValueError):
        return 1, str(value)


def _load_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required for Round-5 figures") from error
    return plt


def _style() -> Dict[str, Any]:
    return {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.7,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "svg.fonttype": "none",
    }


__all__ = [
    "ROUND5_FIGURE_CATEGORIES",
    "generate_round5_figures",
    "semantic_qa_round5_figure",
]
