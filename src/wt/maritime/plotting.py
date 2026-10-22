"""White-background scientific figures for maritime baseline evaluation.

Queue state, task completion, and actual collected payload are absent from the
current environment.  The corresponding required queue figures are explicit
availability notices.  The only communication time series plotted is labelled
as cumulative *link-capacity proxy*, never as collected data.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Union

import numpy as np

from .metrics import UNAVAILABLE_STATUS


FIGURE_FILENAMES = {
    "episode_return": "episode_return_comparison.png",
    "final_remaining_queue": "final_remaining_queue_unavailable.png",
    "cumulative_energy": "cumulative_energy_comparison.png",
    "queue_over_time": "queue_over_time_unavailable.png",
    "cumulative_link_capacity": "cumulative_link_capacity_proxy.png",
    "uav_trajectories": "representative_uav_trajectories.png",
}


def generate_required_figures(
    episodes: Sequence[Any],
    episode_metrics: Sequence[Mapping[str, Any]],
    output_dir: Union[str, Path],
    *,
    delta_t: float,
) -> Dict[str, Path]:
    """Generate the six round-02 figures and return their paths.

    Parameters
    ----------
    episodes:
        Completed structured episode trajectories.
    episode_metrics:
        Per-episode rows from :func:`wt.maritime.metrics.compute_episode_metrics`.
    output_dir:
        A new or empty round-specific figure directory.
    delta_t:
        Communication-slot duration used to integrate rates into the clearly
        labelled link-capacity proxy.
    """

    if not episodes:
        raise ValueError("at least one episode is required for plotting")
    if not episode_metrics:
        raise ValueError("at least one episode metric row is required for plotting")
    delta_t_value = float(delta_t)
    if not math.isfinite(delta_t_value) or delta_t_value <= 0.0:
        raise ValueError("delta_t must be finite and positive")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        key: destination / filename for key, filename in FIGURE_FILENAMES.items()
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing figures: "
            + ", ".join(str(path) for path in existing)
        )

    plt = _load_pyplot()
    with plt.rc_context(_scientific_style()):
        _plot_scalar_comparison(
            plt,
            episode_metrics,
            value_key="episode_return",
            ylabel="Episode return",
            title="Non-learning baseline episode return",
            destination=paths["episode_return"],
        )
        _plot_unavailable_notice(
            plt,
            title="Final remaining buoy queue",
            requested_quantity="Buoy queue state",
            destination=paths["final_remaining_queue"],
        )
        _plot_scalar_comparison(
            plt,
            episode_metrics,
            value_key="cumulative_energy_consumed",
            ylabel="Cumulative energy consumed [energy units]",
            title="Non-learning baseline energy consumption",
            destination=paths["cumulative_energy"],
        )
        _plot_unavailable_notice(
            plt,
            title="Buoy queue over time",
            requested_quantity="Per-buoy queue time series",
            destination=paths["queue_over_time"],
        )
        _plot_cumulative_capacity(
            plt,
            episodes,
            delta_t=delta_t_value,
            destination=paths["cumulative_link_capacity"],
        )
        _plot_representative_trajectories(
            plt,
            episodes,
            destination=paths["uav_trajectories"],
        )
    return paths


def _load_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "matplotlib is required to generate round-02 figures"
        ) from error
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
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "lines.linewidth": 1.4,
    }


def _policy_order(rows: Iterable[Mapping[str, Any]]) -> List[str]:
    return sorted({str(row["policy_id"]) for row in rows})


def _palette(plt: Any, count: int) -> List[Any]:
    colormap = plt.get_cmap("tab10")
    return [colormap(index % 10) for index in range(count)]


def _plot_scalar_comparison(
    plt: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    ylabel: str,
    title: str,
    destination: Path,
) -> None:
    policies = _policy_order(rows)
    colors = _palette(plt, len(policies))
    figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)

    for index, (policy, color) in enumerate(zip(policies, colors)):
        values = np.asarray(
            [
                float(row[value_key])
                for row in rows
                if str(row["policy_id"]) == policy
            ],
            dtype=np.float64,
        )
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid {value_key} values for policy {policy}")
        jitter = (
            np.linspace(-0.12, 0.12, num=values.size)
            if values.size > 1
            else np.zeros(1)
        )
        axis.scatter(
            index + jitter,
            values,
            s=18,
            color=color,
            alpha=0.38,
            edgecolors="none",
            label=f"{policy} episodes" if index == 0 else None,
            zorder=2,
        )
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=0))
        axis.errorbar(
            index,
            mean,
            yerr=std,
            fmt="o",
            markersize=6,
            capsize=4,
            color=color,
            markeredgecolor="#222222",
            markeredgewidth=0.5,
            linewidth=2.1,
            zorder=3,
        )

    axis.set_xticks(np.arange(len(policies)))
    axis.set_xticklabels(policies, rotation=18, ha="right")
    axis.set_xlabel("Policy")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.text(
        0.99,
        0.02,
        "Pale points: episodes; dark markers: mean +/- population SD",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#555555",
    )
    _save_close(plt, figure, destination)


def _plot_unavailable_notice(
    plt: Any,
    *,
    title: str,
    requested_quantity: str,
    destination: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    axis.set_title(title)
    axis.set_axis_off()
    axis.text(
        0.5,
        0.58,
        "DATA UNAVAILABLE",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#9A3A2A",
    )
    axis.text(
        0.5,
        0.42,
        f"{requested_quantity}: {UNAVAILABLE_STATUS}.\n"
        "No queue, service-completion, or payload values were fabricated.",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="#333333",
    )
    _save_close(plt, figure, destination)


def _plot_cumulative_capacity(
    plt: Any,
    episodes: Sequence[Any],
    *,
    delta_t: float,
    destination: Path,
) -> None:
    policies = sorted({str(_field(episode, "policy_id")) for episode in episodes})
    colors = _palette(plt, len(policies))
    figure, axis = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)

    for policy, color in zip(policies, colors):
        policy_episodes = [
            episode
            for episode in episodes
            if str(_field(episode, "policy_id")) == policy
        ]
        curves: List[np.ndarray] = []
        for episode in policy_episodes:
            rates = np.asarray(
                _field(episode, "next_total_rate_bps"), dtype=np.float64
            )
            if rates.ndim != 1 or not np.all(np.isfinite(rates)):
                raise ValueError(
                    "next_total_rate_bps must be a finite 1-D series"
                )
            curve = np.concatenate(
                (np.zeros(1), np.cumsum(rates * delta_t))
            )
            curves.append(curve)
            axis.plot(
                np.arange(curve.size),
                curve,
                color=color,
                alpha=0.22,
                linewidth=0.8,
            )

        maximum_length = max(curve.size for curve in curves)
        padded = np.full((len(curves), maximum_length), np.nan, dtype=np.float64)
        for row_index, curve in enumerate(curves):
            padded[row_index, : curve.size] = curve
        mean_curve = np.nanmean(padded, axis=0)
        axis.plot(
            np.arange(maximum_length),
            mean_curve,
            color=color,
            linewidth=2.4,
            label=f"{policy} mean",
        )

    axis.set_xlabel("Environment step")
    axis.set_ylabel("Cumulative link-capacity proxy [bit]")
    axis.set_title(
        "Potential link capacity over time (not actual collected payload)\n"
        + _episode_context_label(episodes)
    )
    axis.legend(loc="best", frameon=True)
    axis.text(
        0.99,
        0.02,
        "Pale lines: episodes; dark lines: per-policy mean",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#555555",
    )
    _save_close(plt, figure, destination)


def _plot_representative_trajectories(
    plt: Any,
    episodes: Sequence[Any],
    *,
    destination: Path,
) -> None:
    representatives: Dict[str, Any] = {}
    for episode in sorted(
        episodes,
        key=lambda item: (
            str(_field(item, "policy_id")),
            int(_field(item, "seed")),
            int(_field(item, "num_buoys")),
            str(_field(item, "episode_id")),
        ),
    ):
        representatives.setdefault(str(_field(episode, "policy_id")), episode)

    policies = sorted(representatives)
    column_count = min(2, len(policies))
    row_count = int(math.ceil(len(policies) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(6.4 * column_count, 5.0 * row_count),
        squeeze=False,
        constrained_layout=True,
    )
    colors = _palette(plt, len(policies))

    for axis, policy, color in zip(axes.flat, policies, colors):
        episode = representatives[policy]
        positions = np.asarray(
            _field(episode, "uav_positions_m"), dtype=np.float64
        )
        next_positions = np.asarray(
            _field(episode, "next_uav_positions_m"), dtype=np.float64
        )
        buoy_positions = np.asarray(
            _field(episode, "buoy_positions_m"), dtype=np.float64
        )
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("uav_positions_m must have shape (T, 2)")
        if next_positions.shape != positions.shape:
            raise ValueError("next_uav_positions_m shape must match positions")
        if buoy_positions.ndim == 3:
            static_buoys = buoy_positions[0]
        elif buoy_positions.ndim == 2:
            static_buoys = buoy_positions
        else:
            raise ValueError("buoy_positions_m must have shape (K, 2) or (T, K, 2)")
        path = np.vstack((positions[0], next_positions))
        axis.plot(
            path[:, 0],
            path[:, 1],
            color=color,
            linewidth=1.8,
            label="UAV path",
            zorder=2,
        )
        axis.scatter(
            path[0, 0],
            path[0, 1],
            s=55,
            marker="o",
            color="#2A7F62",
            edgecolor="white",
            linewidth=0.7,
            label="Start",
            zorder=4,
        )
        axis.scatter(
            path[-1, 0],
            path[-1, 1],
            s=60,
            marker="X",
            color="#B43C39",
            edgecolor="white",
            linewidth=0.7,
            label="End",
            zorder=4,
        )
        axis.scatter(
            static_buoys[:, 0],
            static_buoys[:, 1],
            s=55,
            marker="^",
            color="#3366A3",
            edgecolor="white",
            linewidth=0.7,
            label="Static buoy",
            zorder=3,
        )
        all_points = np.vstack((path, static_buoys))
        x_center = float(
            0.5 * (np.min(all_points[:, 0]) + np.max(all_points[:, 0]))
        )
        y_center = float(
            0.5 * (np.min(all_points[:, 1]) + np.max(all_points[:, 1]))
        )
        square_span = max(
            float(np.ptp(all_points[:, 0])),
            float(np.ptp(all_points[:, 1])),
            1.0,
        )
        half_extent = 0.58 * square_span
        axis.set_xlim(x_center - half_extent, x_center + half_extent)
        axis.set_ylim(y_center - half_extent, y_center + half_extent)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x position [m]")
        axis.set_ylabel("y position [m]")
        axis.set_title(
            f"{policy}: seed={_field(episode, 'seed')}, "
            f"scenario={_field(episode, 'scenario_id')}, "
            f"buoys={_field(episode, 'num_buoys')}"
        )
        axis.legend(loc="best", frameon=True)

    for unused_axis in list(axes.flat)[len(policies) :]:
        unused_axis.set_axis_off()
    figure.suptitle("Representative single-UAV trajectories", fontsize=12)
    _save_close(plt, figure, destination)


def _buoy_count_label(episodes: Sequence[Any]) -> str:
    values = sorted({int(_field(episode, "num_buoys")) for episode in episodes})
    return ",".join(str(value) for value in values)


def _episode_context_label(episodes: Sequence[Any]) -> str:
    scenarios = sorted(
        {str(_field(episode, "scenario_id")) for episode in episodes}
    )
    seeds = sorted({int(_field(episode, "seed")) for episode in episodes})
    return (
        f"scenarios={','.join(scenarios)}; "
        f"seeds={','.join(str(seed) for seed in seeds)}; "
        f"buoy counts={_buoy_count_label(episodes)}"
    )


def _field(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        if name in record:
            return record[name]
    elif hasattr(record, name):
        return getattr(record, name)
    raise KeyError(f"episode record is missing required field {name!r}")


def _save_close(plt: Any, figure: Any, destination: Path) -> None:
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise RuntimeError(f"failed to generate figure: {destination}")
