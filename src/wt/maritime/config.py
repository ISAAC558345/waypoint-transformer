"""Configuration for the minimal single-UAV maritime communication task."""

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, Optional, Sequence, Tuple


def _require_finite_real(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return numeric_value


def _require_positive(name: str, value: Any) -> float:
    numeric_value = _require_finite_real(name, value)
    if numeric_value <= 0.0:
        raise ValueError(f"{name} must be greater than zero, got {value!r}")
    return numeric_value


def _require_nonnegative(name: str, value: Any) -> float:
    numeric_value = _require_finite_real(name, value)
    if numeric_value < 0.0:
        raise ValueError(f"{name} must be nonnegative, got {value!r}")
    return numeric_value


def _require_positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    numeric_value = int(value)
    if numeric_value < 1:
        raise ValueError(f"{name} must be at least 1, got {value!r}")
    return numeric_value


@dataclass(frozen=True)
class MaritimeEnvConfig:
    """Immutable parameters for the Stage 1A maritime environment.

    Coordinates and lengths are measured in metres, time in seconds, energy in
    abstract nonnegative units, bandwidth in hertz, and powers in watts. This
    class validates configuration but never clips or silently repairs values.
    """

    area_width: float = 200.0
    area_height: float = 200.0

    num_buoys: int = 4
    episode_horizon: int = 20
    delta_t: float = 1.0

    uav_altitude: float = 50.0
    max_speed: float = 10.0

    initial_uav_x: float = 100.0
    initial_uav_y: float = 100.0

    # Round-18 multi-UAV extension.  The default M=1 and None positions leave
    # the completed single-UAV initialization path exactly unchanged.
    num_uavs: int = 1
    initial_uav_positions_m: Optional[Tuple[Tuple[float, float], ...]] = None
    multi_uav_min_separation_m: float = 15.0

    initial_energy: float = 100.0
    hover_energy_per_step: float = 0.1
    move_energy_per_meter: float = 0.01

    total_bandwidth_hz: float = 1.0e6
    buoy_tx_power_w: float = 0.1
    reference_channel_gain: float = 1.0e-3
    pathloss_exponent: float = 2.2
    noise_power_w: float = 1.0e-9

    rate_reward_weight: float = 1.0
    movement_penalty_weight: float = 0.05
    boundary_penalty: float = 0.5
    energy_depletion_penalty: float = 1.0

    randomize_buoys: bool = False
    buoy_margin: float = 20.0
    seed: int = 0

    # Round-16 observable buoy-drift extension. The default static mode is
    # intentionally inert so completed static rounds retain their behavior.
    buoy_drift_mode: str = "static"
    buoy_drift_speed_fraction: float = 0.0
    buoy_drift_rho: float = 0.95
    buoy_drift_innovation_fraction: float = 0.01

    # Round-17 predictable slow-channel extension. Static is deliberately
    # inert so all completed static-channel behavior is reproducible.
    channel_variation_mode: str = "static"
    channel_log_amplitude: float = 0.0
    channel_period_slots: int = 24
    channel_frequency_normalization_max_rad_per_slot: float = 0.3141592653589793
    channel_initial_phase_by_buoy: Optional[Tuple[float, ...]] = None

    # Round-3 task semantics are opt-in so the completed round-01/02
    # environment and datasets remain exactly reproducible.
    task_semantics_version: str = "v0"
    queue_capacity_bits: float = 5.0e6
    initial_queue_bits: float = 2.5e6
    initial_queue_bits_by_buoy: Optional[Tuple[float, ...]] = None
    arrival_mode: str = "constant"
    constant_arrival_bits_per_step: float = 4.5e5
    poisson_arrival_packets_per_second: float = 450.0
    arrival_packet_size_bits: float = 1.0e3
    arrival_trace_bits: Optional[Tuple[Tuple[float, ...], ...]] = None
    arrival_trace_id: Optional[str] = None

    payload_reward_weight: float = 1.0
    energy_penalty_weight: float = 0.05
    invalid_action_penalty: float = 0.2
    unfinished_queue_penalty_weight: float = 0.1
    task_success_bonus: float = 1.0
    target_task_completion_ratio: float = 0.90
    maximum_allowed_residual_queue_ratio: float = 0.10
    minimum_terminal_energy_ratio: float = 0.10
    terminate_on_task_success: bool = True

    def __post_init__(self) -> None:
        """Reject invalid parameters without mutating this frozen instance."""

        area_width = _require_positive("area_width", self.area_width)
        area_height = _require_positive("area_height", self.area_height)
        _require_positive_integer("num_buoys", self.num_buoys)
        _require_positive_integer("episode_horizon", self.episode_horizon)
        _require_positive("delta_t", self.delta_t)
        _require_positive("uav_altitude", self.uav_altitude)
        _require_positive("max_speed", self.max_speed)

        initial_x = _require_finite_real("initial_uav_x", self.initial_uav_x)
        initial_y = _require_finite_real("initial_uav_y", self.initial_uav_y)
        if not 0.0 <= initial_x <= area_width:
            raise ValueError(
                "initial_uav_x must lie in the inclusive interval "
                f"[0, area_width], got {self.initial_uav_x!r}"
            )
        if not 0.0 <= initial_y <= area_height:
            raise ValueError(
                "initial_uav_y must lie in the inclusive interval "
                f"[0, area_height], got {self.initial_uav_y!r}"
            )

        num_uavs = _require_positive_integer("num_uavs", self.num_uavs)
        _require_nonnegative(
            "multi_uav_min_separation_m", self.multi_uav_min_separation_m
        )
        initial_positions = _canonical_optional_positions(
            "initial_uav_positions_m",
            self.initial_uav_positions_m,
            num_uavs,
            area_width,
            area_height,
        )
        if initial_positions is not None:
            object.__setattr__(self, "initial_uav_positions_m", initial_positions)

        for name in (
            "initial_energy",
            "hover_energy_per_step",
            "move_energy_per_meter",
            "rate_reward_weight",
            "movement_penalty_weight",
            "boundary_penalty",
            "energy_depletion_penalty",
        ):
            _require_nonnegative(name, getattr(self, name))

        for name in (
            "total_bandwidth_hz",
            "buoy_tx_power_w",
            "reference_channel_gain",
            "pathloss_exponent",
            "noise_power_w",
        ):
            _require_positive(name, getattr(self, name))

        if not isinstance(self.randomize_buoys, bool):
            raise ValueError(
                "randomize_buoys must be a bool, "
                f"got {self.randomize_buoys!r}"
            )

        margin = _require_nonnegative("buoy_margin", self.buoy_margin)
        if 2.0 * margin > area_width or 2.0 * margin > area_height:
            raise ValueError(
                "buoy_margin leaves no valid placement interval: it must be "
                "at most half of both area dimensions"
            )

        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise ValueError(f"seed must be an integer, got {self.seed!r}")
        if int(self.seed) < 0:
            raise ValueError(f"seed must be nonnegative, got {self.seed!r}")

        if self.buoy_drift_mode not in (
            "static", "constant", "slowly_varying"
        ):
            raise ValueError(
                "buoy_drift_mode must be 'static', 'constant', or "
                "'slowly_varying'"
            )
        drift_fraction = _require_nonnegative(
            "buoy_drift_speed_fraction", self.buoy_drift_speed_fraction
        )
        if drift_fraction > 1.0:
            raise ValueError("buoy_drift_speed_fraction must not exceed 1")
        rho = _require_finite_real("buoy_drift_rho", self.buoy_drift_rho)
        if not 0.0 <= rho <= 1.0:
            raise ValueError("buoy_drift_rho must lie in [0, 1]")
        innovation = _require_nonnegative(
            "buoy_drift_innovation_fraction",
            self.buoy_drift_innovation_fraction,
        )
        if innovation > 1.0:
            raise ValueError(
                "buoy_drift_innovation_fraction must not exceed 1"
            )
        if self.buoy_drift_mode == "static" and drift_fraction != 0.0:
            raise ValueError(
                "static buoy_drift_mode requires buoy_drift_speed_fraction=0"
            )
        if self.buoy_drift_mode != "static" and drift_fraction <= 0.0:
            raise ValueError(
                "dynamic buoy_drift_mode requires positive "
                "buoy_drift_speed_fraction"
            )
        if self.buoy_drift_mode != "static" and self.task_semantics_version != "v1":
            raise ValueError(
                "dynamic buoy drift requires task_semantics_version='v1' so "
                "current velocity is observable"
            )

        if self.channel_variation_mode not in ("static", "slow_oscillation"):
            raise ValueError(
                "channel_variation_mode must be 'static' or 'slow_oscillation'"
            )
        channel_amplitude = _require_nonnegative(
            "channel_log_amplitude", self.channel_log_amplitude
        )
        if channel_amplitude > 1.0:
            raise ValueError("channel_log_amplitude must not exceed 1")
        _require_positive_integer("channel_period_slots", self.channel_period_slots)
        _require_positive(
            "channel_frequency_normalization_max_rad_per_slot",
            self.channel_frequency_normalization_max_rad_per_slot,
        )
        channel_phases = _canonical_optional_finite_vector(
            "channel_initial_phase_by_buoy",
            self.channel_initial_phase_by_buoy,
            int(self.num_buoys),
        )
        if self.channel_variation_mode == "static" and channel_amplitude != 0.0:
            raise ValueError(
                "static channel_variation_mode requires channel_log_amplitude=0"
            )
        if self.channel_variation_mode != "static" and channel_amplitude <= 0.0:
            raise ValueError(
                "slow_oscillation requires positive channel_log_amplitude"
            )
        if self.channel_variation_mode != "static" and self.task_semantics_version != "v1":
            raise ValueError(
                "dynamic channel variation requires task_semantics_version='v1' "
                "so current oscillator state is observable"
            )
        if channel_phases is not None and not all(math.isfinite(value) for value in channel_phases):
            raise ValueError("channel_initial_phase_by_buoy must contain finite values")

        if self.task_semantics_version not in ("v0", "v1"):
            raise ValueError(
                "task_semantics_version must be either 'v0' or 'v1'"
            )
        queue_capacity = _require_positive(
            "queue_capacity_bits", self.queue_capacity_bits
        )
        initial_queue = _require_nonnegative(
            "initial_queue_bits", self.initial_queue_bits
        )
        if initial_queue > queue_capacity:
            raise ValueError(
                "initial_queue_bits must not exceed queue_capacity_bits"
            )
        initial_by_buoy = _canonical_optional_vector(
            "initial_queue_bits_by_buoy",
            self.initial_queue_bits_by_buoy,
            int(self.num_buoys),
        )
        if initial_by_buoy is not None:
            if any(value > queue_capacity for value in initial_by_buoy):
                raise ValueError(
                    "initial_queue_bits_by_buoy values must not exceed "
                    "queue_capacity_bits"
                )
            object.__setattr__(
                self, "initial_queue_bits_by_buoy", initial_by_buoy
            )

        if self.arrival_mode not in ("constant", "poisson", "trace"):
            raise ValueError(
                "arrival_mode must be 'constant', 'poisson', or 'trace'"
            )
        _require_nonnegative(
            "constant_arrival_bits_per_step",
            self.constant_arrival_bits_per_step,
        )
        _require_nonnegative(
            "poisson_arrival_packets_per_second",
            self.poisson_arrival_packets_per_second,
        )
        _require_positive(
            "arrival_packet_size_bits", self.arrival_packet_size_bits
        )
        trace = _canonical_optional_trace(
            self.arrival_trace_bits,
            int(self.episode_horizon),
            int(self.num_buoys),
        )
        if trace is not None:
            object.__setattr__(self, "arrival_trace_bits", trace)
        if self.arrival_mode == "trace" and trace is None:
            raise ValueError(
                "arrival_mode='trace' requires arrival_trace_bits"
            )
        if self.arrival_trace_id is not None:
            if (
                not isinstance(self.arrival_trace_id, str)
                or not self.arrival_trace_id.strip()
            ):
                raise ValueError(
                    "arrival_trace_id must be a non-empty string or None"
                )

        for name in (
            "payload_reward_weight",
            "energy_penalty_weight",
            "invalid_action_penalty",
            "unfinished_queue_penalty_weight",
            "task_success_bonus",
        ):
            _require_nonnegative(name, getattr(self, name))
        for name in (
            "target_task_completion_ratio",
            "maximum_allowed_residual_queue_ratio",
            "minimum_terminal_energy_ratio",
        ):
            value = _require_finite_real(name, getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not isinstance(self.terminate_on_task_success, bool):
            raise ValueError("terminate_on_task_success must be a bool")


def _canonical_optional_vector(
    name: str,
    values: Optional[Sequence[float]],
    expected_length: int,
) -> Optional[Tuple[float, ...]]:
    if values is None:
        return None
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence or None")
    if len(values) != expected_length:
        raise ValueError(
            f"{name} must contain {expected_length} values, got {len(values)}"
        )
    return tuple(_require_nonnegative(name, value) for value in values)


def _canonical_optional_positions(
    name: str,
    values: Optional[Sequence[Sequence[float]]],
    expected_length: int,
    area_width: float,
    area_height: float,
) -> Optional[Tuple[Tuple[float, float], ...]]:
    """Validate optional per-UAV map positions without implicit clipping."""
    if values is None:
        return None
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence or None")
    if len(values) != expected_length:
        raise ValueError(
            f"{name} must contain {expected_length} positions, got {len(values)}"
        )
    result = []
    for index, position in enumerate(values):
        if isinstance(position, (str, bytes)) or not isinstance(position, Sequence):
            raise ValueError(f"{name}[{index}] must be a length-two sequence")
        if len(position) != 2:
            raise ValueError(f"{name}[{index}] must contain exactly two coordinates")
        x = _require_finite_real(f"{name}[{index}][0]", position[0])
        y = _require_finite_real(f"{name}[{index}][1]", position[1])
        if not 0.0 <= x <= area_width or not 0.0 <= y <= area_height:
            raise ValueError(f"{name}[{index}] must lie within the map bounds")
        result.append((x, y))
    return tuple(result)


def _canonical_optional_finite_vector(
    name: str,
    values: Optional[Sequence[float]],
    expected_length: int,
) -> Optional[Tuple[float, ...]]:
    """Validate optional signed finite state values without queue semantics."""
    if values is None:
        return None
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence or None")
    if len(values) != expected_length:
        raise ValueError(
            f"{name} must contain {expected_length} values, got {len(values)}"
        )
    return tuple(_require_finite_real(name, value) for value in values)


def _canonical_optional_trace(
    values: Optional[Sequence[Sequence[float]]],
    expected_steps: int,
    expected_nodes: int,
) -> Optional[Tuple[Tuple[float, ...], ...]]:
    if values is None:
        return None
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("arrival_trace_bits must be a rank-2 sequence or None")
    if len(values) != expected_steps:
        raise ValueError(
            "arrival_trace_bits must have episode_horizon rows, "
            f"expected {expected_steps}, got {len(values)}"
        )
    rows = []
    for row in values:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise ValueError("each arrival_trace_bits row must be a sequence")
        if len(row) != expected_nodes:
            raise ValueError(
                "each arrival_trace_bits row must contain num_buoys values"
            )
        rows.append(
            tuple(
                _require_nonnegative("arrival_trace_bits", value)
                for value in row
            )
        )
    return tuple(rows)
