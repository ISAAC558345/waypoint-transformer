"""Stateless large-scale path-loss calculations for maritime communication."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, ClassVar, Tuple, Union

import numpy as np


MetricValue = Union[np.ndarray, float]


@dataclass(frozen=True)
class ChannelMetrics(Mapping):
    """Batch channel outputs for ``num_buoys`` static transmitters.

    The first five fields are independent ``float64`` arrays with shape
    ``[num_buoys]``. ``total_rate_bps`` is a Python ``float``. Attribute and
    read-only mapping access are both supported, for example
    ``metrics.rates_bps`` and ``metrics["rates_bps"]``.
    """

    horizontal_distances_m: np.ndarray
    distances_3d_m: np.ndarray
    channel_gains: np.ndarray
    snr_linear: np.ndarray
    rates_bps: np.ndarray
    total_rate_bps: float

    _FIELD_NAMES: ClassVar[Tuple[str, ...]] = (
        "horizontal_distances_m",
        "distances_3d_m",
        "channel_gains",
        "snr_linear",
        "rates_bps",
        "total_rate_bps",
    )

    def __getitem__(self, key: str) -> MetricValue:
        if key not in self._FIELD_NAMES:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._FIELD_NAMES)

    def __len__(self) -> int:
        return len(self._FIELD_NAMES)


def _finite_positive_scalar(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return numeric_value


def _positive_num_buoys(num_buoys: Any) -> int:
    if isinstance(num_buoys, bool) or not isinstance(num_buoys, Integral):
        raise ValueError(f"num_buoys must be an integer, got {num_buoys!r}")
    validated_num_buoys = int(num_buoys)
    if validated_num_buoys < 1:
        raise ValueError("num_buoys must be at least 1")
    return validated_num_buoys


def _position_array(
    name: str, value: Any, expected_shape: Tuple[int, ...]
) -> np.ndarray:
    try:
        raw_array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular numeric array") from error
    if np.iscomplexobj(raw_array):
        raise ValueError(f"{name} must contain real-valued coordinates")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric coordinates") from error
    if array.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite coordinates")
    return array


def _buoy_position_array(value: Any) -> np.ndarray:
    try:
        raw_array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "buoy_positions_m must be a rectangular numeric array"
        ) from error
    if raw_array.ndim != 2 or raw_array.shape[1:] != (2,):
        raise ValueError(
            "buoy_positions_m must have shape [num_buoys, 2], "
            f"got {raw_array.shape}"
        )
    num_buoys = _positive_num_buoys(raw_array.shape[0])
    return _position_array(
        "buoy_positions_m", value, (num_buoys, 2)
    )


def _channel_parameters(
    *,
    uav_altitude_m: Any,
    total_bandwidth_hz: Any,
    buoy_tx_power_w: Any,
    reference_channel_gain: Any,
    pathloss_exponent: Any,
    noise_power_w: Any,
) -> Tuple[float, float, float, float, float, float]:
    return (
        _finite_positive_scalar("uav_altitude_m", uav_altitude_m),
        _finite_positive_scalar("total_bandwidth_hz", total_bandwidth_hz),
        _finite_positive_scalar("buoy_tx_power_w", buoy_tx_power_w),
        _finite_positive_scalar(
            "reference_channel_gain", reference_channel_gain
        ),
        _finite_positive_scalar("pathloss_exponent", pathloss_exponent),
        _finite_positive_scalar("noise_power_w", noise_power_w),
    )


def _validate_metric_array(name: str, value: np.ndarray) -> None:
    if not np.all(np.isfinite(value)):
        raise ValueError(f"channel computation produced non-finite {name}")
    if np.any(value < 0.0):
        raise ValueError(f"channel computation produced negative {name}")


def compute_channel_metrics(
    uav_position_m: Any,
    buoy_positions_m: Any,
    *,
    uav_altitude_m: float,
    total_bandwidth_hz: float,
    buoy_tx_power_w: float,
    reference_channel_gain: float,
    pathloss_exponent: float,
    noise_power_w: float,
) -> ChannelMetrics:
    """Compute deterministic orthogonal buoy-to-UAV channel metrics.

    Args:
        uav_position_m: UAV horizontal coordinate with shape ``[2]``.
        buoy_positions_m: Buoy coordinates with shape ``[num_buoys, 2]``.
        uav_altitude_m: Strictly positive fixed UAV altitude.
        total_bandwidth_hz: Strictly positive total orthogonal bandwidth.
        buoy_tx_power_w: Strictly positive transmit power of every buoy.
        reference_channel_gain: Strictly positive path gain at unit distance.
        pathloss_exponent: Strictly positive large-scale path-loss exponent.
        noise_power_w: Strictly positive receiver noise power.

    Returns:
        A ``ChannelMetrics`` object. Its vector fields have shape
        ``[num_buoys]`` and contain only finite nonnegative values.

    Raises:
        ValueError: If an input has the wrong shape, is non-finite, violates a
            positivity constraint, or produces non-finite channel quantities.
    """

    uav_position = _position_array("uav_position_m", uav_position_m, (2,))
    buoy_positions = _buoy_position_array(buoy_positions_m)
    num_buoys = buoy_positions.shape[0]
    (
        altitude,
        bandwidth,
        tx_power,
        reference_gain,
        exponent,
        noise_power,
    ) = _channel_parameters(
        uav_altitude_m=uav_altitude_m,
        total_bandwidth_hz=total_bandwidth_hz,
        buoy_tx_power_w=buoy_tx_power_w,
        reference_channel_gain=reference_channel_gain,
        pathloss_exponent=pathloss_exponent,
        noise_power_w=noise_power_w,
    )

    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            horizontal_offsets = buoy_positions - uav_position[np.newaxis, :]
            horizontal_distances = np.linalg.norm(horizontal_offsets, axis=1)
            distances_3d = np.sqrt(
                np.square(horizontal_distances) + altitude * altitude
            )
            channel_gains = reference_gain / np.power(distances_3d, exponent)
            snr_linear = tx_power * channel_gains / noise_power
            bandwidth_per_buoy = bandwidth / float(num_buoys)
            rates_bps = (
                bandwidth_per_buoy
                * np.log1p(snr_linear)
                / math.log(2.0)
            )
            total_rate_bps = float(np.sum(rates_bps, dtype=np.float64))
    except FloatingPointError as error:
        raise ValueError(
            "channel inputs exceed the finite numerical range"
        ) from error

    for metric_name, metric_value in (
        ("horizontal_distances_m", horizontal_distances),
        ("distances_3d_m", distances_3d),
        ("channel_gains", channel_gains),
        ("snr_linear", snr_linear),
        ("rates_bps", rates_bps),
    ):
        _validate_metric_array(metric_name, metric_value)
    if not math.isfinite(total_rate_bps) or total_rate_bps < 0.0:
        raise ValueError(
            "channel computation produced a non-finite or negative total rate"
        )

    return ChannelMetrics(
        horizontal_distances_m=horizontal_distances.copy(),
        distances_3d_m=distances_3d.copy(),
        channel_gains=channel_gains.copy(),
        snr_linear=snr_linear.copy(),
        rates_bps=rates_bps.copy(),
        total_rate_bps=total_rate_bps,
    )


def theoretical_maximum_rates(
    *,
    num_buoys: int,
    uav_altitude_m: float,
    total_bandwidth_hz: float,
    buoy_tx_power_w: float,
    reference_channel_gain: float,
    pathloss_exponent: float,
    noise_power_w: float,
) -> Tuple[float, float]:
    """Return theoretical per-buoy and total rate maxima in bits per second.

    The calculation uses horizontal distance zero, hence three-dimensional
    distance ``uav_altitude_m``. Orthogonal allocation gives each of the
    ``num_buoys`` transmitters ``total_bandwidth_hz / num_buoys`` bandwidth.

    Returns:
        ``(single_buoy_rate_bps, total_rate_bps)`` as finite Python floats.

    Raises:
        ValueError: If a scalar is invalid or the result is not finite.
    """

    validated_num_buoys = _positive_num_buoys(num_buoys)
    (
        altitude,
        bandwidth,
        tx_power,
        reference_gain,
        exponent,
        noise_power,
    ) = _channel_parameters(
        uav_altitude_m=uav_altitude_m,
        total_bandwidth_hz=total_bandwidth_hz,
        buoy_tx_power_w=buoy_tx_power_w,
        reference_channel_gain=reference_channel_gain,
        pathloss_exponent=pathloss_exponent,
        noise_power_w=noise_power_w,
    )

    try:
        maximum_gain = reference_gain / math.pow(altitude, exponent)
        maximum_snr = tx_power * maximum_gain / noise_power
        maximum_single_rate = (
            bandwidth
            / float(validated_num_buoys)
            * math.log1p(maximum_snr)
            / math.log(2.0)
        )
        maximum_total_rate = maximum_single_rate * validated_num_buoys
    except (OverflowError, ValueError, ZeroDivisionError) as error:
        raise ValueError(
            "channel inputs exceed the finite numerical range"
        ) from error

    if (
        not math.isfinite(maximum_single_rate)
        or not math.isfinite(maximum_total_rate)
        or maximum_single_rate < 0.0
        or maximum_total_rate < 0.0
    ):
        raise ValueError("theoretical maximum rates must be finite and nonnegative")
    return float(maximum_single_rate), float(maximum_total_rate)


def theoretical_max_rates(
    *,
    num_buoys: int,
    uav_altitude_m: float,
    total_bandwidth_hz: float,
    buoy_tx_power_w: float,
    reference_channel_gain: float,
    pathloss_exponent: float,
    noise_power_w: float,
) -> Tuple[float, float]:
    """Alias returning theoretical per-buoy and aggregate rate maxima."""

    return theoretical_maximum_rates(
        num_buoys=num_buoys,
        uav_altitude_m=uav_altitude_m,
        total_bandwidth_hz=total_bandwidth_hz,
        buoy_tx_power_w=buoy_tx_power_w,
        reference_channel_gain=reference_channel_gain,
        pathloss_exponent=pathloss_exponent,
        noise_power_w=noise_power_w,
    )


def theoretical_max_single_buoy_rate_bps(
    *,
    num_buoys: int,
    uav_altitude_m: float,
    total_bandwidth_hz: float,
    buoy_tx_power_w: float,
    reference_channel_gain: float,
    pathloss_exponent: float,
    noise_power_w: float,
) -> float:
    """Return the finite theoretical maximum rate for one buoy."""

    return theoretical_maximum_rates(
        num_buoys=num_buoys,
        uav_altitude_m=uav_altitude_m,
        total_bandwidth_hz=total_bandwidth_hz,
        buoy_tx_power_w=buoy_tx_power_w,
        reference_channel_gain=reference_channel_gain,
        pathloss_exponent=pathloss_exponent,
        noise_power_w=noise_power_w,
    )[0]


def theoretical_max_total_rate_bps(
    *,
    num_buoys: int,
    uav_altitude_m: float,
    total_bandwidth_hz: float,
    buoy_tx_power_w: float,
    reference_channel_gain: float,
    pathloss_exponent: float,
    noise_power_w: float,
) -> float:
    """Return the finite theoretical maximum aggregate rate for all buoys."""

    return theoretical_maximum_rates(
        num_buoys=num_buoys,
        uav_altitude_m=uav_altitude_m,
        total_bandwidth_hz=total_bandwidth_hz,
        buoy_tx_power_w=buoy_tx_power_w,
        reference_channel_gain=reference_channel_gain,
        pathloss_exponent=pathloss_exponent,
        noise_power_w=noise_power_w,
    )[1]
