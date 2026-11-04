"""Link test for M0A. The threshold and airtime stay on the M0 radio contract."""
from __future__ import annotations

import math

from wt.ocean_observation_m0.parameters import (
    BUOY_ANTENNA_M,
    BUOY_GAIN_DBI,
    FLIGHT_ALTITUDE_M,
    SHORE_ANTENNA_M,
    SHORE_GAIN_DBI,
    UAV_GAIN_DBI,
)
from wt.ocean_observation_m0.radio import evaluate_link, lora_time_on_air_s, slot_s
from wt.ocean_observation_m0a.horizon import PRIMARY_K, geometric_horizon_m

Point = tuple[float, float]


def slant_m(a: Point, za: float, b: Point, zb: float) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (za - zb) ** 2)


def link_at(distance_m: float, height_tx_m: float, height_rx_m: float, tx_gain_dbi: float, rx_gain_dbi: float, payload_bytes: int, k: float) -> dict:
    row = evaluate_link(distance_m, height_tx_m, height_rx_m, tx_gain_dbi, rx_gain_dbi, payload_bytes)
    horizon = geometric_horizon_m(height_tx_m, height_rx_m, k)
    beyond = distance_m > horizon
    closes = bool(row["margin_db"] is not None and row["margin_db"] >= 0.0 and not beyond and distance_m > 0.0)
    if distance_m <= 0.0:
        reason = "non_positive_range"
    elif beyond:
        reason = "beyond_radio_horizon"
    elif row["margin_db"] < 0.0:
        reason = "below_threshold"
    else:
        reason = "closed"
    row["historical_horizon_m"] = row["horizon_m"]
    row["horizon_m"] = horizon
    row["k"] = float(k)
    row["closes"] = closes
    row["reason"] = reason
    return row


def buoy_to_shore(buoy: Point, payload_bytes: int, k: float = PRIMARY_K) -> dict:
    return link_at(
        slant_m(buoy, BUOY_ANTENNA_M, (0.0, 0.0), SHORE_ANTENNA_M),
        BUOY_ANTENNA_M,
        SHORE_ANTENNA_M,
        BUOY_GAIN_DBI,
        SHORE_GAIN_DBI,
        payload_bytes,
        k,
    )


def buoy_to_uav(buoy: Point, uav: Point, payload_bytes: int, k: float = PRIMARY_K) -> dict:
    return link_at(
        slant_m(buoy, BUOY_ANTENNA_M, uav, FLIGHT_ALTITUDE_M),
        BUOY_ANTENNA_M,
        FLIGHT_ALTITUDE_M,
        BUOY_GAIN_DBI,
        UAV_GAIN_DBI,
        payload_bytes,
        k,
    )


def uav_to_shore(uav: Point, payload_bytes: int, k: float = PRIMARY_K) -> dict:
    return link_at(
        slant_m(uav, FLIGHT_ALTITUDE_M, (0.0, 0.0), SHORE_ANTENNA_M),
        FLIGHT_ALTITUDE_M,
        SHORE_ANTENNA_M,
        UAV_GAIN_DBI,
        SHORE_GAIN_DBI,
        payload_bytes,
        k,
    )


def service_slot_s(payload_bytes: int) -> float:
    return slot_s(payload_bytes)


def product_airtime_s(payload_bytes: int) -> float:
    return lora_time_on_air_s(payload_bytes)
