"""One LoRa link budget shared by buoy-UAV, UAV-shore and buoy-shore.

Success is a threshold test. There is no second packet-error factor on the rate.
"""

from __future__ import annotations

import math
from typing import Optional

from wt.ocean_observation_m0.parameters import (
    IMPLEMENTATION_MARGIN_DB,
    MAC_GUARD_S,
    RADIO_BW_HZ,
    RADIO_CR,
    RADIO_HZ,
    RADIO_PREAMBLE,
    RADIO_SF,
    RX_CURRENT_A,
    SENSITIVITY_DBM,
    SUPPLY_V,
    TX_CURRENT_A,
    TX_DBM,
)

C_M_S = 299792458.0


def dbm_to_w(dbm: float) -> float:
    return 10.0 ** ((dbm - 30.0) / 10.0)


def w_to_dbm(watts: float) -> float:
    if watts <= 0.0:
        raise ValueError("power must be positive")
    return 10.0 * math.log10(watts) + 30.0


def j_to_wh(joules: float) -> float:
    return joules / 3600.0


def tx_power_w() -> float:
    return SUPPLY_V * TX_CURRENT_A


def rx_power_w() -> float:
    return SUPPLY_V * RX_CURRENT_A


def fspl_db(distance_m: float) -> float:
    if distance_m <= 0.0:
        raise ValueError("distance must be positive")
    return 20.0 * math.log10(4.0 * math.pi * distance_m * RADIO_HZ / C_M_S)


def received_dbm(distance_m: float, tx_gain_dbi: float, rx_gain_dbi: float) -> float:
    return TX_DBM + tx_gain_dbi + rx_gain_dbi - fspl_db(distance_m)


def radio_horizon_m(height_a_m: float, height_b_m: float) -> float:
    """4/3-earth approximation, d_km = 3.57 * (sqrt(h1) + sqrt(h2))."""
    if height_a_m < 0.0 or height_b_m < 0.0:
        raise ValueError("height")
    return 3570.0 * (math.sqrt(height_a_m) + math.sqrt(height_b_m))


def lora_time_on_air_s(payload_bytes: int) -> float:
    """SX1276 payload-symbol formula. CR is the datasheet code 1..4 for 4/5..4/8."""
    if payload_bytes <= 0:
        raise ValueError("payload")
    sf = RADIO_SF
    de = 1 if sf >= 11 else 0
    crc_on = 1
    implicit = 0
    tsym = (2 ** sf) / RADIO_BW_HZ
    numerator = 8 * payload_bytes - 4 * sf + 28 + 16 * crc_on - 20 * implicit
    denominator = 4 * (sf - 2 * de)
    payload_symbols = 8 + max(math.ceil(numerator / denominator) * (RADIO_CR + 4), 0)
    return (RADIO_PREAMBLE + 4.25 + payload_symbols) * tsym


def slot_s(payload_bytes: int) -> float:
    return lora_time_on_air_s(payload_bytes) + MAC_GUARD_S


def evaluate_link(
    distance_m: float,
    height_tx_m: float,
    height_rx_m: float,
    tx_gain_dbi: float,
    rx_gain_dbi: float,
    payload_bytes: int,
) -> dict:
    horizon = radio_horizon_m(height_tx_m, height_rx_m)
    airtime = lora_time_on_air_s(payload_bytes)
    beyond_horizon = distance_m > horizon
    if distance_m <= 0.0:
        pr = None
        margin = None
        closes = False
        reason = "non_positive_range"
    else:
        pr = received_dbm(distance_m, tx_gain_dbi, rx_gain_dbi)
        margin = pr - (SENSITIVITY_DBM + IMPLEMENTATION_MARGIN_DB)
        closes = (not beyond_horizon) and margin >= 0.0
        if beyond_horizon:
            reason = "beyond_radio_horizon"
        elif margin < 0.0:
            reason = "below_threshold"
        else:
            reason = "closed"
    bits = payload_bytes * 8
    return {
        "distance_m": distance_m,
        "horizon_m": horizon,
        "fspl_db": None if distance_m <= 0.0 else fspl_db(distance_m),
        "received_dbm": pr,
        "threshold_dbm": SENSITIVITY_DBM + IMPLEMENTATION_MARGIN_DB,
        "margin_db": margin,
        "airtime_s": airtime,
        "payload_bytes": payload_bytes,
        "payload_bits": bits,
        "rate_bps": bits / airtime,
        "packet_error_model": None,
        "tx_energy_J": tx_power_w() * airtime,
        "rx_energy_J": rx_power_w() * airtime,
        "closes": closes,
        "reason": reason,
        "tx_dbm": TX_DBM,
        "bandwidth_hz": RADIO_BW_HZ,
        "sf": RADIO_SF,
    }


def range_for_zero_margin_m(tx_gain_dbi: float, rx_gain_dbi: float) -> Optional[float]:
    """Friis distance where the implementation-margin threshold is met. Horizon is separate."""
    allowance_db = TX_DBM + tx_gain_dbi + rx_gain_dbi - (SENSITIVITY_DBM + IMPLEMENTATION_MARGIN_DB)
    if allowance_db <= 0.0:
        return 0.0
    # fspl = 20 log10(4 pi d f / c) = allowance
    ratio = 10.0 ** (allowance_db / 20.0)
    return ratio * C_M_S / (4.0 * math.pi * RADIO_HZ)
