"""Frozen M0 contracts.

Numeric case outcomes are not inputs to this module. Cruise airspeed is chosen
later by minimizing zero-wind joules per ground metre on a fixed grid.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

Point = Tuple[float, float]

DESIGN_BASELINE = "76bdd2dd17a22c85626909731eb0319a33baaf84"
FORMAL_SOURCE_COMMIT = "e58e5d190109e5a0e860cb5c6a412321c0d74e31"
HANDOFF_COMMIT = "5b6a7642ffb45b309e3ec09a8b789e55bd5089b7"
E19_RESULT_COMMIT = "f9e9a9e007d7d84046232181fdab6f9a732a4a05"
FORMAL_CHECKPOINT = (
    "src/wt/benchmark_v2/phase6g_lowlevel_mappo/checkpoints/phase6g_seed0.pt"
)

AREA_WIDTH_M = 2000.0
AREA_HEIGHT_M = 2000.0
FLIGHT_ALTITUDE_M = 100.0
MOTION_STEP_S = 1.0
PLANNING_INTERVAL_S = 30.0
OBSERVATION_WINDOW_S = 1200.0
PRODUCT_INTERVAL_S = 1200.0
EVALUATION_HORIZON_S = 1800.0
WARMUP_START_S = -1200.0
PRODUCT_TIMES_S = (0.0, 1200.0)
SAMPLE_INTERVAL_S = 60.0
PRODUCT_VALIDITY_S = 3600.0

SHORE_XY: Point = (0.0, 0.0)
SHORE_ANTENNA_M = 10.0
DEPOT_XY: Point = (5.0, 0.0)
LANDED_ANTENNA_M = 1.5
BUOY_ANTENNA_M = 1.0

# Rotary-wing numerical example paired with Zeng, Xu and Zhang, IEEE TWC 2019,
# arXiv:1804.02238, equation (6). P0 and Pi are computed from these primitives.
# The PDF table was not re-read in this run, so the primitive vector is a
# literature-reproduction contract rather than a new measurement.
ROTOR_RADIUS_M = 0.4
ROTOR_OMEGA_RAD_S = 300.0
TIP_SPEED_M_S = ROTOR_OMEGA_RAD_S * ROTOR_RADIUS_M
ROTOR_AREA_M2 = math.pi * ROTOR_RADIUS_M * ROTOR_RADIUS_M
SOLIDITY = 0.05
PROFILE_DRAG = 0.012
PARASITE_D0 = 0.6
INDUCED_K = 0.1
INDUCED_VELOCITY_M_S = 4.03
AIR_DENSITY_KG_M3 = 1.225
WEIGHT_N = 20.0
AIRSPEED_GRID_M_S = tuple(float(v) for v in range(1, 21))
MAX_AIRSPEED_M_S = 20.0
ACCEL_M_S2 = 2.0
TAKEOFF_HOVER_S = 20.0
LANDING_HOVER_S = 20.0
HOVER_ENDURANCE_CONTRACT_S = 1200.0
RESERVE_FRACTION = 0.20
COMPUTE_W = 5.0
GROUND_IDLE_W = 1.0
BUOY_IDLE_W = 0.05

# Semtech SX1276 datasheet, LoRa, 3.3 V table conditions.
# Sensitivity: Table 10, RFS_L125_HF, SF7, 125 kHz, Band 1, typical -123 dBm.
# TX current: IDDT_L, +13 dBm on the RFO pin, typical 28 mA.
# RX current: IDDR_L, Band 1, 125 kHz, LnaBoost off, typical 10.3 mA.
# The sensitivity row uses LnaBoost, so 10.3 mA is a lower bound on RX draw.
RADIO_HZ = 868.0e6
RADIO_BW_HZ = 125000.0
RADIO_SF = 7
RADIO_CR = 2  # 4/6, the datasheet electrical-spec coding rate
RADIO_PREAMBLE = 8
TX_DBM = 13.0
SUPPLY_V = 3.3
TX_CURRENT_A = 0.028
RX_CURRENT_A = 0.0103
BUOY_GAIN_DBI = 2.0
UAV_GAIN_DBI = 2.0
SHORE_GAIN_DBI = 6.0
SENSITIVITY_DBM = -123.0
IMPLEMENTATION_MARGIN_DB = 6.0
MAC_GUARD_S = 0.01

# Operational schedules. They do not change transmit power or bandwidth.
ABUNDANT_MIN_FRAME_S = 1.0
JOINT_CYCLE_S = PLANNING_INTERVAL_S
BACKHAUL_PERIOD_S = 600.0
BACKHAUL_OPEN_S = 60.0
BACKHAUL_CLOSED_S = BACKHAUL_PERIOD_S - BACKHAUL_OPEN_S

# Synthetic drift split in the OpenDrift sense: current, Stokes and windage
# are separate terms. Stokes is not copied into the current.
CURRENT_M_S = 0.10
CURRENT_FROM_EAST_RAD = math.pi / 4.0
STOKES_M_S = 0.02
WINDAGE_FRACTION = 0.01
CORRELATION_LENGTH_M = 1000.0
COVERAGE_GRID_M = 100.0

# Synthetic significant-wave-height field used only for the numeric update
# example. It is not a wave model and it does not decide link feasibility.
HS_PRIOR_MEAN_M = 1.2
HS_PRIOR_VAR_M2 = 0.25
HS_NOISE_STD_M = 0.05
HS_FIELD_AMP_M = 0.3
EXAMPLE_SEED = 20261104

WIND_CALM = (0.0, 0.0)
WIND_EAST_8 = (8.0, 0.0)

SCALES = ((2, 4), (6, 12))
REGIMES = (
    "communication_abundant",
    "joint_access_and_communication_constraint",
    "intermittent_backhaul",
)
WIND_NAMES = ("calm", "wind_8mps_east")
LAYOUT_NAMES = ("dispersed", "two_clusters")

PERFORMANCE_PASS = False
ADOPTED = False
RELEASE_ELIGIBLE = False

RAW_CHANNELS = 3
RAW_HZ = 4.0
RAW_BITS = 16
RAW_RECORD_BITS = int(RAW_CHANNELS * RAW_HZ * RAW_BITS * OBSERVATION_WINDOW_S)


def dispersed_buoys(count: int) -> Tuple[Point, ...]:
    if count == 4:
        return ((500.0, 500.0), (500.0, 1500.0), (1500.0, 500.0), (1500.0, 1500.0))
    if count == 12:
        xs = (400.0, 1000.0, 1600.0)
        ys = (300.0, 800.0, 1300.0, 1800.0)
        return tuple((x, y) for y in ys for x in xs)
    raise ValueError(count)


def clustered_buoys(count: int) -> Tuple[Point, ...]:
    if count == 4:
        return ((350.0, 350.0), (450.0, 450.0), (1550.0, 1550.0), (1650.0, 1650.0))
    if count == 12:
        offsets = (
            (-80.0, -60.0),
            (0.0, -60.0),
            (80.0, -60.0),
            (-80.0, 60.0),
            (0.0, 60.0),
            (80.0, 60.0),
        )
        points: List[Point] = []
        for origin in ((400.0, 500.0), (1600.0, 1500.0)):
            for dx, dy in offsets:
                points.append((origin[0] + dx, origin[1] + dy))
        return tuple(points)
    raise ValueError(count)


def layout_buoys(name: str, count: int) -> Tuple[Point, ...]:
    if name == "dispersed":
        return dispersed_buoys(count)
    if name == "two_clusters":
        return clustered_buoys(count)
    raise KeyError(name)


def wind_vector(name: str) -> Point:
    if name == "calm":
        return WIND_CALM
    if name == "wind_8mps_east":
        return WIND_EAST_8
    raise KeyError(name)


def inside_domain(point: Point) -> bool:
    x, y = point
    return 0.0 <= x <= AREA_WIDTH_M and 0.0 <= y <= AREA_HEIGHT_M


def parameter_records(computed: Dict[str, float]) -> List[dict]:
    """Source table. ``computed`` supplies values derived from the primitives."""
    rows = [
        _row("area_width", AREA_WIDTH_M, "m", "synthetic_assumption", "M0 contract starting square", "flat task square", "contract", False),
        _row("area_height", AREA_HEIGHT_M, "m", "synthetic_assumption", "M0 contract starting square", "flat task square", "contract", False),
        _row("flight_altitude", FLIGHT_ALTITUDE_M, "m", "synthetic_assumption", "M0 contract", "level cruise; no climb model", "contract", False),
        _row("motion_step", MOTION_STEP_S, "s", "synthetic_assumption", "M0 contract", "analytical legs are not integrated at this step", "contract", False),
        _row("planning_interval", PLANNING_INTERVAL_S, "s", "synthetic_assumption", "M0 contract", "joint-regime cycle", "contract", False),
        _row("observation_window", OBSERVATION_WINDOW_S, "s", "synthetic_assumption", "aligned with the NDBC 20-minute record length; not an NDBC download", "one Hs mean per window", "contract", False),
        _row("evaluation_horizon", EVALUATION_HORIZON_S, "s", "synthetic_assumption", "M0 contract", "two product times", "contract", False),
        _row("shore_xy", list(SHORE_XY), "m", "synthetic_assumption", "registered before the case run", "corner antenna", "contract", False),
        _row("shore_antenna_height", SHORE_ANTENNA_M, "m", "synthetic_assumption", "no surveyed mast", "link geometry only", "low", False),
        _row("depot_xy", list(DEPOT_XY), "m", "synthetic_assumption", "pad offset so the landed range is not zero", "ground state", "contract", False),
        _row("landed_antenna_height", LANDED_ANTENNA_M, "m", "synthetic_assumption", "no airframe drawing", "ground backhaul", "low", False),
        _row("buoy_antenna_height", BUOY_ANTENNA_M, "m", "synthetic_assumption", "no buoy drawing", "link geometry only", "low", False),
        _row("weight", WEIGHT_N, "N", "literature", "Zeng/Xu/Zhang numerical example paired with arXiv:1804.02238 eq. (6)", "steady axial rotary-wing flight at sea-level density", "equation high; table not re-read from the PDF", False),
        _row("rotor_radius", ROTOR_RADIUS_M, "m", "literature", "same numerical example", "same", "equation high; table not re-read from the PDF", False),
        _row("rotor_omega", ROTOR_OMEGA_RAD_S, "rad/s", "literature", "same numerical example", "same", "equation high; table not re-read from the PDF", False),
        _row("solidity", SOLIDITY, "1", "literature", "same numerical example", "same", "equation high; table not re-read from the PDF", False),
        _row("profile_drag", PROFILE_DRAG, "1", "literature", "same numerical example", "same", "equation high; table not re-read from the PDF", False),
        _row("parasite_d0", PARASITE_D0, "1", "literature", "same numerical example", "same", "equation high; table not re-read from the PDF", False),
        _row("induced_k", INDUCED_K, "1", "literature", "same numerical example", "same", "equation high; table not re-read from the PDF", False),
        _row("induced_velocity", INDUCED_VELOCITY_M_S, "m/s", "literature", "same numerical example", "same", "equation high; table not re-read from the PDF", False),
        _row("air_density", AIR_DENSITY_KG_M3, "kg/m^3", "literature", "same numerical example", "not a measured marine profile", "equation high; table not re-read from the PDF", False),
        _row("P0", computed["P0_W"], "W", "literature", "computed from the primitives; not copied from a rounded watt claim", "hover and axial flight", "derived", False),
        _row("Pi", computed["Pi_W"], "W", "literature", "computed from the primitives", "hover and axial flight", "derived", False),
        _row("cruise_airspeed", computed["cruise_airspeed_m_s"], "m/s", "literature", "argmin of zero-wind propulsion joule per metre on 1..20 m/s", "steady legs; chosen before case outcomes", "derived", False),
        _row("max_airspeed", MAX_AIRSPEED_M_S, "m/s", "synthetic_assumption", "operating cap inside the model, far below tip speed", "grid upper bound", "contract", False),
        _row("acceleration", ACCEL_M_S2, "m/s^2", "synthetic_assumption", "no airframe measurement", "one accel and one decel per leg at hover power", "low", False),
        _row("takeoff_hover", TAKEOFF_HOVER_S, "s", "synthetic_assumption", "momentum model has no climb", "once per sortie", "low", False),
        _row("landing_hover", LANDING_HOVER_S, "s", "synthetic_assumption", "momentum model has no ground effect", "once per sortie", "low", False),
        _row("usable_energy", computed["usable_energy_J"], "J", "synthetic_assumption", "1200 s times computed hover power; not a measured pack", "no recharge", "contract", False),
        _row("reserve_fraction", RESERVE_FRACTION, "1", "synthetic_assumption", "energy after the planned returns", "guard", "contract", False),
        _row("compute_power", COMPUTE_W, "W", "synthetic_assumption", "no flight-computer measurement", "airborne time only", "low", False),
        _row("ground_idle_power", GROUND_IDLE_W, "W", "synthetic_assumption", "motors off, avionics on; includes listen-mode radio", "ground waits", "low", False),
        _row("buoy_idle_power", BUOY_IDLE_W, "W", "synthetic_assumption", "no buoy power measurement", "direct and upload waits", "low", False),
        _row("radio_hz", RADIO_HZ, "Hz", "device_specification", "EU868 sits in SX1276 Band 1", "all three links", "datasheet band", False),
        _row("tx_power", TX_DBM, "dBm", "device_specification", "SX1276 IDDT_L condition on the RFO pin", "all transmitters", "datasheet typical", False),
        _row("tx_current", TX_CURRENT_A, "A", "device_specification", "SX1276 IDDT_L typical at +13 dBm", "TX joules = 3.3 V * current * airtime", "datasheet typical", False),
        _row("rx_current", RX_CURRENT_A, "A", "device_specification", "SX1276 IDDR_L Band 1 125 kHz, LnaBoost off", "lower bound versus the LnaBoost sensitivity row", "datasheet typical", False),
        _row("sensitivity", SENSITIVITY_DBM, "dBm", "device_specification", "SX1276 RFS_L125_HF SF7 typical", "threshold plus one implementation margin; no second PER", "datasheet typical", False),
        _row("implementation_margin", IMPLEMENTATION_MARGIN_DB, "dB", "synthetic_assumption", "cables and polarization; not a fading draw and not a second rate cut", "all links", "contract", False),
        _row("buoy_gain", BUOY_GAIN_DBI, "dBi", "synthetic_assumption", "no antenna measurement", "all buoy links", "low", False),
        _row("uav_gain", UAV_GAIN_DBI, "dBi", "synthetic_assumption", "no antenna measurement", "all UAV links", "low", False),
        _row("shore_gain", SHORE_GAIN_DBI, "dBi", "synthetic_assumption", "no antenna measurement", "shore links", "low", False),
        _row("mac_guard", MAC_GUARD_S, "s", "synthetic_assumption", "fixed guard per slot", "time only; not a path-loss term", "contract", False),
        _row("backhaul_period", BACKHAUL_PERIOD_S, "s", "synthetic_assumption", "operational TDMA, not a measured outage and not a bandwidth change", "intermittent regime", "contract", False),
        _row("backhaul_open", BACKHAUL_OPEN_S, "s", "synthetic_assumption", "must be checked against the epoch length", "intermittent regime", "contract", False),
        _row("current_speed", CURRENT_M_S, "m/s", "synthetic_assumption", "not a Copernicus sample", "horizontal drift", "low", False),
        _row("stokes_speed", STOKES_M_S, "m/s", "synthetic_assumption", "explicit term; not added again inside the current", "wind direction; zero when calm", "low", False),
        _row("windage_fraction", WINDAGE_FRACTION, "1", "synthetic_assumption", "OpenDrift-style coefficient, not fitted", "times wind speed", "low", False),
        _row("correlation_length", CORRELATION_LENGTH_M, "m", "synthetic_assumption", "diagnostic only; not estimated", "coverage and redundancy", "low", False),
        _row("hs_noise_std", HS_NOISE_STD_M, "m", "synthetic_assumption", "numeric example only", "one worked packet", "low", False),
        _row("wind_east", 8.0, "m/s", "synthetic_assumption", "Beaufort 5 starts at 8.0 m/s; the vector is a scenario, not ERA5", "steady horizontal wind", "scenario", False),
    ]
    return rows


def _row(name, value, unit, source_class, source, applicability, confidence, missing) -> dict:
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "source_class": source_class,
        "source": source,
        "applicability": applicability,
        "confidence": confidence,
        "missing": missing,
    }


def assert_registered_layouts() -> None:
    for count in (4, 12):
        for name in LAYOUT_NAMES:
            points = layout_buoys(name, count)
            if len(points) != count:
                raise RuntimeError(name)
            if len(set(points)) != count:
                raise RuntimeError("duplicate buoy")
            for point in points:
                if not inside_domain(point):
                    raise RuntimeError(point)
    if RAW_RECORD_BITS != 230400:
        raise RuntimeError(RAW_RECORD_BITS)
