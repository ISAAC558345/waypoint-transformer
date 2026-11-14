"""Complete-business cost shared by admission, terminal value, and budgets.

The quote replays the fixed low-level controller for one named business:
service collects that buoy, backhaul sends the cache, return lands.
Calibration-then-service is the same replay with one leading calibration
interval. Path and airtime do not depend on theta, so one nominal replay
gives z and auxiliary joules and the cost at another theta is theta * z + aux.
That identity is checked against execution. No constant is fitted to reward.

The other UAV waits and still uses LoRa for its own command and telemetry.
Budget probes and coordinate-wise scoring use that same channel assumption.
Immediate return-to-home remains a separate check and is not this quote.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from wt.ocean_observation_m0.radio import rx_power_w, tx_power_w

from wt.active_model_identification.belief import normal_cdf
from wt.active_model_identification.world import (
    Action,
    Uav,
    World,
    _backhaul,
    _charge,
    _collect_bulk,
    _copy_packet,
    _copy_uav,
    _move,
    _try_lora,
    make_world,
    pad_distance_m,
    pad_radius_m,
)

_CACHE: dict[tuple, "Nominal"] = {}
_TAIL_CACHE: dict[tuple, "ReturnTail"] = {}
_TAIL_STATS = {
    "return_quotes": 0,
    "return_cache_hits": 0,
    "return_simulated_seconds": 0,
    "analytic_ground_intervals": 0,
}
_BUCKETS = ("service", "backhaul", "return", "landing", "ground", "calibrate")
LEGACY_MODE = "legacy"
TAIL_MODE = "return_tail_when_no_business"


@dataclass
class Nominal:
    z_j: float
    aux_j: float
    seconds: float
    delivered_bits: float
    on_time_bits: float
    landed: bool
    finished: bool
    on_time: bool
    components: dict
    radio_j: float
    quote_id: str


@dataclass
class CompletionQuote:
    quote_id: str
    energy_j: float
    z_j: float
    aux_j: float
    seconds: float
    delivered_bits: float
    on_time_bits: float
    landed: bool
    finished: bool
    on_time: bool
    components: dict
    radio_j: float


def clear_completion_cache() -> None:
    _CACHE.clear()


def clear_return_tail_cache() -> None:
    _TAIL_CACHE.clear()


def reset_tail_stats() -> None:
    for key in _TAIL_STATS:
        _TAIL_STATS[key] = 0


def tail_stats() -> dict:
    return {key: int(value) for key, value in _TAIL_STATS.items()}


def terminal_mode(spec: dict) -> str:
    """Missing key stays on the legacy zero terminal."""
    mode = LEGACY_MODE
    block = spec.get("terminal")
    if isinstance(block, dict) and block.get("mode"):
        mode = str(block["mode"])
    if mode not in (LEGACY_MODE, TAIL_MODE):
        raise KeyError(mode)
    return mode


def admits(uav: Uav, quote: CompletionQuote, spec: dict) -> bool:
    """Full business fits the battery, the landing, and the deadline."""
    reserve = float(spec["energy"]["reserve_j"])
    if not quote.finished or not quote.landed or not quote.on_time:
        return False
    return uav.energy_j + 1e-6 >= quote.energy_j + reserve


def completion_quote(world: World, uav_index: int, buoy_id: int | None, theta: float, spec: dict) -> CompletionQuote:
    return _scale(_nominal(world, uav_index, buoy_id, spec, "business"), theta)


def chain_quote(world: World, uav_index: int, buoy_id: int, theta: float, spec: dict) -> CompletionQuote:
    """One calibration interval, then the named service business."""
    return _scale(_nominal(world, uav_index, buoy_id, spec, "chain"), theta)


def _scale(nominal: Nominal, theta: float) -> CompletionQuote:
    components = {}
    for name, row in nominal.components.items():
        components[name] = {
            "z_j": row["z_j"],
            "aux_j": row["aux_j"],
            "seconds": row["seconds"],
            "energy_j": theta * row["z_j"] + row["aux_j"],
        }
    return CompletionQuote(
        quote_id=nominal.quote_id,
        energy_j=theta * nominal.z_j + nominal.aux_j,
        z_j=nominal.z_j,
        aux_j=nominal.aux_j,
        seconds=nominal.seconds,
        delivered_bits=nominal.delivered_bits,
        on_time_bits=nominal.on_time_bits,
        landed=nominal.landed,
        finished=nominal.finished,
        on_time=nominal.on_time,
        components=components,
        radio_j=nominal.radio_j,
    )


def _nominal(world: World, uav_index: int, buoy_id: int | None, spec: dict, mode: str) -> Nominal:
    key = _cache_key(world, uav_index, buoy_id, spec, mode)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    nominal = _replay(world, uav_index, buoy_id, spec, mode, key)
    _CACHE[key] = nominal
    return nominal


def _cache_key(world: World, uav_index: int, buoy_id: int | None, spec: dict, mode: str) -> tuple:
    def rounded(value: float) -> float:
        return round(float(value), 5)

    uavs = tuple(
        (
            uav.index,
            rounded(uav.x),
            rounded(uav.y),
            rounded(uav.vx),
            rounded(uav.vy),
            bool(uav.airborne),
            rounded(uav.land_timer_s),
        )
        for uav in world.uavs
    )
    packets = tuple(
        (packet.id, packet.buoy, packet.holder, rounded(packet.remaining_bits), rounded(packet.bits), rounded(packet.deadline_s))
        for packet in world.packets
    )
    return (
        mode,
        uav_index,
        None if buoy_id is None else int(buoy_id),
        rounded(world.t),
        rounded(world.channel_busy_until),
        uavs,
        packets,
        int(spec["time"]["decision_s"]),
        int(spec["time"]["landing_s"]),
        int(spec["radio"]["command_bytes"]),
        int(spec["radio"]["telemetry_bytes"]),
        int(spec["radio"]["backhaul_chunk_bytes"]),
        float(spec["energy"]["bulk_range_m"]),
    )


def _replay(world: World, uav_index: int, buoy_id: int | None, spec: dict, mode: str, key: tuple) -> Nominal:
    sim = World(
        t=world.t,
        uavs=[_copy_uav(uav) for uav in world.uavs],
        packets=[_copy_packet(packet) for packet in world.packets],
        buoy_xy=list(world.buoy_xy),
        channel_busy_until=world.channel_busy_until,
    )
    for uav in sim.uavs:
        uav.energy_j = 1.0e8
        uav.initial_energy_j = 1.0e8
        uav.failed = False
        uav.failure_recorded = False
        uav.pending_failure = False
        uav.z_j = 0.0
        uav.aux_j = 0.0
        uav.true_drop_j = 0.0
        uav.consumed_j = 0.0
    components = {name: {"z_j": 0.0, "aux_j": 0.0, "seconds": 0.0} for name in _BUCKETS}
    radio_j = 0.0
    decision = int(spec["time"]["decision_s"])
    mission = float(spec["time"]["mission_s"])
    limit = int(mission // decision) + 4
    start = sim.t
    for interval in range(limit):
        action = _script_action(sim, uav_index, buoy_id, spec, mode, interval)
        if action is None:
            break
        actions: list[Action] = [("wait", None) for _ in sim.uavs]
        actions[uav_index] = action
        radio_j += _step_interval(sim, actions, uav_index, spec, components)
    focus = sim.uavs[uav_index]
    total_z = math.fsum(row["z_j"] for row in components.values())
    total_aux = math.fsum(row["aux_j"] for row in components.values())
    if abs(total_z - focus.z_j) > 1e-4 or abs(total_aux - focus.aux_j) > 1e-4:
        raise RuntimeError(f"completion buckets do not match the ledger z={focus.z_j} aux={focus.aux_j}")
    finished = _business_done(sim, uav_index, buoy_id, spec)
    landed = (not focus.airborne) and pad_distance_m(focus) <= pad_radius_m(spec)
    delivered_bits = math.fsum(item["bits"] for item in sim.deliveries if item["uav"] == uav_index)
    on_time_bits = math.fsum(item["bits"] for item in sim.deliveries if item["uav"] == uav_index and item["on_time"])
    on_time = finished and landed and sim.t <= mission + 1e-9
    digest = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:16]
    return Nominal(
        z_j=focus.z_j,
        aux_j=focus.aux_j,
        seconds=sim.t - start,
        delivered_bits=delivered_bits,
        on_time_bits=on_time_bits,
        landed=landed,
        finished=finished,
        on_time=on_time,
        components=components,
        radio_j=radio_j,
        quote_id=digest,
    )


def _business_done(world: World, uav_index: int, buoy_id: int | None, spec: dict) -> bool:
    uav = world.uavs[uav_index]
    if buoy_id is not None and any(packet.buoy == buoy_id and packet.holder != "shore" for packet in world.packets):
        return False
    if any(packet.holder == f"uav:{uav_index}" for packet in world.packets):
        return False
    if uav.cache_ids or uav.airborne:
        return False
    return pad_distance_m(uav) <= pad_radius_m(spec)


def _script_action(world: World, uav_index: int, buoy_id: int | None, spec: dict, mode: str, interval: int) -> Action | None:
    if mode == "chain" and interval == 0:
        return ("calibrate", None)
    if _business_done(world, uav_index, buoy_id, spec):
        return None
    if buoy_id is not None and any(packet.buoy == buoy_id and packet.holder == "buoy" for packet in world.packets):
        return ("service", int(buoy_id))
    if any(packet.holder == f"uav:{uav_index}" for packet in world.packets):
        return ("backhaul", None)
    return ("return", None)


def _step_interval(world: World, actions: list[Action], focus: int, spec: dict, components: dict) -> float:
    seconds = int(spec["time"]["decision_s"])
    start = world.t
    radio = 0.0
    kind = actions[focus][0]
    for phase in range(seconds):
        if phase == 0:
            for uav in world.uavs:
                radio += _lora(world, uav, spec, start + 1.0, int(spec["radio"]["command_bytes"]), True, components if uav.index == focus else None, kind)
        step_end = start + phase + 1.0
        for uav in world.uavs:
            action = actions[uav.index]
            before_z = uav.z_j
            before_aux = uav.aux_j
            before_air = uav.airborne
            timer = uav.land_timer_s
            _move(uav, action, spec, 1.0, phase)
            if uav.index == focus:
                bucket = _move_bucket(kind, before_air, timer, uav, spec)
                components[bucket]["z_j"] += uav.z_j - before_z
                components[bucket]["aux_j"] += uav.aux_j - before_aux
                components[bucket]["seconds"] += 1.0
            if phase == seconds - 1:
                radio += _lora(
                    world,
                    uav,
                    spec,
                    step_end,
                    int(spec["radio"]["telemetry_bytes"]),
                    False,
                    components if uav.index == focus else None,
                    kind,
                )
            if uav.failed:
                continue
            if action[0] == "service" and action[1] is not None:
                buoy_x, buoy_y = world.buoy_xy[int(action[1])]
                if math.hypot(uav.x - buoy_x, uav.y - buoy_y) <= float(spec["energy"]["bulk_range_m"]) and math.hypot(uav.vx, uav.vy) < 1.5:
                    before_aux = uav.aux_j
                    _collect_bulk(world, uav, int(action[1]), spec)
                    if uav.index == focus:
                        delta = uav.aux_j - before_aux
                        components["service"]["aux_j"] += delta
                        radio += delta
            if action[0] == "backhaul":
                before_aux = uav.aux_j
                _backhaul(world, uav, spec, step_end)
                if uav.index == focus:
                    delta = uav.aux_j - before_aux
                    components["backhaul"]["aux_j"] += delta
                    radio += delta
        world.t = step_end
    return radio


def _move_bucket(kind: str, before_air: bool, timer_before: float, uav: Uav, spec: dict) -> str:
    if kind == "return":
        if (not before_air) and pad_distance_m(uav) <= pad_radius_m(spec):
            return "ground"
        if timer_before > 0.0 or uav.land_timer_s > 0.0:
            return "landing"
        return "return"
    if kind not in _BUCKETS:
        raise KeyError(kind)
    return kind


def _lora(world: World, uav: Uav, spec: dict, step_end: float, payload: int, receive: bool, components: dict | None, kind: str) -> float:
    before_aux = uav.aux_j
    airtime = _try_lora(world, step_end, payload)
    if airtime is None:
        return 0.0
    watts = rx_power_w() if receive else tx_power_w()
    _charge(uav, 1.0, 0.0, watts * airtime)
    if receive:
        world.shore_radio_j += tx_power_w() * airtime
    else:
        world.shore_radio_j += rx_power_w() * airtime
    delta = uav.aux_j - before_aux
    if components is not None:
        bucket = kind if kind in _BUCKETS else "return"
        components[bucket]["aux_j"] += delta
    return delta


@dataclass(frozen=True)
class ReturnTail:
    """Nominal RETURN-to-deadline cost. Energy is theta * z + aux, then clipped by the battery."""

    z_j: float
    aux_j: float
    flight_z_j: float
    flight_aux_j: float
    ground_z_j: float
    ground_aux_j: float
    radio_j: float
    seconds: float
    landing_s: float | None
    landed: bool
    clock_truncated: bool
    quote_id: str


def tail_utility(z_j: float, aux_j: float, theta: float, energy_j: float, coef: float) -> tuple[float, bool, float, float]:
    """Future utility of a return tail. The sign stays negative. No reserve is added.

    Returns value, energy_truncated, intercept A, and slope B for value = A - B * theta
    on the unclipped side. Past the battery the value is the constant -coef * energy.
    """
    demand = theta * z_j + aux_j
    available = max(0.0, energy_j)
    if demand <= available + 1e-6:
        return -coef * demand, False, -coef * aux_j, coef * z_j
    return -coef * available, True, -coef * available, 0.0


def return_tail_value(world: World, uav_index: int, theta: float, spec: dict) -> float:
    uav = world.uavs[uav_index]
    if uav.failed:
        return 0.0
    nominal = return_tail_nominal(world, uav_index, spec)
    coef = float(spec["utility"]["lambda_e"]) / float(spec["utility"]["e_ref_j"])
    value, _truncated, _intercept, _slope = tail_utility(nominal.z_j, nominal.aux_j, theta, uav.energy_j, coef)
    return value


def return_tail_nominal(world: World, uav_index: int, spec: dict) -> ReturnTail:
    """RETURN until the pad landing, then ground load until the original deadline.

    This is not completion_quote(..., None): that quote backhauls the cache and
    stops when the business is finished. The tail keeps RETURN's physics, does
    not invent a delivery reward, and prices the remaining ground interval.
    """
    key = _tail_key(world, uav_index, spec)
    cached = _TAIL_CACHE.get(key)
    if cached is not None:
        _TAIL_STATS["return_cache_hits"] += 1
        return cached
    _TAIL_STATS["return_quotes"] += 1
    nominal = _replay_return_tail(world, uav_index, spec, key)
    _TAIL_CACHE[key] = nominal
    return nominal


def _tail_key(world: World, uav_index: int, spec: dict) -> tuple:
    def rounded(value: float) -> float:
        return round(float(value), 5)

    focus = world.uavs[uav_index]
    others = tuple((uav.index, bool(uav.failed)) for uav in world.uavs if uav.index != uav_index)
    motion = spec["motion"]
    radio = spec["radio"]
    energy = spec["energy"]
    return (
        TAIL_MODE,
        uav_index,
        rounded(world.t),
        rounded(world.channel_busy_until),
        rounded(focus.x),
        rounded(focus.y),
        rounded(focus.vx),
        rounded(focus.vy),
        rounded(focus.home_x),
        rounded(focus.home_y),
        bool(focus.airborne),
        rounded(focus.land_timer_s),
        bool(focus.failed),
        others,
        int(spec["time"]["decision_s"]),
        int(spec["time"]["mission_s"]),
        int(spec["time"]["landing_s"]),
        int(radio["command_bytes"]),
        int(radio["telemetry_bytes"]),
        float(energy["ground_idle_w"]),
        float(energy["compute_w"]),
        float(motion.get("pad_radius_m", 12.0)),
        float(motion.get("pad_speed_m_s", 1.0)),
        float(motion["cruise_m_s"]),
        float(motion["accel_m_s2"]),
    )


def _focus_landed(world: World, focus: int, spec: dict) -> bool:
    uav = world.uavs[focus]
    return (not uav.failed) and (not uav.airborne) and pad_distance_m(uav) <= pad_radius_m(spec)


def _new_buckets() -> dict:
    return {
        "flight_z": 0.0,
        "flight_aux": 0.0,
        "ground_z": 0.0,
        "ground_aux": 0.0,
        "radio": 0.0,
        "landing_s": None,
    }


def _add_buckets(total: dict, part: dict, times: float = 1.0) -> None:
    for key in ("flight_z", "flight_aux", "ground_z", "ground_aux", "radio"):
        total[key] += times * part[key]


def _radio_charge(world: World, uav: Uav, spec: dict, step_end: float, payload: int, receive: bool, buckets: dict, focus: int) -> None:
    before = uav.aux_j
    airtime = _try_lora(world, step_end, payload)
    if airtime is None:
        return
    watts = rx_power_w() if receive else tx_power_w()
    _charge(uav, 1.0, 0.0, watts * airtime)
    if receive:
        world.shore_radio_j += tx_power_w() * airtime
    else:
        world.shore_radio_j += rx_power_w() * airtime
    if uav.index != focus:
        return
    delta = uav.aux_j - before
    buckets["radio"] += delta
    ground = (not uav.airborne) and pad_distance_m(uav) <= pad_radius_m(spec)
    if ground:
        buckets["ground_aux"] += delta
    else:
        buckets["flight_aux"] += delta


def _motion_charge(world: World, uav: Uav, action: Action, spec: dict, phase: int, buckets: dict, focus: int, t0: float, start: float) -> None:
    before_air = uav.airborne
    before_z = uav.z_j
    before_aux = uav.aux_j
    distance = pad_distance_m(uav)
    _move(uav, action, spec, 1.0, phase)
    if uav.index != focus:
        return
    ground = (not before_air) and distance <= pad_radius_m(spec)
    if ground:
        buckets["ground_z"] += uav.z_j - before_z
        buckets["ground_aux"] += uav.aux_j - before_aux
    else:
        buckets["flight_z"] += uav.z_j - before_z
        buckets["flight_aux"] += uav.aux_j - before_aux
    if before_air and (not uav.airborne) and buckets["landing_s"] is None:
        buckets["landing_s"] = (start + phase + 1.0) - t0


def _simulate_span(world: World, focus: int, spec: dict, seconds: int, buckets: dict, t0: float) -> None:
    start = world.t
    actions: list[Action] = [("wait", None) for _ in world.uavs]
    actions[focus] = ("return", None)
    command_bytes = int(spec["radio"]["command_bytes"])
    telemetry_bytes = int(spec["radio"]["telemetry_bytes"])
    for phase in range(int(seconds)):
        if phase == 0:
            for uav in world.uavs:
                _radio_charge(world, uav, spec, start + 1.0, command_bytes, True, buckets, focus)
        step_end = start + phase + 1.0
        for uav in world.uavs:
            _motion_charge(world, uav, actions[uav.index], spec, phase, buckets, focus, t0, start)
            if uav.failed:
                continue
            if phase == seconds - 1:
                _radio_charge(world, uav, spec, step_end, telemetry_bytes, False, buckets, focus)
        world.t = step_end
        _TAIL_STATS["return_simulated_seconds"] += 1


def _prepare_tail_world(world: World) -> World:
    sim = world.snapshot()
    for uav in sim.uavs:
        uav.energy_j = 1.0e8
        uav.initial_energy_j = 1.0e8
        uav.failed = False
        uav.failure_recorded = False
        uav.pending_failure = False
        uav.z_j = 0.0
        uav.aux_j = 0.0
        uav.true_drop_j = 0.0
        uav.consumed_j = 0.0
    return sim


def _ground_repeat(sim: World, focus: int, spec: dict, mission: float, decision: int, buckets: dict, t0: float) -> None:
    remain = mission - sim.t
    if remain <= 1e-9:
        return
    full = int(math.floor(remain / decision + 1e-9))
    leftover = remain - full * decision
    if leftover < 1e-6:
        leftover = 0.0
    if full <= 0:
        if leftover > 0.0:
            _simulate_span(sim, focus, spec, int(round(leftover)), buckets, t0)
        return
    probe = sim.snapshot()
    first = _new_buckets()
    _simulate_span(probe, focus, spec, decision, first, t0)
    repeat = abs(first["flight_z"]) < 1e-6 and abs(first["ground_z"]) < 1e-6
    if full >= 2 and repeat:
        second = _new_buckets()
        _simulate_span(probe, focus, spec, decision, second, t0)
        repeat = (
            abs(first["ground_aux"] - second["ground_aux"]) < 1e-6
            and abs(first["radio"] - second["radio"]) < 1e-6
            and abs(second["ground_z"]) < 1e-6
            and abs(second["flight_z"]) < 1e-6
        )
    if repeat:
        _add_buckets(buckets, first, full)
        _TAIL_STATS["analytic_ground_intervals"] += full
    else:
        for _ in range(full):
            _simulate_span(sim, focus, spec, decision, buckets, t0)
    if leftover > 0.0:
        anchor = sim if not repeat else sim.snapshot()
        _simulate_span(anchor, focus, spec, int(round(leftover)), buckets, t0)


def _replay_return_tail(world: World, uav_index: int, spec: dict, key: tuple) -> ReturnTail:
    mission = float(spec["time"]["mission_s"])
    decision = int(spec["time"]["decision_s"])
    digest = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:16]
    started_landed = _focus_landed(world, uav_index, spec)
    if world.uavs[uav_index].failed or world.t >= mission - 1e-9:
        return ReturnTail(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 if started_landed else None, started_landed, False, digest)
    sim = _prepare_tail_world(world)
    t0 = sim.t
    buckets = _new_buckets()
    if _focus_landed(sim, uav_index, spec):
        buckets["landing_s"] = 0.0
    while sim.t < mission - 1e-9 and buckets["landing_s"] is None:
        span = int(min(decision, math.floor(mission - sim.t + 1e-9)))
        if span <= 0:
            break
        _simulate_span(sim, uav_index, spec, span, buckets, t0)
    landed = buckets["landing_s"] is not None
    if landed and sim.t < mission - 1e-9:
        _ground_repeat(sim, uav_index, spec, mission, decision, buckets, t0)
    flight_z = buckets["flight_z"]
    flight_aux = buckets["flight_aux"]
    ground_z = buckets["ground_z"]
    ground_aux = buckets["ground_aux"]
    radio = buckets["radio"]
    z_j = flight_z + ground_z
    aux_j = flight_aux + ground_aux
    if abs(radio) > abs(aux_j) + 1e-4:
        raise RuntimeError("return tail radio is not part of the auxiliary ledger")
    seconds = max(0.0, mission - t0)
    return ReturnTail(
        z_j=z_j,
        aux_j=aux_j,
        flight_z_j=flight_z,
        flight_aux_j=flight_aux,
        ground_z_j=ground_z,
        ground_aux_j=ground_aux,
        radio_j=radio,
        seconds=seconds,
        landing_s=None if buckets["landing_s"] is None else float(buckets["landing_s"]),
        landed=landed,
        clock_truncated=bool((not landed) and seconds > 0.0),
        quote_id=digest,
    )


def standard_quantile(probability: float) -> float:
    left, right = -8.0, 8.0
    for _ in range(80):
        mid = 0.5 * (left + right)
        if normal_cdf(mid) < probability:
            left = mid
        else:
            right = mid
    return 0.5 * (left + right)


def accounting_cases(spec: dict) -> list[dict]:
    """Quote against the same controller on the registered states and two thetas."""
    cases = []
    thetas = (1.0, 1.25)

    def start(world: World) -> None:
        return None

    def midroute(world: World) -> None:
        uav = world.uavs[0]
        uav.x = 800.0
        uav.y = 400.0
        uav.vx = 18.0
        uav.vy = 0.0
        uav.airborne = True

    def holding(world: World) -> None:
        uav = world.uavs[0]
        uav.x = 1850.0
        uav.y = 400.0
        uav.vx = 0.0
        uav.vy = 0.0
        uav.airborne = True
        for packet in world.packets:
            if packet.buoy == 1:
                packet.holder = "uav:0"
                packet.remaining_bits = packet.bits
                uav.cache_ids.append(packet.id)

    def near_deadline(world: World) -> None:
        holding(world)
        world.t = float(spec["time"]["mission_s"]) - 20.0

    plans = (
        ("pad_far", start, 1, "business"),
        ("midroute_far", midroute, 1, "business"),
        ("holding_cache", holding, None, "business"),
        ("near_deadline_cache", near_deadline, None, "business"),
        ("pad_chain", start, 1, "chain"),
    )
    for name, prepare, buoy_id, mode in plans:
        for theta in thetas:
            world = make_world(spec, [1.0e7, 1.0e7], f"acct-{name}")
            prepare(world)
            if mode == "chain":
                quote = chain_quote(world, 0, int(buoy_id), theta, spec)
            else:
                quote = completion_quote(world, 0, buoy_id, theta, spec)
            executed = _execute_script(world, spec, theta, 0, buoy_id, mode)
            groups = _groups(quote)
            errors = {key: quote_value - executed["groups"].get(key, 0.0) for key, quote_value in groups.items()}
            errors["total"] = quote.energy_j - executed["energy_j"]
            cases.append(
                {
                    "name": name,
                    "theta": theta,
                    "quote_id": quote.quote_id,
                    "quote_j": quote.energy_j,
                    "executed_j": executed["energy_j"],
                    "seconds": quote.seconds,
                    "executed_seconds": executed["seconds"],
                    "finished": quote.finished,
                    "on_time": quote.on_time,
                    "interventions": executed["interventions"],
                    "components_j": {key: row["energy_j"] for key, row in quote.components.items()},
                    "group_errors_j": errors,
                    "affine_z_j": quote.z_j,
                    "affine_aux_j": quote.aux_j,
                }
            )
    return cases


def _groups(quote: CompletionQuote) -> dict[str, float]:
    energy = {name: row["energy_j"] for name, row in quote.components.items()}
    return {
        "service": energy.get("service", 0.0),
        "backhaul": energy.get("backhaul", 0.0),
        "return": energy.get("return", 0.0) + energy.get("landing", 0.0) + energy.get("ground", 0.0),
        "calibrate": energy.get("calibrate", 0.0),
    }


def _execute_script(world: World, spec: dict, theta: float, uav_index: int, buoy_id: int | None, mode: str) -> dict:
    from wt.active_model_identification.belief import Belief
    from wt.active_model_identification.scenarios import _actions
    from wt.active_model_identification.world import advance

    sim = world.snapshot()
    lo, hi = (float(value) for value in spec["belief"]["support"])
    belief = Belief(theta, 1.0 / (1.0e-4 ** 2), lo, hi)
    beliefs = [belief.copy(), belief.copy()]
    groups = {"service": 0.0, "backhaul": 0.0, "return": 0.0, "calibrate": 0.0}
    decision = int(spec["time"]["decision_s"])
    limit = int(float(spec["time"]["mission_s"]) // decision) + 4
    start = sim.t
    for interval in range(limit):
        action = _script_action(sim, uav_index, buoy_id, spec, mode, interval)
        if action is None:
            break
        before = sim.uavs[uav_index].consumed_j
        advance(
            sim,
            _actions(spec, uav_index, action),
            [theta, theta],
            beliefs,
            spec,
            1.0,
            None,
            planning=True,
            interval_index=interval,
            noise_j=lambda _uav, _interval: 0.0,
        )
        kind = action[0]
        if kind not in groups:
            kind = "return"
        groups[kind] += sim.uavs[uav_index].consumed_j - before
    return {
        "energy_j": sim.uavs[uav_index].consumed_j,
        "seconds": sim.t - start,
        "groups": groups,
        "interventions": sim.interventions,
        "finished": _business_done(sim, uav_index, buoy_id, spec),
    }
