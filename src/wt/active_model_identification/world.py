"""Physical mission state for one stationary identification segment.

Truth theta is an argument of the step, not a field of the state that the
planner copies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from wt.ocean_observation_m0.parameters import MAC_GUARD_S
from wt.ocean_observation_m0.radio import lora_time_on_air_s, rx_power_w, tx_power_w
from wt.ocean_observation_m0.flight import hover_power_w

from wt.active_model_identification.belief import Belief, telemetry_y
from wt.active_model_identification.physics import (
    hover_energy_j,
    seek_velocity,
    step_kinematics,
    straight_energy_j,
    trip_energy_j,
)

Action = tuple[str, int | None]


@dataclass
class Uav:
    index: int
    x: float
    y: float
    home_x: float
    home_y: float
    initial_energy_j: float
    vx: float = 0.0
    vy: float = 0.0
    energy_j: float = 0.0
    airborne: bool = True
    failed: bool = False
    failure_recorded: bool = False
    pending_failure: bool = False
    cache_ids: list[str] = field(default_factory=list)
    z_j: float = 0.0
    aux_j: float = 0.0
    true_drop_j: float = 0.0
    consumed_j: float = 0.0
    land_timer_s: float = 0.0

    def __post_init__(self) -> None:
        if self.energy_j == 0.0:
            self.energy_j = self.initial_energy_j


@dataclass
class Packet:
    id: str
    buoy: int
    bits: float
    remaining_bits: float
    created_s: float
    deadline_s: float
    holder: str = "buoy"
    delivered_s: float | None = None


@dataclass
class World:
    t: float
    uavs: list[Uav]
    packets: list[Packet]
    buoy_xy: list[tuple[float, float]]
    channel_busy_until: float = 0.0
    interventions: int = 0
    return_failures: int = 0
    calibration_s: list[float] = field(default_factory=lambda: [0.0, 0.0])
    calibration_j: list[float] = field(default_factory=lambda: [0.0, 0.0])
    min_separation_m: float = 1.0e9
    shore_radio_j: float = 0.0
    buoy_radio_j: float = 0.0
    interval_windows: list[tuple[int, float, float]] = field(default_factory=list)

    def snapshot(self) -> "World":
        cloned = World(
            t=self.t,
            uavs=[_copy_uav(uav) for uav in self.uavs],
            packets=[_copy_packet(packet) for packet in self.packets],
            buoy_xy=list(self.buoy_xy),
            channel_busy_until=self.channel_busy_until,
            interventions=self.interventions,
            return_failures=self.return_failures,
            calibration_s=list(self.calibration_s),
            calibration_j=list(self.calibration_j),
            min_separation_m=self.min_separation_m,
            shore_radio_j=self.shore_radio_j,
            buoy_radio_j=self.buoy_radio_j,
            interval_windows=list(self.interval_windows),
        )
        return cloned


def _copy_uav(uav: Uav) -> Uav:
    return Uav(
        index=uav.index,
        x=uav.x,
        y=uav.y,
        home_x=uav.home_x,
        home_y=uav.home_y,
        initial_energy_j=uav.initial_energy_j,
        vx=uav.vx,
        vy=uav.vy,
        energy_j=uav.energy_j,
        airborne=uav.airborne,
        failed=uav.failed,
        failure_recorded=uav.failure_recorded,
        pending_failure=uav.pending_failure,
        cache_ids=list(uav.cache_ids),
        z_j=uav.z_j,
        aux_j=uav.aux_j,
        true_drop_j=uav.true_drop_j,
        consumed_j=uav.consumed_j,
        land_timer_s=uav.land_timer_s,
    )


def _copy_packet(packet: Packet) -> Packet:
    return Packet(
        packet.id,
        packet.buoy,
        packet.bits,
        packet.remaining_bits,
        packet.created_s,
        packet.deadline_s,
        packet.holder,
        packet.delivered_s,
    )


def on_time_bits(world: World) -> float:
    return math.fsum(
        packet.bits
        for packet in world.packets
        if packet.holder == "shore" and packet.delivered_s is not None and packet.delivered_s <= packet.deadline_s + 1e-9
    )


def energy_used_j(world: World) -> list[float]:
    return [uav.consumed_j for uav in world.uavs]


def utility(world: World, spec: dict) -> float:
    block = spec["utility"]
    used = math.fsum(energy_used_j(world))
    return on_time_bits(world) / float(block["b_ref_bits"]) - float(block["lambda_e"]) * used / float(block["e_ref_j"])


def _charge(uav: Uav, theta: float, propulsion_z: float, aux_j: float) -> None:
    """Draw theta * nominal propulsion plus auxiliary joules from the battery.

    ``propulsion_z`` is the nominal integral. The ledger stores it separately
    from auxiliary joules so the regression observation is theta * z.
    """
    demand = theta * propulsion_z + aux_j
    if demand <= 0.0:
        return
    taken = min(demand, max(0.0, uav.energy_j))
    scale = taken / demand
    uav.energy_j -= taken
    uav.consumed_j += taken
    uav.true_drop_j += taken
    uav.z_j += propulsion_z * scale
    uav.aux_j += aux_j * scale
    if demand > taken + 1e-6:
        away = math.hypot(uav.x - uav.home_x, uav.y - uav.home_y) > 25.0
        if away and not uav.failure_recorded:
            uav.failed = True
            uav.failure_recorded = True
            uav.pending_failure = True
            uav.airborne = False
            uav.vx = 0.0
            uav.vy = 0.0


def _aux_w(spec: dict, airborne: bool) -> float:
    energy = spec["energy"]
    return float(energy["compute_w"] if airborne else energy["ground_idle_w"])


def action_cost_j(uav: Uav, action: Action, theta: float, spec: dict, factor: float) -> float:
    aux = _aux_w(spec, True)
    landing = float(spec["time"]["landing_s"])
    kind, target = action
    home_gap = math.hypot(uav.x - uav.home_x, uav.y - uav.home_y)
    if kind == "wait":
        return hover_energy_j(landing, theta, aux) if uav.airborne else 0.0
    if kind == "return":
        return straight_energy_j(home_gap, theta, aux, factor) + hover_energy_j(landing, theta, aux)
    if kind == "backhaul":
        return 80.0
    if kind == "calibrate":
        # Measured closed pattern is about one decision of mixed cruise and hover.
        # The guard uses hover power as a conservative stand-in plus the trip home.
        nominal = hover_power_w() * float(spec["time"]["decision_s"])
        return theta * nominal + aux * float(spec["time"]["decision_s"]) + straight_energy_j(home_gap, theta, aux, factor)
    if kind == "service":
        if target is None:
            raise ValueError("service target")
        buoy_x, buoy_y = _buoy(spec, target)
        return trip_energy_j(
            uav.x,
            uav.y,
            buoy_x,
            buoy_y,
            uav.home_x,
            uav.home_y,
            theta,
            aux,
            factor,
            landing,
        )
    raise KeyError(kind)


def _buoy(spec: dict, buoy_id: int) -> tuple[float, float]:
    row = spec["geometry"]["buoys"][buoy_id]
    return float(row["xy"][0]), float(row["xy"][1])


def uav_target(uav: Uav, buoy_id: int, spec: dict) -> tuple[float, float]:
    return _buoy(spec, buoy_id)


def action_allowed(uav: Uav, action: Action, theta: float, spec: dict, factor: float) -> bool:
    if uav.failed:
        return action[0] == "wait"
    reserve = float(spec["energy"]["reserve_j"])
    if action[0] in ("wait", "return"):
        return True
    return uav.energy_j >= action_cost_j(uav, action, theta, spec, factor) + reserve


def apply_option_guard(
    uav: Uav,
    action: Action,
    belief: Belief,
    spec: dict,
    factor: float,
) -> tuple[Action, bool]:
    if uav.failed:
        return ("wait", None), action != ("wait", None)
    theta = belief.quantile(float(spec["belief"]["guard_probability"]))
    if action_allowed(uav, action, theta, spec, factor):
        return action, False
    return ("return", None), True


def _return_guard(uav: Uav, action: Action, belief: Belief, spec: dict, factor: float) -> tuple[Action, bool]:
    if uav.failed or action[0] in ("wait", "return"):
        return action, False
    theta = belief.quantile(float(spec["belief"]["guard_probability"]))
    aux = _aux_w(spec, True)
    landing = float(spec["time"]["landing_s"])
    need = straight_energy_j(math.hypot(uav.x - uav.home_x, uav.y - uav.home_y), theta, aux, factor)
    need += hover_energy_j(landing, theta, aux) + float(spec["energy"]["reserve_j"])
    if uav.energy_j < need:
        return ("return", None), True
    return action, False


def _desired(uav: Uav, action: Action, spec: dict, phase_s: int) -> tuple[float, float, bool]:
    kind, target = action
    if uav.failed:
        return 0.0, 0.0, False
    if kind == "wait":
        if uav.airborne and uav.land_timer_s < float(spec["time"]["landing_s"]):
            return 0.0, 0.0, True
        return 0.0, 0.0, False
    if kind == "calibrate":
        if phase_s < 10:
            return 12.0, 0.0, True
        if phase_s < 20:
            return -12.0, 0.0, True
        return 0.0, 0.0, True
    if kind == "backhaul":
        return 0.0, 0.0, False
    if kind == "return":
        if math.hypot(uav.x - uav.home_x, uav.y - uav.home_y) < 12.0 and math.hypot(uav.vx, uav.vy) < 1.0:
            return 0.0, 0.0, uav.airborne and uav.land_timer_s < float(spec["time"]["landing_s"])
        vx, vy = seek_velocity(uav.x, uav.y, uav.home_x, uav.home_y)
        return vx, vy, True
    if kind == "service":
        buoy_x, buoy_y = _buoy(spec, int(target))
        if math.hypot(buoy_x - uav.x, buoy_y - uav.y) <= float(spec["energy"]["bulk_range_m"]) and math.hypot(uav.vx, uav.vy) < 1.5:
            return 0.0, 0.0, True
        vx, vy = seek_velocity(uav.x, uav.y, buoy_x, buoy_y)
        return vx, vy, True
    raise KeyError(kind)


def _try_lora(world: World, step_end: float, payload_bytes: int) -> float | None:
    airtime = lora_time_on_air_s(payload_bytes) + MAC_GUARD_S
    start = max(world.t, world.channel_busy_until)
    if start + airtime > step_end + 1e-9:
        return None
    world.channel_busy_until = start + airtime
    return airtime


def _collect_bulk(world: World, uav: Uav, buoy_id: int, spec: dict) -> None:
    energy = spec["energy"]
    rate = float(energy["bulk_rate_bps"])
    for packet in world.packets:
        if packet.holder != "buoy" or packet.buoy != buoy_id:
            continue
        duration = packet.bits / rate
        _charge(uav, 1.0, 0.0, float(energy["bulk_rx_w"]) * duration)
        world.buoy_radio_j += float(energy["bulk_tx_w"]) * duration
        packet.holder = f"uav:{uav.index}"
        packet.remaining_bits = packet.bits
        uav.cache_ids.append(packet.id)


def _backhaul(world: World, uav: Uav, spec: dict, step_end: float) -> None:
    chunk = int(spec["radio"]["backhaul_chunk_bytes"])
    for packet in world.packets:
        if packet.holder != f"uav:{uav.index}" or packet.remaining_bits <= 0.0:
            continue
        while packet.remaining_bits > 0.0:
            payload_bits = min(packet.remaining_bits, chunk * 8.0)
            payload_bytes = max(1, int(math.ceil(payload_bits / 8.0)))
            airtime = _try_lora(world, step_end, payload_bytes)
            if airtime is None:
                return
            _charge(uav, 1.0, 0.0, tx_power_w() * airtime)
            world.shore_radio_j += rx_power_w() * airtime
            packet.remaining_bits -= payload_bits
            if packet.remaining_bits <= 1e-6:
                packet.remaining_bits = 0.0
                packet.holder = "shore"
                packet.delivered_s = world.t
                if packet.id in uav.cache_ids:
                    uav.cache_ids.remove(packet.id)


def _move(uav: Uav, action: Action, spec: dict, theta: float, phase_s: int) -> None:
    if uav.failed:
        return
    desired_vx, desired_vy, propulsion = _desired(uav, action, spec, phase_s)
    kind, _target = action
    if kind in ("return", "wait") and math.hypot(uav.x - uav.home_x, uav.y - uav.home_y) < 12.0 and math.hypot(uav.vx, uav.vy) < 1.0:
        uav.land_timer_s += 1.0
        if uav.land_timer_s > float(spec["time"]["landing_s"]):
            propulsion = False
            uav.airborne = False
    elif propulsion:
        uav.airborne = True
        uav.land_timer_s = 0.0
    else:
        uav.airborne = False
    aux = _aux_w(spec, propulsion or uav.airborne)
    x, y, vx, vy, z_j = step_kinematics(uav.x, uav.y, uav.vx, uav.vy, desired_vx, desired_vy, propulsion, 1.0)
    uav.x, uav.y, uav.vx, uav.vy = x, y, vx, vy
    _charge(uav, theta, z_j, aux * 1.0)


def advance(
    world: World,
    actions: list[Action],
    dynamics_theta: list[float],
    beliefs: list[Belief],
    spec: dict,
    factor: float,
    stats: dict | None,
    *,
    planning: bool,
    interval_index: int,
    noise_j,
) -> list[dict]:
    """Advance one decision interval and assimilate one observation per UAV."""
    if len(actions) != len(world.uavs) or len(dynamics_theta) != len(world.uavs):
        raise ValueError("action vector")
    seconds = int(spec["time"]["decision_s"])
    start = world.t
    for uav in world.uavs:
        uav.z_j = 0.0
        uav.aux_j = 0.0
        uav.true_drop_j = 0.0
    executed = list(actions)
    option_overrode = [False] * len(world.uavs)
    telemetry_sent = [False] * len(world.uavs)
    observations: list[dict] = []
    for phase in range(seconds):
        if stats is not None:
            key = "planning_steps" if planning else "physical_steps"
            stats[key] = int(stats.get(key, 0)) + 1
        if phase == 0:
            for index, uav in enumerate(world.uavs):
                updated, changed = apply_option_guard(uav, executed[index], beliefs[index], spec, factor)
                executed[index] = updated
                if changed:
                    option_overrode[index] = True
                    world.interventions += 1
            for index, uav in enumerate(world.uavs):
                airtime = _try_lora(world, start + 1.0, int(spec["radio"]["command_bytes"]))
                if airtime is not None:
                    _charge(uav, 1.0, 0.0, rx_power_w() * airtime)
                    world.shore_radio_j += tx_power_w() * airtime
        for index, uav in enumerate(world.uavs):
            updated, changed = _return_guard(uav, executed[index], beliefs[index], spec, factor)
            executed[index] = updated
            if changed and not option_overrode[index]:
                option_overrode[index] = True
                world.interventions += 1
        step_end = start + phase + 1.0
        for index, uav in enumerate(world.uavs):
            action = executed[index]
            spent_before = uav.consumed_j
            if action[0] == "calibrate":
                world.calibration_s[index] += 1.0
            _move(uav, action, spec, dynamics_theta[index], phase)
            if action[0] == "calibrate":
                world.calibration_j[index] += uav.consumed_j - spent_before
            if uav.failed:
                continue
            if phase == seconds - 1:
                airtime = _try_lora(world, step_end, int(spec["radio"]["telemetry_bytes"]))
                if airtime is not None:
                    _charge(uav, 1.0, 0.0, tx_power_w() * airtime)
                    world.shore_radio_j += rx_power_w() * airtime
                    telemetry_sent[index] = True
            if action[0] == "service" and action[1] is not None:
                buoy_x, buoy_y = _buoy(spec, int(action[1]))
                if math.hypot(uav.x - buoy_x, uav.y - buoy_y) <= float(spec["energy"]["bulk_range_m"]) and math.hypot(uav.vx, uav.vy) < 1.5:
                    _collect_bulk(world, uav, int(action[1]), spec)
            if action[0] == "backhaul":
                _backhaul(world, uav, spec, step_end)
            if uav.pending_failure:
                world.return_failures += 1
                uav.pending_failure = False
        if len(world.uavs) == 2:
            gap = math.hypot(world.uavs[0].x - world.uavs[1].x, world.uavs[0].y - world.uavs[1].y)
            world.min_separation_m = min(world.min_separation_m, gap)
        world.t = step_end
    world.interval_windows.append((interval_index, start, world.t))
    sigma = float(spec["belief"]["sigma_j"])
    for index, uav in enumerate(world.uavs):
        epsilon = 0.0 if planning else float(noise_j(index, interval_index))
        mean_before, var_before = beliefs[index].moments()
        y_j = telemetry_y(uav.true_drop_j, uav.aux_j, epsilon)
        predictive = uav.z_j * mean_before
        scale = math.sqrt(max(0.0, (uav.z_j ** 2) * var_before + sigma ** 2))
        hit = abs(y_j - predictive) <= 1.6448536269514722 * scale
        if telemetry_sent[index]:
            beliefs[index] = beliefs[index].update(uav.z_j, y_j, sigma)
        observations.append(
            {
                "uav": index,
                "interval": interval_index,
                "z_j": uav.z_j,
                "y_j": y_j,
                "prediction_error_j": y_j - predictive,
                "mean_before": mean_before,
                "variance_before": var_before,
                "covered_90": hit,
                "telemetry_sent": telemetry_sent[index],
                "executed": list(executed[index]),
                "requested": list(actions[index]),
            }
        )
    _check_windows(world)
    return observations


def _check_windows(world: World) -> None:
    previous_end = -1.0
    for _index, start, end in world.interval_windows:
        if start < previous_end - 1e-9:
            raise RuntimeError("overlapping energy intervals")
        previous_end = end


def make_world(spec: dict, battery_j: list[float], seed_tag: str) -> World:
    geometry = spec["geometry"]
    uavs = []
    for index, row in enumerate(geometry["uavs"]):
        home_x, home_y = float(row["home_xy"][0]), float(row["home_xy"][1])
        uavs.append(
            Uav(
                index=index,
                x=home_x,
                y=home_y,
                home_x=home_x,
                home_y=home_y,
                initial_energy_j=float(battery_j[index]),
                energy_j=float(battery_j[index]),
                airborne=False,
            )
        )
    packets = []
    buoy_xy = []
    deadline = float(spec["time"]["mission_s"])
    for buoy in geometry["buoys"]:
        buoy_xy.append((float(buoy["xy"][0]), float(buoy["xy"][1])))
        for number in range(int(buoy["packets"])):
            bits = float(buoy["bits_each"])
            packets.append(
                Packet(
                    id=f"{seed_tag}-b{buoy['id']}-p{number}",
                    buoy=int(buoy["id"]),
                    bits=bits,
                    remaining_bits=bits,
                    created_s=0.0,
                    deadline_s=deadline,
                )
            )
    return World(t=0.0, uavs=uavs, packets=packets, buoy_xy=buoy_xy)
