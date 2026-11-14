"""One physical rollout per theta, then a segmented noise integral.

Measurement noise is applied only when the interval's telemetry is assimilated,
so every epsilon of one theta reuses that rollout. Posterior q95 is monotone in
epsilon. Admission, net-value crossings, and zero crossings become noise
thresholds and the pieces are integrated on the standard-normal CDF coordinate.
An independent scan locates those pieces without using the threshold algebra.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from wt.active_model_identification.belief import (
    Belief,
    equal_width_quadrature,
    expectation,
    gauss_legendre_unit,
    gaussian_quadrature,
    normal_cdf,
    prior_belief,
    quadrature,
)
from wt.active_model_identification.completion import completion_quote
from wt.active_model_identification.world import Action, World, advance, energy_used_j, make_world, on_time_bits

_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_P_LOW = 0.02425


class BudgetExceeded(RuntimeError):
    """The registered rollout cap was reached. Callers do not raise the cap."""


@dataclass(frozen=True)
class Plan:
    z_j: float
    aux_j: float
    bits: float
    ok: bool
    label: str = ""


@dataclass(frozen=True)
class Roll:
    immediate: float
    z_j: float
    y0: float
    energy_j: float
    telemetry: bool
    failed: bool
    intervened: bool
    executed: tuple
    plans: tuple
    mu: float
    lam: float
    lo: float
    hi: float
    sigma: float
    aux_j: float = 0.0
    radio_j: float = 0.0
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    calibration_s: float = 0.0


_ROLL_CACHE: dict[tuple, Roll] = {}


def clear_rollout_cache() -> None:
    _ROLL_CACHE.clear()


def inv_norm(probability: float) -> float:
    """Acklam's inverse standard-normal CDF. Checked against the erf bisection."""
    if not 0.0 < probability < 1.0:
        raise ValueError("probability")
    if probability < _P_LOW:
        q = math.sqrt(-2.0 * math.log(probability))
        num = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5])
        den = ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
        return num / den
    if probability > 1.0 - _P_LOW:
        q = math.sqrt(-2.0 * math.log(1.0 - probability))
        num = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5])
        den = ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
        return -num / den
    q = probability - 0.5
    r = q * q
    num = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q
    den = (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
    return num / den


def fast_truncated_quantile(center: float, precision: float, lo: float, hi: float, probability: float) -> float:
    sigma = 1.0 / math.sqrt(precision)
    alpha = (lo - center) / sigma
    beta = (hi - center) / sigma
    mass = normal_cdf(beta) - normal_cdf(alpha)
    if mass < 1e-18:
        if center <= lo:
            return lo
        if center >= hi:
            return hi
        return min(max(center, lo), hi)
    target = normal_cdf(alpha) + probability * mass
    target = min(max(target, 1e-16), 1.0 - 1e-16)
    return center + sigma * inv_norm(target)


def posterior_quantile(mu: float, lam: float, lo: float, hi: float, z_j: float, y_j: float, sigma: float, probability: float) -> float:
    if z_j == 0.0:
        return fast_truncated_quantile(mu, lam, lo, hi, probability)
    precision = lam + (z_j * z_j) / (sigma * sigma)
    center = (lam * mu + z_j * y_j / (sigma * sigma)) / precision
    return fast_truncated_quantile(center, precision, lo, hi, probability)


def extract_plans(world: World, focus: int, spec: dict) -> tuple:
    uav = world.uavs[focus]
    if uav.failed:
        return ()
    plans = []
    cached = math.fsum(packet.bits for packet in world.packets if packet.holder == f"uav:{focus}")
    if cached > 0.0:
        quote = completion_quote(world, focus, None, 1.0, spec)
        plans.append(Plan(quote.z_j, quote.aux_j, quote.on_time_bits, bool(quote.finished and quote.landed and quote.on_time), "cache"))
    for buoy in spec["geometry"]["buoys"]:
        buoy_id = int(buoy["id"])
        pending = math.fsum(packet.bits for packet in world.packets if packet.holder == "buoy" and packet.buoy == buoy_id)
        if pending <= 0.0:
            continue
        quote = completion_quote(world, focus, buoy_id, 1.0, spec)
        plans.append(Plan(quote.z_j, quote.aux_j, quote.on_time_bits, bool(quote.finished and quote.landed and quote.on_time), f"service:{buoy_id}"))
    return tuple(plans)


def _scales(spec: dict) -> tuple[float, float, float]:
    block = spec["utility"]
    reserve = float(spec["energy"]["reserve_j"])
    bits = float(block["b_ref_bits"])
    coef = float(block["lambda_e"]) / float(block["e_ref_j"])
    return reserve, bits, coef


def describe(theta: float, energy: float, plans: tuple, spec: dict, failed: bool) -> tuple[int, float, float, float]:
    """Active plan as value = A - B * theta, or zero when nothing clears the reserve."""
    if failed:
        return -1, 0.0, 0.0, 0.0
    reserve, bits_scale, coef = _scales(spec)
    best_id = -1
    best_net = 0.0
    best_a = 0.0
    best_b = 0.0
    for index, plan in enumerate(plans):
        if not plan.ok:
            continue
        cost = theta * plan.z_j + plan.aux_j
        if energy + 1e-6 < cost + reserve:
            continue
        net = plan.bits / bits_scale - coef * cost
        if net > best_net:
            best_net = net
            best_id = index
            best_a = plan.bits / bits_scale - coef * plan.aux_j
            best_b = coef * plan.z_j
    return best_id, best_a, best_b, best_net


def value_at(theta: float, energy: float, plans: tuple, spec: dict, failed: bool) -> float:
    return describe(theta, energy, plans, spec, failed)[3]


def _utility_delta(before: World, after: World, spec: dict) -> float:
    block = spec["utility"]
    bits = on_time_bits(after) - on_time_bits(before)
    energy = math.fsum(energy_used_j(after)) - math.fsum(energy_used_j(before))
    return bits / float(block["b_ref_bits"]) - float(block["lambda_e"]) * energy / float(block["e_ref_j"])


def _cache_key(world: World, beliefs: list[Belief], action: Action, uav_index: int, theta: float, spec: dict, factor: float) -> tuple:
    def rounded(value: float, digits: int = 6) -> float:
        return round(float(value), digits)

    uavs = tuple(
        (
            uav.index,
            rounded(uav.x),
            rounded(uav.y),
            rounded(uav.vx),
            rounded(uav.vy),
            bool(uav.airborne),
            bool(uav.failed),
            rounded(uav.energy_j),
            rounded(uav.land_timer_s),
            tuple(uav.cache_ids),
        )
        for uav in world.uavs
    )
    packets = tuple((packet.id, packet.holder, rounded(packet.remaining_bits), packet.delivered_s) for packet in world.packets)
    belief_key = tuple((round(item.mu, 10), round(item.lam, 8)) for item in beliefs)
    return (
        uav_index,
        action[0],
        None if action[1] is None else int(action[1]),
        round(float(theta), 10),
        round(float(factor), 8),
        rounded(world.t),
        rounded(world.channel_busy_until),
        uavs,
        packets,
        belief_key,
        int(spec["time"]["decision_s"]),
        float(spec["belief"]["sigma_j"]),
        float(spec["belief"]["guard_probability"]),
    )


def roll_action(
    world: World,
    beliefs: list[Belief],
    action: Action,
    uav_index: int,
    theta: float,
    spec: dict,
    factor: float,
    stats: dict | None,
) -> Roll:
    key = _cache_key(world, beliefs, action, uav_index, theta, spec, factor)
    cached = _ROLL_CACHE.get(key)
    if cached is not None:
        if stats is not None:
            stats["cache_hits"] = int(stats.get("cache_hits", 0)) + 1
        return cached
    if stats is not None and stats.get("rollout_cap") is not None and int(stats.get("rollouts", 0)) >= int(stats["rollout_cap"]):
        stats["budget_exhausted"] = True
        raise BudgetExceeded("rollout cap")
    snap = world.snapshot()
    if stats is not None:
        stats["rollouts"] = int(stats.get("rollouts", 0)) + 1
        stats["planning_copies"] = int(stats.get("planning_copies", 0)) + 1
    copied = [item.copy() for item in beliefs]
    dynamics = [item.mean() for item in beliefs]
    dynamics[uav_index] = float(theta)
    actions: list[Action] = [("wait", None) for _ in world.uavs]
    actions[uav_index] = action
    observations = advance(
        snap,
        actions,
        dynamics,
        copied,
        spec,
        factor,
        stats,
        planning=True,
        interval_index=0,
        noise_j=lambda _uav, _interval: 0.0,
    )
    obs = observations[uav_index]
    uav = snap.uavs[uav_index]
    belief = beliefs[uav_index]
    radio_book = getattr(snap, "interval_uav_radio_j", None)
    roll = Roll(
        immediate=_utility_delta(world, snap, spec),
        z_j=float(uav.z_j),
        y0=float(uav.true_drop_j - uav.aux_j),
        energy_j=float(uav.energy_j),
        telemetry=bool(obs["telemetry_sent"]),
        failed=bool(uav.failed),
        intervened=bool(obs["intervened"]),
        executed=(obs["executed"][0], None if obs["executed"][1] is None else int(obs["executed"][1])),
        plans=extract_plans(snap, uav_index, spec),
        mu=belief.mu,
        lam=belief.lam,
        lo=belief.lo,
        hi=belief.hi,
        sigma=float(spec["belief"]["sigma_j"]),
        aux_j=float(uav.aux_j),
        radio_j=0.0 if radio_book is None else float(radio_book[uav_index]),
        x=float(uav.x),
        y=float(uav.y),
        vx=float(uav.vx),
        vy=float(uav.vy),
        calibration_s=float(snap.calibration_s[uav_index] - world.calibration_s[uav_index]),
    )
    _ROLL_CACHE[key] = roll
    return roll


def _guard(spec: dict) -> float:
    return float(spec["belief"]["guard_probability"])


def _quantile_of(roll: Roll, epsilon: float, probability: float) -> float:
    return posterior_quantile(roll.mu, roll.lam, roll.lo, roll.hi, roll.z_j, roll.y0 + epsilon, roll.sigma, probability)


def _invert_epsilon(roll: Roll, target: float, guard: float, stats: dict | None) -> float | None:
    if stats is not None:
        stats["root_solves"] = int(stats.get("root_solves", 0)) + 1
    lo = -12.0 * roll.sigma
    hi = 12.0 * roll.sigma
    q_lo = _quantile_of(roll, lo, guard)
    q_hi = _quantile_of(roll, hi, guard)
    if target <= q_lo or target >= q_hi:
        return None
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if _quantile_of(roll, mid, guard) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _theta_breaks(roll: Roll, spec: dict) -> list[float]:
    reserve, bits_scale, coef = _scales(spec)
    lo, hi = roll.lo, roll.hi
    usable: list[Plan] = []
    points: list[float] = []
    for plan in roll.plans:
        if not plan.ok:
            continue
        if plan.z_j <= 1e-9:
            if roll.energy_j + 1e-6 >= plan.aux_j + reserve:
                net = plan.bits / bits_scale - coef * plan.aux_j
                if net > 0.0:
                    usable.append(plan)
            continue
        theta_max = (roll.energy_j + 1e-6 - reserve - plan.aux_j) / plan.z_j
        if theta_max <= lo:
            continue
        usable.append(plan)
        if lo < theta_max < hi:
            points.append(theta_max)
        theta_zero = (plan.bits / bits_scale / coef - plan.aux_j) / plan.z_j
        if lo < theta_zero < hi and theta_zero <= theta_max + 1e-9:
            points.append(theta_zero)
    for left in range(len(usable)):
        for right in range(left + 1, len(usable)):
            first, second = usable[left], usable[right]
            b_first = coef * first.z_j
            b_second = coef * second.z_j
            denom = b_second - b_first
            if abs(denom) < 1e-15:
                continue
            a_first = first.bits / bits_scale - coef * first.aux_j
            a_second = second.bits / bits_scale - coef * second.aux_j
            theta = (a_second - a_first) / denom
            if not lo < theta < hi:
                continue
            if _admitted(first, theta, roll.energy_j, reserve) and _admitted(second, theta, roll.energy_j, reserve):
                if (a_first - b_first * theta) > 0.0 and (a_second - b_second * theta) > 0.0:
                    points.append(theta)
    return points


def _admitted(plan: Plan, theta: float, energy: float, reserve: float) -> bool:
    if not plan.ok:
        return False
    return energy + 1e-6 >= theta * plan.z_j + plan.aux_j + reserve


def _q_of_u(roll: Roll, probability_u: float, guard: float) -> float:
    clamped = min(max(probability_u, 1e-12), 1.0 - 1e-12)
    epsilon = roll.sigma * inv_norm(clamped)
    return _quantile_of(roll, epsilon, guard)


def _merge(points: list[float]) -> list[float]:
    ordered = sorted(points)
    merged = [ordered[0]]
    for value in ordered[1:]:
        if value - merged[-1] > 1e-9:
            merged.append(value)
    return merged


def _integrate_linear(u0: float, u1: float, nodes: int, roll: Roll, spec: dict, intercept: float, slope: float, guard: float, stats: dict | None) -> float:
    if stats is not None:
        stats["algebraic_integrals"] = int(stats.get("algebraic_integrals", 0)) + 1
    width = u1 - u0
    if width <= 1e-15:
        return 0.0
    if abs(slope) < 1e-15:
        return width * intercept
    total = 0.0
    for node, weight in gauss_legendre_unit(nodes):
        probability = u0 + width * node
        total += weight * (intercept - slope * _q_of_u(roll, probability, guard))
    return width * total


def _integrate_sampled(u0: float, u1: float, nodes: int, roll: Roll, spec: dict, guard: float, stats: dict | None) -> float:
    if stats is not None:
        stats["algebraic_integrals"] = int(stats.get("algebraic_integrals", 0)) + 1
    width = u1 - u0
    if width <= 1e-15:
        return 0.0
    total = 0.0
    for node, weight in gauss_legendre_unit(nodes):
        probability = u0 + width * node
        total += weight * value_at(_q_of_u(roll, probability, guard), roll.energy_j, roll.plans, spec, roll.failed)
    return width * total


def _constant_terminal(roll: Roll, spec: dict, guard: float) -> float:
    theta = fast_truncated_quantile(roll.mu, roll.lam, roll.lo, roll.hi, guard)
    return value_at(theta, roll.energy_j, roll.plans, spec, roll.failed)


def expectation_pair(roll: Roll, spec: dict, inner_nodes: int, stats: dict | None) -> tuple[float, float]:
    """Terminal with this interval assimilated, and the same terminal with that update removed.

    Motion, energy and the other UAV's belief are untouched. The removed case
    evaluates the pre-update quantile on the post-interval state.
    """
    plus = segment_expectation(roll, spec, inner_nodes, stats)
    minus = _constant_terminal(roll, spec, _guard(spec))
    if stats is not None:
        stats["algebraic_integrals"] = int(stats.get("algebraic_integrals", 0)) + 1
    return plus, minus


def plan_snapshot(roll: Roll, spec: dict) -> dict:
    """Business quotes at the pre-update q95 and at the zero-noise posterior q95."""
    guard = _guard(spec)
    reserve, _bits, _coef = _scales(spec)
    prior = fast_truncated_quantile(roll.mu, roll.lam, roll.lo, roll.hi, guard)
    posterior = posterior_quantile(roll.mu, roll.lam, roll.lo, roll.hi, roll.z_j, roll.y0, roll.sigma, guard)
    plans = []
    for plan in roll.plans:
        plans.append(
            {
                "label": plan.label,
                "bits": plan.bits,
                "z_j": plan.z_j,
                "aux_j": plan.aux_j,
                "ok": bool(plan.ok),
                "quote_prior_j": prior * plan.z_j + plan.aux_j,
                "quote_posterior_j": posterior * plan.z_j + plan.aux_j,
                "admitted_prior_q95": _admitted(plan, prior, roll.energy_j, reserve),
                "admitted_posterior_q95_eps0": _admitted(plan, posterior, roll.energy_j, reserve),
            }
        )
    prior_id = describe(prior, roll.energy_j, roll.plans, spec, roll.failed)[0]
    post_id = describe(posterior, roll.energy_j, roll.plans, spec, roll.failed)[0]

    def _label(index: int) -> str:
        if index < 0 or index >= len(roll.plans):
            return "none"
        return roll.plans[index].label or str(index)

    def _bits(index: int) -> float:
        if index < 0 or index >= len(roll.plans):
            return 0.0
        return float(roll.plans[index].bits)

    return {
        "prior_q95": prior,
        "posterior_q95_eps0": posterior,
        "predicted_bits_prior": _bits(prior_id),
        "predicted_bits_posterior": _bits(post_id),
        "active_prior": _label(prior_id),
        "active_posterior": _label(post_id),
        "plans": plans,
        "telemetry": bool(roll.telemetry),
        "z_j": roll.z_j,
        "aux_j": roll.aux_j,
        "radio_j": roll.radio_j,
        "energy_j": roll.energy_j,
        "executed": roll.executed,
        "intervened": bool(roll.intervened),
        "x": roll.x,
        "y": roll.y,
        "vx": roll.vx,
        "vy": roll.vy,
        "calibration_s": roll.calibration_s,
    }


def segment_expectation(roll: Roll, spec: dict, inner_nodes: int, stats: dict | None) -> float:
    started = time.perf_counter()
    try:
        guard = _guard(spec)
        if roll.failed or not roll.plans:
            return 0.0
        if (not roll.telemetry) or abs(roll.z_j) < 1e-9:
            if stats is not None:
                stats["algebraic_integrals"] = int(stats.get("algebraic_integrals", 0)) + 1
            return _constant_terminal(roll, spec, guard)
        cuts = [0.0, 1.0]
        for theta in _theta_breaks(roll, spec):
            epsilon = _invert_epsilon(roll, theta, guard, stats)
            if epsilon is None:
                continue
            probability = normal_cdf(epsilon / roll.sigma)
            if 0.0 < probability < 1.0:
                cuts.append(probability)
        cuts = _refine_cuts(cuts, roll, spec, guard, depth=0)
        total = 0.0
        for left, right in zip(cuts, cuts[1:]):
            probe = 0.5 * (left + right)
            _plan_id, intercept, slope, _value = describe(_q_of_u(roll, probe, guard), roll.energy_j, roll.plans, spec, roll.failed)
            total += _integrate_linear(left, right, inner_nodes, roll, spec, intercept, slope, guard, stats)
        return total
    finally:
        if stats is not None:
            stats["algebraic_seconds"] = float(stats.get("algebraic_seconds", 0.0)) + (time.perf_counter() - started)


def _refine_cuts(cuts: list[float], roll: Roll, spec: dict, guard: float, depth: int) -> list[float]:
    merged = _merge(cuts)
    if depth >= 6:
        return merged
    extra: list[float] = []
    for left, right in zip(merged, merged[1:]):
        if right - left < 1e-6:
            continue
        left_id = describe(_q_of_u(roll, left + 0.25 * (right - left), guard), roll.energy_j, roll.plans, spec, roll.failed)[0]
        right_id = describe(_q_of_u(roll, left + 0.75 * (right - left), guard), roll.energy_j, roll.plans, spec, roll.failed)[0]
        if left_id != right_id:
            extra.append(0.5 * (left + right))
    if not extra:
        return merged
    return _refine_cuts(merged + extra, roll, spec, guard, depth + 1)


def reference_expectation(roll: Roll, spec: dict, stats: dict | None) -> float:
    """Piecewise integral whose breaks come from a scan, not from the threshold solver."""
    started = time.perf_counter()
    try:
        guard = _guard(spec)
        if roll.failed or not roll.plans:
            return 0.0
        if (not roll.telemetry) or abs(roll.z_j) < 1e-9:
            if stats is not None:
                stats["algebraic_integrals"] = int(stats.get("algebraic_integrals", 0)) + 1
            return _constant_terminal(roll, spec, guard)
        rule = spec["integral"]
        scan = int(rule["reference_scan_nodes"])
        piece_nodes = int(rule["reference_piece_gl_nodes"])
        steps = int(rule["reference_bisect_steps"])
        points = [(index + 0.5) / scan for index in range(scan)]
        points.extend(node for node, _weight in gauss_legendre_unit(min(81, scan)))
        points = _merge(points)

        def plan_id(probability: float) -> int:
            return describe(_q_of_u(roll, probability, guard), roll.energy_j, roll.plans, spec, roll.failed)[0]

        identities = [plan_id(point) for point in points]
        cuts = [0.0, 1.0]
        for left, right, left_id, right_id in zip(points, points[1:], identities, identities[1:]):
            if left_id == right_id:
                continue
            lo, hi = left, right
            for _ in range(steps):
                mid = 0.5 * (lo + hi)
                if plan_id(mid) == left_id:
                    lo = mid
                else:
                    hi = mid
            cuts.append(0.5 * (lo + hi))
        cuts = _merge(cuts)
        total = 0.0
        for left, right in zip(cuts, cuts[1:]):
            total += _integrate_sampled(left, right, piece_nodes, roll, spec, guard, stats)
        return total
    finally:
        if stats is not None:
            stats["algebraic_seconds"] = float(stats.get("algebraic_seconds", 0.0)) + (time.perf_counter() - started)


def _noise_nodes(roll: Roll, spec: dict, nodes: int, stats: dict | None) -> float:
    guard = _guard(spec)
    if roll.failed or not roll.plans:
        return 0.0
    if (not roll.telemetry) or abs(roll.z_j) < 1e-9:
        return _constant_terminal(roll, spec, guard)
    total = 0.0
    for epsilon, weight in gaussian_quadrature(roll.sigma, nodes):
        total += weight * value_at(_quantile_of(roll, epsilon, guard), roll.energy_j, roll.plans, spec, roll.failed)
    if stats is not None:
        stats["algebraic_integrals"] = int(stats.get("algebraic_integrals", 0)) + 1
    return total


def score_segment(
    world: World,
    beliefs: list[Belief],
    spec: dict,
    factor: float,
    uav_index: int,
    *,
    include_calibrate: bool,
    stats: dict | None,
) -> list[tuple[float, Action]]:
    from wt.active_model_identification.policies import candidates

    mode = str(spec["integral"]["scoring"])
    if stats is not None:
        stats["score_calls"] = int(stats.get("score_calls", 0)) + 1
        if mode == "frozen":
            stats["fallback_score_calls"] = int(stats.get("fallback_score_calls", 0)) + 1
        else:
            stats["segment_score_calls"] = int(stats.get("segment_score_calls", 0)) + 1
    grid = quadrature(beliefs[uav_index], int(spec["belief"]["quadrature_nodes"]))
    inner = int(spec["integral"].get("segment_inner_nodes", 7))
    noise_nodes = int(spec["integral"].get("noise_nodes", 5))
    scored: list[tuple[float, Action]] = []
    for action in candidates(spec, world, uav_index, include_calibrate=include_calibrate):
        total = 0.0
        parts: list[float] = []
        weights: list[float] = []
        for theta, weight in grid:
            roll = roll_action(world, beliefs, action, uav_index, theta, spec, factor, stats)
            if mode == "frozen":
                future = _noise_nodes(roll, spec, noise_nodes, stats)
            else:
                future = segment_expectation(roll, spec, inner, stats)
            parts.append(weight * (roll.immediate + future))
            weights.append(weight)
        total = math.fsum(parts)
        _ = weights
        scored.append((total, action))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def action_values(
    world: World,
    beliefs: list[Belief],
    spec: dict,
    factor: float,
    uav_index: int,
    *,
    include_calibrate: bool,
    theta_nodes: int,
    noise: str,
    stats: dict | None,
) -> list[tuple[float, Action]]:
    from wt.active_model_identification.policies import candidates

    grid = quadrature(beliefs[uav_index], theta_nodes)
    inner = int(spec["integral"]["segment_inner_nodes"])
    scored: list[tuple[float, Action]] = []
    for action in candidates(spec, world, uav_index, include_calibrate=include_calibrate):
        parts = []
        for theta, weight in grid:
            roll = roll_action(world, beliefs, action, uav_index, theta, spec, factor, stats)
            if noise == "segment":
                future = segment_expectation(roll, spec, inner, stats)
            elif noise == "reference":
                future = reference_expectation(roll, spec, stats)
            else:
                raise KeyError(noise)
            parts.append(weight * (roll.immediate + future))
        scored.append((math.fsum(parts), action))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def _development_belief(spec: dict, state: dict) -> Belief:
    named = state["belief"]
    if named == "prior":
        return prior_belief(spec)
    lo, hi = (float(value) for value in spec["belief"]["support"])
    std = float(named["std"])
    return Belief(float(named["mu"]), 1.0 / (std * std), lo, hi)


def _development_world(spec: dict, state: dict) -> tuple[World, list[Belief]]:
    battery = [float(state["battery_j"]), float(state["battery_j"])]
    world = make_world(spec, battery, f"m13-dev-{state['name']}")
    if "x" in state:
        probe = world.uavs[0]
        probe.x = float(state["x"])
        probe.y = float(state["y"])
        probe.vx = float(state.get("vx", 0.0))
        probe.vy = float(state.get("vy", 0.0))
        probe.airborne = bool(state.get("airborne", False))
    if "cache_buoy" in state:
        moved = False
        for packet in world.packets:
            if packet.buoy == int(state["cache_buoy"]) and packet.holder == "buoy" and not moved:
                packet.holder = "uav:0"
                world.uavs[0].cache_ids.append(packet.id)
                moved = True
    belief = _development_belief(spec, state)
    return world, [belief.copy(), belief.copy()]


def _moment_row(spec: dict, nodes: int) -> dict:
    rule = spec["integral"]
    rows = []
    passed = True
    mean_tol = float(rule["theta_mean_abs_tol"])
    variance_tol = float(rule["theta_variance_abs_tol"])
    for case in rule["moment_cases"]:
        lo, hi = (float(value) for value in spec["belief"]["support"])
        std = float(case["std"])
        belief = Belief(float(case["mu"]), 1.0 / (std * std), lo, hi)
        mean, variance = belief.moments()
        approx_mean = expectation(belief, nodes, lambda theta: theta)
        approx_second = expectation(belief, nodes, lambda theta, _mean=mean: (theta - _mean) ** 2)
        width = equal_width_quadrature(belief, 7)
        row = {
            "case": case["name"],
            "analytic_mean": mean,
            "analytic_variance": variance,
            "mean_abs_error": abs(approx_mean - mean),
            "variance_abs_error": abs(approx_second - variance),
            "equal_width_7_mean": math.fsum(weight * theta for theta, weight in width),
        }
        rows.append(row)
        if row["mean_abs_error"] > mean_tol or row["variance_abs_error"] > variance_tol:
            passed = False
    return {"theta_nodes": nodes, "passed": passed, "cases": rows}


def _action_label(action: Action) -> str:
    return f"{action[0]}:{action[1]}"


def _rank_changed(left: list[tuple[float, Action]], right: list[tuple[float, Action]], tau: float) -> bool:
    from wt.active_model_identification.policies import pick_action

    return pick_action(left, tau)[1] != pick_action(right, tau)[1]


def select_integral_m13(spec: dict, factor: float) -> dict:
    """Choose the segmented rule, or record NOT_MET without promoting a failed grid."""
    from wt.active_model_identification.policies import candidates, pick_action

    rule = spec["integral"]
    tol = float(rule["q_abs_tol"])
    tau = float(rule["tie_abs"])
    stats: dict = {
        "rollouts": 0,
        "cache_hits": 0,
        "root_solves": 0,
        "algebraic_integrals": 0,
        "algebraic_seconds": 0.0,
        "planning_steps": 0,
        "planning_copies": 0,
        "rollout_cap": int(rule["development_rollout_budget"]),
        "budget_exhausted": False,
    }
    moment_checks = [_moment_row(spec, int(nodes)) for nodes in rule["theta_candidates"]]
    theta_counts = [int(nodes) for nodes in rule["theta_candidates"]]
    reference_count = int(rule["theta_reference_nodes"])
    inner_candidates = [int(nodes) for nodes in rule["segment_inner_candidates"]]
    grids_needed = theta_counts + [reference_count]
    states_out = []
    budget_hit = False
    started = time.perf_counter()
    moments_ok = all(bool(moment["passed"]) for moment in moment_checks)
    try:
        for state in rule["development_states"]:
            if not moments_ok:
                break
            print(f"segment development {state['name']}", flush=True)
            world, beliefs = _development_world(spec, state)
            uav_rows = []
            for uav_index, belief in enumerate(beliefs):
                grids = {count: quadrature(belief, count) for count in grids_needed}
                action_rows = []
                branches = set()
                for action in candidates(spec, world, uav_index, include_calibrate=True):
                    rolls: dict[float, Roll] = {}
                    for grid in grids.values():
                        for theta, _weight in grid:
                            if theta not in rolls:
                                rolls[theta] = roll_action(world, beliefs, action, uav_index, theta, spec, factor, stats)
                                branches.add((rolls[theta].intervened, rolls[theta].executed))
                    expectations = {count: {} for count in inner_candidates}
                    references = {}
                    for theta, roll in rolls.items():
                        for count in inner_candidates:
                            expectations[count][theta] = segment_expectation(roll, spec, count, stats)
                        references[theta] = reference_expectation(roll, spec, stats)

                    def combine(count: int, inner: int) -> float:
                        parts = []
                        for theta, weight in grids[count]:
                            future = expectations[inner][theta]
                            parts.append(weight * (rolls[theta].immediate + future))
                        return math.fsum(parts)

                    def combine_reference(count: int) -> float:
                        parts = []
                        for theta, weight in grids[count]:
                            parts.append(weight * (rolls[theta].immediate + references[theta]))
                        return math.fsum(parts)

                    by_theta = {str(count): {str(inner): combine(count, inner) for inner in inner_candidates} for count in grids_needed}
                    ref_value = combine_reference(reference_count)
                    action_rows.append(
                        {
                            "action": _action_label(action),
                            "q_by_theta": by_theta,
                            "q_reference": ref_value,
                        }
                    )
                uav_rows.append(
                    {
                        "uav": uav_index,
                        "actions": action_rows,
                        "execution_branches": len(branches),
                        "gaps_filled_later": True,
                    }
                )
            states_out.append({"state": state["name"], "uavs": uav_rows})
    except BudgetExceeded:
        budget_hit = True
    _ = pick_action, tol, tau
    return _finalize_selection(spec, stats, moment_checks, states_out, budget_hit, time.perf_counter() - started, factor)


def _parse_action(label: str) -> Action:
    kind, raw = label.split(":", 1)
    if raw == "None":
        return (kind, None)
    return (kind, int(raw))


def _finalize_selection(spec: dict, stats: dict, moment_checks: list[dict], states: list[dict], budget_hit: bool, elapsed: float, factor: float) -> dict:
    from wt.active_model_identification.policies import pick_action

    rule = spec["integral"]
    tol = float(rule["q_abs_tol"])
    tau = float(rule["tie_abs"])
    theta_counts = [int(nodes) for nodes in rule["theta_candidates"]]
    reference_count = int(rule["theta_reference_nodes"])
    inner_candidates = [int(nodes) for nodes in rule["segment_inner_candidates"]]
    frozen_theta = int(rule["frozen_theta_nodes"])
    frozen_noise = int(rule["frozen_noise_nodes"])
    per_state = []
    inner_worst = {count: 0.0 for count in inner_candidates}
    outer_worst = {count: 0.0 for count in theta_counts}
    rank_changed = {count: False for count in theta_counts}
    complete = (not budget_hit) and len(states) == len(rule["development_states"]) and all(moment["passed"] for moment in moment_checks)
    if complete:
        for state in states:
            state_gaps = {str(count): 0.0 for count in theta_counts}
            state_ranks = {str(count): False for count in theta_counts}
            state_inner = {str(count): 0.0 for count in inner_candidates}
            scored_actions = []
            for uav in state["uavs"]:
                for row in uav["actions"]:
                    action = _parse_action(row["action"])
                    for inner in inner_candidates:
                        gap = abs(float(row["q_by_theta"][str(reference_count)][str(inner)]) - float(row["q_reference"]))
                        state_inner[str(inner)] = max(state_inner[str(inner)], gap)
                        inner_worst[inner] = max(inner_worst[inner], gap)
                    scored_actions.append((uav["uav"], action, row))
                for count in theta_counts:
                    for inner in inner_candidates:
                        left = []
                        right = []
                        for row in uav["actions"]:
                            action = _parse_action(row["action"])
                            left.append((float(row["q_by_theta"][str(count)][str(inner)]), action))
                            right.append((float(row["q_reference"]), action))
                            gap = abs(left[-1][0] - right[-1][0])
                            state_gaps[str(count)] = max(state_gaps[str(count)], gap)
                        if _rank_changed(left, right, tau):
                            state_ranks[str(count)] = True
                            rank_changed[count] = True
                        outer_worst[count] = max(outer_worst[count], state_gaps[str(count)])
                        _ = pick_action
            per_state.append(
                {
                    "state": state["state"],
                    "execution_branches": [uav["execution_branches"] for uav in state["uavs"]],
                    "inner_max_abs_q": state_inner,
                    "outer_max_abs_q": state_gaps,
                    "rank_changed": state_ranks,
                    "actions": [
                        {
                            "uav": uav["uav"],
                            "action": row["action"],
                            "q_reference": row["q_reference"],
                            "q_by_theta": row["q_by_theta"],
                        }
                        for uav in state["uavs"]
                        for row in uav["actions"]
                    ],
                }
            )
    chosen_inner = None
    chosen_theta = None
    chosen_rank_changed = False
    attempted: list[dict] = []
    if complete and not stats.get("budget_exhausted"):
        for inner in inner_candidates:
            if inner_worst[inner] <= tol:
                chosen_inner = inner
                break
        if chosen_inner is not None:
            for count in theta_counts:
                worst = 0.0
                changed = False
                for state in per_state:
                    for uav_id in (0, 1):
                        rows = [row for row in state["actions"] if row["uav"] == uav_id]
                        left = [(float(row["q_by_theta"][str(count)][str(chosen_inner)]), _parse_action(row["action"])) for row in rows]
                        right = [(float(row["q_reference"]), _parse_action(row["action"])) for row in rows]
                        if left:
                            worst = max(worst, max(abs(a[0] - b[0]) for a, b in zip(left, right)))
                            if _rank_changed(left, right, tau):
                                changed = True
                attempted.append({"theta_nodes": count, "inner_nodes": chosen_inner, "max_abs_q": worst, "rank_changed": changed, "passed": worst <= tol})
                if worst <= tol:
                    chosen_theta = count
                    chosen_rank_changed = changed
                    break
    passed = chosen_inner is not None and chosen_theta is not None and not budget_hit
    if passed:
        status = "PASS"
        scoring = "segment"
        selection = "smallest segmented theta and inner node counts inside q_abs_tol against the independent piecewise reference"
    else:
        status = "NOT_MET"
        scoring = "frozen"
        chosen_theta = None
        chosen_inner = None
        if budget_hit or stats.get("budget_exhausted"):
            selection = "NOT_MET: development rollout budget was reached; the cap was not raised and the last grid was not kept"
        else:
            selection = "NOT_MET: no segmented candidate stayed inside q_abs_tol; missions use the frozen 15-point GL and 5-point noise rule"
    measured = None
    if passed:
        matched = next(item for item in attempted if item["passed"])
        measured = float(matched["max_abs_q"])
    elif attempted:
        measured = min(float(item["max_abs_q"]) for item in attempted)
    kept_inner = chosen_inner
    kept_theta = chosen_theta
    return {
        "numerical_status": status,
        "scoring": scoring,
        "theta_nodes": kept_theta if passed else None,
        "segment_inner_nodes": kept_inner if passed else None,
        "frozen_theta_nodes": frozen_theta,
        "frozen_noise_nodes": frozen_noise,
        "tie_abs": tau,
        "q_abs_tol": tol,
        "measured_q_gap": measured,
        "development_rank_changed": chosen_rank_changed if passed else None,
        "candidate_checks": attempted,
        "moment_checks": moment_checks,
        "states": per_state,
        "development_rollouts": int(stats.get("rollouts", 0)),
        "development_planning_steps": int(stats.get("planning_steps", 0)),
        "development_planning_copies": int(stats.get("planning_copies", 0)),
        "cache_hits": int(stats.get("cache_hits", 0)),
        "root_solves": int(stats.get("root_solves", 0)),
        "algebraic_integrals": int(stats.get("algebraic_integrals", 0)),
        "algebraic_seconds": float(stats.get("algebraic_seconds", 0.0)),
        "budget_exhausted": bool(budget_hit or stats.get("budget_exhausted")),
        "elapsed_s": elapsed,
        "integration_factor": factor,
        "selection": selection,
        "numerically_sensitive": status != "PASS" or chosen_rank_changed,
    }
