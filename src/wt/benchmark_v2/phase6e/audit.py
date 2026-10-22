"""Decision-state audits on the max-rate fly-toward trajectory."""
from __future__ import annotations

import numpy as np

from wt.benchmark_v2.bit_queue_v2 import L3BBitQueueEnvV2
from wt.benchmark_v2.phase6c.channel import geometry, link_rates
from wt.benchmark_v2.phase6c.rollout import choose_assignment
from wt.l3b.policies import fly_toward

from .generators import build_pools
from .protocol import GENERATORS, HOLD, KS, REFERENCE_GENERATOR, REFERENCE_K, assert_allowed
from .value import argmax_value, batch_j15, rollout_edges


def _key(assignment) -> tuple[int, ...]:
    return tuple(int(item) for item in np.asarray(assignment, dtype=np.int64).reshape(-1))


def _mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if np.isfinite(value)]
    if not clean:
        return None
    return float(np.mean(clean))


def _edge_stats(features: dict, assignment: tuple[int, ...], rollout: dict) -> dict:
    distances, etas, queues, rates, lifetimes, firsts = [], [], [], [], [], []
    for uav, buoy in enumerate(assignment):
        if int(buoy) < 0:
            continue
        distances.append(float(features["distance"][uav, buoy]))
        etas.append(float(features["eta"][uav, buoy]))
        queues.append(float(features["queue"][buoy]))
        rates.append(float(features["rate"][uav, buoy]))
        lifetimes.append(float(rollout["lifetime"][buoy]))
        first = int(rollout["first_service"][uav])
        if first >= 0:
            firsts.append(float(first))
    return {
        "edges": len(distances),
        "mean_distance": _mean(distances),
        "mean_eta": _mean(etas),
        "mean_queue": _mean(queues),
        "mean_rate": _mean(rates),
        "mean_lifetime": _mean(lifetimes),
        "mean_first_service": _mean(firsts),
    }


def _coverage(reference: tuple[int, ...], top: np.ndarray) -> tuple[float, int, int]:
    edges = 0
    covered = 0
    for uav, buoy in enumerate(reference):
        if int(buoy) < 0:
            continue
        edges += 1
        if int(buoy) in {int(item) for item in top[uav]}:
            covered += 1
    if edges == 0:
        return 1.0, 0, 0
    return float(covered / edges), int(covered), int(edges)


def evaluate_scenario(scenario: int, rules: tuple[tuple[str, int], ...], *, diagnostics: bool) -> dict:
    assert_allowed(int(scenario), split="design" if diagnostics else "confirm")
    env = L3BBitQueueEnvV2()
    env.reset(seed=int(scenario))
    if int(env.config.option_hold) != HOLD:
        raise RuntimeError("phase-6e refuses a changed hold")
    rng = np.random.default_rng(17_000_003 + int(scenario))
    pending = 0
    assignment = np.full(env.config.num_uavs, -1, dtype=np.int64)
    rows: list[dict] = []
    reachability: list[dict] = []
    maxrate_rows: list[dict] = []
    interference_rows: list[dict] = []
    idle_rows: list[dict] = []
    conservation_error = 0.0
    illegal = 0
    while int(env.t) < int(env.config.episode_horizon):
        if pending <= 0:
            geo = geometry(env)
            assignment, _info = choose_assignment("max_rate_independent", env, rng, geo)
            packet, error, illegal_now = _evaluate_decision(env, assignment, rules, diagnostics=diagnostics)
            conservation_error = max(conservation_error, error)
            illegal += int(illegal_now)
            rows.extend(packet["rows"])
            reachability.extend(packet["reachability"])
            maxrate_rows.extend(packet["maxrate"])
            interference_rows.extend(packet["interference"])
            idle_rows.extend(packet["idle"])
            pending = HOLD
        env.step(fly_toward(env, assignment), assignment)
        pending -= 1
    conservation_error = max(conservation_error, float(np.max(np.abs(env.conservation_error_bits()))))
    packet = {
        "scenario_id": int(scenario),
        "rows": rows,
        "reachability": reachability,
        "maxrate": maxrate_rows,
        "interference": interference_rows,
        "idle": idle_rows,
        "conservation_error": float(conservation_error),
        "conservation_ok": bool(conservation_error <= float(env.tol_bits)),
        "illegal_assignments": int(illegal),
    }
    for name in ("rows", "reachability", "maxrate", "interference", "idle"):
        for row in packet[name]:
            row["scenario_id"] = int(scenario)
    return packet


def _evaluate_decision(env, maxrate_assignment, rules, *, diagnostics: bool) -> tuple[dict, float, int]:
    horizon = min(HOLD, int(env.config.episode_horizon) - int(env.t))
    needed = set(rules)
    if diagnostics:
        needed.update((name, int(k)) for name in GENERATORS for k in KS)
    needed.add((REFERENCE_GENERATOR, REFERENCE_K))
    pools, packed, illegal = build_pools(env, tuple(sorted(needed)))
    features = packed["features"]
    tops = packed["tops"]
    union = []
    seen: set[tuple[int, ...]] = set()
    for key in needed:
        for assignment in pools[key]:
            identity = _key(assignment)
            if identity not in seen:
                seen.add(identity)
                union.append(assignment)
    values, elapsed, shadow_error = batch_j15(env, union, horizon, blind_interference=False)
    seconds_per = float(elapsed / max(len(values), 1))
    reference, j_ref = argmax_value(pools[(REFERENCE_GENERATOR, REFERENCE_K)], values)
    rows = []
    for name, k in sorted(needed):
        pool = pools[(name, int(k))]
        _best, j_best = argmax_value(pool, values)
        edge_coverage, covered, edges = _coverage(reference, tops[(name, int(k))])
        regret = float(j_ref - j_best)
        rows.append(
            {
                "time_step": int(env.t),
                "generator": name,
                "k": int(k),
                "edge_coverage": float(edge_coverage),
                "covered_edges": int(covered),
                "reference_edges": int(edges),
                "full_assignment_coverage": float(1.0 if edges == 0 or covered == edges else 0.0),
                "j_ref": float(j_ref),
                "j_best": float(j_best),
                "regret_bits": regret,
                "nregret": float(regret / max(j_ref, 1.0)),
                "zero_regret": int(regret <= 1.0),
                "pool_size": int(len(pool)),
                "per_uav_buoy_candidates": int(k),
                "runtime_s": float(len(pool) * seconds_per),
                "illegal_assignments": int(illegal),
                "reference_assignment": ",".join(str(item) for item in reference),
                "self_reference": int(name == REFERENCE_GENERATOR and int(k) == REFERENCE_K),
                "global_oracle": False,
                "non_causal_upper_bound": False,
            }
        )
    packet = {"rows": rows, "reachability": [], "maxrate": [], "interference": [], "idle": []}
    if diagnostics:
        packet.update(
            _diagnostics(env, features, tops, reference, pools[(REFERENCE_GENERATOR, REFERENCE_K)], maxrate_assignment, horizon, rows)
        )
        packet["rows"] = rows
    return packet, float(shadow_error), int(illegal)


def _diagnostics(env, features, tops, reference, reference_pool, maxrate_assignment, horizon, rows) -> dict:
    reference_rollout = rollout_edges(env, reference, horizon)
    maxrate_key = _key(maxrate_assignment)
    maxrate_rollout = rollout_edges(env, maxrate_key, horizon)
    rate_top3 = tops[(REFERENCE_GENERATOR, 3)]
    reachability = []
    for uav, buoy in enumerate(reference):
        if int(buoy) < 0:
            continue
        eta = float(features["eta"][uav, buoy])
        reachability.append(
            {
                "time_step": int(env.t),
                "uav": int(uav),
                "buoy": int(buoy),
                "distance_m": float(features["distance"][uav, buoy]),
                "eta_steps": eta,
                "in_service_radius": int(bool(features["in_range"][uav, buoy])),
                "rate_bps": float(features["rate"][uav, buoy]),
                "q_bits": float(features["queue"][buoy]),
                "in_rate_top3": int(int(buoy) in {int(item) for item in rate_top3[uav]}),
                "geometric_service_fraction": float(max(0.0, horizon - eta) / max(horizon, 1)),
                "first_service_step": int(reference_rollout["first_service"][uav]),
                "queue_lifetime_steps": int(reference_rollout["lifetime"][buoy]),
                "global_oracle": False,
            }
        )
    ref_stats = _edge_stats(features, reference, reference_rollout)
    rate_stats = _edge_stats(features, maxrate_key, maxrate_rollout)
    ref_edges = {(uav, int(buoy)) for uav, buoy in enumerate(reference) if int(buoy) >= 0}
    rate_edges = {(uav, int(buoy)) for uav, buoy in enumerate(maxrate_key) if int(buoy) >= 0}
    shared = len(ref_edges & rate_edges)
    maxrate = [
        {
            "time_step": int(env.t),
            "ref_edges": int(ref_stats["edges"]),
            "maxrate_edges": int(rate_stats["edges"]),
            "shared_edge_fraction": float(shared / ref_stats["edges"]) if ref_stats["edges"] else 1.0,
            "ref_mean_distance": ref_stats["mean_distance"],
            "ref_mean_eta": ref_stats["mean_eta"],
            "ref_mean_queue": ref_stats["mean_queue"],
            "ref_mean_rate": ref_stats["mean_rate"],
            "ref_mean_lifetime": ref_stats["mean_lifetime"],
            "ref_mean_first_service": ref_stats["mean_first_service"],
            "maxrate_mean_distance": rate_stats["mean_distance"],
            "maxrate_mean_eta": rate_stats["mean_eta"],
            "maxrate_mean_queue": rate_stats["mean_queue"],
            "maxrate_mean_rate": rate_stats["mean_rate"],
            "maxrate_mean_lifetime": rate_stats["mean_lifetime"],
            "maxrate_mean_first_service": rate_stats["mean_first_service"],
            "global_oracle": False,
        }
    ]
    blinded, _elapsed, _error = batch_j15(env, reference_pool, horizon, blind_interference=True)
    blinded_assignment, _blind_value = argmax_value(reference_pool, blinded)
    disagreement = int(blinded_assignment != reference)
    geo = geometry(env)
    _sinr, _rate, mean_interference = link_rates(
        geo["gain"],
        np.asarray(maxrate_assignment, dtype=np.int64),
        env.queue,
        geo["in_range"],
        bandwidth_hz=float(env.config.bandwidth_hz),
        tx_power=float(env.config.tx_power),
        noise_power=float(env.config.noise_power),
        kappa_same=float(env.config.kappa_same_cluster),
        kappa_cross=float(env.config.kappa_cross_cluster),
    )
    interference = [
        {
            "time_step": int(env.t),
            "generator": row["generator"],
            "k": int(row["k"]),
            "assignment_disagreement": int(disagreement),
            "mean_interference": float(mean_interference),
            "full_assignment_coverage": float(row["full_assignment_coverage"]),
            "edge_coverage": float(row["edge_coverage"]),
            "non_causal_upper_bound": False,
        }
        for row in rows
    ]
    idle = [
        {
            "time_step": int(env.t),
            "idle_always_legal": 1,
            "idle_occupies_buoy_slot": 0,
            "per_uav_buoy_candidates_are_buoys_only": 1,
            "idle_encoding": -1,
        }
    ]
    return {"reachability": reachability, "maxrate": maxrate, "interference": interference, "idle": idle}


def audit_design(scenario: int) -> dict:
    rules = tuple((name, int(k)) for name in GENERATORS for k in KS)
    return evaluate_scenario(int(scenario), rules, diagnostics=True)


def audit_confirm(job: tuple) -> dict:
    scenario, generator, k = job
    rules = [(REFERENCE_GENERATOR, REFERENCE_K)]
    if (str(generator), int(k)) != (REFERENCE_GENERATOR, REFERENCE_K):
        rules.append((str(generator), int(k)))
    return evaluate_scenario(int(scenario), tuple(rules), diagnostics=False)
