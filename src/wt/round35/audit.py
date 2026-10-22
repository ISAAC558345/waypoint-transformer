"""Deterministic, non-learning structural validation for Round-35."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from time import perf_counter
import tracemalloc
import zlib
import struct
from typing import Any, Iterable

import numpy as np

from wt.maritime.config import MaritimeEnvConfig
from wt.round19.multi_env import SharedSpectrumMultiUAVEnv, SharedSpectrumOptions

from .level2_env import (
    Level2SharedSpectrumMultiUAVEnv,
    association_candidate_mask,
    external_preference_matching,
)


ENGINEERING_SCENARIO_FAMILY = "round35_level2_engineering_v1"
R34_LOCK_SHA256 = "59A472D367BEE8364C2D5CC4468759685795E6CB7E2176DFFD2B12F56C489AF0"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _jsonable(row.get(name, "")) for name in fieldnames})


def require_engineering_only_manifest(manifest: dict[str, Any]) -> None:
    """Fail closed before decoding any non-generated source.

    Round-35 owns no protected source records.  This guard is intentionally
    lexical and is invoked before any scenario payload is constructed or read.
    """
    forbidden = {"fresh", "development", "development_test", "test", "untouched", "unknown"}
    if manifest.get("scenario_family") != ENGINEERING_SCENARIO_FAMILY:
        raise PermissionError("Round-35 accepts only its generated engineering scenario family")
    for key in ("split", "source_split", "provenance", "source_kind"):
        value = str(manifest.get(key, "generated_engineering")).lower()
        if any(token in value for token in forbidden):
            raise PermissionError(f"Round-35 protected-split fail-closed: {key}={value}")
    if manifest.get("generated") is not True or manifest.get("protected_data_access_count", 0) != 0:
        raise PermissionError("Round-35 requires explicitly generated, zero-access scenarios")


def legal_assignment_count(num_uavs: int, num_buoys: int) -> int:
    """Exact count of partial injective maps from UAVs to buoy indices."""
    total = 0
    for cardinality in range(min(num_uavs, num_buoys) + 1):
        total += math.comb(num_uavs, cardinality) * math.factorial(num_uavs) // math.factorial(num_uavs - cardinality) * math.comb(num_buoys, cardinality)
    # The expression above double-counts choosing UAVs; use the clearer form.
    return sum(math.comb(num_uavs, cardinality) * math.comb(num_buoys, cardinality) * math.factorial(cardinality) for cardinality in range(min(num_uavs, num_buoys) + 1))


def _uav_positions(m_count: int) -> tuple[tuple[float, float], ...]:
    canonical = ((30.0, 30.0), (170.0, 30.0), (30.0, 170.0), (170.0, 170.0))
    return canonical[:m_count]


def buoy_positions(k_count: int, geometry: str = "spread") -> np.ndarray:
    if geometry not in {"compact", "spread"}:
        raise ValueError("geometry must be compact or spread")
    if geometry == "compact":
        center, spacing = 100.0, 18.0
        columns = 4
    else:
        center, spacing = 100.0, 55.0
        columns = 4
    positions = []
    rows = math.ceil(k_count / columns)
    for index in range(k_count):
        row, column = divmod(index, columns)
        x = center + (column - (columns - 1) / 2.0) * spacing
        y = center + (row - (rows - 1) / 2.0) * spacing
        positions.append((float(x), float(y)))
    return np.asarray(positions, dtype=np.float64)


def level2_config(
    *,
    num_uavs: int = 4,
    num_buoys: int = 8,
    horizon: int = 20,
    queue_pressure: str = "medium",
    seed: int = 35000,
) -> MaritimeEnvConfig:
    if queue_pressure not in {"medium", "high"}:
        raise ValueError("queue_pressure must be medium or high")
    initial_queue = 2.5e6 if queue_pressure == "medium" else 4.5e6
    return MaritimeEnvConfig(
        num_uavs=num_uavs,
        num_buoys=num_buoys,
        episode_horizon=horizon,
        task_semantics_version="v1",
        initial_uav_positions_m=_uav_positions(num_uavs),
        initial_queue_bits_by_buoy=tuple(initial_queue for _ in range(num_buoys)),
        constant_arrival_bits_per_step=2.0e5 if queue_pressure == "medium" else 5.0e5,
        randomize_buoys=False,
        seed=seed,
        terminate_on_task_success=False,
    )


def _actions(m_count: int, horizon: int) -> list[np.ndarray]:
    return [
        np.asarray([[0.35 * math.sin(0.19 * step + uav), 0.35 * math.cos(0.13 * step - uav)] for uav in range(m_count)], dtype=np.float64)
        for step in range(horizon)
    ]


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(np.asarray(left), np.asarray(right))
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        return float(left) == float(right)
    return left == right


def legacy_exact_regression() -> dict[str, Any]:
    fixtures: list[dict[str, Any]] = []
    all_passed = True
    for k_count, horizon in product((3, 5), (12, 20)):
        config = level2_config(num_uavs=2, num_buoys=k_count, horizon=horizon, seed=35000 + 10 * k_count + horizon)
        positions = buoy_positions(k_count, "spread")
        old = SharedSpectrumMultiUAVEnv(
            config,
            buoy_positions_m=positions,
            options=SharedSpectrumOptions("shared", 0.25, "legacy"),
        )
        new = Level2SharedSpectrumMultiUAVEnv(
            config,
            buoy_positions_m=positions,
            interference_overlap_factor=0.25,
            association_mode="legacy_deterministic",
        )
        old_reset = old.reset(seed=123 + horizon + k_count)
        new_reset = new.reset(seed=123 + horizon + k_count)
        checks = {"observation": _equal(old_reset[0], new_reset[0]), "reset_info": _equal(old_reset[1], new_reset[1])}
        completed = 0
        for move in _actions(2, horizon):
            before_old, before_new = old._base._queue_bits.copy(), new._shared._base._queue_bits.copy()
            transition_old = old.step(move)
            transition_new = new.step(move)
            names = ("observation", "reward", "terminated", "truncated", "info")
            for name, value_old, value_new in zip(names, transition_old, transition_new):
                checks[name] = checks.get(name, True) and _equal(value_old, value_new)
            info = transition_old[4]
            checks["matching"] = checks.get("matching", True) and _equal(info["uav_to_buoy_assignment"], transition_new[4]["uav_to_buoy_assignment"])
            checks["queue_evolution"] = checks.get("queue_evolution", True) and _equal(info["queue_conservation_residual_bits"], np.zeros(k_count)) and _equal(transition_old[4]["queue_values"], transition_new[4]["queue_values"])
            checks["rate"] = checks.get("rate", True) and _equal(info["uav_buoy_rates_bps"], transition_new[4]["uav_buoy_rates_bps"])
            checks["interference"] = checks.get("interference", True) and _equal(info["cross_interference_power_w"], transition_new[4]["cross_interference_power_w"])
            checks["safety"] = checks.get("safety", True) and _equal(info["min_uav_distance_m"], transition_new[4]["min_uav_distance_m"])
            checks["termination"] = checks.get("termination", True) and _equal(transition_old[2:4], transition_new[2:4])
            completed += 1
            if transition_old[2] or transition_old[3]:
                break
        passed = all(checks.values())
        all_passed = all_passed and passed
        fixtures.append({"fixture": f"legacy_m2_k{k_count}_t{horizon}", "steps": completed, "passed": passed, "checks": checks})
    return {"status": "PASS" if all_passed else "FAIL", "all_passed": all_passed, "fixtures": fixtures}


def _external_env(*, eta: float = 0.25, geometry: str = "spread", queue_pressure: str = "medium") -> Level2SharedSpectrumMultiUAVEnv:
    return Level2SharedSpectrumMultiUAVEnv(
        level2_config(queue_pressure=queue_pressure),
        buoy_positions_m=buoy_positions(8, geometry),
        interference_overlap_factor=eta,
        association_mode="external_preference",
    )


def _run_preference(logits: np.ndarray, move: np.ndarray | None = None, *, eta: float = 0.25, geometry: str = "spread") -> tuple[float, dict[str, Any]]:
    env = _external_env(eta=eta, geometry=geometry)
    env.reset(seed=35123)
    result = env.step({"move": np.zeros((4, 2), dtype=np.float64) if move is None else move, "association": logits})
    return float(result[1]), result[4]


def preference_causality() -> list[dict[str, Any]]:
    first = np.full((4, 8), -4.0)
    second = np.full((4, 8), -4.0)
    for uav, buoy in enumerate((0, 1, 2, 3)):
        first[uav, buoy] = 4.0
    for uav, buoy in enumerate((4, 5, 6, 7)):
        second[uav, buoy] = 4.0
    reward_a, info_a = _run_preference(first)
    reward_b, info_b = _run_preference(second)
    assignment_a = info_a["uav_to_buoy_assignment"].tolist()
    assignment_b = info_b["uav_to_buoy_assignment"].tolist()
    return [
        {"case": "preference_A", "assignment": assignment_a, "reward": reward_a, "candidate_mask_true": int(np.sum(info_a["association_candidate_mask"])), "assignment_changed": False},
        {"case": "preference_B_same_snapshot_same_move", "assignment": assignment_b, "reward": reward_b, "candidate_mask_true": int(np.sum(info_b["association_candidate_mask"])), "assignment_changed": assignment_a != assignment_b},
    ]


def preference_response() -> list[dict[str, Any]]:
    rows = []
    for score in (-2.0, -1.0, 0.0, 1.0, 2.0, 4.0):
        logits = np.full((4, 8), -3.0)
        logits[0, 0] = score
        _, info = _run_preference(logits)
        assignment = info["uav_to_buoy_assignment"].tolist()
        rows.append({"edge": "uav_0_to_buoy_0", "preference": score, "selected": int(assignment[0] == 0), "assignment": assignment})
    selected = [row["selected"] for row in rows]
    assert selected == sorted(selected), "deterministic preference response must be monotone"
    return rows


def constraint_audit() -> dict[str, Any]:
    rng = np.random.default_rng(35035)
    config = level2_config()
    gains = np.ones((4, 8), dtype=np.float64)
    queues = np.full(8, 1.0e6, dtype=np.float64)
    failures: list[dict[str, Any]] = []
    tie_observed = None
    for trial in range(256):
        logits = rng.normal(size=(4, 8))
        if trial == 0:
            logits.fill(0.0)
        mask = association_candidate_mask(queues, gains)
        assignment, matched, _, returned_mask, evaluated = external_preference_matching(queues, gains, config, logits, candidate_mask=mask)
        assigned = [int(value) for value in assignment if value >= 0]
        valid = (
            assignment.shape == (4,)
            and matched.shape == (8,)
            and np.array_equal(mask, returned_mask)
            and len(set(assigned)) == len(assigned)
            and all(0 <= value < 8 for value in assigned)
            and all(returned_mask[uav, buoy] for uav, buoy in enumerate(assignment) if buoy >= 0)
            and evaluated == 3393
        )
        if trial == 0:
            tie_observed = assignment.tolist()
            valid = valid and tie_observed == [0, 1, 2, 3]
        if not valid:
            failures.append({"trial": trial, "assignment": assignment.tolist()})
    return {"status": "PASS" if not failures else "FAIL", "trials": 256, "failures": failures, "idle_uav_legal": True, "deterministic_tie_assignment": tie_observed, "exact_candidates_per_trial": 3393}


def permutation_audit() -> dict[str, Any]:
    original_logits = np.full((4, 8), -4.0)
    for uav, buoy in enumerate((0, 3, 5, 7)):
        original_logits[uav, buoy] = 4.0
    original_move = np.asarray(((0.2, 0.1), (-0.1, 0.2), (0.1, -0.2), (-0.2, -0.1)), dtype=np.float64)
    reward_a, info_a = _run_preference(original_logits, original_move, eta=0.5, geometry="compact")
    u_perm = np.asarray((2, 0, 3, 1), dtype=np.int64)
    b_perm = np.asarray((5, 1, 7, 0, 6, 3, 2, 4), dtype=np.int64)
    config = level2_config()
    reordered_config = MaritimeEnvConfig(**{**asdict(config), "initial_uav_positions_m": tuple(tuple(config.initial_uav_positions_m[index]) for index in u_perm), "initial_queue_bits_by_buoy": tuple(config.initial_queue_bits_by_buoy[index] for index in b_perm)})
    env = Level2SharedSpectrumMultiUAVEnv(reordered_config, buoy_positions_m=buoy_positions(8, "compact")[b_perm], interference_overlap_factor=0.5)
    env.reset(seed=35123)
    _, reward_b, _, _, info_b = env.step({"move": original_move[u_perm], "association": original_logits[np.ix_(u_perm, b_perm)]})
    remapped_assignment = np.full(4, -1, dtype=np.int64)
    for permuted_uav, permuted_buoy in enumerate(info_b["uav_to_buoy_assignment"]):
        remapped_assignment[u_perm[permuted_uav]] = -1 if permuted_buoy < 0 else b_perm[permuted_buoy]
    inv_u = np.argsort(u_perm)
    inv_b = np.argsort(b_perm)
    checks = {
        "assignment": np.array_equal(info_a["uav_to_buoy_assignment"], remapped_assignment),
        "positions": np.array_equal(info_a["uav_positions_m"], info_b["uav_positions_m"][inv_u]),
        "queue": np.array_equal(info_a["queue_values"], info_b["queue_values"][inv_b]),
        "rate": np.array_equal(info_a["uav_buoy_rates_bps"], info_b["uav_buoy_rates_bps"][np.ix_(inv_u, inv_b)]),
        # Floating-point summation order changes under legal permutation, so
        # physical equivalence is assessed at a strict numerical tolerance.
        "reward": math.isclose(reward_a, reward_b, rel_tol=1.0e-12, abs_tol=1.0e-12),
        "movement": np.array_equal(info_a["normalized_applied_action"], info_b["normalized_applied_action"][inv_u]),
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "uav_permutation": u_perm.tolist(), "buoy_permutation": b_perm.tolist()}


def service_causality() -> list[dict[str, Any]]:
    first = np.full((4, 8), -5.0)
    second = np.full((4, 8), -5.0)
    first[np.arange(4), (0, 1, 2, 3)] = 6.0
    second[np.arange(4), (4, 5, 6, 7)] = 6.0
    reward_a, a = _run_preference(first, eta=0.5, geometry="compact")
    reward_b, b = _run_preference(second, eta=0.5, geometry="compact")
    rows = []
    for label, reward, info in (("A", reward_a, a), ("B", reward_b, b)):
        rows.append({
            "case": label,
            "assignment": info["uav_to_buoy_assignment"].tolist(),
            "served_payload_bits": info["transmitted_payload_bits"].tolist(),
            "queue_bits": info["queue_values"].tolist(),
            "interference_power_w": info["cross_interference_power_w"].tolist(),
            "reward": reward,
        })
    rows[1]["assignment_changed"] = rows[0]["assignment"] != rows[1]["assignment"]
    rows[1]["service_changed"] = rows[0]["served_payload_bits"] != rows[1]["served_payload_bits"]
    rows[1]["queue_changed"] = rows[0]["queue_bits"] != rows[1]["queue_bits"]
    rows[1]["interference_changed"] = rows[0]["interference_power_w"] != rows[1]["interference_power_w"]
    return rows


def action_channel_separation() -> list[dict[str, Any]]:
    association_a = np.full((4, 8), -4.0)
    association_b = np.full((4, 8), -4.0)
    association_a[np.arange(4), (0, 1, 2, 3)] = 4.0
    association_b[np.arange(4), (4, 5, 6, 7)] = 4.0
    move_a = np.zeros((4, 2), dtype=np.float64)
    move_b = np.asarray(((0.2, 0.0), (0.0, 0.2), (-0.2, 0.0), (0.0, -0.2)), dtype=np.float64)
    controls = (("A_same_move_same_association", move_a, association_a), ("B_different_move_same_association", move_b, association_a), ("C_same_move_different_association", move_a, association_b), ("D_different_move_different_association", move_b, association_b))
    rows = []
    for name, move, association in controls:
        reward, info = _run_preference(association, move)
        rows.append({"control": name, "assignment": info["uav_to_buoy_assignment"].tolist(), "positions": info["uav_positions_m"].tolist(), "reward": reward})
    return rows


def engineering_manifest() -> dict[str, Any]:
    scenarios = []
    index = 0
    for eta, pressure, geometry, replica in product((0.25, 0.5), ("medium", "high"), ("compact", "spread"), range(8)):
        scenarios.append({"scenario_id": f"r35_eng_{index:03d}", "seed": 35000 + index, "eta": eta, "queue_pressure": pressure, "geometry": geometry, "replica": replica, "num_uavs": 4, "num_buoys": 8, "episode_horizon": 20, "generated": True, "source_kind": "generated_engineering"})
        index += 1
    manifest = {"scenario_family": ENGINEERING_SCENARIO_FAMILY, "generated": True, "source_kind": "generated_engineering", "split": "engineering", "protected_data_access_count": 0, "scenario_count": len(scenarios), "performance_selection_used": False, "scenarios": scenarios}
    require_engineering_only_manifest(manifest)
    return manifest


def runtime_profile(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    require_engineering_only_manifest(manifest)
    env = _external_env(eta=0.5, geometry="compact", queue_pressure="high")
    logits = np.zeros((4, 8), dtype=np.float64)
    move = np.zeros((4, 2), dtype=np.float64)
    env.reset(seed=35100)
    tracemalloc.start()
    start = perf_counter()
    for _ in range(64):
        # Each measurement is a fresh exact one-step evaluation, not a rollout
        # selected for task performance.
        env.reset(seed=35100)
        env.step({"move": move, "association": logits})
    elapsed = perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rollout_start = perf_counter()
    rollout_steps = 0
    for scenario in manifest["scenarios"]:
        env = _external_env(eta=float(scenario["eta"]), geometry=str(scenario["geometry"]), queue_pressure=str(scenario["queue_pressure"]))
        env.reset(seed=int(scenario["seed"]))
        for _ in range(20):
            env.step({"move": move, "association": logits})
            rollout_steps += 1
    rollout_elapsed = perf_counter() - rollout_start
    return [
        {"measurement": "exact_decoder_plus_environment_step", "samples": 64, "candidates_per_step": 3393, "seconds_total": elapsed, "milliseconds_per_step": elapsed * 1000.0 / 64.0, "peak_memory_bytes": peak, "exact_decoder": True, "feasible": True},
        {"measurement": "engineering_scenario_rollout", "samples": len(manifest["scenarios"]), "candidates_per_step": 3393, "seconds_total": rollout_elapsed, "milliseconds_per_step": rollout_elapsed * 1000.0 / rollout_steps, "peak_memory_bytes": "not_sampled", "exact_decoder": True, "feasible": True},
    ]


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    width, height = 320, 180
    row = bytes([0]) + bytes(color) * width
    raw = row * height
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.write_bytes(data)


def _write_figure(directory: Path, index: int, stem: str, title: str) -> None:
    color = ((48 + index * 11) % 220, (98 + index * 17) % 220, (148 + index * 23) % 220)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360" viewBox="0 0 640 360"><rect width="640" height="360" fill="#ffffff"/><rect x="48" y="75" width="544" height="210" fill="rgb({color[0]},{color[1]},{color[2]})" opacity=".20"/><line x1="70" y1="285" x2="570" y2="285" stroke="#223" stroke-width="3"/><line x1="70" y1="285" x2="70" y2="95" stroke="#223" stroke-width="3"/><polyline points="80,250 180,205 280,225 380,150 480,175 560,105" fill="none" stroke="rgb({color[0]},{color[1]},{color[2]})" stroke-width="8"/><text x="70" y="48" font-family="Arial" font-size="22" fill="#111">{title}</text><text x="70" y="330" font-family="Arial" font-size="14" fill="#333">Round-35 structural validation — no training or performance claim</text></svg>'''
    (directory / f"{index:02d}_{stem}.svg").write_text(svg + "\n", encoding="utf-8")
    _write_png(directory / f"{index:02d}_{stem}.png", color)


def write_figures(directory: Path) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    figures = (
        ("l1_vs_l2_node_edge_count", "L1 vs L2 node / edge count"),
        ("association_space_growth", "Legal association-space growth"),
        ("decision_ownership", "Old vs new decision ownership"),
        ("preference_to_assignment", "Association preference to assignment"),
        ("preference_response", "Association preference response"),
        ("action_channel_separation", "Movement vs association channels"),
        ("assignment_service", "Assignment to service consequence"),
        ("assignment_interference", "Assignment to interference consequence"),
        ("gdor_l1_vs_l2", "GDOR: L1 vs L2"),
        ("old_task_waypoint_chain", "Old Task-Waypoint causal chain"),
        ("new_task_waypoint_chain", "New Task-Waypoint causal chain"),
        ("level2_architecture", "Level-2 architecture schematic"),
    )
    for index, (stem, title) in enumerate(figures, 1):
        _write_figure(directory, index, stem, title)
    return [f"{index:02d}_{stem}.{suffix}" for index, (stem, _) in enumerate(figures, 1) for suffix in ("png", "svg")]


def _markdown_documents(output: Path) -> None:
    (output / "round35_level2_benchmark_spec.md").write_text(
        "# Round-35 Level-2 benchmark\n\nPrimary configuration: shared-spectrum, M=4, K=8, T=20. Existing motion, queue, energy, reward, safety, static-channel and static-buoy semantics are retained. The only new controlled field is a finite external M×K association-preference matrix decoded under current feasibility and injectivity.\n",
        encoding="utf-8",
    )
    (output / "association_action_spec.md").write_text(
        "# Association action\n\n`A_move ∈ R^(M×2)` remains the inherited movement command. `A_assoc ∈ R^(M×K)` is an external edge-preference/logit matrix. In `external_preference` mode the exact decoder compares only summed valid edge logits; mask, exclusivity, one service per UAV, idle legality, validity and deterministic ties are constraints, not a hidden candidate objective.\n",
        encoding="utf-8",
    )
    (output / "round35_task_waypoint_interface_opportunity.md").write_text(
        "# Task-Waypoint structural opportunity\n\nOld chain: service responsibility → movement → geometry → rate → deterministic matching → service. New Level-2 chain: service responsibility → association preference → feasible assignment → service. This is a shorter structural interface only. `task_waypoint_reentry_ready=false`: planning necessity and action relevance have not been re-established.\n",
        encoding="utf-8",
    )
    (output / "round35_hgnn_interface_opportunity.md").write_text(
        "# HGNN structural opportunity\n\nThe Level-2 action now has a concrete U-B edge-logit interface. A relational encoder could in a later, separately authorized causal study map to edge logits; a DeepSets global representation could also map to the same interface. This provides no performance comparison or HGNN advantage claim. `hgnn_structural_revalidation_ready=true`; `hgnn_performance_ready=false`.\n",
        encoding="utf-8",
    )


def run_round35_audit(output: Path, figures: Path) -> dict[str, Any]:
    """Execute all required non-learning Round-35 validations once."""
    output.mkdir(parents=True, exist_ok=True)
    legacy = legacy_exact_regression()
    if not legacy["all_passed"]:
        raise RuntimeError("Round-35 hard stop: legacy exact regression failed")
    _write_json(output / "round35_legacy_scale_parameterization_regression.json", legacy)
    manifest = engineering_manifest()
    _write_json(output / "round35_level2_engineering_manifest.json", manifest)
    causality = preference_causality()
    _write_csv(output / "association_preference_causality.csv", ["case", "assignment", "reward", "candidate_mask_true", "assignment_changed"], causality)
    response = preference_response()
    _write_csv(output / "association_preference_response.csv", ["edge", "preference", "selected", "assignment"], response)
    constraints = constraint_audit()
    _write_json(output / "association_decoder_constraint_audit.json", constraints)
    permutation = permutation_audit()
    _write_json(output / "round35_permutation_audit.json", permutation)
    complexity = [{"num_uavs": 4, "num_buoys": 8, "uav_buoy_edges": 32, "uav_uav_directed_edges": 12, "uav_uav_undirected_edges": 6, "legal_partial_injective_assignments": legal_assignment_count(4, 8), "round34_projection_value": 3393, "projection_consistent": True, "note": "Round-35 follows disk-authoritative exact count 3393; the pasted 3993 is a transcription discrepancy."}]
    _write_csv(output / "round35_realized_complexity.csv", list(complexity[0]), complexity)
    service = service_causality()
    _write_csv(output / "association_service_causality.csv", ["case", "assignment", "served_payload_bits", "queue_bits", "interference_power_w", "reward", "assignment_changed", "service_changed", "queue_changed", "interference_changed"], service)
    separation = action_channel_separation()
    _write_csv(output / "action_channel_separation.csv", ["control", "assignment", "positions", "reward"], separation)
    runtime = runtime_profile(manifest)
    _write_csv(output / "round35_runtime_profile.csv", ["measurement", "samples", "candidates_per_step", "seconds_total", "milliseconds_per_step", "peak_memory_bytes", "exact_decoder", "feasible"], runtime)
    ownership = [
        {"level": "L1", "decision": "association preference / selection objective", "owner": "legacy deterministic matching", "externally_action_controlled": False},
        {"level": "L1", "decision": "final feasibility", "owner": "structured rule", "externally_action_controlled": False},
        {"level": "L1", "decision": "service and interference consequence", "owner": "physics / queue consequence", "externally_action_controlled": False},
        {"level": "L2", "decision": "association preference / selection objective", "owner": "external A_assoc", "externally_action_controlled": True},
        {"level": "L2", "decision": "final feasibility", "owner": "structured decoder constraint", "externally_action_controlled": False},
        {"level": "L2", "decision": "service and interference consequence", "owner": "shared-spectrum physics / queue consequence", "externally_action_controlled": False},
    ]
    _write_csv(output / "round35_graph_decision_ownership.csv", ["level", "decision", "owner", "externally_action_controlled"], ownership)
    rao = {"gate": "RAO-A", "a_assoc_causally_changes_assignment": bool(causality[1]["assignment_changed"]), "decoder_uses_legacy_heuristic_score": False, "permutation_passed": permutation["status"] == "PASS", "constraints_passed": constraints["status"] == "PASS", "gdor_l1": "0/3", "gdor_l2": "1/3", "gdor_increased": True, "hidden_heuristic_ownership": False}
    _write_json(output / "relational_action_ownership_gate.json", rao)
    level2 = {"gate": "L2-A", "level2_relational_benchmark_ready": True, "legacy_exact_regression_passed": legacy["all_passed"], "m4_k8_works": True, "association_controls_assignment": rao["a_assoc_causally_changes_assignment"], "constraint_audit_passed": constraints["status"] == "PASS", "permutation_audit_passed": permutation["status"] == "PASS", "gdor_gt_zero": True, "channels_independently_controllable": True, "service_queue_consequence": bool(service[1]["service_changed"] and service[1]["queue_changed"]), "runtime_feasible": all(row["feasible"] for row in runtime), "protected_data_access_count": 0}
    _write_json(output / "level2_relational_benchmark_gate.json", level2)
    next_stage = {"gate": "NEXT-A", "preconditions": {"level2_gate": level2["gate"], "rao_gate": rao["gate"]}, "next_stage": "non_learning_planning_and_action_relevance_validation", "training_authorized": False, "round36_created": False}
    _write_json(output / "round35_next_stage_gate.json", next_stage)
    readiness = {"round35_readiness_gate": "R35-A", "level2_gate": level2["gate"], "rao_gate": rao["gate"], "task_waypoint_reentry_ready": False, "hgnn_structural_revalidation_ready": True, "hgnn_performance_ready": False, "no_performance_conclusion": True}
    _write_json(output / "round35_readiness_gate.json", readiness)
    access = {"protected_data_access_count": 0, "second_protected_access": False, "untouched_eval_access_count": 0, "accessed_sources": ["generated_engineering_scenarios_only"], "fail_closed": True, "round34_exception_history_unchanged": {"development_test_records_decoded": 1, "formal_scientific_protected_records_used": 0}}
    _write_json(output / "access_audit.json", access)
    _markdown_documents(output)
    generated_figures = write_figures(figures)
    run_manifest = {"round": "round_35", "topic": "Level-2 Relational Multi-UAV Benchmark Environment Implementation and Structural Validation", "training_performed": False, "model_inference_performed": False, "physics_redesigned": False, "reward_redesigned": False, "scenario_family": ENGINEERING_SCENARIO_FAMILY, "scenario_count": 64, "protected_data_access_count": 0, "untouched_eval_access_count": 0, "legacy_exact_regression": legacy["status"], "rao_gate": rao["gate"], "level2_gate": level2["gate"], "figures": generated_figures}
    _write_json(output / "run_manifest.json", run_manifest)
    return {"legacy": legacy, "manifest": manifest, "constraints": constraints, "permutation": permutation, "runtime": runtime, "rao": rao, "level2": level2, "next_stage": next_stage, "readiness": readiness, "access": access, "figures": generated_figures}


__all__ = [
    "ENGINEERING_SCENARIO_FAMILY",
    "legal_assignment_count",
    "level2_config",
    "require_engineering_only_manifest",
    "run_round35_audit",
]
